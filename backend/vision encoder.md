# Qwen3-VL Roofline 分析公式详解

> 本文档基于 `backend/model_analyzer.py` 中 `VLMAnalyzer` 类和 `backend/models/qwen3_vl.py` 整理，
> 系统梳理 Qwen3-VL 模型在 Roofline 模型下的所有计算公式，按 **Vision Encoder → Merger → LLM** 三部分展开。

---

## 符号说明

| 符号 | 含义 | 来源参数 |
|------|------|----------|
| $B$ | batch size，批次大小 | `batchsize` |
| $S$ | 文本序列长度 | `seqlen` |
| $H$ | LLM 隐藏层维度 | `hidden_size` |
| $H_{head}$ | LLM 每个注意力头的维度，$H_{head} = H / N_h$ | 计算得到 |
| $N_h$ | LLM Query 注意力头数量 | `num_attention_heads` |
| $N_{kv}$ | LLM KV 注意力头数量（GQA） | `num_key_value_heads` |
| $L$ | LLM Transformer 层数 | `num_hidden_layers` |
| $I$ | LLM MLP 中间层维度 | `intermediate_size` |
| $V$ | 词表大小 | `vocab_size` |
| $tp$ | 张量并行度 | `tp_size` |
| $H_v$ | Vision Encoder 隐藏层维度 | `vision_hidden_size` |
| $H_{v,head}$ | Vision 每个注意力头维度，$H_{v,head} = H_v / N_{v,h}$ | 计算得到 |
| $N_{v,h}$ | Vision 注意力头数量 | `vision_num_heads` |
| $L_v$ | Vision Encoder Transformer 层数 | `vision_num_hidden_layers` (depth) |
| $I_v$ | Vision MLP 中间层维度 | `vision_intermediate_size` |
| $C$ | 图像输入通道数（RGB=3） | `in_channels` |
| $P$ | Patch 空间大小（如 16，即 16×16 像素） | `patch_size` |
| $T_p$ | Patch 时间维度大小（Conv3d 时间 kernel） | `temporal_patch_size` |
| $S_m$ | 空间合并因子 | `spatial_merge_size` |
| $W_{img}, H_{img}$ | 图像宽度和高度（像素） | `image_size` |
| $N_p$ | 总 patch 数量 | 计算得到 |
| $N_m$ | 空间合并后的 token 数量 | 计算得到 |
| $H_{out,v}$ | Vision 投影输出维度（对齐 LLM） | `out_hidden_size` |
| $H_{merger}$ | Merger 输入维度，$H_{merger} = S_m^2 \times H_v$ | 计算得到 |
| $N_{merger}$ | Merger 总数量，$N_{merger} = 1 + |\text{deepstack\_visual\_indexes}|$ | 计算得到 |
| $w_{byte}$ | 权重每元素字节数，$w_{byte} = w\_bit / 8$ | `w_bit` |
| $a_{byte}$ | 激活值每元素字节数，$a_{byte} = a\_bit / 8$ | `a_bit` |
| $kv_{byte}$ | KV Cache 每元素字节数，$kv_{byte} = kv\_bit / 8$ | `kv_bit` |

---

## Roofline 模型基础

Roofline 模型用于判断一个算子是**内存瓶颈**还是**计算瓶颈**：

$$
\text{算术强度} = \frac{OPs}{\text{内存访问量（bytes）}}
$$

$$
\text{转折点} = \frac{\text{峰值算力（OPS）}}{\text{内存带宽（bytes/s）}}
$$

- 若 $\text{算术强度} < \text{转折点}$：**内存瓶颈**，实际性能 $= \text{算术强度} \times \text{带宽}$
- 若 $\text{算术强度} \geq \text{转折点}$：**计算瓶颈**，实际性能 $= \text{峰值算力}$

$$
\text{推理时间} = \frac{OPs}{\text{实际性能}}
$$

---

## 第一部分：Vision Encoder（视觉编码器）

Vision Encoder 将输入图像转换为一系列视觉 token，供后续 LLM 处理。
整体流程：**图像 → Patch Embedding → Spatial Merge → $L_v$ 层 Transformer → 输出视觉 token**

### 1.1 图像分块（Patch 数量计算）

$$
N_{p,w} = \left\lceil \frac{W_{img}}{P} \right\rceil, \quad N_{p,h} = \left\lceil \frac{H_{img}}{P} \right\rceil, \quad N_p = N_{p,w} \times N_{p,h}
$$

**含义**：将宽 $W_{img}$、高 $H_{img}$ 的图像按 $P \times P$ 像素切分，得到 $N_p$ 个 patch。

**空间合并后 token 数**：

$$
N_m = \left\lceil \frac{N_p}{S_m^2} \right\rceil
$$

**含义**：将相邻 $S_m \times S_m$ 个 patch 合并为 1 个 token，减少序列长度，降低后续计算量。例如 $S_m=2$ 时，4 个 patch 合并为 1 个 token。后续所有 Vision Encoder 的计算都以 $N_m$ 为序列长度。

> 代码位置：[backend/model_analyzer.py:1165-1168](backend/model_analyzer.py#L1165)

---

### 1.2 Patch Embedding（图像块嵌入）

实际使用 `Conv3d(3, 1152, kernel_size=(T_p, P, P), stride=(T_p, P, P))`，将时间+空间维度的像素展平后投影到视觉隐藏空间。

- 输入维度：$patch\_ic = C \times T_p \times P \times P$（如 $3 \times 2 \times 16 \times 16 = 1536$）
- 输出维度：$patch\_oc = H_v$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $C \cdot T_p \cdot P^2 \cdot H_v \cdot B \cdot N_p \times 2$ | Conv3d 等价线性投影，×2 表示乘法+加法 |
| **权重加载** | $C \cdot T_p \cdot P^2 \cdot H_v \cdot w_{byte}$ | 卷积核权重大小，与 token 数无关 |
| **激活加载** | $C \cdot T_p \cdot P^2 \cdot B \cdot N_p \cdot a_{byte}$ | 输入：$B \times N_p$ 个 patch，每个 $C \cdot T_p \cdot P^2$ 维 |
| **激活存储** | $H_v \cdot B \cdot N_p \cdot a_{byte}$ | 输出：$B \times N_p$ 个 embedding，每个 $H_v$ 维 |

**公式解释**：Conv3d 的 kernel 为 $(T_p, P, P)$，每个输出 token 对应 $C \times T_p \times P^2$ 个输入元素的线性组合。$T_p=2$ 表示每次处理 2 帧（视频）或将图像复制 2 份（图像），$N_p$ 是空间合并前的 patch 总数。

> 代码位置：[backend/model_analyzer.py:1171-1184](backend/model_analyzer.py#L1171)

---

### 1.3 Vision Encoder 线性层（Q/K/V/O 投影 + MLP）

每个 Transformer 块包含 6 个线性层，共重复 $L_v$ 次。注意：线性层作用在空间合并后的 $N_m$ 个 token 上。

Vision MLP 使用标准结构（fc1 + GELUTanh + fc2），**不是 SwiGLU**，无 gate_proj。

| 层名 | 输入维 $IC$ | 输出维 $OC$ | 作用 |
|------|------------|------------|------|
| `vision_q_proj` | $H_v$ | $H_v / tp$ | Query 投影 |
| `vision_k_proj` | $H_v$ | $H_v / tp$ | Key 投影 |
| `vision_v_proj` | $H_v$ | $H_v / tp$ | Value 投影 |
| `vision_out_proj` | $H_v / tp$ | $H_v$ | 注意力输出投影 |
| `vision_up_proj` | $H_v$ | $I_v / tp$ | MLP fc1（唯一的升维投影） |
| `vision_down_proj` | $I_v / tp$ | $H_v$ | MLP fc2（降维投影） |

对每个线性层，公式如下：

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $IC \times OC \times B \times N_m \times 2$ | 矩阵乘法，$B \times N_m$ 个 token 并行 |
| **权重加载** | $IC \times OC \times w_{byte}$ | 权重矩阵大小，与 batch/token 数无关 |
| **激活加载** | $IC \times B \times N_m \times a_{byte}$ | 输入激活张量 |
| **激活存储** | $OC \times B \times N_m \times a_{byte}$ | 输出激活张量 |

**公式解释**：线性层 $Y = XW$，$X$ 形状为 $(B \cdot N_m,\ IC)$，$W$ 形状为 $(IC,\ OC)$，每个输出元素需要 $IC$ 次乘法和 $IC$ 次加法，共 $IC \times OC \times B \times N_m \times 2$ 次操作。权重只需加载一次，与 token 数无关，这是 decode 阶段内存瓶颈的根本原因。

> 代码位置：[backend/model_analyzer.py:1185-1195](backend/model_analyzer.py#L1185)

---

### 1.4 Vision Encoder 注意力机制

Vision Encoder 使用标准多头自注意力（无 GQA/MQA），序列长度为 $N_m$。

#### QK 矩阵乘法（$Q \cdot K^T$）

$$
OPs_{qk} = N_m \times N_m \times H_{v,head} \times N_{v,h} \times B \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $N_m^2 \cdot H_{v,head} \cdot N_{v,h} \cdot B \times 2$ | 每个 head 计算 $N_m \times N_m$ 的注意力分数矩阵 |
| **激活加载** | $N_m \cdot H_{v,head} \cdot B \cdot N_{v,h} \cdot a_{byte}$ | 加载 Q 矩阵（形状 $B \times N_{v,h} \times N_m \times H_{v,head}$） |
| **激活存储** | $N_m^2 \cdot B \cdot N_{v,h} \cdot a_{byte}$ | 存储注意力分数矩阵（形状 $B \times N_{v,h} \times N_m \times N_m$） |
| **KV Cache 加载** | 无（Vision Encoder 不使用 KV Cache） | — |

**公式解释**：$Q$ 形状为 $(B, N_{v,h}, N_m, H_{v,head})$，$K^T$ 形状为 $(B, N_{v,h}, H_{v,head}, N_m)$，矩阵乘法结果形状为 $(B, N_{v,h}, N_m, N_m)$，每个元素需要 $H_{v,head}$ 次乘加，故总 OPs = $N_m \times N_m \times H_{v,head} \times N_{v,h} \times B \times 2$。

#### SV 矩阵乘法（$\text{Softmax}(QK^T) \cdot V$）

$$
OPs_{sv} = N_m \times H_{v,head} \times N_m \times N_{v,h} \times B \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $N_m \cdot H_{v,head} \cdot N_m \cdot N_{v,h} \cdot B \times 2$ | 注意力权重矩阵乘以 V |
| **激活加载** | $N_m^2 \cdot B \cdot N_{v,h} \cdot a_{byte}$ | 加载 Softmax 后的注意力权重 |
| **激活存储** | $N_m \cdot H_{v,head} \cdot B \cdot N_{v,h} \cdot a_{byte}$ | 存储注意力输出（形状同 Q） |

**公式解释**：注意力权重形状 $(B, N_{v,h}, N_m, N_m)$ 乘以 $V$ 形状 $(B, N_{v,h}, N_m, H_{v,head})$，结果形状 $(B, N_{v,h}, N_m, H_{v,head})$，每个元素需要 $N_m$ 次乘加。

#### Softmax

$$
OPs_{softmax} = B \times N_{v,h} \times N_m \times N_m \times 5
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_{v,h} \cdot N_m^2 \times 5$ | 5 步：max、sub、exp、sum、div |
| **激活加载** | $B \cdot N_{v,h} \cdot N_m^2 \cdot a_{byte}$ | 加载注意力分数矩阵 |
| **激活存储** | $B \cdot N_{v,h} \cdot N_m^2 \cdot a_{byte}$ | 存储归一化后的注意力权重 |

**公式解释**：Softmax 对每行 $N_m$ 个元素执行 5 步操作（求最大值用于数值稳定、减最大值、取指数、求和、除以和），共 $N_m \times 5$ 次操作，对 $B \times N_{v,h} \times N_m$ 行执行。

> 代码位置：[backend/model_analyzer.py:1199-1253](backend/model_analyzer.py#L1199)

---

### 1.5 Vision Encoder 归一化层（LayerNorm）

Vision Encoder 使用 LayerNorm（而非 LLM 的 RMSNorm），每个 Transformer 块有 2 个：`vision_norm1`（注意力前）和 `vision_norm2`（MLP 前）。

$$
OPs_{layernorm} = B \times H_v \times N_m \times 7
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot H_v \cdot N_m \times 7$ | 7 步：均值、方差、减均值、除标准差、缩放、偏移、输出 |
| **激活加载** | $B \cdot H_v \cdot N_m \cdot a_{byte}$ | 输入激活 |
| **激活存储** | $B \cdot H_v \cdot N_m \cdot a_{byte}$ | 归一化后的激活 |

**公式解释**：LayerNorm 对每个 token 的 $H_v$ 维向量执行归一化，需要 7 步基本操作（均值计算 2 步、方差计算 2 步、归一化 1 步、仿射变换 2 步）。与 RMSNorm（4 步，无均值计算）相比计算量更大。

> 代码位置：[backend/model_analyzer.py:1256-1267](backend/model_analyzer.py#L1256)

---

### 1.6 Vision Encoder 残差连接

每个 Transformer 块有 2 个残差加法：`vision_attn_add`（注意力后）和 `vision_mlp_add`（MLP 后）。

$$
OPs_{add} = B \times H_v \times N_m
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot H_v \cdot N_m$ | 逐元素加法，每个元素 1 次操作 |
| **激活加载** | $B \cdot H_v \cdot N_m \cdot a_{byte}$ | 加载残差输入 |
| **激活存储** | $B \cdot H_v \cdot N_m \cdot a_{byte}$ | 存储残差输出 |

**公式解释**：残差连接 $x = x + \text{sublayer}(x)$，对 $B \times N_m \times H_v$ 个元素各执行 1 次加法。

> 代码位置：[backend/model_analyzer.py:1270-1280](backend/model_analyzer.py#L1270)

---

### 1.7 Vision Encoder MLP 激活函数（GELUTanh）

Vision MLP 使用 `GELUTanh`（即 `GELU(approximate='tanh')`），作用在 fc1 输出上，维度为 $I_v$。

$$
OPs_{act} = B \times I_v \times N_m \times 5
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot I_v \cdot N_m \times 5$ | GELUTanh 近似计算约 5 步 |
| **激活加载** | $B \cdot I_v \cdot N_m \cdot a_{byte}$ | vision_up_proj（fc1）输出，维度 $I_v$ |
| **激活存储** | $B \cdot I_v \cdot N_m \cdot a_{byte}$ | 激活后结果，送入 vision_down_proj（fc2） |

**公式解释**：GELUTanh 使用 tanh 近似：$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left(\sqrt{2/\pi}(x + 0.044715x^3)\right)\right)$，约 5 步基本操作。注意维度是 $I_v=4304$（fc1 输出），而非 $H_v=1152$。

> 代码位置：[backend/model_analyzer.py:1284-1294](backend/model_analyzer.py#L1284)

---

### 1.8 Vision Encoder 层重复汇总

上述 1.3～1.7 中的算子在每个 Transformer 块中重复，共 $L_v$ 层。Patch Embedding（1.2）只执行一次。

$$
\text{总计算量}_{vision} = OPs_{patch\_embed} + L_v \times \sum_{\text{重复层}} OPs_i
$$

重复层集合：`vision_q/k/v/out_proj`、`vision_up/down_proj`、`vision_qk/sv_matmul`、`vision_softmax`、`vision_norm1/2`、`vision_attn/mlp_add`、`vision_mlp_act`

> 代码位置：[backend/model_analyzer.py:1295-1318](backend/model_analyzer.py#L1295)

---

## 第二部分：Merger（视觉-语言对齐投影层）

Merger 将 Vision Encoder 输出的视觉 token 投影到 LLM 的隐藏空间，使视觉 token 可以拼接到文本序列中。

实际结构为 `Qwen3VLVisionPatchMerger`，共包含 **4 个 merger**，分为两种类型：

### 主 merger（1 个）

$$
\text{LayerNorm}(H_v) \;\to\; \text{concat}(S_m^2 \text{ 个 patch}) \;\to\; \text{Linear}(H_{merger}, H_{merger}) \;\to\; \text{GELU} \;\to\; \text{Linear}(H_{merger}, H_{out,v})
$$

- LayerNorm 作用在 concat **之前**，输入维度为 $H_v$（单个 patch 的特征维度）
- concat 将相邻 $S_m^2 = 4$ 个 patch 的特征拼接，维度从 $H_v$ 扩展为 $H_{merger}$
- 随后经过两个线性层和 GELU 激活，输出对齐 LLM 的 $H_{out,v}$ 维度

### deepstack mergers（3 个，对应 ViT 层 8/16/24）

$$
\text{LayerNorm}(H_{merger}) \;\to\; \text{Linear}(H_{merger}, H_{merger}) \;\to\; \text{GELU} \;\to\; \text{Linear}(H_{merger}, H_{out,v})
$$

- LayerNorm 作用在 concat **之后**，输入维度已经是 $H_{merger}$（4 个 patch 拼接后的维度）
- 无独立的 concat 步骤，直接对已合并的特征做投影
- 线性层结构与主 merger 相同

### 两种 merger 的关键区别

| | 主 merger | deepstack merger |
|---|---|---|
| 数量 | 1 个 | 3 个（ViT 层 8/16/24） |
| LayerNorm 输入维度 | $H_v$（concat 前） | $H_{merger}$（concat 后） |
| 是否含独立 concat | 是 | 否 |

### merger 输入维度

$$
H_{merger} = S_m^2 \times H_v = 4 \times 1152 = 4608
$$

**含义**：空间合并将相邻 $S_m \times S_m = 4$ 个 patch 的特征拼接，输入维度从 $H_v$ 扩展为 $H_{merger}$。

### 总 merger 数量

设 `deepstack_visual_indexes` 为 deepstack merger 对应的 ViT 层索引列表（默认为 $[8, 16, 24]$，共 3 个），则：

$$
N_{merger} = 1 + |\,\text{deepstack visual indexes}\,| = 1 + 3 = 4
$$

> 代码位置：[backend/models/qwen3_vl.py:155](backend/models/qwen3_vl.py#L155)，[backend/model_analyzer.py:1321-1327](backend/model_analyzer.py#L1321)

---

### 2.1 LayerNorm（归一化层）

每个 merger 的第一步，对 $H_{merger}$ 维向量做 LayerNorm。

$$
OPs_{ln\_q} = B \times N_m \times H_{merger} \times 7
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_m \cdot H_{merger} \times 7$ | 7 步：均值(2)、方差(2)、归一化(1)、仿射变换(2) |
| **权重加载** | $H_{merger} \cdot w_{byte}$ | scale/bias 参数，大小为 $H_{merger}$ |
| **激活加载** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | 输入激活 |
| **激活存储** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | 归一化后激活 |

**公式解释**：LayerNorm 对每个 token 的向量计算均值和方差，再做归一化和仿射变换，共 7 步基本操作。两种 merger 的 norm 输入维度不同：主 merger 的 norm 在 concat 之前，输入维度为 $H_v=1152$；deepstack mergers 的 norm 在 concat 之后，输入维度为 $H_{merger}=4608$。代码中统一使用 `merger_input_size`（即 $H_{merger}$）计算，对主 merger 是保守估计。

---

### 2.2 fc1（第一个线性层）

$$
OPs_{fc1} = B \times N_m \times H_{merger} \times H_{merger} \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_m \cdot H_{merger}^2 \times 2$ | 方阵乘法，输入输出维度均为 $H_{merger}$ |
| **权重加载** | $H_{merger}^2 \cdot w_{byte}$ | 权重矩阵 $H_{merger} \times H_{merger}$ |
| **激活加载** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | LayerNorm 输出 |
| **激活存储** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | fc1 输出 |

**公式解释**：`Linear(4608, 4608)` 是一个方阵投影，计算量为 $H_{merger}^2 \times 2 \times B \times N_m$。权重矩阵 $4608^2 \approx 21M$ 参数，是 merger 中最大的权重。

---

### 2.3 GELU 激活函数

$$
OPs_{act} = B \times N_m \times H_{merger} \times 5
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_m \cdot H_{merger} \times 5$ | GELU 近似计算约 5 步 |
| **权重加载** | $0$ | 无可学习参数 |
| **激活加载** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | fc1 输出 |
| **激活存储** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | GELU 激活后结果 |

**公式解释**：GELU 激活函数 $\text{GELU}(x) = x \cdot \Phi(x)$，其中 $\Phi$ 为标准正态分布 CDF，实际使用 tanh 近似，约需 5 步基本操作。

---

### 2.4 fc2（第二个线性层，输出投影）

$$
OPs_{fc2} = B \times N_m \times H_{merger} \times H_{out,v} \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_m \cdot H_{merger} \cdot H_{out,v} \times 2$ | 将 $H_{merger}$ 维投影到 LLM 隐藏维度 $H_{out,v}$ |
| **权重加载** | $H_{merger} \cdot H_{out,v} \cdot w_{byte}$ | 权重矩阵 $H_{merger} \times H_{out,v}$ |
| **激活加载** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | GELU 输出 |
| **激活存储** | $B \cdot N_m \cdot H_{out,v} \cdot a_{byte}$ | 投影后视觉 token，维度对齐 LLM（$H_{out,v}=4096$） |

**公式解释**：`Linear(4608, 4096)` 将 merger 特征投影到 LLM 的隐藏维度，输出的 $N_m$ 个视觉 token 会被拼接到文本 token 序列中，作为 LLM prefill 阶段的输入。

---

### 2.5 单个 Merger 总计算量

$$
OPs_{merger} = OPs_{ln\_q} + OPs_{fc1} + OPs_{act} + OPs_{fc2}
$$

$$
= B \cdot N_m \cdot H_{merger} \cdot (7 + 2H_{merger} + 5 + 2H_{out,v})
$$

以 Qwen3-VL-8B 参数代入（$H_{merger}=4608$，$H_{out,v}=4096$）：

$$
= B \cdot N_m \cdot 4608 \cdot (7 + 9216 + 5 + 8192) \approx B \cdot N_m \cdot 4608 \times 17420
$$

---

### 2.6 全部 Merger 总计算量

$$
OPs_{all\_mergers} = N_{merger} \times OPs_{merger} = 4 \times OPs_{merger}
$$

**公式解释**：4 个 merger（1 主 + 3 deepstack）结构相同，计算量直接乘以 4。

---

### 2.7 Merger 内存占用

Merger 层归属于 `vision` stage，其内存统计与 Vision Encoder 合并：

$$
\text{Memory}_{vision} = \text{Weight}_{vision} + \text{TmpAct}_{vision}
$$

其中：
- $\text{Weight}_{vision}$：所有视觉层权重之和（含 4 个 merger 的权重）
- $\text{TmpAct}_{vision}$：所有视觉层临时激活值之和（重复层 × $L_v$，非重复层 × 1）
- Vision Encoder **不使用 KV Cache**，故 $\text{KVCache}_{vision} = 0$

> 代码位置：[backend/model_analyzer.py:1329-1338](backend/model_analyzer.py#L1329)

---

## 第三部分：LLM（语言模型）

LLM 部分与标准 Transformer 解码器相同，分为 **Prefill（预填充）** 和 **Decode（解码）** 两个阶段。
- **Prefill**：一次性处理整个输入序列（文本 token + 视觉 token），序列长度为 $S$
- **Decode**：每次生成 1 个新 token，需要访问历史 KV Cache

每层结构：RMSNorm → Q/K/V 投影 → 注意力 → O 投影 → 残差 → RMSNorm → Gate/Up 投影 → SwiGLU → Down 投影 → 残差，共 $L$ 层。

---

### 3.1 LLM 线性层（Q/K/V/O 投影 + MLP）

每个 Transformer 块包含 7 个线性层，共重复 $L$ 次：

| 层名 | 输入维 $IC$ | 输出维 $OC$ | 作用 |
|------|------------|------------|------|
| `q_proj` | $H$ | $N_h \cdot H_{head} / tp$ | Query 投影 |
| `k_proj` | $H$ | $N_{kv} \cdot H_{head} / tp$ | Key 投影（GQA） |
| `v_proj` | $H$ | $N_{kv} \cdot H_{head} / tp$ | Value 投影（GQA） |
| `out_proj` | $N_h \cdot H_{head} / tp$ | $H$ | 注意力输出投影 |
| `gate_proj` | $H$ | $I / tp$ | SwiGLU gate 分支 |
| `up_proj` | $H$ | $I / tp$ | SwiGLU up 分支 |
| `down_proj` | $I / tp$ | $H$ | MLP 输出投影 |

#### Decode 阶段（每次处理 1 个 token）

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $IC \times OC \times B \times 2$ | 单 token，序列维度为 1 |
| **权重加载** | $IC \times OC \times w_{byte}$ | 每次 decode 都需重新加载权重 |
| **激活加载** | $IC \times B \times a_{byte}$ | 当前 token 的输入激活 |
| **激活存储（非 KV 层）** | $OC \times B \times a_{byte}$ | q/out/gate/up/down_proj 的输出 |
| **KV Cache 存储（KV 层）** | $OC \times B \times kv_{byte}$ | k_proj/v_proj 输出写入 KV Cache |
| **激活存储（KV 层）** | $0$ | KV 层输出直接写入 Cache，不额外存激活 |

**公式解释**：Decode 阶段每次只处理 1 个新 token，序列维度为 1。权重每次都要从 HBM 加载，而计算量极小（仅 $IC \times OC \times 2$），算术强度 $\approx 2B$（极低），几乎总是**内存瓶颈**。这是 LLM decode 阶段的核心性能瓶颈。

#### Prefill 阶段（一次处理 $S$ 个 token）

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $IC \times OC \times B \times S \times 2$ | 整个序列并行计算 |
| **权重加载** | $IC \times OC \times w_{byte}$ | 权重只加载一次（与 decode 相同） |
| **激活加载** | $IC \times B \times S \times a_{byte}$ | 整个序列的输入激活 |
| **激活存储（非 KV 层）** | $OC \times B \times S \times a_{byte}$ | 整个序列的输出激活 |
| **KV Cache 存储（KV 层）** | $OC \times B \times S \times kv_{byte}$ | 整个序列的 KV 写入 Cache |

**公式解释**：Prefill 阶段序列长度为 $S$，计算量是 decode 的 $S$ 倍，而权重加载量不变，算术强度 $\approx 2S$，$S$ 足够大时容易达到**计算瓶颈**。

> 代码位置：[backend/model_analyzer.py:868-893](backend/model_analyzer.py#L868)

---

### 3.2 LLM 注意力机制

LLM 使用 GQA（Grouped Query Attention），$N_h$ 个 Query head 共享 $N_{kv}$ 个 KV head。令 $N_h' = \max(1, N_h/tp)$，$N_{kv}' = \max(1, N_{kv}/tp)$。

#### Decode 阶段注意力

**QK 矩阵乘法**（当前 token 的 Q 与历史所有 K 做点积）：

$$
OPs_{qk} = S \times H_{head} \times N_h' \times B \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $S \cdot H_{head} \cdot N_h' \cdot B \times 2$ | 当前 Q（1 token）与历史 $S$ 个 K 做点积 |
| **激活加载** | $1 \cdot H_{head} \cdot B \cdot N_h' \cdot a_{byte}$ | 当前 token 的 Q |
| **激活存储** | $1 \cdot S \cdot B \cdot N_h' \cdot a_{byte}$ | 注意力分数向量（长度 $S$） |
| **KV Cache 加载** | $S \cdot H_{head} \cdot B \cdot N_{kv}' \cdot kv_{byte}$ | 从 Cache 加载历史 K |

**SV 矩阵乘法**（注意力权重与 V 加权求和）：

$$
OPs_{sv} = 1 \times H_{head} \times S \times N_h' \times B \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $H_{head} \cdot S \cdot N_h' \cdot B \times 2$ | 注意力权重（长度 $S$）与 V 加权求和 |
| **激活加载** | $S \cdot B \cdot N_h' \cdot a_{byte}$ | Softmax 后的注意力权重 |
| **激活存储** | $H_{head} \cdot B \cdot N_h' \cdot a_{byte}$ | 注意力输出（1 token） |
| **KV Cache 加载** | $S \cdot H_{head} \cdot B \cdot N_{kv}' \cdot kv_{byte}$ | 从 Cache 加载历史 V |

**Softmax**：

$$
OPs_{softmax} = B \times N_h' \times S \times 5
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_h' \cdot S \times 5$ | 对长度 $S$ 的注意力分数做 Softmax（5 步） |
| **激活加载/存储** | $B \cdot N_h' \cdot S \cdot a_{byte}$ | 各一次 |

**公式解释**：Decode 阶段 Q 只有 1 个 token，但需要与历史 $S$ 个 K/V 做计算，KV Cache 的加载量随序列长度线性增长，是 decode 阶段内存访问的主要来源之一。

> 代码位置：[backend/model_analyzer.py:900-958](backend/model_analyzer.py#L900)

---

#### Prefill 阶段注意力

**QK 矩阵乘法**（序列内所有 token 两两计算注意力）：

$$
OPs_{qk} = S \times S \times H_{head} \times N_h' \times B \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $S^2 \cdot H_{head} \cdot N_h' \cdot B \times 2$ | $S$ 个 Q 与 $S$ 个 K 两两点积 |
| **激活加载** | $S \cdot H_{head} \cdot B \cdot N_{kv}' \cdot a_{byte}$ | 加载 Q |
| **激活存储** | $S^2 \cdot B \cdot N_h' \cdot a_{byte}$ | 注意力分数矩阵 $S \times S$ |
| **KV Cache 加载** | $S \cdot H_{head} \cdot B \cdot N_{kv}' \cdot kv_{byte}$ | 加载 K |

**SV 矩阵乘法**：

$$
OPs_{sv} = S \times H_{head} \times S \times N_h' \times B \times 2
$$

**Softmax**：

$$
OPs_{softmax} = B \times N_h' \times S \times S \times 5
$$

**公式解释**：Prefill 注意力计算量为 $O(S^2)$，是整个推理中计算量最密集的部分，通常是**计算瓶颈**。

> 代码位置：[backend/model_analyzer.py:1001-1058](backend/model_analyzer.py#L1001)

---

### 3.3 LLM 归一化层（RMSNorm）

LLM 使用 RMSNorm，每个 Transformer 块有 2 个：`attn_norm`（注意力前）和 `mlp_norm`（MLP 前）。

#### Decode 阶段

$$
OPs_{rmsnorm} = B \times (H / tp) \times 4
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot (H/tp) \times 4$ | 4 步：平方求和、除以 $H$、开方、逐元素除法+缩放 |
| **激活加载** | $B \cdot (H/tp) \cdot a_{byte}$ | 输入激活（1 token） |
| **激活存储** | $B \cdot (H/tp) \cdot a_{byte}$ | 归一化后激活 |

#### Prefill 阶段

$$
OPs_{rmsnorm} = B \times (H / tp) \times S \times 4
$$

**公式解释**：RMSNorm 计算 $y = \frac{x}{\text{RMS}(x)} \cdot \gamma$，其中 $\text{RMS}(x) = \sqrt{\frac{1}{H}\sum x_i^2}$，共 4 步。相比 LayerNorm（7 步）省去了均值计算，计算量更小。

> 代码位置：[backend/model_analyzer.py:961-973](backend/model_analyzer.py#L961)（decode），[backend/model_analyzer.py:1061-1072](backend/model_analyzer.py#L1061)（prefill）

---

### 3.4 LLM 残差连接

每个 Transformer 块有 2 个残差加法：`attn_add` 和 `mlp_add`。

#### Decode 阶段

$$
OPs_{add} = B \times (H / tp)
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot (H/tp)$ | 逐元素加法，1 token，每元素 1 次操作 |
| **激活加载** | $B \cdot (H/tp) \cdot a_{byte}$ | 残差输入 |
| **激活存储** | $B \cdot (H/tp) \cdot a_{byte}$ | 残差输出 |

#### Prefill 阶段

$$
OPs_{add} = B \times (H / tp) \times S
$$

> 代码位置：[backend/model_analyzer.py:976-986](backend/model_analyzer.py#L976)（decode），[backend/model_analyzer.py:1074-1084](backend/model_analyzer.py#L1074)（prefill）

---

### 3.5 LLM MLP 激活函数（SwiGLU）

#### Decode 阶段

$$
OPs_{act} = B \times (H / tp) \times 5
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot (H/tp) \times 5$ | Swish 激活 5 步：sigmoid(4步) + 乘法(1步) |
| **激活加载** | $B \cdot (H/tp) \cdot a_{byte}$ | gate_proj 输出 |
| **激活存储** | $B \cdot (H/tp) \cdot a_{byte}$ | 激活后结果（与 up_proj 输出逐元素相乘） |

#### Prefill 阶段

$$
OPs_{act} = B \times (H / tp) \times S \times 5
$$

**公式解释**：SwiGLU 激活 $\text{SwiGLU}(x, g) = x \cdot \text{Swish}(g) = x \cdot g \cdot \sigma(g)$，Swish 需要 sigmoid（约 4 步）+ 乘法（1 步）共 5 步。

> 代码位置：[backend/model_analyzer.py:988-998](backend/model_analyzer.py#L988)（decode），[backend/model_analyzer.py:1086-1096](backend/model_analyzer.py#L1086)（prefill）

---

### 3.6 LM Head（语言模型输出头）

将最后一层隐藏状态映射到词表概率分布。

#### Decode 阶段

$$
OPs_{lm\_head} = B \times H \times V \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot H \cdot V \times 2$ | 线性层，输出词表大小 $V$ |
| **权重加载** | $H \cdot V \cdot w_{byte}$ | 词表投影矩阵 |
| **激活加载** | $B \cdot H \cdot a_{byte}$ | 最后一层输出（1 token） |
| **激活存储** | $B \cdot V \cdot a_{byte}$ | logits 向量 |

#### Prefill 阶段

$$
OPs_{lm\_head} = B \times S \times H \times V \times 2
$$

**公式解释**：LM Head 是 $H \to V$ 的线性层，词表 $V$ 通常很大（如 32000+），权重矩阵 $H \times V$ 占用大量内存，是 decode 阶段内存访问的重要来源。

> 代码位置：[backend/models/qwen3_vl.py:49-70](backend/models/qwen3_vl.py#L49)

---

### 3.7 LLM 内存占用

#### Decode 阶段

$$
\text{Memory}_{decode} = \text{Weight}_{total} + \text{KVCache}_{total} + \text{TmpAct}_{decode}
$$

- $\text{Weight}_{total}$：所有层权重之和（乘以 $L$ 层）
- $\text{KVCache}_{total}$：prefill 阶段写入的 KV Cache 总量
- $\text{TmpAct}_{decode}$：decode 阶段所有层临时激活值之和

#### Prefill 阶段

$$
\text{Memory}_{prefill} = \text{Weight}_{total} + \text{KVCache}_{total} + \text{TmpAct}_{prefill}
$$

> 代码位置：[backend/model_analyzer.py:1110-1123](backend/model_analyzer.py#L1110)

---

## 第四部分：多模态整体统计

### 4.1 TTFT（Time To First Token，首 token 时延）

TTFT 包含视觉编码和文本 prefill 两个阶段的总开销：

$$
\text{TTFT}_{OPs} = \text{Vision}_{OPs} + \text{Prefill}_{OPs}
$$

$$
\text{Memory}_{TTFT} = \text{Weight}_{vision} + \text{Weight}_{LLM} + \max(\text{TmpAct}_{vision},\ \text{TmpAct}_{prefill}) + \text{KVCache}_{prefill}
$$

**公式解释**：
- 计算量直接相加，视觉编码和文本 prefill 串行执行
- 权重内存相加，两部分权重需同时驻留显存
- 临时激活值取**最大值**而非相加：视觉编码完成后其激活值可释放，再执行文本 prefill，两者不会同时占用内存
- KV Cache 来自文本 prefill 阶段

> 代码位置：[backend/model_analyzer.py:1341-1366](backend/model_analyzer.py#L1341)

---

### 4.2 TPOT（Time Per Output Token，每 token 生成时延）

TPOT 只包含文本 decode 阶段的开销：

$$
\text{TPOT}_{OPs} = \text{Decode}_{OPs}
$$

$$
\text{Memory}_{TPOT} = \text{Memory}_{decode}
$$

**公式解释**：生成每个新 token 时，视觉编码器不再参与，只有 LLM decode 阶段在运行。

> 代码位置：[backend/model_analyzer.py:1349-1372](backend/model_analyzer.py#L1349)

---

## 附录 A：Flash Attention 优化

当 `use_flashattention=True` 时，QK、SV、Softmax 三个算子融合为一个 `fused_attention`，通过分块计算减少 HBM 访问。

### 分块参数

$$
\text{block\_size\_r} = \min\!\left(\left\lceil \frac{\text{OnChip Buffer}}{kv_{byte} \times H_{head}} \right\rceil,\ H_{head}\right), \quad n\_blocks\_r = \left\lceil \frac{S}{\text{block\_size\_r}} \right\rceil
$$

**含义**：根据片上缓存大小确定每次能处理的 token 块大小，$n\_blocks\_r$ 是需要分多少块处理。

### Decode 阶段 Flash Attention

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $OPs_{qk} + OPs_{sv} + OPs_{softmax}$（同标准注意力） | 计算量不变 |
| **激活加载** | $1 \cdot H_{head} \cdot B \cdot N_h' \cdot a_{byte}$ | 只加载 Q |
| **激活存储** | $1 \cdot S \cdot B \cdot N_h' \cdot a_{byte} \times 2$ | 初始化 O + 保存 O |
| **KV Cache 加载** | $n\_blocks\_r \cdot S \cdot H_{head} \cdot B \cdot N_{kv}' \cdot kv_{byte} \times 2 / \text{kv\_reuse}$ | 分块加载 K 和 V |

**GQA KV 复用因子**：

$$
\text{kv\_reuse} = \min\!\left(\frac{N_h'}{N_{kv}'},\ \max\!\left(1,\ \frac{\text{block\_size\_r}}{H_{head}}\right)\right)
$$

**公式解释**：Flash Attention 将 K/V 分块加载到片上缓存，避免将完整 $S \times S$ 注意力矩阵写回 HBM。GQA 时多个 Q head 共享同一组 K/V，可进一步减少 KV Cache 加载次数。

> 代码位置：[backend/model_analyzer.py:903-926](backend/model_analyzer.py#L903)（decode），[backend/model_analyzer.py:1004-1026](backend/model_analyzer.py#L1004)（prefill）

---

## 附录 B：Roofline 判断逻辑汇总

| 算子类型 | 典型算术强度 | 常见瓶颈 | 原因 |
|----------|------------|---------|------|
| Decode 线性层 | $\approx 2B$（极低） | 内存瓶颈 | 每次只处理 1 token，权重反复从 HBM 加载 |
| Prefill 线性层 | $\approx 2S$（中等） | 计算瓶颈（$S$ 大时） | 序列长时计算量大，权重复用率高 |
| Decode 注意力 | 低 | 内存瓶颈 | KV Cache 随序列长度线性增长 |
| Prefill 注意力 | $O(S)$（高） | 计算瓶颈 | $S^2$ 计算量，内存访问相对少 |
| 归一化/残差/激活 | 极低 | 内存瓶颈 | 计算量小，内存访问量相对大 |
| Vision 线性层 | 中等（$\approx 2N_m$） | 计算瓶颈（$N_m$ 大时） | 类似 LLM prefill，$N_m$ 个 token 并行 |
| Vision 注意力 | $O(N_m)$（高） | 计算瓶颈 | $N_m^2$ 计算量 |
