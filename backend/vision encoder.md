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

每个 Transformer 块包含 6 个线性层，共重复 $L_v$ 次。Vision MLP 使用标准结构（fc1 + GELUTanh + fc2），**不是 SwiGLU**，无 gate_proj。

| 层名 | OPs 公式 | 权重加载 | 激活加载 | 激活存储 |
|------|----------|----------|----------|----------|
| `vision_q_proj` | $H_v \cdot (H_v/tp) \cdot B \cdot N_m \times 2$ | $H_v \cdot (H_v/tp) \cdot w_{byte}$ | $H_v \cdot B \cdot N_m \cdot a_{byte}$ | $(H_v/tp) \cdot B \cdot N_m \cdot a_{byte}$ |
| `vision_k_proj` | $H_v \cdot (H_v/tp) \cdot B \cdot N_m \times 2$ | $H_v \cdot (H_v/tp) \cdot w_{byte}$ | $H_v \cdot B \cdot N_m \cdot a_{byte}$ | $(H_v/tp) \cdot B \cdot N_m \cdot a_{byte}$ |
| `vision_v_proj` | $H_v \cdot (H_v/tp) \cdot B \cdot N_m \times 2$ | $H_v \cdot (H_v/tp) \cdot w_{byte}$ | $H_v \cdot B \cdot N_m \cdot a_{byte}$ | $(H_v/tp) \cdot B \cdot N_m \cdot a_{byte}$ |
| `vision_out_proj` | $(H_v/tp) \cdot H_v \cdot B \cdot N_m \times 2$ | $(H_v/tp) \cdot H_v \cdot w_{byte}$ | $(H_v/tp) \cdot B \cdot N_m \cdot a_{byte}$ | $H_v \cdot B \cdot N_m \cdot a_{byte}$ |
| `vision_up_proj`（fc1） | $H_v \cdot (I_v/tp) \cdot B \cdot N_m \times 2$ | $H_v \cdot (I_v/tp) \cdot w_{byte}$ | $H_v \cdot B \cdot N_m \cdot a_{byte}$ | $(I_v/tp) \cdot B \cdot N_m \cdot a_{byte}$ |
| `vision_down_proj`（fc2） | $(I_v/tp) \cdot H_v \cdot B \cdot N_m \times 2$ | $(I_v/tp) \cdot H_v \cdot w_{byte}$ | $(I_v/tp) \cdot B \cdot N_m \cdot a_{byte}$ | $H_v \cdot B \cdot N_m \cdot a_{byte}$ |

**公式解释**：线性层 $Y = XW$，每个输出元素需要 $IC$ 次乘法和 $IC$ 次加法，共 $IC \times OC \times B \times N_m \times 2$ 次操作。权重只需加载一次，与 token 数无关，这是 decode 阶段内存瓶颈的根本原因。

> 代码位置：[backend/model_analyzer.py:1186-1197](backend/model_analyzer.py#L1186)，[backend/models/qwen3_vl.py:136-152](backend/models/qwen3_vl.py#L136)

---

### 1.4 Vision Encoder 注意力机制

Vision Encoder 使用标准多头自注意力（无 GQA/MQA），序列长度为 $N_m$，每个 Transformer 块重复 $L_v$ 次。

| 算子 | OPs 公式 | 激活加载 | 激活存储 |
|------|----------|----------|----------|
| `vision_qk_matmul` | $N_m^2 \cdot H_{v,head} \cdot N_{v,h} \cdot B \times 2$ | $N_m \cdot H_{v,head} \cdot N_{v,h} \cdot B \cdot a_{byte}$（Q） | $N_m^2 \cdot N_{v,h} \cdot B \cdot a_{byte}$ |
| `vision_softmax` | $B \cdot N_{v,h} \cdot N_m^2 \times 5$ | $N_m^2 \cdot N_{v,h} \cdot B \cdot a_{byte}$ | $N_m^2 \cdot N_{v,h} \cdot B \cdot a_{byte}$ |
| `vision_sv_matmul` | $N_m^2 \cdot H_{v,head} \cdot N_{v,h} \cdot B \times 2$ | $N_m^2 \cdot N_{v,h} \cdot B \cdot a_{byte}$（attn weights） | $N_m \cdot H_{v,head} \cdot N_{v,h} \cdot B \cdot a_{byte}$ |

**说明**：
- `vision_qk_matmul`：$Q(B, N_{v,h}, N_m, H_{v,head}) \times K^T(B, N_{v,h}, H_{v,head}, N_m)$，每个输出元素需 $H_{v,head}$ 次乘加
- `vision_softmax`：对每行 $N_m$ 个元素执行 5 步（max、sub、exp、sum、div）
- `vision_sv_matmul`：注意力权重 $(B, N_{v,h}, N_m, N_m) \times V(B, N_{v,h}, N_m, H_{v,head})$，每个输出元素需 $N_m$ 次乘加
- Vision Encoder **不使用 KV Cache**，无 KV Cache 加载

> 代码位置：[backend/model_analyzer.py:1200-1255](backend/model_analyzer.py#L1200)

---

### 1.5 Vision Encoder 归一化层（LayerNorm）

每个 Transformer 块有 2 个：`vision_norm1`（注意力前）和 `vision_norm2`（MLP 前），共重复 $L_v$ 次。

| 算子 | OPs 公式 | 激活加载 | 激活存储 |
|------|----------|----------|----------|
| `vision_norm1` / `vision_norm2` | $B \cdot H_v \cdot N_m \times 7$ | $B \cdot H_v \cdot N_m \cdot a_{byte}$ | $B \cdot H_v \cdot N_m \cdot a_{byte}$ |

**说明**：LayerNorm 对每个 token 的 $H_v$ 维向量执行 7 步操作（均值 2 步、方差 2 步、归一化 1 步、仿射变换 2 步）。与 LLM 的 RMSNorm（4 步）相比多了均值计算。

> 代码位置：[backend/model_analyzer.py:1257-1269](backend/model_analyzer.py#L1257)

---

### 1.6 Vision Encoder 残差连接

每个 Transformer 块有 2 个：`vision_attn_add`（注意力后）和 `vision_mlp_add`（MLP 后），共重复 $L_v$ 次。

| 算子 | OPs 公式 | 激活加载 | 激活存储 |
|------|----------|----------|----------|
| `vision_attn_add` / `vision_mlp_add` | $B \cdot H_v \cdot N_m$ | $B \cdot H_v \cdot N_m \cdot a_{byte}$ | $B \cdot H_v \cdot N_m \cdot a_{byte}$ |

**说明**：逐元素加法 $x = x + \text{sublayer}(x)$，每个元素 1 次操作，无权重。

> 代码位置：[backend/model_analyzer.py:1271-1282](backend/model_analyzer.py#L1271)

---

### 1.7 Vision Encoder MLP 激活函数（GELUTanh）

作用在 `vision_up_proj`（fc1）输出上，维度为 $I_v$（不是 $H_v$），重复 $L_v$ 次。

| 算子 | OPs 公式 | 激活加载 | 激活存储 |
|------|----------|----------|----------|
| `vision_mlp_act` | $B \cdot I_v \cdot N_m \times 5$ | $B \cdot I_v \cdot N_m \cdot a_{byte}$ | $B \cdot I_v \cdot N_m \cdot a_{byte}$ |

**说明**：$\text{GELU}(x) \approx 0.5x\!\left(1 + \tanh\!\left(\sqrt{2/\pi}(x + 0.044715x^3)\right)\right)$，约 5 步。注意维度是 $I_v$（如 4304），不是 $H_v$（1152）。

> 代码位置：[backend/model_analyzer.py:1284-1294](backend/model_analyzer.py#L1284)

---

### 1.8 Vision Encoder 层重复汇总

上述 1.3～1.7 中的算子在每个 Transformer 块中重复，共 $L_v$ 层。Patch Embedding（1.2）只执行一次。

$$
\text{总计算量}_{vision} = OPs_{patch\_embed} + L_v \times \sum_{\text{重复层}} OPs_i
$$

**所有算子 OPs 公式一览**（重复层均乘以 $L_v$）：

| 算子 | OPs 公式 | 重复 |
|------|----------|------|
| `vision_patch_embed` | $C \cdot T_p \cdot P^2 \cdot H_v \cdot B \cdot N_p \times 2$ | ×1 |
| `vision_norm1` / `vision_norm2` | $B \cdot H_v \cdot N_m \times 7$ | ×$L_v$ |
| `vision_q_proj` | $H_v \cdot (H_v/tp) \cdot B \cdot N_m \times 2$ | ×$L_v$ |
| `vision_k_proj` | $H_v \cdot (H_v/tp) \cdot B \cdot N_m \times 2$ | ×$L_v$ |
| `vision_v_proj` | $H_v \cdot (H_v/tp) \cdot B \cdot N_m \times 2$ | ×$L_v$ |
| `vision_qk_matmul` | $N_m^2 \cdot H_{v,head} \cdot N_{v,h} \cdot B \times 2$ | ×$L_v$ |
| `vision_softmax` | $B \cdot N_{v,h} \cdot N_m^2 \times 5$ | ×$L_v$ |
| `vision_sv_matmul` | $N_m \cdot H_{v,head} \cdot N_m \cdot N_{v,h} \cdot B \times 2$ | ×$L_v$ |
| `vision_out_proj` | $(H_v/tp) \cdot H_v \cdot B \cdot N_m \times 2$ | ×$L_v$ |
| `vision_attn_add` | $B \cdot H_v \cdot N_m$ | ×$L_v$ |
| `vision_up_proj`（fc1） | $H_v \cdot (I_v/tp) \cdot B \cdot N_m \times 2$ | ×$L_v$ |
| `vision_mlp_act`（GELUTanh） | $B \cdot I_v \cdot N_m \times 5$ | ×$L_v$ |
| `vision_down_proj`（fc2） | $(I_v/tp) \cdot H_v \cdot B \cdot N_m \times 2$ | ×$L_v$ |
| `vision_mlp_add` | $B \cdot H_v \cdot N_m$ | ×$L_v$ |

> 代码位置：[backend/model_analyzer.py:1186-1318](backend/model_analyzer.py#L1186)，[backend/models/qwen3_vl.py:136-152](backend/models/qwen3_vl.py#L136)

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

4 个 merger 的 LayerNorm 在代码中统一使用 $H_{merger}$ 计算（`merger_input_size`）：

$$
OPs_{ln} = B \times N_m \times H_{merger} \times 7
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_m \cdot H_{merger} \times 7$ | 7 步：均值(2)、方差(2)、归一化(1)、仿射变换(2) |
| **权重加载** | $H_{merger} \cdot w_{byte}$ | scale/bias 参数，大小为 $H_{merger}$ |
| **激活加载** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | 输入激活 |
| **激活存储** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | 归一化后激活 |

**公式解释**：从模型架构上，主 merger 的 LayerNorm 作用在 concat 之前（维度 $H_v=1152$），deepstack mergers 的 LayerNorm 作用在 concat 之后（维度 $H_{merger}=4608$）。但代码实现中对所有 merger 统一使用 `merger_input_size`（即 $H_{merger}$）计算，对主 merger 是保守估计（高估了 $S_m^2=4$ 倍）。

> 代码位置：[backend/models/qwen3_vl.py:174-181](backend/models/qwen3_vl.py#L174)

---

### 2.2 fc1（第一个线性层）

$$
OPs_{fc1} = B \times N_m \times H_{merger}^2 \times 2
$$

| 指标 | 公式 | 说明 |
|------|------|------|
| **计算量 OPs** | $B \cdot N_m \cdot H_{merger}^2 \times 2$ | 方阵乘法，输入输出维度均为 $H_{merger}$ |
| **权重加载** | $H_{merger}^2 \cdot w_{byte}$ | 权重矩阵 $H_{merger} \times H_{merger}$ |
| **激活加载** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | LayerNorm 输出 |
| **激活存储** | $B \cdot N_m \cdot H_{merger} \cdot a_{byte}$ | fc1 输出 |

**公式解释**：`Linear(4608, 4608)` 是一个方阵投影，权重矩阵 $4608^2 \approx 21M$ 参数，是 merger 中最大的权重。

> 代码位置：[backend/models/qwen3_vl.py:183-190](backend/models/qwen3_vl.py#L183)

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

**公式解释**：GELU 激活函数 $\text{GELU}(x) = x \cdot \Phi(x)$，实际使用 tanh 近似，约需 5 步基本操作。

> 代码位置：[backend/models/qwen3_vl.py:192-199](backend/models/qwen3_vl.py#L192)

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

> 代码位置：[backend/models/qwen3_vl.py:201-208](backend/models/qwen3_vl.py#L201)

---

### 2.5 单个 Merger 总计算量

代码中 4 个 merger 使用相同公式（统一以 $H_{merger}$ 计算 LayerNorm）：

$$
OPs_{merger} = OPs_{ln} + OPs_{fc1} + OPs_{act} + OPs_{fc2}
$$

$$
= B \cdot N_m \cdot H_{merger} \cdot (7 + 2H_{merger} + 5 + 2H_{out,v})
$$

以 Qwen3-VL-8B 参数代入（$H_{merger}=4608$，$H_{out,v}=4096$）：

$$
= B \cdot N_m \cdot 4608 \times (7 + 9216 + 5 + 8192) \approx B \cdot N_m \times 80{,}254{,}464
$$

其中各项占比：LayerNorm（$\times 7$）$\approx 0.04\%$，fc1（$\times 9216$）$\approx 52.9\%$，GELU（$\times 5$）$\approx 0.03\%$，fc2（$\times 8192$）$\approx 47.0\%$。**fc1 和 fc2 主导计算量。**

---

### 2.6 全部 Merger 总计算量

$$
OPs_{all} = N_{merger} \times OPs_{merger} = 4 \times OPs_{merger}
$$

$$
\approx B \cdot N_m \times 321{,}017{,}856
$$

**公式解释**：代码对 4 个 merger 循环执行相同计算（[backend/models/qwen3_vl.py:171](backend/models/qwen3_vl.py#L171)），计算量直接乘以 $N_{merger}=4$。架构上主 merger 的 LayerNorm 输入维度为 $H_v$（concat 前），代码统一用 $H_{merger}$ 是保守估计，误差约 $3 \times H_v \times 7 \times B \times N_m$，占总量不足 $0.01\%$，可忽略。

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

---

## 第五部分：端到端推理时间计算示例

> 本节以 **Qwen3-VL-32B-Instruct，1024×1024 图像，Batchsize=1** 为例，完整演示如何从模型参数和硬件规格出发，逐步计算出 Vision Stage、Prefill、Decode 三个阶段的 `inference_time`。

---

### 5.1 示例配置与参数来源

#### 输入配置

| 参数 | 值 | 来源 |
|------|-----|------|
| 模型 | Qwen3-VL-32B-Instruct | 用户选择 |
| 图像尺寸 $W_{img} \times H_{img}$ | $1024 \times 1024$ | `image_size` |
| Batchsize $B$ | $1$ | `batchsize` |
| 文本序列长度 $S_{text}$ | $128$（示例提示词） | `seqlen` |
| 精度 | BF16（$w\_bit=16, a\_bit=16, kv\_bit=16$） | 默认 |
| 张量并行 $tp$ | $1$ | 默认 |
| 硬件 | NVIDIA H100 SXM | 用户选择 |

#### 从 config.json 读取的模型参数

**LLM（text_config）：**

| 符号 | 值 | 字段名 |
|------|-----|--------|
| $H$ | 5120 | `hidden_size` |
| $H_{head}$ | 128 | `head_dim` |
| $N_h$ | 64 | `num_attention_heads` |
| $N_{kv}$ | 8 | `num_key_value_heads` |
| $L$ | 64 | `num_hidden_layers` |
| $I$ | 25600 | `intermediate_size` |
| $V$ | 151936 | `vocab_size` |

**Vision（vision_config）：**

| 符号 | 值 | 字段名 |
|------|-----|--------|
| $H_v$ | 1152 | `hidden_size` |
| $N_{v,h}$ | 16 | `num_heads` |
| $H_{v,head}$ | $1152/16=72$ | 计算得到 |
| $L_v$ | 27 | `depth` |
| $I_v$ | 4304 | `intermediate_size` |
| $C$ | 3 | `in_channels` |
| $P$ | 16 | `patch_size` |
| $T_p$ | 2 | `temporal_patch_size` |
| $S_m$ | 2 | `spatial_merge_size` |
| $H_{out,v}$ | 5120 | `out_hidden_size` |
| $H_{merger}$ | $2^2 \times 1152 = 4608$ | 计算得到 |
| $N_{merger}$ | $1+3=4$ | `deepstack_visual_indexes` 长度+1 |

#### 从 hardwares.py 读取的硬件参数

| 参数 | 值 |
|------|-----|
| 内存带宽 | $3072\ \text{GB/s} = 3.072 \times 10^{12}\ \text{bytes/s}$ |
| 峰值算力（FP16） | $989.5\ \text{TFLOPS} = 9.895 \times 10^{14}\ \text{OPS}$ |
| 转折点 | $989.5\text{T} / 3.072\text{T} \approx 322\ \text{OPs/byte}$ |

字节数：$w_{byte} = a_{byte} = kv_{byte} = 2$

---

### 5.2 关键中间量推导

**Patch 数量：**

$$N_{p,w} = \left\lfloor \frac{1024}{16} \right\rfloor = 64, \quad N_{p,h} = 64, \quad N_p = 64 \times 64 = 4096$$

**空间合并后 token 数：**

$$N_m = \frac{N_p}{S_m^2} = \frac{4096}{4} = 1024$$

**LLM Prefill 序列长度：**

视觉 token 经 Merger 投影后拼接到文本序列，LLM 看到的总序列长度为：

$$S = S_{text} + N_m = 128 + 1024 = 1152$$

---

### 5.3 Roofline 判断逻辑（每个算子通用）

对每个算子，按以下步骤计算推理时间：

$$\text{算术强度（AI）} = \frac{OPs}{\text{内存访问量（bytes）}}$$

$$\text{实际性能} = \begin{cases} AI \times \text{带宽} & \text{若 } AI < 322\ \text{（内存瓶颈）} \\ 989.5\ \text{TFLOPS} & \text{若 } AI \geq 322\ \text{（计算瓶颈）} \end{cases}$$

$$\text{算子推理时间} = \frac{OPs}{\text{实际性能}}$$

**整个阶段的推理时间 = 该阶段所有算子推理时间之和。**

---

### 5.4 Vision Stage 推理时间计算

#### 5.4.1 Patch Embedding（执行 1 次）

$$OPs = C \cdot T_p \cdot P^2 \cdot H_v \cdot B \cdot N_p \times 2 = 3 \times 2 \times 256 \times 1152 \times 1 \times 4096 \times 2 \approx 14.5\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{C \cdot T_p \cdot P^2 \cdot H_v \cdot w_{byte}}_{\text{权重 3.4 MB}} + \underbrace{C \cdot T_p \cdot P^2 \cdot B \cdot N_p \cdot a_{byte}}_{\text{输入激活 12 MB}} + \underbrace{H_v \cdot B \cdot N_p \cdot a_{byte}}_{\text{输出激活 9 MB}} \approx 24.4\ \text{MB}$$

$$AI = \frac{14.5\ \text{G}}{24.4\ \text{M}} \approx 594\ \text{OPs/byte} > 322 \Rightarrow \text{计算瓶颈}$$

$$t_{patch\_embed} = \frac{14.5\ \text{G}}{989.5\ \text{T}} \approx 14.7\ \mu s$$

#### 5.4.2 Vision Encoder 每层代表性算子（重复 $L_v=27$ 次）

以下展示每层中计算量最大的算子：

**vision\_up\_proj（fc1，$H_v \to I_v$）：**

$$OPs = H_v \cdot I_v \cdot B \cdot N_m \times 2 = 1152 \times 4304 \times 1 \times 1024 \times 2 \approx 10.16\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{1152 \times 4304 \times 2}_{\text{权重 9.46 MB}} + \underbrace{1152 \times 1024 \times 2}_{\text{输入 2.25 MB}} + \underbrace{4304 \times 1024 \times 2}_{\text{输出 8.41 MB}} \approx 20.1\ \text{MB}$$

$$AI = \frac{10.16\ \text{G}}{20.1\ \text{M}} \approx 505\ \text{OPs/byte} > 322 \Rightarrow \text{计算瓶颈}$$

$$t_{up\_proj} = \frac{10.16\ \text{G}}{989.5\ \text{T}} \approx 10.3\ \mu s$$

**vision\_qk\_matmul（注意力 QK 点积）：**

$$OPs = N_m^2 \cdot H_{v,head} \cdot N_{v,h} \cdot B \times 2 = 1024^2 \times 72 \times 16 \times 1 \times 2 \approx 2.42\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{N_m \cdot H_{v,head} \cdot N_{v,h} \cdot B \cdot a_{byte}}_{\text{Q: 2.25 MB}} + \underbrace{N_m^2 \cdot N_{v,h} \cdot B \cdot a_{byte}}_{\text{输出注意力分数: 32 MB}} \approx 34.3\ \text{MB}$$

$$AI = \frac{2.42\ \text{G}}{34.3\ \text{M}} \approx 70.6\ \text{OPs/byte} < 322 \Rightarrow \text{内存瓶颈}$$

$$\text{实际性能} = 70.6 \times 3.072\ \text{T} \approx 216.9\ \text{TFLOPS}$$

$$t_{qk\_matmul} = \frac{2.42\ \text{G}}{216.9\ \text{T}} \approx 11.2\ \mu s$$

#### 5.4.3 Vision Encoder 每层总时间汇总

| 算子 | OPs | 内存访问 | AI | 瓶颈 | 时间 |
|------|-----|----------|----|------|------|
| `vision_norm1` | 8.26 MFLOPs | 4.5 MB | 1.84 | 内存 | 1.46 μs |
| `vision_q_proj` | 2.72 GFLOPs | 7.03 MB | 387 | 计算 | 2.75 μs |
| `vision_k_proj` | 2.72 GFLOPs | 7.03 MB | 387 | 计算 | 2.75 μs |
| `vision_v_proj` | 2.72 GFLOPs | 7.03 MB | 387 | 计算 | 2.75 μs |
| `vision_qk_matmul` | 2.42 GFLOPs | 34.3 MB | 70.6 | 内存 | 11.2 μs |
| `vision_softmax` | 83.9 MFLOPs | 64 MB | 1.31 | 内存 | 20.9 μs |
| `vision_sv_matmul` | 2.42 GFLOPs | 34.3 MB | 70.6 | 内存 | 11.2 μs |
| `vision_out_proj` | 2.72 GFLOPs | 7.03 MB | 387 | 计算 | 2.75 μs |
| `vision_attn_add` | 1.18 MFLOPs | 4.5 MB | 0.26 | 内存 | 1.48 μs |
| `vision_norm2` | 8.26 MFLOPs | 4.5 MB | 1.84 | 内存 | 1.46 μs |
| `vision_up_proj` | 10.16 GFLOPs | 20.1 MB | 505 | 计算 | 10.3 μs |
| `vision_mlp_act` | 22 MFLOPs | 16.8 MB | 1.31 | 内存 | 5.47 μs |
| `vision_down_proj` | 10.16 GFLOPs | 20.1 MB | 505 | 计算 | 10.3 μs |
| `vision_mlp_add` | 1.18 MFLOPs | 4.5 MB | 0.26 | 内存 | 1.48 μs |
| **每层合计** | **≈ 36.1 GFLOPs** | | | | **≈ 86.2 μs** |

$$t_{vision\_encoder} = t_{patch\_embed} + L_v \times t_{per\_layer} = 14.7 + 27 \times 86.2 \approx 2,342\ \mu s \approx 2.34\ \text{ms}$$

#### 5.4.4 Merger 时间（4 个 merger）

每个 merger 的主导算子是 fc1 和 fc2：

**fc1（$H_{merger} \to H_{merger}$，方阵投影）：**

$$OPs = B \cdot N_m \cdot H_{merger}^2 \times 2 = 1 \times 1024 \times 4608^2 \times 2 \approx 43.5\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{4608^2 \times 2}_{\text{权重 40.5 MB}} + \underbrace{1024 \times 4608 \times 2}_{\text{输入 9 MB}} + \underbrace{1024 \times 4608 \times 2}_{\text{输出 9 MB}} \approx 58.5\ \text{MB}$$

$$AI = \frac{43.5\ \text{G}}{58.5\ \text{M}} \approx 743\ \text{OPs/byte} > 322 \Rightarrow \text{计算瓶颈}$$

$$t_{fc1} = \frac{43.5\ \text{G}}{989.5\ \text{T}} \approx 44.0\ \mu s$$

**fc2（$H_{merger} \to H_{out,v}$，输出投影）：**

$$OPs = B \cdot N_m \cdot H_{merger} \cdot H_{out,v} \times 2 = 1 \times 1024 \times 4608 \times 5120 \times 2 \approx 48.3\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{4608 \times 5120 \times 2}_{\text{权重 45 MB}} + \underbrace{1024 \times 4608 \times 2}_{\text{输入 9 MB}} + \underbrace{1024 \times 5120 \times 2}_{\text{输出 10 MB}} \approx 64\ \text{MB}$$

$$AI = \frac{48.3\ \text{G}}{64\ \text{M}} \approx 755\ \text{OPs/byte} > 322 \Rightarrow \text{计算瓶颈}$$

$$t_{fc2} = \frac{48.3\ \text{G}}{989.5\ \text{T}} \approx 48.8\ \mu s$$

每个 merger 总时间 $\approx 44.0 + 48.8 + \text{(ln+act 约 1 μs)} \approx 93.8\ \mu s$

$$t_{merger} = N_{merger} \times t_{per\_merger} = 4 \times 93.8 \approx 375\ \mu s$$

**Vision Stage 总推理时间：**

$$\boxed{t_{vision} = t_{vision\_encoder} + t_{merger} \approx 2342 + 375 \approx 2717\ \mu s \approx 2.72\ \text{ms}}$$

---

### 5.5 Prefill 推理时间计算（$S=1152$）

Prefill 阶段 LLM 一次性处理全部 1152 个 token（128 文本 + 1024 视觉）。

#### 5.5.1 线性层（每层 7 个，重复 $L=64$ 次）

**q\_proj（$H \to N_h \cdot H_{head}$，即 $5120 \to 8192$）：**

$$OPs = H \cdot (N_h \cdot H_{head}) \cdot B \cdot S \times 2 = 5120 \times 8192 \times 1 \times 1152 \times 2 \approx 96.6\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{5120 \times 8192 \times 2}_{\text{权重 80 MB}} + \underbrace{5120 \times 1152 \times 2}_{\text{输入 11.25 MB}} + \underbrace{8192 \times 1152 \times 2}_{\text{输出 18 MB}} \approx 109.3\ \text{MB}$$

$$AI = \frac{96.6\ \text{G}}{109.3\ \text{M}} \approx 884\ \text{OPs/byte} > 322 \Rightarrow \text{计算瓶颈}$$

$$t_{q\_proj} = \frac{96.6\ \text{G}}{989.5\ \text{T}} \approx 97.6\ \mu s$$

**gate\_proj / up\_proj（$H \to I$，即 $5120 \to 25600$）：**

$$OPs = 5120 \times 25600 \times 1 \times 1152 \times 2 \approx 302\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{5120 \times 25600 \times 2}_{\text{权重 250 MB}} + \underbrace{5120 \times 1152 \times 2}_{\text{输入 11.25 MB}} + \underbrace{25600 \times 1152 \times 2}_{\text{输出 56.25 MB}} \approx 317.5\ \text{MB}$$

$$AI = \frac{302\ \text{G}}{317.5\ \text{M}} \approx 951\ \text{OPs/byte} > 322 \Rightarrow \text{计算瓶颈}$$

$$t_{gate\_proj} = t_{up\_proj} = \frac{302\ \text{G}}{989.5\ \text{T}} \approx 305\ \mu s$$

#### 5.5.2 注意力（每层，$S=1152$）

**qk\_matmul（$S^2$ 计算）：**

$$OPs = S^2 \cdot H_{head} \cdot N_h \cdot B \times 2 = 1152^2 \times 128 \times 64 \times 1 \times 2 \approx 22.0\ \text{GFLOPs}$$

$$\text{内存访问} = \underbrace{S \cdot H_{head} \cdot N_h \cdot B \cdot a_{byte}}_{\text{Q: 18 MB}} + \underbrace{S \cdot H_{head} \cdot N_{kv} \cdot B \cdot kv_{byte}}_{\text{K cache: 2.25 MB}} + \underbrace{S^2 \cdot N_h \cdot B \cdot a_{byte}}_{\text{注意力分数: 162 MB}} \approx 182.3\ \text{MB}$$

$$AI = \frac{22.0\ \text{G}}{182.3\ \text{M}} \approx 120.7\ \text{OPs/byte} < 322 \Rightarrow \text{内存瓶颈}$$

$$\text{实际性能} = 120.7 \times 3.072\ \text{T} \approx 370.8\ \text{TFLOPS}$$

$$t_{qk\_matmul} = \frac{22.0\ \text{G}}{370.8\ \text{T}} \approx 59.3\ \mu s$$

#### 5.5.3 Prefill 每层时间汇总

| 算子 | OPs | AI | 瓶颈 | 时间 |
|------|-----|----|------|------|
| `attn_norm` | 5.31 MFLOPs | 1.84 | 内存 | 0.94 μs |
| `q_proj` | 96.6 GFLOPs | 884 | 计算 | 97.6 μs |
| `k_proj` | 12.1 GFLOPs | 515 | 计算 | 12.2 μs |
| `v_proj` | 12.1 GFLOPs | 515 | 计算 | 12.2 μs |
| `qk_matmul` | 22.0 GFLOPs | 120.7 | 内存 | 59.3 μs |
| `softmax` | 676 MFLOPs | 1.31 | 内存 | 168 μs |
| `sv_matmul` | 22.0 GFLOPs | 120.7 | 内存 | 59.3 μs |
| `out_proj` | 96.6 GFLOPs | 877 | 计算 | 97.6 μs |
| `attn_add` | 5.90 MFLOPs | 0.26 | 内存 | 0.95 μs |
| `mlp_norm` | 5.31 MFLOPs | 1.84 | 内存 | 0.94 μs |
| `gate_proj` | 302 GFLOPs | 951 | 计算 | 305 μs |
| `up_proj` | 302 GFLOPs | 951 | 计算 | 305 μs |
| `mlp_act` | 147 MFLOPs | 1.31 | 内存 | 36.6 μs |
| `down_proj` | 302 GFLOPs | 951 | 计算 | 305 μs |
| `mlp_add` | 5.90 MFLOPs | 0.26 | 内存 | 0.95 μs |
| **每层合计** | **≈ 1167 GFLOPs** | | | **≈ 1461 μs** |

加上 lm\_head（$H \to V$，$5120 \to 151936$）：

$$OPs_{lm\_head} = B \cdot S \cdot H \cdot V \times 2 = 1 \times 1152 \times 5120 \times 151936 \times 2 \approx 1793\ \text{GFLOPs}$$

$$AI_{lm\_head} = \frac{1793\ \text{G}}{(5120 \times 151936 \times 2) + (1152 \times 5120 \times 2) + (1152 \times 151936 \times 2)\ \text{bytes}} \approx \frac{1793\ \text{G}}{1908\ \text{MB}} \approx 940 > 322 \Rightarrow \text{计算瓶颈}$$

$$t_{lm\_head} = \frac{1793\ \text{G}}{989.5\ \text{T}} \approx 1812\ \mu s$$

**Prefill 总推理时间：**

$$\boxed{t_{prefill} = L \times t_{per\_layer} + t_{lm\_head} = 64 \times 1461 + 1812 \approx 95,316\ \mu s \approx 95.3\ \text{ms}}$$

---

### 5.6 Decode 推理时间计算（单步，KV Cache 长度 $S=1152$）

Decode 阶段每次只处理 **1 个新 token**，但需要从 KV Cache 加载历史 $S=1152$ 个 token 的 K/V。

#### 5.6.1 线性层（每层，序列维度=1）

**q\_proj（$5120 \to 8192$）：**

$$OPs = H \cdot (N_h \cdot H_{head}) \cdot B \times 2 = 5120 \times 8192 \times 1 \times 2 \approx 83.9\ \text{MFLOPs}$$

$$\text{内存访问} = \underbrace{5120 \times 8192 \times 2}_{\text{权重 80 MB}} + \underbrace{5120 \times 1 \times 2}_{\text{输入 0.01 MB}} + \underbrace{8192 \times 1 \times 2}_{\text{输出 0.016 MB}} \approx 80.03\ \text{MB}$$

$$AI = \frac{83.9\ \text{M}}{80.03\ \text{M}} \approx 1.05\ \text{OPs/byte} \ll 322 \Rightarrow \text{内存瓶颈}$$

$$\text{实际性能} = 1.05 \times 3.072\ \text{T} \approx 3.23\ \text{TFLOPS}$$

$$t_{q\_proj} = \frac{83.9\ \text{M}}{3.23\ \text{T}} \approx 26.0\ \mu s$$

> **关键洞察**：Decode 阶段线性层的算术强度 $AI \approx 2B = 2$（极低），几乎完全由权重加载量决定时间，与序列长度无关。这是 LLM decode 阶段的核心性能瓶颈。

**gate\_proj / up\_proj（$5120 \to 25600$）：**

$$OPs = 5120 \times 25600 \times 1 \times 2 \approx 262.1\ \text{MFLOPs}$$

$$\text{内存访问} \approx 5120 \times 25600 \times 2 = 250\ \text{MB（权重主导）}$$

$$AI \approx \frac{262.1\ \text{M}}{250\ \text{M}} \approx 1.05 \Rightarrow \text{内存瓶颈}$$

$$t_{gate\_proj} = t_{up\_proj} = \frac{262.1\ \text{M}}{1.05 \times 3.072\ \text{T}} \approx 81.3\ \mu s$$

#### 5.6.2 注意力（每层，KV Cache 长度 $S=1152$）

**qk\_matmul（当前 Q 与历史 $S$ 个 K 做点积）：**

$$OPs = S \cdot H_{head} \cdot N_h \cdot B \times 2 = 1152 \times 128 \times 64 \times 1 \times 2 \approx 19.1\ \text{MFLOPs}$$

$$\text{内存访问} = \underbrace{1 \cdot H_{head} \cdot N_h \cdot B \cdot a_{byte}}_{\text{Q: 0.016 MB}} + \underbrace{S \cdot H_{head} \cdot N_{kv} \cdot B \cdot kv_{byte}}_{\text{K cache: 2.25 MB}} + \underbrace{1 \cdot S \cdot N_h \cdot B \cdot a_{byte}}_{\text{注意力分数: 0.14 MB}} \approx 2.41\ \text{MB}$$

$$AI = \frac{19.1\ \text{M}}{2.41\ \text{M}} \approx 7.9\ \text{OPs/byte} \ll 322 \Rightarrow \text{内存瓶颈}$$

$$t_{qk\_matmul} = \frac{19.1\ \text{M}}{7.9 \times 3.072\ \text{T}} \approx 0.79\ \mu s$$

**sv\_matmul（注意力权重与历史 $S$ 个 V 加权求和）：**

$$OPs = H_{head} \cdot S \cdot N_h \cdot B \times 2 = 128 \times 1152 \times 64 \times 1 \times 2 \approx 19.1\ \text{MFLOPs}$$

$$\text{内存访问} = \underbrace{S \cdot N_h \cdot B \cdot a_{byte}}_{\text{注意力权重: 0.14 MB}} + \underbrace{S \cdot H_{head} \cdot N_{kv} \cdot B \cdot kv_{byte}}_{\text{V cache: 2.25 MB}} + \underbrace{H_{head} \cdot N_h \cdot B \cdot a_{byte}}_{\text{输出: 0.016 MB}} \approx 2.41\ \text{MB}$$

$$t_{sv\_matmul} \approx 0.79\ \mu s$$

#### 5.6.3 Decode 每层时间汇总

| 算子 | OPs | AI | 瓶颈 | 时间 |
|------|-----|----|------|------|
| `attn_norm` | 4.61 KFLOPs | 0.26 | 内存 | 0.82 μs |
| `q_proj` | 83.9 MFLOPs | 1.05 | 内存 | 26.0 μs |
| `k_proj` | 10.5 MFLOPs | 1.05 | 内存 | 3.25 μs |
| `v_proj` | 10.5 MFLOPs | 1.05 | 内存 | 3.25 μs |
| `qk_matmul` | 19.1 MFLOPs | 7.9 | 内存 | 0.79 μs |
| `softmax` | 0.59 MFLOPs | 1.31 | 内存 | 0.15 μs |
| `sv_matmul` | 19.1 MFLOPs | 7.9 | 内存 | 0.79 μs |
| `out_proj` | 83.9 MFLOPs | 1.05 | 内存 | 26.0 μs |
| `attn_add` | 5.12 KFLOPs | 0.26 | 内存 | 0.82 μs |
| `mlp_norm` | 4.61 KFLOPs | 0.26 | 内存 | 0.82 μs |
| `gate_proj` | 262.1 MFLOPs | 1.05 | 内存 | 81.3 μs |
| `up_proj` | 262.1 MFLOPs | 1.05 | 内存 | 81.3 μs |
| `mlp_act` | 0.13 MFLOPs | 1.31 | 内存 | 0.032 μs |
| `down_proj` | 262.1 MFLOPs | 1.05 | 内存 | 81.3 μs |
| `mlp_add` | 5.12 KFLOPs | 0.26 | 内存 | 0.82 μs |
| **每层合计** | **≈ 1013 MFLOPs** | | | **≈ 307.4 μs** |

加上 lm\_head（decode 阶段只处理 1 个 token）：

$$OPs_{lm\_head} = B \cdot H \cdot V \times 2 = 1 \times 5120 \times 151936 \times 2 \approx 1557\ \text{MFLOPs}$$

$$\text{内存访问} \approx 5120 \times 151936 \times 2 = 1557\ \text{MB（权重主导）}$$

$$AI \approx 1.0 \Rightarrow \text{内存瓶颈}, \quad t_{lm\_head} = \frac{1557\ \text{M}}{1.0 \times 3.072\ \text{T}} \approx 507\ \mu s$$

**Decode 总推理时间（单步 TPOT）：**

$$\boxed{t_{decode} = L \times t_{per\_layer} + t_{lm\_head} = 64 \times 307.4 + 507 \approx 20,181\ \mu s \approx 20.2\ \text{ms/token}}$$

---

### 5.7 三阶段时间汇总

| 阶段 | 推理时间 | 说明 |
|------|----------|------|
| **Vision Stage**（TTFT 的视觉部分） | $\approx 2.72\ \text{ms}$ | Vision Encoder + Merger，处理 1024×1024 图像 |
| **Prefill**（LLM 预填充） | $\approx 95.3\ \text{ms}$ | 处理 1152 个 token（128 文本 + 1024 视觉） |
| **TTFT**（首 token 时延） | $\approx 2.72 + 95.3 \approx 98.0\ \text{ms}$ | Vision + Prefill 串行执行 |
| **Decode**（每 token 生成，TPOT） | $\approx 20.2\ \text{ms/token}$ | 单步 decode，KV Cache 长度 1152 |

**关键结论：**

1. **Vision Stage 占 TTFT 约 2.8%**：视觉编码计算量虽大（约 1.36 TFLOPs），但 $N_m=1024$ 个 token 并行处理，算术强度高，多数算子达到计算瓶颈，效率较高。

2. **Prefill 主导 TTFT**：LLM prefill 的 MLP 线性层（gate/up/down\_proj）计算量最大，每层约 906 GFLOPs，占每层总时间的 62%。

3. **Decode 完全内存瓶颈**：所有线性层 $AI \approx 1.05$，远低于转折点 322，时间由权重加载量决定。增大 batchsize 可线性提升算术强度，改善 decode 效率。

4. **注意力在 Decode 中占比小**：KV Cache 加载量（$S=1152$ 时约 2.25 MB/层）远小于权重加载量（约 500 MB/层），注意力不是 decode 瓶颈。随着生成 token 增多，$S$ 增大，KV Cache 加载量线性增长，最终可能成为瓶颈。
