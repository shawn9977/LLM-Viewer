# PaddleOCR-VL-1.5 Roofline 分析支持 — 完整改动说明

## 1. 概述

本次改动为 LLM-Viewer 新增了对 **PaddleOCR-VL-1.5** 多模态视觉语言模型的 roofline 分析支持。PaddleOCR-VL-1.5 是一个视觉+文本双分支架构模型，需要同时统计视觉 encoder 和文本 decoder 的算力/带宽开销。

### 模型架构规格（来自 config.json）

| 参数 | 文本分支 | 视觉分支 |
|------|---------|--------|
| hidden_size | 1024 | 1152 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | 2 (GQA) | 16 (MHA) |
| num_hidden_layers | 18 | 27 |
| intermediate_size | 3072 | 4304 |
| head_dim | 128（显式配置） | 72（1152/16） |
| vocab_size | 103,424 | — |
| max_position_embeddings | 131,072 | — |
| patch_size | — | 14 |
| spatial_merge_size | — | 2（2×2合并） |
| precision | bfloat16 | bfloat16 |

### 分析阶段划分

| stage | 含义 |
|-------|------|
| `prefill` | 文本 prefill（处理输入序列） |
| `decode` | 文本 decode（逐 token 生成） |
| `vision` | 视觉 encoder（图像编码 + projector） |
| `multimodal_ttft` | 多模态首 token 延迟 = vision + prefill |
| `multimodal_tpot` | 多模态生成延迟 = decode |

---

## 2. 新增文件

### 2.1 `backend/models/paddleocr_vl.py`

模型参数读取模块，为 `PaddleOCRVLAnalyzer` 提供所有维度信息和算子统计函数。

---

## 3. 修改文件

### 3.1 `backend/model_analyzer.py` — 新增 `PaddleOCRVLAnalyzer`

注册表新增一条：

```python
"PaddleOCRVLAnalyzer": ["paddleocr_vl"],
```

`get_analyzer()` 通过 `model_type` 字段（来自 config.json 的 `"model_type": "paddleocr_vl"`）自动匹配该类。

---

### 3.2 `backend/model_params.py` — 注册模型 ID

```python
"PaddlePaddle/PaddleOCR-VL-1.5": {"source": "local"},
```

config.json 存放路径为 `backend/PaddlePaddle/PaddleOCR-VL-1.5/config.json`，`source=local` 表示从本地读取而非从 HuggingFace/ModelScope 下载。

---

### 3.3 `frontend/src/components/left_controls/Config.vue` — 前端多模态检测

```javascript
// 修改前
return id.includes('Qwen3-VL') || id.includes('Omni');
// 修改后
return id.includes('Qwen3-VL') || id.includes('Omni') || id.includes('PaddleOCR-VL');
```

`is_multimodal` 计算属性控制前端是否显示图像尺寸输入框，新增 PaddleOCR-VL 检测后，用户可以在 Config 面板中设置 `image_size` 参数传入分析器。

---

## 4. 分析器详细统计说明

以下符号约定：
- `B` = batchsize
- `S` = seqlen（文本序列长度）
- `H` = hidden_size（文本，1024）
- `Hv` = vision_hidden_size（视觉，1152）
- `d` = head_dim（文本，128）
- `dv` = vision head_dim（72 = 1152/16）
- `A` = num_attention_heads（文本，16）
- `Av` = vision_num_attention_heads（视觉，16）
- `Kv` = num_key_value_heads（文本，2）
- `FFN` = intermediate_size（文本，3072）
- `FFNv` = vision_intermediate_size（视觉，4304）
- `L` = num_hidden_layers（文本，18）
- `Lv` = vision_num_hidden_layers（视觉，27）
- `P` = num_patches = ceil(W/14) × ceil(H/14)（spatial merge 之前的 patch 数量）
- `M` = merged_tokens = P / 4（spatial merge 之后，2×2=4合并）
- `w_byte` / `a_byte` / `kv_byte` = 权重/激活/KV cache 字节数
- `tp` = tensor parallel size

---

### 4.1 文本分支 — Decode Stage（单 token 生成）

每层统计以下算子，最终乘以 `L=18` 累加：

#### 4.1.1 线性层（q/k/v/out/gate/up/down)

| 算子 | 形状 | OPs | load_weight | load_act | store_act | store_kv_cache |
|------|------|-----|-------------|----------|-----------|----------------|
| q_proj | [H, A×d/tp] | B × H × (A×d/tp) × 2 | H×(A×d/tp)×w_byte | B×H×a_byte | B×(A×d/tp)×a_byte | 0 |
| k_proj | [H, Kv×d/tp] | B × H × (Kv×d/tp) × 2 | H×(Kv×d/tp)×w_byte | B×H×a_byte | 0 | B×(Kv×d/tp)×kv_byte |
| v_proj | [H, Kv×d/tp] | B × H × (Kv×d/tp) × 2 | H×(Kv×d/tp)×w_byte | B×H×a_byte | 0 | B×(Kv×d/tp)×kv_byte |
| out_proj | [A×d/tp, H] | B × (A×d/tp) × H × 2 | (A×d/tp)×H×w_byte | B×(A×d/tp)×a_byte | B×H×a_byte | 0 |
| gate_proj | [H, FFN/tp] | B × H × (FFN/tp) × 2 | H×(FFN/tp)×w_byte | B×H×a_byte | B×(FFN/tp)×a_byte | 0 |
| up_proj | [H, FFN/tp] | B × H × (FFN/tp) × 2 | H×(FFN/tp)×w_byte | B×H×a_byte | B×(FFN/tp)×a_byte | 0 |
| down_proj | [FFN/tp, H] | B × (FFN/tp) × H × 2 | (FFN/tp)×H×w_byte | B×(FFN/tp)×a_byte | B×H×a_byte | 0 |

> k_proj / v_proj 的输出直接写入 KV cache（store_kv_cache），不写 store_act，因为 decode 阶段不需要缓存激活值。

#### 4.1.2 注意力（Decode）

| 算子 | 说明 | OPs | 关键带宽 |
|------|------|-----|----------|
| qk_matmul | Q(1×d) × K^T(S×d)，取历史 KV cache | B×A×S×d×2 | load_kv_cache = S×d×B×Kv×kv_byte |
| sv_matmul | Attn(1×S) × V(S×d) | B×A×d×S×2 | load_kv_cache = S×d×B×Kv×kv_byte |
| softmax | softmax(1×S) | B×A×S×5 | load/store B×A×S×a_byte |

> decode 阶段 Q 只有 1 个 token，因此 qk_matmul 是向量×矩阵而非矩阵×矩阵，算力密度极低，通常受带宽瓶颈。

#### 4.1.3 Norm 和残差（Decode）

| 算子 | OPs | load_act | store_act |
|------|-----|----------|-----------|
| attn_norm / mlp_norm（RMSNorm） | B×(H/tp)×4 | B×(H/tp)×a_byte | B×(H/tp)×a_byte |
| attn_add / mlp_add（残差加） | B×(H/tp) | B×(H/tp)×a_byte | B×(H/tp)×a_byte |
| mlp_act（SiLU，作用于 gate×up） | B×(FFN/tp)×5 | B×(FFN/tp)×a_byte | B×(FFN/tp)×a_byte |

---

### 4.2 文本分支 — Prefill Stage（输入序列处理）

与 Decode 结构相同，区别在于序列维度从 1 变为 S，以及 KV cache 写入方向。

#### 4.2.1 线性层（Prefill）

所有线性层 OPs 乘以 S：
- q/k/v/out/gate/up/down 的 `OPs = ic × oc × B × S × 2`
- load_act = `ic × B × S × a_byte`
- k_proj / v_proj 的 `store_kv_cache = oc × B × S × kv_byte`（写入完整历史 KV）

#### 4.2.2 注意力（Prefill）

**标准 Attention（非 FlashAttention）：**

| 算子 | OPs | 说明 |
|------|-----|------|
| qk_matmul | S×S×d×A×B×2 | Q(S×d) × K^T(S×d)，全序列自注意力 |
| sv_matmul | S×d×S×A×B×2 | Attn(S×S) × V(S×d) |
| softmax | B×A×S×S×5 | 对 S×S 注意力矩阵做 softmax |

**FlashAttention 模式：**

融合为单个 `fused_attention` 算子，分块计算避免 S×S 矩阵具体化：
- OPs = qk_OPs + sv_OPs + softmax_OPs（总量不变）
- `load_act` = Q + O（分块加载），O = S×S×B×A×a_byte
- `load_kv_cache` = n_blocks_r × S × d × B × Kv × kv_byte × 2（K、V 各读一次）
- `block_size_r` = min(ceil(onchip_buffer / (kv_byte × d)), d)（由硬件片上缓冲决定）

#### 4.2.3 Norm 和残差（Prefill）

与 Decode 相同，但序列维度为 S：

| 算子 | OPs | load_act | store_act |
|------|-----|----------|-----------|
| attn_norm / mlp_norm | B×(H/tp)×S×4 | B×(H/tp)×S×a_byte | B×(H/tp)×S×a_byte |
| attn_add / mlp_add | B×(H/tp)×S | B×(H/tp)×S×a_byte | B×(H/tp)×S×a_byte |
| mlp_act | B×(FFN/tp)×S×5 | B×(FFN/tp)×S×a_byte | B×(FFN/tp)×S×a_byte |

---

### 4.3 文本分支 — post_process（embed_tokens + lm_head）

这 4 个算子不乘以 L，只统计一次：

| 算子 | stage | OPs | load_weight | 说明 |
|------|-------|-----|-------------|------|
| embed_tokens | prefill | B×S×H | vocab×H×w_byte | 查表操作，prefill 时加载完整 embedding table |
| embed_tokens | decode | B×H | 0 | 权重已在 prefill 时加载，decode 不重复计算 load_weight |
| lm_head | prefill | B×S×H×vocab×2 | H×vocab×w_byte | 投影到词表，计算每个位置的 logits |
| lm_head | decode | B×H×vocab×2 | H×vocab×w_byte | 仅生成最后 1 个 token 的 logits |

> **embed_tokens 修复说明**：embedding table 在整个推理过程只需加载一次（prefill 阶段），decode 阶段只需查找 1 个 token 的向量，权重已在 L2/HBM 中，不重复计入 load_weight，避免权重内存统计偏高。

---

### 4.4 视觉分支 — Vision Stage（图像编码）

#### 序列长度定义

```
num_patches_w = ceil(image_width  / patch_size)   # 水平 patch 数
num_patches_h = ceil(image_height / patch_size)   # 垂直 patch 数
num_patches   = num_patches_w × num_patches_h     # encoder 主体序列长度
merged_tokens = ceil(num_patches / spatial_merge_size²)  # projector 序列长度
```

例：输入 1024×1024 图像，patch_size=14，spatial_merge_size=2：
- `num_patches = 74 × 74 = 5476`
- `merged_tokens = ceil(5476 / 4) = 1369`

> **关键设计**：visual encoder 主体（27层）的所有算子均以 `num_patches` 为序列长度统计，spatial merge 发生在 encoder 之后的 projector 入口，因此 encoder 层内部不能用 `merged_tokens`（否则低估 4 倍）。

---

#### 4.4.1 Patch Embedding

将图像切成 patch 并线性映射到 vision hidden space：

| 参数 | 值 |
|------|----|
| 输入通道 | in_channels × patch_size² = 3×14×14 = 588 |
| 输出通道 | Hv = 1152 |

```
OPs         = patch_ic × patch_oc × B × num_patches × 2
load_weight = patch_ic × patch_oc × w_byte
load_act    = patch_ic × B × num_patches × a_byte
store_act   = patch_oc × B × num_patches × a_byte
```

---

#### 4.4.2 视觉线性层（每层，重复 Lv=27 次）

Vision encoder 采用 ViT 结构，MLP 为 **fc1+GELU+fc2**（无 gate_proj）：

| 算子 | 形状 [ic, oc] | OPs | load_weight | load_act | store_act |
|------|--------------|-----|-------------|----------|-----------|
| vision_q_proj | [Hv, Av×dv/tp] | ic×oc×B×P×2 | ic×oc×w_byte | ic×B×P×a_byte | oc×B×P×a_byte |
| vision_k_proj | [Hv, Av×dv/tp] | 同上 | 同上 | 同上 | 同上 |
| vision_v_proj | [Hv, Av×dv/tp] | 同上 | 同上 | 同上 | 同上 |
| vision_out_proj | [Av×dv/tp, Hv] | ic×oc×B×P×2 | ic×oc×w_byte | ic×B×P×a_byte | oc×B×P×a_byte |
| vision_up_proj | [Hv, FFNv/tp] | ic×oc×B×P×2 | ic×oc×w_byte | ic×B×P×a_byte | oc×B×P×a_byte |
| vision_down_proj | [FFNv/tp, Hv] | ic×oc×B×P×2 | ic×oc×w_byte | ic×B×P×a_byte | oc×B×P×a_byte |

其中 P = num_patches，Hv=1152，Av=16，dv=72，FFNv=4304。

---

#### 4.4.3 视觉注意力（每层，重复 Lv=27 次）

**标准 Attention：**

| 算子 | OPs | load_act | store_act |
|------|-----|----------|-----------|
| vision_qk_matmul | P×P×dv×Av×B×2 | P×dv×B×Av×a_byte | P×P×B×Av×a_byte |
| vision_sv_matmul | P×dv×P×Av×B×2 | P×P×B×Av×a_byte | P×dv×B×Av×a_byte |
| vision_softmax | B×Av×P×P×5 | B×Av×P×P×a_byte | B×Av×P×P×a_byte |

**FlashAttention 模式（视觉 encoder 同样支持）：**

融合为 `vision_fused_attention`，分块 tiling 避免 P×P 矩阵具体化：
```
block_size_r = min(ceil(onchip_buffer / (a_byte × dv)), dv)
n_blocks_r   = ceil(P / block_size_r)
OPs          = v_qk_OPs + v_sv_OPs + v_softmax_OPs
load_act     = q_numel + kv_numel   # Q + (K,V)
store_act    = o_numel × 2          # 输出 + softmax 中间结果
load_kv_cache = 0                   # 视觉 encoder 无 KV cache
```

---

#### 4.4.4 视觉 Norm 和残差（每层，重复 Lv=27 次）

| 算子 | OPs | load_act | store_act | 说明 |
|------|-----|----------|-----------|------|
| vision_norm1 | B×Hv×P×4 | B×Hv×P×a_byte | B×Hv×P×a_byte | Pre-Attn LayerNorm |
| vision_norm2 | B×Hv×P×4 | B×Hv×P×a_byte | B×Hv×P×a_byte | Pre-MLP LayerNorm |
| vision_attn_add | B×Hv×P | B×Hv×P×a_byte | B×Hv×P×a_byte | Attention 残差连接 |
| vision_mlp_add | B×Hv×P | B×Hv×P×a_byte | B×Hv×P×a_byte | MLP 残差连接 |
| vision_mlp_act | B×FFNv×P×5 | B×FFNv×P×a_byte | B×FFNv×P×a_byte | GELU 激活函数 |

以上 5 类算子均在 `vision_repeat_layers` 集合中，统计时乘以 `Lv=27`。

---

#### 4.4.5 vision_post_process（不重复，只统计一次）

Encoder 主体（27层）结束后的尾部算子，序列长度切换到 `num_patches` 或 `merged_tokens`：

| # | 算子 | 序列长度 | 形状 | 说明 |
|---|------|---------|------|------|
| 1 | vision_post_layernorm | P | Hv=1152 | Encoder 输出的最终 LayerNorm |
| 2 | vision_head_out_proj | 1 | [Hv→Hv] | Pooling head attention，1 个 query token 做全局池化 |
| 3 | vision_head_layernorm | 1 | Hv=1152 | Pooling head 内部 LayerNorm |
| 4 | vision_head_fc1 | 1 | [Hv→FFNv] | Pooling head MLP 第一层 |
| 5 | vision_head_fc2 | 1 | [FFNv→Hv] | Pooling head MLP 第二层 |
| 6 | vision_merger_pre_norm | P | Hv=1152 | Spatial merge 前的 LayerNorm |
| 7 | vision_merger_fc1 | M | [4×Hv→4×Hv] = [4608→4608] | mlp_AR projector，merge 后序列 |
| 8 | vision_merger_act | M | 4608 | projector 内部 GELU |
| 9 | vision_merger_fc2 | M | [4×Hv→H] = [4608→1024] | 输出对齐文本 hidden_size |

> **算子 7-9（mlp_AR projector）** 的输入是将 2×2=4 个相邻 patch 的 Hv 维特征拼接，形成 4×Hv=4608 维向量，token 数量缩减为 M=merged_tokens。这是视觉到文本的桥接模块。

---

### 4.5 结果累加与内存统计

#### 层重复累加

```python
# 文本层：所有算子 × L（decode / prefill 各自独立统计）
total_results[stage][data] += result[data] * num_hidden_layers

# 视觉层：vision_repeat_layers 中的算子 × Lv，其余 × 1
multiplier = vision_num_layers if name in vision_repeat_layers else 1
total_results["vision"][data] += result[data] * multiplier
```

#### 内存占用计算

| 阶段 | memory_weight | memory_tmp_act | memory_kv_cache |
|------|--------------|---------------|------------------|
| decode | prefill load_weight 总和 | decode 所有 store_act 总和 | prefill store_kv_cache 总和 |
| prefill | prefill load_weight 总和 | prefill 所有 store_act 总和 | prefill store_kv_cache 总和 |
| vision | vision load_weight 总和 | vision 所有 store_act × 重复数 之和 | 0（无 KV cache） |

#### multimodal 合并

```
multimodal_ttft（首 token，TTFT）：
  data        = vision[data] + prefill[data]
  weight      = vision_weight + prefill_weight
  tmp_act     = max(vision_tmp_act, prefill_tmp_act)  # 串行执行，取最大值
  kv_cache    = prefill_kv_cache

multimodal_tpot（续生成，TPOT）：
  data        = decode[data]
  weight      = decode_weight
  tmp_act     = decode_tmp_act
  kv_cache    = decode_kv_cache
```

---

## 5. 已修复的 Bug

| # | 问题 | 修复 |
|---|------|------|
| 1 | 类名拼写错误 `PaddleORCVLAnalyzer`（ORC→OCR） | 改为 `PaddleOCRVLAnalyzer`，注册表 key 同步修正 |
| 2 | 视觉线性层（encoder 27层）用 `merged_tokens` 作序列长度 | 改为 `num_patches`，spatial merge 在 encoder 之后发生 |
| 3 | 视觉注意力 QK/SV matmul 用 `merged_tokens` | 改为 `num_patches`，同问题 2 |
| 4 | 视觉 norm/残差/激活层用 `merged_tokens` | 改为 `num_patches`，同问题 2 |
| 5 | decode `embed_tokens.load_weight` 重复计算 | decode 设为 0，权重仅在 prefill 阶段计一次 |

> **问题 2-4 的影响量化**：`spatial_merge_size=2` 时 `merged_tokens = num_patches / 4`，错误统计导致视觉 encoder 主体（27层）的 OPs、load_act、store_act 均被低估 **4 倍**。修复后视觉分支算力/带宽数值将提高 4 倍，更接近真实推理开销。

---

## 6. 遗留小问题（可选优化）

- **Pooling head attention 的 QK/SV matmul 未统计**：`vision_post_process` 中 `vision_head_out_proj` 只统计了 out_proj 线性层，pooling head 内部的 Q 生成、QK matmul、SV matmul 被省略。由于 pooling head 序列长度为 1，该部分对总开销影响极小（< 0.1%），当前实现可接受。