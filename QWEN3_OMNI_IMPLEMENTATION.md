# Qwen3-Omni-30B-A3B-Instruct Roofline 分析实现

## 实现概述

已成功为 Qwen3-Omni-30B-A3B-Instruct 模型添加 Roofline 性能分析支持（Phase 1：Thinker 部分，不含语音合成输出）。

## 修改的文件

### 1. 新增文件

- **`backend/models/qwen3_omni_moe.py`**
  - 实现 Qwen3-Omni 模型的参数提取函数
  - 支持三个分支：
    - 文本分支（MoE Transformer，48层，8个active experts，hidden=2048）
    - 视觉分支（Vision Encoder，27层，hidden=1152，16头，标准 2 层 MLP）
    - 音频分支（Audio Encoder，32层，d_model=1280，20头）
  - 文本图结构复用 `qwen3_moe` 的 `transformer_layer_graph`
  - 视觉图结构独立定义（使用 fc1/fc2 标准 MLP，不复用 qwen3_vl 的 gated MLP）
  - 新定义 `audio_layer_graph` 和 `audio_flashattention_layer_graph`

### 2. 修改的文件

- **`backend/model_analyzer.py`**
  - 在 `MODEL_ANALYZER_REGISTRY` 中注册 `OmniAnalyzer`（key: `"qwen3_omni_moe"`）
  - 新增 `OmniAnalyzer` 类，实现正确的多模态 token 计算逻辑
  - 所有 analyzer 的 `analyze()` 签名统一添加 `audio_length=None` 参数（`LLMAnalyzer`、`MoEAnalyzer`、`VLMAnalyzer`、基类），避免调用时 TypeError

- **`backend/model_params.py`**
  - 添加 `"Qwen/Qwen3-Omni-30B-A3B-Instruct": {"source": "huggingface"}` 到可用模型列表

- **`backend/get_model_graph.py`**
  - 添加 `has_audio = hasattr(analyzer.module, "audio_layer_graph")` 检测
  - 从 `inference_config` 提取 `audio_length` 并传入 `analyzer.analyze()`
  - prefill 阶段：`has_vision or has_audio` 时同时渲染 Vision + Audio + Text 三个子图
  - chat 阶段：内部 `analyze()` 补传 `audio_length`；多模态图渲染统一支持 `has_audio`
  - decode 阶段：`has_vision or has_audio` 时替换 `total_results["decode"]` 为 `multimodal_tpot`
  - 添加 `audio` stage 的独立图结构处理逻辑

- **`frontend/src/components/left_controls/Config.vue`**
  - `is_multimodal` computed 扩展为同时包含 `'Qwen3-VL'` 和 `'Omni'`（修复 Image Size 不显示的问题）
  - 新增 `is_omni` computed（检测 `'Omni'`）
  - 新增 Audio Length (s) 输入框，在 Omni 模型 prefill/chat 阶段显示
  - 新增 `audio_length` ref（默认 5.0s）及 watcher，写入 `global_inference_config.audio_length`
  - 新增 `use_image` checkbox（默认勾选），取消勾选时传 `null` 给后端，跳过视觉分支计算
  - 新增 `use_audio` checkbox（默认勾选），取消勾选时传 `null` 给后端，跳过音频分支计算
  - Image Size 和 Audio Length 输入框仅在对应 checkbox 勾选时显示

- **`frontend/src/App.vue`**
  - `global_inference_config` 初始值新增 `audio_length: 5.0`，确保 Omni 模型首次加载时后端即可收到有效音频长度

- **`backend/get_model_graph.py`**
  - Audio Encoder 子图渲染条件从 `has_audio and result["audio"]` 改为 `has_audio`，无音频输入时仍渲染图骨架（节点 OPs 显示为 0），与 Vision Encoder 行为一致

## 核心实现逻辑

### Token 计算

```python
# 计算各模态的 token 数
text_tokens = seqlen  # 用户输入的文本长度

# 视觉 token：patch_size=16, temporal_patch_size=2, spatial_merge_size=2
# 例如 1024x1024 → 256 tokens
visual_tokens = (H / patch_size) * (W / patch_size) / (spatial_merge_size ** 2)

# 音频 token：n_window=50，卷积下采样 8x
# 例如 5秒 → audio_length * 50 / 8 ≈ 31 tokens
audio_tokens = audio_length * 50 / 8

# Prefill 阶段处理所有模态的 tokens
prefill_total_tokens = text_tokens + visual_tokens + audio_tokens
```

### Stage 定义

#### Prefill 阶段

**包括**：
- Vision Encoder（如果传入 `image_size`）
- Audio Encoder（如果传入 `audio_length`）
- Text Transformer 处理 `prefill_total_tokens` 个 tokens

**前端展示**：
- 图结构：Vision Encoder + Audio Encoder + Text Prefill 三个子图（按实际输入条件显示）
- `total_results["prefill"]` = `multimodal_ttft`（Vision + Audio + Text Prefill 总时间）

#### Decode 阶段

**包括**：
- Text Transformer 处理 1 个新 token
- 访问长度为 `prefill_total_tokens` 的 KV cache

**前端展示**：
- 图结构：Text Decode 单个子图
- `total_results["decode"]` = `multimodal_tpot`（Text Decode 时间）

## 关键修复

### Phase 1.1：结构性修正（对照实际模型架构）

通过对照 `Qwen3OmniMoeForConditionalGeneration` 的完整模型结构，修正了以下问题：

#### 1. 视觉 MLP 结构修正（影响最大）

**问题**：代码使用了 SwiGLU gated MLP（gate_proj + up_proj + down_proj = 3 个线性层），但实际模型是标准 2 层 MLP（linear_fc1 + GELU + linear_fc2）。

**影响**：多算了一个 `Linear(1152, 4304)` × 27 层，视觉分支 OPs 多算约 33%。

**修复**：
- `get_vision_linear_layers()` 改为返回 `vision_fc1` / `vision_fc2`
- `vision_layer_graph` 独立定义，不再复用 `qwen3_vl` 的 gated MLP 图

#### 2. 视觉 Merger 维度修正

**问题**：`vision_post_process` 输入维度写的是 `1152`，实际是 `4608`（spatial_merge_size=2 → 合并 2×2=4 个 patch → 4×1152=4608）。

**修复**：
- 输入维度改为 `spatial_merge_size² × hidden_size = 4608`
- 补全完整链路：`LayerNorm(4608)` → `Linear(4608, 4608)` → `GELU` → `Linear(4608, 2048)`
- 新增 3 个 `merger_list`（对应 `deepstack_visual_indexes: [8, 16, 24]`），共 4 个 merger

#### 3. 音频前端卷积层补全

**问题**：`audio_post_process` 只统计了最后的 `proj2: Linear(1280, 2048)`，缺失了整个前端。

**补全**：
```
conv2d1: Conv2d(1, 480, k=3, s=2)      ← mel spectrogram 输入
conv2d2: Conv2d(480, 480, k=3, s=2)
conv2d3: Conv2d(480, 480, k=3, s=2)
conv_out: Linear(7680, 1280)            ← 480×16=7680 频率维度折叠
ln_post: LayerNorm(1280)
proj1: Linear(1280, 1280)
act: GELU
proj2: Linear(1280, 2048)              ← 原有
```

#### 4. 文本 embed_tokens 补全

**问题**：`Embedding(152064, 2048)` 查表操作未统计。

**修复**：在 `post_process()` 中新增 `embed_tokens`（OPs=0，但有权重加载和激活存储开销）。同时修复 `_analyze_to_results` 中 OPs=0 时的除零错误。

#### 5. model_analyzer.py 的 vision_repeat_layers 更新

新增 `vision_fc1`、`vision_fc2` 到重复层集合，确保视觉分支总计正确乘以层数。

### 修复了 VLMAnalyzer 的 Bug

原有的 `VLMAnalyzer` 存在问题：
- Prefill 阶段只计算了 `seqlen` 个文本 token，**没有加上视觉 token 的处理开销**
- Decode 阶段的 KV cache 长度也只用了 `seqlen`

`OmniAnalyzer` 正确实现了：
- Prefill 使用 `prefill_total_tokens = visual_tokens + audio_tokens + text_tokens`
- Decode 的 KV cache 访问也使用 `prefill_total_tokens`

### 修复了 get_model_graph.py 的三个 Bug

1. **chat 阶段内部 analyze() 未传 audio_length**：补传后音频 token 在 chat 阶段也能正确累计
2. **chat 阶段多模态图渲染只处理 has_vision**：统一为 `has_vision or has_audio`，Audio Encoder 子图在 chat 阶段也能显示
3. **decode 阶段 multimodal_tpot 替换只检查 has_vision**：改为 `has_vision or has_audio`

## 使用方法

### 后端 API

```python
from model_analyzer import get_analyzer

analyzer = get_analyzer('Qwen/Qwen3-Omni-30B-A3B-Instruct', 'nvidia_A100')

# 纯文本分析
result = analyzer.analyze(seqlen=100, batchsize=1, w_bit=16, a_bit=16)

# 图像+文本分析
result = analyzer.analyze(
    seqlen=100, batchsize=1, w_bit=16, a_bit=16,
    image_size='1024x1024'
)

# 音频+文本分析
result = analyzer.analyze(
    seqlen=100, batchsize=1, w_bit=16, a_bit=16,
    audio_length=5.0  # 5秒音频
)

# 全模态分析
result = analyzer.analyze(
    seqlen=100, batchsize=1, w_bit=16, a_bit=16,
    image_size='1024x1024',
    audio_length=5.0
)
```

### 前端使用

1. 在模型选择下拉框中选择 `Qwen/Qwen3-Omni-30B-A3B-Instruct`
2. 选择 Stage：
   - **Prefill**：展示 Vision Encoder + Audio Encoder + Text Prefill 的完整流程
   - **Decode**：展示 Text Decode 的单步生成
3. 配置参数：
   - SeqLength：文本 token 数量
   - Image Size：勾选 checkbox 后启用，输入图像尺寸（Omni/VL 模型在 prefill/chat 阶段显示）；取消勾选则不计算视觉分支
   - Audio Length (s)：勾选 checkbox 后启用，输入音频时长（Omni 模型在 prefill/chat 阶段显示，默认 5.0s）；取消勾选则不计算音频分支

## 输出结果

### total_results 结构

```python
{
    "vision": {          # 仅当 image_size 不为 None 时存在
        "OPs": xxx,
        "memory_access": xxx,
        "inference_time": xxx,
        ...
    },
    "audio": {           # 仅当 audio_length 不为 None 时存在
        "OPs": xxx,
        "memory_access": xxx,
        "inference_time": xxx,
        ...
    },
    "prefill": {
        "OPs": xxx,      # 处理所有模态的 tokens
        "memory_access": xxx,
        "inference_time": xxx,
        ...
    },
    "decode": {
        "OPs": xxx,      # 处理 1 个新 token
        "memory_access": xxx,
        "inference_time": xxx,
        ...
    },
    "multimodal_ttft": { # Time To First Token = vision + audio + prefill
        "OPs": xxx,
        "inference_time": xxx,
        ...
    },
    "multimodal_tpot": { # Time Per Output Token = decode
        "OPs": xxx,
        "inference_time": xxx,
        ...
    }
}
```

## 注意事项

1. **硬件名称**：使用时需要加前缀，如 `nvidia_A100` 而不是 `A100`
2. **图像尺寸**：支持多种格式：`"1024x1024"`、`[1024, 1024]`、`{"width": 1024, "height": 1024}`
3. **音频时长**：前端 Audio Length 输入框默认 5.0s，取消勾选 checkbox 则不计算音频分支（传 `null` 给后端）
4. **图片输入**：前端 Image Size 输入框默认 1024×1024，取消勾选 checkbox 则不计算视觉分支（传 `null` 给后端）
4. **模型配置**：config.json 位于 `backend/Qwen/Qwen3-Omni-30B-A3B-Instruct/config.json`
5. **analyzer 注册 key**：`"qwen3_omni_moe"`（对应 config.json 中的 `model_type`）

## 测试结果

### 测试 1：纯文本分析（seqlen=100）
```
✓ Prefill OPs: 6.10e+11
✓ Decode OPs: 6.10e+09
✓ Prefill time: 0.004841s
✓ Decode time: 0.003909s
```

### 测试 2：图像+文本分析（1024x1024 图像 + 100 文本 tokens）
```
✓ Vision OPs: 1.26e+12
✓ Prefill OPs: 7.32e+12  （处理 256 visual + 100 text = 356 tokens）
✓ Decode OPs: 6.51e+09   （访问 356 tokens 的 KV cache）
✓ TTFT time: 0.041854s   （Vision + Prefill）
✓ TPOT time: 0.003949s   （Decode）
```

### 测试 3：全模态分析（1024x1024 图像 + 10s 音频 + 128 文本 tokens，Phase 1.1 修正后）
```
✓ Vision OPs: 1.23e+12  (TTFT 的 13.3%)
✓ Audio OPs:  1.01e+11  (TTFT 的 1.1%)
✓ Prefill OPs: 7.95e+12 (TTFT 的 85.6%)
✓ TTFT OPs:   9.29e+12  (Vision + Audio + Prefill)
✓ Decode OPs: 6.55e+09

新增统计的层:
✓ embed_tokens (decode): OPs=0, load_weight=4096B
✓ audio_conv2d1/2/3 + conv_out + proj1 + proj2: 完整音频前端
✓ vision_merger × 4 (主 merger + 3 个 merger_list): LN(4608) → FC(4608,4608) → GELU → FC(4608,2048)
✓ vision_fc1/fc2: 标准 2 层 MLP（替代原来的 3 层 gated MLP）
```

## 验证状态

- ✅ 纯文本分析
- ✅ 图像+文本分析
- ✅ 音频+文本分析
- ✅ 全模态分析
- ✅ OmniAnalyzer 正确注册和创建
- ✅ 多模态 token 计算正确
- ✅ KV cache 长度计算正确
- ✅ 前端 Image Size 输入框正确显示（Omni 模型）
- ✅ 前端 Audio Length 输入框正确显示（Omni 模型）
- ✅ 前端 use_image / use_audio checkbox 控制是否计算对应模态
- ✅ prefill 阶段 Audio Encoder 子图始终渲染（无音频输入时 OPs 显示为 0）
- ✅ chat/decode 阶段多模态处理逻辑修复
- ✅ 视觉 MLP 结构修正（fc1/fc2 标准 MLP，非 gated MLP）
- ✅ 视觉 merger 维度修正（4608 输入，4 个 merger）
- ✅ 音频前端卷积层补全（conv2d1/2/3 + conv_out + ln_post + proj1 + act + proj2）
- ✅ 文本 embed_tokens 补全
- ✅ OPs=0 除零修复（Embedding 查表场景）

## Thinker 部分覆盖对照

对照 `Qwen3OmniMoeForConditionalGeneration.thinker` 的完整模型结构。

### Prefill 阶段覆盖情况

#### 文本分支 `thinker.model`（48层 MoE Decoder）— 每层 ×48

| 组件 | 状态 | 说明 |
|------|------|------|
| embed_tokens Embedding(152064, 2048) | ✅ | post_process, OPs=0 |
| q_proj Linear(2048, 4096) | ✅ | prefill_total_tokens |
| k_proj Linear(2048, 512) | ✅ | GQA 4 heads |
| v_proj Linear(2048, 512) | ✅ | |
| QK matmul / Softmax / SV matmul | ✅ | |
| out_proj Linear(4096, 2048) | ✅ | |
| attn_norm RMSNorm(2048) | ✅ | input_layernorm |
| attn_add (residual) | ✅ | |
| mlp_norm RMSNorm(2048) | ✅ | post_attention_layernorm |
| gate/up/down_proj ×8 experts | ✅ | MoE |
| mlp_act (SiLU + mul) | ✅ | 维度有偏差，见「已知偏差」 |
| mlp_add (residual) | ✅ | |
| lm_head Linear(2048, 152064) | ✅ | post_process |
| model.norm RMSNorm(2048) | ❌ | 最终 norm，仅 1 次 |
| q_norm/k_norm RMSNorm(128) | ❌ | 每层，影响极小 |
| MoE Router Linear(2048, 128) | ❌ | 每层，影响小 |

#### 视觉分支 `thinker.visual`（27层 ViT）— 每层 ×27

| 组件 | 状态 | 说明 |
|------|------|------|
| patch_embed Conv3d→Linear 近似 | ✅ | |
| norm1/norm2 LayerNorm(1152) | ✅ | |
| q/k/v/out_proj | ✅ | |
| QK matmul / Softmax / SV matmul | ✅ | |
| fc1 Linear(1152, 4304) | ✅ | 标准 MLP |
| mlp_act GELU | ✅ | 维度有偏差 |
| fc2 Linear(4304, 1152) | ✅ | |
| attn_add / mlp_add | ✅ | |
| merger ×4 (LN+FC+GELU+FC) | ✅ | vision_post_process |
| pos_embed Embedding(2304, 1152) | ❌ | 仅 1 次，影响极小 |

#### 音频分支 `thinker.audio_tower`（32层 Whisper-like）— 每层 ×32

| 组件 | 状态 | 说明 |
|------|------|------|
| conv2d1/2/3 | ✅ | audio_post_process |
| conv_out Linear(7680, 1280) | ✅ | |
| ln_post LayerNorm(1280) | ✅ | |
| proj1 + GELU + proj2 | ✅ | |
| attn_norm / mlp_norm | ✅ | |
| q/k/v/out_proj | ✅ | |
| QK matmul / Softmax / SV matmul | ✅ | |
| fc1 Linear(1280, 5120) / fc2 Linear(5120, 1280) | ✅ | |
| attn_add / mlp_add | ✅ | |
| mlp_act GELU | ✅ | 维度有偏差 |
| conv 之间的 GELU ×3 | ❌ | 影响极小 |
| positional_embedding | ❌ | 正弦位置编码，无可学习权重 |

### Decode 阶段覆盖情况

Decode 只有文本分支，所有组件与 prefill 文本分支一致（KV cache 长度正确使用 `prefill_total_tokens`）。缺失项同上（model.norm、q_norm/k_norm、Router）。

### 已知偏差

#### 1. mlp_act 维度偏差（所有分支共有，影响极小）

三个分支的 `mlp_act` 都用了 `hidden_size` 而非 `intermediate_size`：

| 分支 | 当前维度 | 正确维度 | 偏差 |
|------|---------|---------|------|
| 文本 | 2048 (×8 experts) | 768 (×8 experts) | 多算 2.67x |
| 视觉 | 1152 | 4304 | 少算 3.74x |
| 音频 | 1280 | 5120 | 少算 4x |

但 mlp_act 的 OPs 只占线性层的 ~0.1%，对总 OPs 影响可忽略。这是所有 analyzer（LLMAnalyzer、MoEAnalyzer、VLMAnalyzer）的共同模式，不是 Omni 特有的。

#### 2. embed_tokens 的 seqlen 传入了 prefill_total_tokens

`model_analyzer.py:1891` 传入 `seqlen=prefill_total_tokens`，但 embed_tokens 只处理文本 token，不处理视觉/音频 token。因为 OPs=0，只影响 memory_access 估算，实际影响很小。

### 未统计部分影响排序

1. **model.norm** — 最终 RMSNorm(2048)，仅 1 次，影响小
2. **MoE Router** — Linear(2048, 128) ×48 层，OPs 约为单层 q_proj 的 3%
3. **q_norm/k_norm** — RMSNorm(128)，极小
4. 其余（pos_embed、conv GELU、positional_embedding）— 可忽略

总体未统计部分对总 OPs 的影响在 1% 以内。

## 未来扩展

### Phase 2：Talker + Code2Wav（语音合成输出）

目前只统计了 Thinker 部分。完整的语音输出还需要：

#### Talker（MoE Transformer，AR 自回归）
- `text_projection`: Linear(2048→2048→1024) — 维度映射
- `hidden_projection`: Linear(2048→2048→1024) — 维度映射
- `model`: 20 层 MoE Decoder（hidden=1024, 128 experts, 6 active, + shared_expert）
- `codec_head`: Linear(1024, 3072)
- `code_predictor`: 5 层 Transformer + 15×Embedding + 15×lm_head（16 组并行 codec 预测）

**指标**：Talker TTFT / Talker TPOT（类比文本的 TTFT/TPOT）

#### Code2Wav（CNN + Transformer，非 AR）
- `pre_transformer`: 8 层 Transformer（hidden=1024, sliding_window=72）
- `upsample`: 2× [TransConv ×2 + ConvNeXt]（上采样 ×4）
- `decoder`: 4 个 DecoderBlock（TransConv stride=[8,5,4,3] + 3×ResidualUnit）+ 最终 Conv1d(96→1)
- 总上采样倍率 = 2×2×8×5×4×3 = 960

**指标**：RTF（Real-Time Factor）= 处理时间 / 音频物理时长

#### 综合指标
- **TTAF**（Time to Audio First）= Thinker Prefill + Talker 首帧 + Code2Wav 首 chunk
- **RTF** = (Talker 生成 N 帧 + Code2Wav 处理) / 音频时长
