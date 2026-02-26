# Qwen3-Omni-30B-A3B-Instruct Roofline 分析实现

## 实现概述

已成功为 Qwen3-Omni-30B-A3B-Instruct 模型添加 Roofline 性能分析支持（Phase 1：Thinker 部分，不含语音合成输出）。

## 修改的文件

### 1. 新增文件

- **`backend/models/qwen3_omni_moe.py`**
  - 实现 Qwen3-Omni 模型的参数提取函数
  - 支持三个分支：
    - 文本分支（MoE Transformer，48层，8个active experts，hidden=2048）
    - 视觉分支（Vision Encoder，27层，hidden=1152，16头）
    - 音频分支（Audio Encoder，32层，d_model=1280，20头）
  - 文本图结构复用 `qwen3_moe` 的 `transformer_layer_graph`
  - 视觉图结构复用 `qwen3_vl` 的 `vision_layer_graph`
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
   - Image Size：图像尺寸（Omni/VL 模型在 prefill/chat 阶段自动显示）
   - Audio Length (s)：音频时长（Omni 模型在 prefill/chat 阶段自动显示，默认 5.0s）

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
3. **音频时长**：前端 Audio Length 输入框默认 5.0s，设为 0 则不计算音频分支
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
- ✅ prefill 阶段 Audio Encoder 子图正确渲染
- ✅ chat/decode 阶段多模态处理逻辑修复

## 未来扩展

### Phase 2：Talker + Code2Wav（语音合成）

如果需要支持语音合成分析，需要：
1. 在 `qwen3_omni_moe.py` 中添加 Talker 和 Code2Wav 的参数提取
2. 在 `OmniAnalyzer` 中添加对应的分析逻辑
3. 在前端添加 "Output Mode" 选项（Text / Audio）
