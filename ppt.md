# Qwen3-VL 多模态模型 Roofline 性能分析
## 演讲提纲 & PPT 文稿（约15分钟）

---

## 演讲逻辑与顺序

```
1. 背景引入：为什么多模态模型与纯LLM不同？（2分钟）
2. 架构对比：VLM vs LLM vs MoE 的核心差异（3分钟）
3. 核心分析：三阶段 Roofline 模型拆解（5分钟）
4. 工具演示：LLM-Viewer 对 Qwen3-VL 的可视化分析（3分钟）
5. 总结与洞察（2分钟）
```

---

# PPT 正文（4页）

---

## 第1页：为什么多模态模型需要单独分析？

### 标题：从文本到图文——多模态推理的新挑战

**核心问题：**
> 纯 LLM 只处理 Token 序列，而 VLM 需要先"看懂图片"再"生成文字"。
> 这两个过程的计算特征完全不同，不能用同一把尺子衡量。

**三类模型的本质区别：**

| 模型类型 | 代表模型 | 输入 | 计算阶段 | 核心瓶颈 |
|---------|---------|------|---------|---------|
| **LLM** | Qwen3-32B, Llama | 纯文本 Token | prefill + decode | KV Cache 带宽 |
| **MoE** | Qwen3-MoE-30B | 纯文本 Token | prefill + decode | Expert 路由 + 稀疏激活 |
| **VLM** | Qwen3-VL-32B | 图像 + 文本 | **vision + prefill + decode** | 视觉编码器 + 跨模态对齐 |

**关键差异（重点）：**
- LLM/MoE：单一 Transformer 流水线
- VLM：**双分支架构** = 视觉编码器（ViT）+ 语言解码器（LLM）
- VLM 新增 `vision` 阶段：图像 → Patch → Visual Token → 语言空间投影

**演讲要点：**
> "今天我们重点看的是 Qwen3-VL-8B 和 32B，它们比同参数量的纯 LLM 多了一整个视觉编码器，
> 这个编码器的性能特征和 LLM 完全不同，必须单独建模。"

---

## 第2页：VLM 架构拆解——三阶段计算流水线

### 标题：Qwen3-VL 的三阶段推理流程

**完整推理流程：**

```
输入图像 (1024×1024)
    │
    ▼
[Stage 1: Vision]  ──── ViT 视觉编码器
    │  patch_size=14 → 73×73=5329 patches
    │  spatial_merge → visual tokens
    │
    ▼
[Stage 2: Prefill] ──── LLM Prefill（文本 + Visual Tokens 拼接）
    │  seqlen = text_tokens + visual_tokens
    │
    ▼
[Stage 3: Decode]  ──── 自回归生成（与纯 LLM 完全相同）
    │
    ▼
输出文本
```

**VLM 专属算子（与 LLM 的核心区别）：**

| 算子类别 | LLM 有 | VLM 新增 |
|---------|--------|---------|
| Patch Embedding | ✗ | ✅ `vision_patch_embed` |
| Vision Attention | ✗ | ✅ `vision_qk_matmul`, `vision_sv_matmul` |
| Vision FFN | ✗ | ✅ `vision_gate/up/down_proj` |
| Vision Norm | ✗ | ✅ `vision_norm1`, `vision_norm2` |
| 跨模态投影 | ✗ | ✅ `vision_proj`（ViT→LLM 空间） |
| 文本 Attention | ✅ | ✅（相同） |
| 文本 FFN | ✅ | ✅（相同） |

**关键指标：**
- **TTFT**（Time To First Token）= `vision` 延迟 + `prefill` 延迟
- **TPOT**（Time Per Output Token）= `decode` 延迟（与纯 LLM 相同）
- VLM 的 TTFT 远高于同参数 LLM，因为多了视觉编码阶段

---

## 第3页：Roofline 模型分析——三阶段性能特征对比

### 标题：用 Roofline 模型量化三阶段瓶颈

**Roofline 核心公式：**

```
算术强度 (AI) = 计算量(OPs) / 内存访问量(Bytes)
转折点       = 峰值算力(FLOPS) / 内存带宽(GB/s)

if AI < 转折点:  → 内存带宽瓶颈（Memory-Bound）
else:            → 计算瓶颈（Compute-Bound）
```

**三阶段 Roofline 特征（以 H100 + Qwen3-VL-32B 为例）：**

| 阶段 | 典型算术强度 | 瓶颈类型 | 原因 |
|------|------------|---------|------|
| **Vision** | 中等 | 内存带宽为主 | Patch 数量有限，ViT 层数少 |
| **Prefill** | 高（长序列时） | 计算为主 | Visual Token + 文本 Token 序列长，矩阵乘法密集 |
| **Decode** | 低 | 内存带宽 | 每步只生成1个 Token，权重复用率极低 |

**与纯 LLM 的 Roofline 差异（重点）：**

```
纯 LLM:
  prefill ──→ [Compute-Bound]
  decode  ──→ [Memory-Bound]

VLM:
  vision  ──→ [Memory-Bound]  ← 新增！ViT 编码器独立分析
  prefill ──→ [Compute-Bound] ← 序列更长（含 Visual Tokens）
  decode  ──→ [Memory-Bound]  ← 与 LLM 相同
```

**图像尺寸对性能的影响：**

```
image_size = 1024×1024:
  patches = ceil(1024/14) × ceil(1024/14) = 73×73 = 5,329
  visual_tokens ≈ 5,329（spatial_merge_size=1 时）

image_size = 512×512:
  patches = 37×37 = 1,369
  visual_tokens ≈ 1,369（减少 75%！）
```

> **结论：** 图像分辨率直接决定 Vision 阶段计算量和 Prefill 阶段序列长度，
> 是 VLM 性能调优的第一旋钮。

---

## 第4页：LLM-Viewer 工具演示 & 优化洞察

### 标题：LLM-Viewer 对 Qwen3-VL 的可视化分析与优化建议

**工具核心能力（针对 VLM）：**

```
LLM-Viewer 对 Qwen3-VL 的分析维度：

┌─────────────────────────────────────────────┐
│  模型选择：Qwen3-VL-8B / Qwen3-VL-32B       │
│  硬件选择：H100 / A100 / RTX4090 / ...      │
├─────────────────────────────────────────────┤
│  分析参数：                                  │
│    image_size: 1024×1024（触发视觉分支）     │
│    seqlen: 文本序列长度                      │
│    tp_size: 张量并行度（视觉+文本均支持）    │
│    w_bit/a_bit/kv_bit: 量化配置             │
│    use_flashattention: Flash Attention 开关  │
├─────────────────────────────────────────────┤
│  输出结果：                                  │
│    - 三阶段 Roofline 图（vision/prefill/decode）│
│    - 逐算子延迟分解（视觉算子 vs 文本算子）  │
│    - TTFT / TPOT 端到端延迟估算             │
│    - 内存占用：权重 + KV Cache + 激活值      │
└─────────────────────────────────────────────┘
```

**关键优化洞察（VLM 专属）：**

| 优化手段 | 影响阶段 | 效果 |
|---------|---------|------|
| 降低图像分辨率 | Vision + Prefill | Visual Token 数量↓，TTFT 大幅降低 |
| 增大 spatial_merge_size | Vision → Prefill | 合并相邻 Patch，减少 Visual Token |
| Flash Attention（视觉分支） | Vision | 减少 Attention 内存访问 |
| 张量并行（TP=2/4/8） | 全部阶段 | 线性扩展算力，降低延迟 |
| KV Cache 量化（kv_bit=4） | Decode | 减少 KV Cache 内存，提升 Decode 带宽利用率 |
| 权重量化（w_bit=4） | 全部阶段 | 减少权重加载量，对 Memory-Bound 阶段效果显著 |

**8B vs 32B 对比洞察：**

```
Qwen3-VL-8B:
  - 视觉编码器参数量相近（ViT 参数与模型规模相对独立）
  - 文本解码器更小 → Decode 阶段更快
  - 适合实时交互场景

Qwen3-VL-32B:
  - 文本理解能力更强
  - Decode 阶段内存带宽压力更大
  - 需要更大 TP 或量化来保持实时性
```

**总结：**
> VLM 的性能分析不能简单套用 LLM 的框架。
> LLM-Viewer 通过三阶段独立 Roofline 建模，
> 精确定位视觉编码、文本预填充、自回归解码各自的瓶颈，
> 为多模态模型的硬件选型和推理优化提供量化依据。

---

## 演讲时间分配

| 页码 | 内容 | 时间 |
|------|------|------|
| 第1页 | 背景引入 + 三类模型对比 | 3分钟 |
| 第2页 | VLM 架构 + 三阶段流程 | 4分钟 |
| 第3页 | Roofline 分析 + 与LLM对比 | 4分钟 |
| 第4页 | 工具演示 + 优化洞察 | 4分钟 |
| **合计** | | **约15分钟** |
