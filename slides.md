---
marp: true
theme: default
paginate: true
backgroundColor: "#0f172a"
color: "#e2e8f0"
style: |
  section {
    font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    padding: 48px 64px;
  }
  h1 {
    color: #38bdf8;
    font-size: 2em;
    border-bottom: 2px solid #38bdf8;
    padding-bottom: 12px;
  }
  h2 {
    color: #7dd3fc;
    font-size: 1.4em;
  }
  h3 {
    color: #94a3b8;
    font-size: 1em;
    font-weight: normal;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
  }
  th {
    background: #1e3a5f;
    color: #38bdf8;
    padding: 8px 12px;
    text-align: left;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid #1e293b;
  }
  tr:nth-child(even) td { background: #1e293b; }
  code {
    background: #1e293b;
    color: #7dd3fc;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.88em;
  }
  pre {
    background: #1e293b;
    border-left: 3px solid #38bdf8;
    padding: 16px 20px;
    border-radius: 6px;
    font-size: 0.78em;
    line-height: 1.6;
  }
  blockquote {
    border-left: 4px solid #f59e0b;
    background: #1c1a0f;
    padding: 12px 20px;
    margin: 16px 0;
    color: #fcd34d;
    font-style: normal;
  }
  .highlight { color: #f59e0b; font-weight: bold; }
  .green { color: #4ade80; }
  .red { color: #f87171; }
  .dim { color: #64748b; }
---

<!-- _backgroundColor: "#0a0f1e" -->
<!-- _color: "#e2e8f0" -->

# Qwen3-VL 多模态模型
# Roofline 性能分析

### 基于 LLM-Viewer 的三阶段推理性能建模

<br>

**模型：** Qwen3-VL-8B · Qwen3-VL-32B
**工具：** LLM-Viewer · Roofline Model

---

# 第一页：为什么 VLM 需要单独分析？

### 纯 LLM 只处理 Token 序列，VLM 需要先"看懂图片"再"生成文字"

<br>

| 模型类型 | 代表模型 | 输入 | 计算阶段 | 核心瓶颈 |
|---------|---------|------|---------|---------|
| **LLM** | Qwen3-32B, Llama | 纯文本 Token | prefill + decode | KV Cache 带宽 |
| **MoE** | Qwen3-MoE-30B | 纯文本 Token | prefill + decode | Expert 路由 + 稀疏激活 |
| **VLM** | **Qwen3-VL-32B** | **图像 + 文本** | **vision + prefill + decode** | **视觉编码器 + 跨模态对齐** |

<br>

> **关键差异：** LLM/MoE 是单一 Transformer 流水线；VLM 是**双分支架构** = ViT 视觉编码器 + LLM 语言解码器，两个分支的计算特征完全不同，必须分开建模。

---

# 第二页：三阶段推理流水线

### Qwen3-VL 推理 = Vision → Prefill → Decode 三个串行阶段

<br>

```
输入图像 (1024×1024)
    │
    ▼  patch_size=14 → 73×73 = 5,329 patches
[Stage 1: Vision]  ─── ViT 视觉编码器（VLM 专属）
    │  spatial_merge → visual tokens 注入语言空间
    │
    ▼  seqlen = text_tokens + visual_tokens（序列比纯LLM更长）
[Stage 2: Prefill] ─── LLM Prefill（文本 + Visual Tokens 拼接）
    │
    ▼  每步生成 1 个 Token（与纯 LLM 完全相同）
[Stage 3: Decode]  ─── 自回归生成
    │
    ▼
输出文本
```

**端到端指标：**
- **TTFT**（首 Token 延迟）= `vision` + `prefill`　　← VLM 比 LLM 多了 vision 阶段
- **TPOT**（每 Token 延迟）= `decode`　　　　　　← 与纯 LLM 相同

---

# 第二页（续）：VLM 专属算子

### Vision 阶段引入了 LLM 完全没有的算子

<br>

| 算子类别 | LLM | VLM 新增 |
|---------|-----|---------|
| Patch Embedding | ✗ | ✅ `vision_patch_embed` |
| Vision Attention | ✗ | ✅ `vision_qk_matmul` · `vision_sv_matmul` |
| Vision FFN | ✗ | ✅ `vision_gate/up/down_proj` |
| Vision Norm | ✗ | ✅ `vision_norm1` · `vision_norm2` |
| 跨模态投影 | ✗ | ✅ `vision_proj`（ViT → LLM 空间） |
| 文本 Attention | ✅ | ✅（完全相同） |
| 文本 FFN | ✅ | ✅（完全相同） |

<br>

> **结论：** VLM 的 Vision 分支是一个完整的 ViT，有独立的 Attention、FFN、Norm，最后通过 `vision_proj` 投影到语言空间，这部分在 LLM/MoE 中完全不存在。

---

# 第三页：Roofline 模型——三阶段瓶颈量化

### 用算术强度判断每个阶段是算力瓶颈还是带宽瓶颈

<br>

```
算术强度 (AI) = 计算量 (OPs) / 内存访问量 (Bytes)
转折点        = 峰值算力 (FLOPS) / 内存带宽 (GB/s)

AI < 转折点  →  Memory-Bound（带宽瓶颈）
AI ≥ 转折点  →  Compute-Bound（算力瓶颈）
```

<br>

| 阶段 | 算术强度 | 瓶颈 | 原因 |
|------|---------|------|------|
| **Vision** | 中等 | Memory-Bound | ViT 层数少，Patch 数量有限 |
| **Prefill** | 高（长序列） | Compute-Bound | Visual Token + 文本 Token，矩阵乘法密集 |
| **Decode** | 低 | Memory-Bound | 每步仅生成 1 Token，权重复用率极低 |

---

# 第三页（续）：与纯 LLM 的 Roofline 对比

### VLM 多了一个 Memory-Bound 的 Vision 阶段，Prefill 序列也更长

<br>

```
纯 LLM:
  prefill ──→  Compute-Bound
  decode  ──→  Memory-Bound

VLM（Qwen3-VL）:
  vision  ──→  Memory-Bound   ← 新增！ViT 编码器独立建模
  prefill ──→  Compute-Bound  ← 序列更长（含 Visual Tokens）
  decode  ──→  Memory-Bound   ← 与 LLM 完全相同
```

<br>

**图像分辨率的影响（VLM 独有变量）：**

| 图像尺寸 | Patches 数量 | Visual Tokens | 对 TTFT 的影响 |
|---------|------------|--------------|--------------|
| 1024×1024 | 73×73 = 5,329 | ~5,329 | 基准 |
| 512×512 | 37×37 = 1,369 | ~1,369 | TTFT 大幅降低 |

> **图像分辨率是 VLM 性能调优的第一旋钮**，直接决定 Vision 计算量和 Prefill 序列长度。

---

# 第四页：LLM-Viewer 分析能力 & 优化洞察

### 工具对 Qwen3-VL 的三阶段独立建模

<br>

**LLM-Viewer 分析参数（VLM 专属）：**

| 参数 | 说明 | VLM 特有 |
|-----|------|---------|
| `image_size` | 图像尺寸，触发 Vision 分支分析 | ✅ |
| `seqlen` | 文本序列长度 | — |
| `tp_size` | 张量并行度（视觉+文本均支持） | — |
| `w_bit / kv_bit` | 权重/KV Cache 量化 | — |
| `use_flashattention` | 视觉+文本分支均可开启 | — |

<br>

**关键优化手段：**

| 优化 | 影响阶段 | 效果 |
|-----|---------|------|
| 降低图像分辨率 | Vision + Prefill | TTFT 大幅降低 |
| 增大 spatial_merge_size | Vision → Prefill | 减少 Visual Token 数量 |
| 权重量化 w_bit=4 | 全部阶段 | Memory-Bound 阶段效果显著 |
| 张量并行 TP=4/8 | 全部阶段 | 线性扩展算力 |

---

<!-- _backgroundColor: "#0a0f1e" -->

# 总结

<br>

**VLM ≠ LLM + 图像输入，而是完全不同的性能模型**

<br>

| | LLM | VLM |
|--|-----|-----|
| 推理阶段 | 2 个 | **3 个**（多 Vision） |
| TTFT 构成 | prefill | **vision + prefill** |
| 新增算子 | — | **ViT 全套 + vision_proj** |
| 性能调优变量 | seqlen | seqlen + **image_size** |

<br>

> LLM-Viewer 通过三阶段独立 Roofline 建模，精确定位视觉编码、文本预填充、自回归解码各自的瓶颈，为多模态模型的硬件选型和推理优化提供量化依据。

<br>

### 演讲时间：约 15 分钟 · 4+1 页
