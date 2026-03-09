# CNN Roofline 分析模块

## 背景与目标

LLM-Viewer 原本只支持 Transformer 类模型（LLM/VLM）的 roofline 分析。本次改动在**完全不影响原有 LLM 功能**的前提下，新增了对 CNN 类模型的 roofline 分析支持，目前已集成 MobileNetV2 和 YOLOv8n，后续可按统一规范扩展任意 CNN 模型。

---

## 整体设计方案

### 核心原则

- **完全隔离**：所有 CNN 代码放在新文件中，原有 LLM 代码零修改（除 `app.py` 新增两个端点）
- **可扩展**：新增模型只需新建一个文件 + 一行 import，无需改动框架代码
- **复用基础设施**：复用现有的 `roofline_analyze()`、`get_hardware_info()`、硬件配置等

### 架构图

```
前端 /cnn.html
    └── CnnApp.vue (provide 全局状态)
        ├── CnnHeader.vue     → GET /get_available_cnn
        ├── CnnLeftPanel.vue
        │   └── CnnConfig.vue (batchsize / w_quant / a_quant)
        └── CnnGraph.vue      → POST /get_cnn_graph

后端 app.py
    ├── POST /get_cnn_graph    → get_cnn_graph.py → CNNAnalyzer.analyze()
    └── GET  /get_available_cnn → cnn_analyzer.get_available_cnn_models()

数据层
    backend/cnn_config/
        ├── mobilenet-v2.json   (由 csv_to_json.py 从 CSV 生成)
        └── yolo_v8n.json
```

---

## 新增文件说明

### 后端

#### `backend/cnn_config/csv_to_json.py`

将 OpenVINO 导出的 `computational_complexity.csv` 转换为结构化 JSON。

关键处理：
- 解析 `f32(1x3x640x640)` 格式的 tensor shape，计算 `numel` 和 `size_bytes`
- `GFLOPs = -1.0000` 的层（Interpolate、ReduceMean 等）存为 `null`
- 支持 5D 权重张量（GroupConvolution 的 depthwise 卷积）

使用方式：
```bash
cd backend/cnn_config
python csv_to_json.py mobilenet-v2.csv mobilenet-v2.json
python csv_to_json.py yolo_v8n.csv yolo_v8n.json
```

生成的 JSON 格式：
```json
{
  "layers": [
    {
      "layer_type": "Convolution",
      "layer_name": "/model.0/conv/Conv",
      "gflops": 0.2295,
      "input_tensors": [
        {"dtype": "f32", "shape": [1,3,640,640], "numel": 1228800, "size_bytes": 4915200},
        {"dtype": "f32", "shape": [16,3,3,3],    "numel": 432,     "size_bytes": 1728}
      ],
      "output_tensors": [
        {"dtype": "f32", "shape": [1,16,320,320], "numel": 1638400, "size_bytes": 6553600}
      ]
    }
  ]
}
```

---

#### `backend/cnn_analyzer.py`

CNN 分析框架的核心，包含：

**全局注册表 + 装饰器**
```python
CNN_MODEL_REGISTRY: dict[str, type] = {}

def register_cnn_model(model_id: str):
    def decorator(cls):
        CNN_MODEL_REGISTRY[model_id] = cls
        return cls
    return decorator
```

**`CNNAnalyzer` 基类**

子类只需实现两个方法：

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `get_layer_graph()` | `dict[str, list[str]]` | DAG 拓扑，key 为层名，value 为输入层名列表 |
| `get_layers()` | `list[dict]` | 每层的 OPs、权重/激活字节数（FP32 基准） |

**`analyze(w_bit, a_bit, batchsize)` 通用分析逻辑**

量化缩放方式与 LLM 侧完全一致：JSON 中 `size_bytes` 基于 FP32（4 bytes/element），量化后按比例缩小：

```python
w_scale = w_bit / 32   # FP16 → 0.5, INT8 → 0.25, INT4 → 0.125
a_scale = a_bit / 32

load_weight   = layer["load_weight_bytes"] * w_scale        # 权重访存，与 batch 无关
load_act      = layer["load_act_bytes"]    * a_scale * batchsize
store_act     = layer["store_act_bytes"]   * a_scale * batchsize
memory_access = load_weight + load_act + store_act
```

对比 LLM 侧的写法（从维度出发）：
```python
w_byte = w_bit / 8
load_weight = ic * oc * w_byte   # 等价，只是基准不同
```

每层调用 `roofline_analyze(bandwidth, max_OPS, OPs, memory_access)` 得到：
- `arithmetic_intensity`：算术强度（OPs/byte）
- `performance`：实际性能（受算力或带宽限制）
- `bound`：`"compute"` 或 `"memory"`
- `inference_time`：理论推理时间

---

#### `backend/cnn_models/mobilenet_v2.py`

MobileNetV2 子类，线性拓扑（无分支）。

```python
@register_cnn_model("mobilenet_v2")
class MobileNetV2Analyzer(CNNAnalyzer):
    ...
```

特殊处理：
- 跳过 `/fq_weights_0`、`/fq_weights_1` 量化辅助层（不产生实际计算）
- `Add`、`Subtract`、`Multiply`、`ReduceMean` 等无权重层，`load_weight = 0`

---

#### `backend/cnn_models/yolov8n.py`

YOLOv8n 子类，带分支的 DAG 拓扑（FPN 结构）。

网络拓扑：
```
input
  └── Backbone (model.0 → model.9, 线性)
        ├── Neck 上采样 1: Resize(model.10) + concat(model.6) → model.12
        │     └── Neck 上采样 2: Resize(model.13) + concat(model.4) → model.15
        │           ├── Head P3 (80×80): model.15 → cv2.0, cv3.0
        │           └── Downsample: model.15 → model.16
        │                 └── Head P4 (40×40): model.16 + concat(model.12) → model.18 → cv2.1, cv3.1
        │                       └── Downsample: model.18 → model.19
        │                             └── Head P5 (20×20): model.19 + concat(model.9) → model.21 → cv2.2, cv3.2
        │                                   └── DFL: dfl/conv → Sub → Add → Div → Mul
        └── output ← [P3, P4, P5]
```

Concat 节点（多输入）在 `get_layer_graph()` 中显式指定两个 parent：
```python
if i == 0:
    parents = [prev, m6_out]  # Resize 输出 + backbone feature
else:
    parents = [prev]
graph[name] = parents
```

---

#### `backend/get_cnn_graph.py`

对接后端 API 与分析框架，将分析结果转换为前端所需的节点/边格式：

```python
def get_cnn_graph(model_id, hardware, cnn_config):
    w_bit = get_quant_bit(cnn_config["w_quant"])  # "FP16"→16, "8-bit"→8, "4-bit"→4
    a_bit = get_quant_bit(cnn_config["a_quant"])
    batchsize = int(cnn_config["batchsize"])

    analyzer = get_cnn_analyzer(model_id, hardware)
    result = analyzer.analyze(w_bit, a_bit, batchsize)
    # 构建 nodes / edges 返回给前端
```

---

### 后端修改文件

#### `backend/app.py`（新增 2 个端点）

```python
@app.route("/get_cnn_graph", methods=["POST"])
def get_cnn_graph_api():
    # 接收 model_id, hardware, cnn_config
    # 返回 nodes, edges, total_results, hardware_info

@app.route("/get_available_cnn", methods=["GET"])
def get_available_cnn():
    # 返回 available_hardwares, available_model_ids（CNN 模型列表）
```

---

### 前端

CNN 前端完全独立，通过 Vite 多页构建实现：

| 访问地址 | 功能 |
|----------|------|
| `http://host:5173/` | 原 LLM Viewer（不受影响） |
| `http://host:5173/cnn.html` | CNN Roofline 分析页面 |

#### `frontend/cnn.html` + `frontend/src/cnn_main.js`

CNN 页面独立入口，挂载 `CnnApp.vue`。

#### `frontend/src/CnnApp.vue`

根组件，通过 `provide()` 向子组件注入全局状态：

```js
provide("model_id", ref('mobilenet_v2'))
provide("hardware", ref('nvidia_A100'))
provide("global_cnn_config", ref({ w_quant: "FP16", a_quant: "FP16", batchsize: 1 }))
provide("total_results", ref({}))
provide("ip_port", ref('127.0.0.1:5000'))
provide("global_update_trigger", ref(1))
```

#### `frontend/src/components/CnnHeader.vue`

调用 `/get_available_cnn` 获取模型和硬件列表，提供 Model / Hardware / Server 下拉选择。

#### `frontend/src/components/left_controls/CnnConfig.vue`

CNN 专用配置面板，相比 LLM 侧去掉了 SeqLength、KV Cache、FlashAttention 等无关参数，只保留：
- **Batchsize**（1~256）
- **Weight Quantization**（FP16 / 8-bit / 4-bit / 2-bit）
- **Activation Quantization**（FP16 / 8-bit）

配置变更时通过 `global_update_trigger += 1` 触发 `CnnGraph.vue` 重新请求后端。

#### `frontend/src/components/CnnGraph.vue`

复用 G6 图可视化和 Chart.js roofline 图，调用 `/get_cnn_graph` 端点。与 `Graph.vue` 的主要区别：
- 请求体发送 `cnn_config` 而非 `inference_config`
- G6 容器 id 改为 `cnnGraphContainer`（避免与 LLM 页面冲突）
- Chart canvas id 改为 `cnnLineChart`

#### `frontend/vite.config.js`（修改）

新增多页构建入口：
```js
build: {
  rollupOptions: {
    input: {
      main: resolve(__dirname, 'index.html'),
      cnn:  resolve(__dirname, 'cnn.html'),
    }
  }
}
```

---

## 数据文件

所有 CNN 相关数据文件统一存放在 `backend/cnn_config/`：

| 文件 | 说明 |
|------|------|
| `csv_to_json.py` | CSV → JSON 转换工具 |
| `mobilenet-v2.csv` | MobileNetV2 原始 computational_complexity 数据 |
| `mobilenet-v2.json` | 解析后的结构化层数据（161 层） |
| `yolo_v8n.csv` | YOLOv8n 原始数据 |
| `yolo_v8n.json` | 解析后的结构化层数据（133 层） |

---

## 量化对推理时间的影响

以 YOLOv8n 在 `intel_Ultra_225H iGPU` 上为例：

| w_quant / a_quant | 访存量 | 推理时间 |
|-------------------|--------|----------|
| FP16 / FP16 | 141 MB | 1.344 ms |
| INT8 / INT8 | 71 MB  | 0.672 ms |
| INT4 / INT8 | 69 MB  | 0.659 ms |

- **OPs 不变**：量化不改变计算量
- **访存量减少**：权重量化减少权重访存，激活量化减少激活访存
- **推理时间变化**：取决于模型是 memory-bound 还是compute-bound；memory-bound 模型量化收益更显著

---

## 新增 CNN 模型的步骤

1. **准备数据**：用 OpenVINO 导出 `computational_complexity.csv`，运行转换工具：
   ```bash
   cd backend/cnn_config
   python csv_to_json.py resnet50.csv resnet50.json
   ```

2. **新建模型文件** `backend/cnn_models/resnet50.py`：
   ```python
   from pathlib import Path
   from cnn_analyzer import CNNAnalyzer, register_cnn_model
   import json

   _JSON_PATH = Path(__file__).parent.parent / "cnn_config" / "resnet50.json"

   @register_cnn_model("resnet50")
   class ResNet50Analyzer(CNNAnalyzer):
       def _load_json_layers(self):
           with open(_JSON_PATH) as f:
               return json.load(f)["layers"]

       def get_layer_graph(self):
           # 线性结构直接用循环构建，有残差连接需手动指定多 parent
           ...

       def get_layers(self):
           # 遍历层，返回 OPs / load_weight_bytes / load_act_bytes / store_act_bytes
           ...
   ```

3. **注册到 `backend/cnn_models/__init__.py`**：
   ```python
   from . import mobilenet_v2
   from . import yolov8n
   from . import resnet50   # 新增这一行
   ```

4. 重启后端，前端 CNN 页面的 Model 下拉框自动出现 `resnet50`。
