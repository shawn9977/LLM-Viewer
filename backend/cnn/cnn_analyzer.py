"""
cnn_analyzer.py
CNN roofline 分析框架：基类 + 子类注册表。

新增模型只需：
  1. 在 cnn_models/ 下新建文件，继承 CNNAnalyzer
  2. 加 @register_cnn_model("your_model_id") 装饰器
  3. 在 app.py 中 import cnn_models.your_model
"""

import sys
import os

# 添加父目录到路径，以便导入 hardwares, roofline_model
_parent_dir = os.path.dirname(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from hardwares import get_hardware_info
from roofline_model import roofline_analyze

# ── 全局注册表 ────────────────────────────────────────────────
CNN_MODEL_REGISTRY: dict[str, type] = {}


def register_cnn_model(model_id: str):
    """类装饰器，将 CNNAnalyzer 子类注册到全局注册表。"""
    def decorator(cls):
        CNN_MODEL_REGISTRY[model_id] = cls
        cls.model_id = model_id
        return cls
    return decorator


def get_cnn_analyzer(model_id: str, hardware: str) -> "CNNAnalyzer":
    if model_id not in CNN_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown CNN model: '{model_id}'. "
            f"Available: {list(CNN_MODEL_REGISTRY.keys())}"
        )
    return CNN_MODEL_REGISTRY[model_id](hardware)


def get_available_cnn_models() -> list[str]:
    return list(CNN_MODEL_REGISTRY.keys())


# ── 基类 ──────────────────────────────────────────────────────
class CNNAnalyzer:
    """
    CNN roofline 分析基类。

    子类必须实现：
      - get_layer_graph() -> dict[str, list[str]]
          返回 DAG：{layer_name: [input_layer_names]}
          特殊节点 "input" / "output" 不参与计算。

      - get_layers() -> list[dict]
          返回每层参数列表，每个 dict 包含：
            name             : str   层名（与 layer_graph 的 key 对应）
            OPs              : float 单 batch 的浮点运算量（FLOPs）
            load_weight_bytes: int   权重读取字节数（与 batch 无关）
            load_act_bytes   : int   单 batch 输入激活读取字节数
            store_act_bytes  : int   单 batch 输出激活写入字节数
    """

    model_id: str = ""

    def __init__(self, hardware: str):
        self.hardware = hardware
        self.w_bit = 16
        self.a_bit = 16
        self.batchsize = 1

    # ── 子类必须实现 ──────────────────────────────────────────
    def get_layer_graph(self) -> dict[str, list[str]]:
        raise NotImplementedError

    def get_layers(self) -> list[dict]:
        raise NotImplementedError

    # ── 通用分析逻辑（子类无需覆盖）─────────────────────────
    def analyze(self, w_bit: int = 16, a_bit: int = 16, batchsize: int = 1) -> dict:
        """
        对所有层执行 roofline 分析。

        返回格式：
        {
            "layers": {
                "layer_name": {
                    "OPs", "memory_access", "arithmetic_intensity",
                    "performance", "bound",
                    "load_weight", "load_act", "store_act", "inference_time"
                }, ...
            },
            "total_results": {
                "inference": { "OPs", "memory_access", "inference_time" }
            }
        }
        """
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.batchsize = batchsize

        bandwidth, max_OPS, _ = get_hardware_info(self.hardware, w_bit, a_bit, a_bit)
        layer_results: dict[str, dict] = {}

        # JSON 中 size_bytes 基于 FP32（4 bytes），量化后按比例缩小
        w_scale = w_bit / 32
        a_scale = a_bit / 32

        for layer in self.get_layers():
            name         = layer["name"]
            OPs          = layer["OPs"] * batchsize
            load_weight  = layer["load_weight_bytes"] * w_scale
            load_act     = layer["load_act_bytes"]  * a_scale * batchsize
            store_act    = layer["store_act_bytes"]  * a_scale * batchsize
            memory_access = load_weight + load_act + store_act

            if memory_access > 0 and OPs > 0:
                ai, perf, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
            elif memory_access > 0:
                ai, perf, bound = 0.0, bandwidth, "memory"
            else:
                ai, perf, bound = 0.0, 0.0, "memory"

            inf_time = (OPs / perf if perf > 0
                        else (memory_access / bandwidth if bandwidth > 0 else 0.0))

            layer_results[name] = {
                "OPs":                  OPs,
                "memory_access":        memory_access,
                "arithmetic_intensity": ai,
                "performance":          perf,
                "bound":                bound,
                "load_weight":          load_weight,
                "load_act":             load_act,
                "store_act":            store_act,
                "inference_time":       inf_time,
            }

        total_ops  = sum(r["OPs"]           for r in layer_results.values())
        total_mem  = sum(r["memory_access"]  for r in layer_results.values())
        total_time = sum(r["inference_time"] for r in layer_results.values())

        return {
            "layers": layer_results,
            "total_results": {
                "inference": {
                    "OPs":            total_ops,
                    "memory_access":  total_mem,
                    "inference_time": total_time,
                }
            },
        }
