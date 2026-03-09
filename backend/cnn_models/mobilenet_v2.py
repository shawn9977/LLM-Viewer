"""
cnn_models/mobilenet_v2.py
MobileNetV2 roofline 分析子类。
从 mobilenet-v2.json 读取层信息，自动构建线性 DAG。
"""

import json
from pathlib import Path
from cnn_analyzer import CNNAnalyzer, register_cnn_model

# JSON 文件路径（相对于项目根目录）
_JSON_PATH = Path(__file__).parent.parent / "cnn_config" / "mobilenet-v2.json"

# 量化辅助层后缀，不计入分析
_SKIP_SUFFIXES = ("/fq_weights_1", "/fq_weights_0")

# 无权重层类型（访存只有激活）
_NO_WEIGHT_TYPES = {"Add", "Subtract", "Multiply", "ReduceMean", "Softmax"}


@register_cnn_model("mobilenet_v2")
class MobileNetV2Analyzer(CNNAnalyzer):

    def _load_json_layers(self) -> list[dict]:
        with open(_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)["layers"]

    def _is_skip(self, layer: dict) -> bool:
        return any(layer["layer_name"].endswith(s) for s in _SKIP_SUFFIXES)

    def get_layer_graph(self) -> dict[str, list[str]]:
        """线性 DAG：input → layer0 → layer1 → ... → output"""
        graph: dict[str, list[str]] = {"input": []}
        prev = "input"
        for layer in self._load_json_layers():
            if self._is_skip(layer):
                continue
            name = layer["layer_name"]
            graph[name] = [prev]
            prev = name
        graph["output"] = [prev]
        return graph

    def get_layers(self) -> list[dict]:
        result = []
        for layer in self._load_json_layers():
            if self._is_skip(layer):
                continue

            gflops = layer.get("gflops") or 0.0
            OPs = gflops * 1e9

            inputs  = layer.get("input_tensors", [])
            outputs = layer.get("output_tensors", [])

            if layer["layer_type"] in _NO_WEIGHT_TYPES or len(inputs) < 2:
                load_weight = 0
                load_act    = inputs[0]["size_bytes"] if inputs else 0
            else:
                load_weight = inputs[1]["size_bytes"]
                load_act    = inputs[0]["size_bytes"]

            store_act = outputs[0]["size_bytes"] if outputs else 0

            result.append({
                "name":              layer["layer_name"],
                "OPs":               OPs,
                "load_weight_bytes": load_weight,
                "load_act_bytes":    load_act,
                "store_act_bytes":   store_act,
            })
        return result
