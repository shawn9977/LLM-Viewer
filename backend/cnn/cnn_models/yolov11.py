"""
cnn_models/yolov11.py
YOLOv11 roofline 分析子类 (n/s/m 共用)。
从 yolov11{n,s,m}.json 读取层信息，构建 YOLOv11 的 DAG。

网络结构 (PyTorch module indices):
  Backbone: model.0-10
    0: stem, 1: down, 2: C3k2, 3: down, 4: C3k2(P3),
    5: down, 6: C3k2(P4), 7: down, 8: C3k2(P5), 9: SPPF, 10: C2PSA
  Neck FPN+PAN: model.11-22
    11: Upsample, 12: Concat(10+6), 13: C3k2,
    14: Upsample, 15: Concat(13+4), 16: C3k2,
    17: Conv(down), 18: Concat(17+13), 19: C3k2,
    20: Conv(down), 21: Concat(20+10), 22: C3k2
  Detect Head: model.23  (3 scales × reg/cls + dfl)
"""

import json
from pathlib import Path
from ..cnn_analyzer import CNNAnalyzer, register_cnn_model

_NO_WEIGHT_TYPES = {"Add", "Swish", "Concat", "Interpolate", "MaxPool",
                    "MatMul", "Identity"}

# Concat skip connections: concat_module -> skip_source_block
_YOLO11_CONCATS = {
    "model.12": "model.6",    # C2PSA out (via upsample) + P4
    "model.15": "model.4",    # neck1 out (via upsample) + P3
    "model.18": "model.13",   # down(neck2) + neck1
    "model.21": "model.10",   # down(neck3) + C2PSA
}

# Detect head: first layer prefix -> source block
_YOLO11_HEAD = {
    "model.23.cv2.0.": "model.16",  # reg P3
    "model.23.cv3.0.": "model.16",  # cls P3
    "model.23.cv2.1.": "model.19",  # reg P4
    "model.23.cv3.1.": "model.19",  # cls P4
    "model.23.cv2.2.": "model.22",  # reg P5
    "model.23.cv3.2.": "model.22",  # cls P5
}


class YOLOv11Analyzer(CNNAnalyzer):
    """Base class for all YOLOv11 variants (n/s/m)."""

    _json_name: str = ""

    def _json_path(self) -> Path:
        return Path(__file__).parent.parent / "cnn_config" / self._json_name

    def _load_json_layers(self) -> list[dict]:
        with open(self._json_path(), encoding="utf-8") as f:
            return json.load(f)["layers"]

    @staticmethod
    def _block_output(layers, names, block_prefix):
        """Find the last layer belonging to a block, including any shared activation after it."""
        last_idx = None
        for i, n in enumerate(names):
            if n.startswith(block_prefix + ".") or n == block_prefix:
                last_idx = i
        if last_idx is None:
            return names[-1]
        # Include the activation right after (shared SiLU named model.0.act_N)
        if last_idx + 1 < len(names) and layers[last_idx + 1]["layer_type"] in ("Swish", "Identity"):
            return names[last_idx + 1]
        return names[last_idx]

    @staticmethod
    def _is_first_with_prefix(names, idx, prefix):
        if not names[idx].startswith(prefix):
            return False
        return not any(names[j].startswith(prefix) for j in range(idx))

    def get_layer_graph(self) -> dict[str, list[str]]:
        layers = self._load_json_layers()
        names = [l["layer_name"] for l in layers]

        graph: dict[str, list[str]] = {"input": []}
        prev = "input"

        for i, name in enumerate(names):
            # Concat: two parents (prev + skip connection)
            if name in _YOLO11_CONCATS:
                skip_block = _YOLO11_CONCATS[name]
                skip_out = self._block_output(layers, names, skip_block)
                graph[name] = [prev, skip_out]
            # Detect head branch entries
            elif any(self._is_first_with_prefix(names, i, p) for p in _YOLO11_HEAD):
                for prefix, src_block in _YOLO11_HEAD.items():
                    if self._is_first_with_prefix(names, i, prefix):
                        src_out = self._block_output(layers, names, src_block)
                        graph[name] = [src_out]
                        break
            else:
                graph[name] = [prev]

            prev = name

        graph["output"] = [prev]
        return graph

    def get_layers(self) -> list[dict]:
        result = []
        for layer in self._load_json_layers():
            gflops = layer.get("gflops") or 0.0
            OPs = gflops * 1e9

            inputs = layer.get("input_tensors", [])
            outputs = layer.get("output_tensors", [])

            if layer["layer_type"] in _NO_WEIGHT_TYPES or len(inputs) < 2:
                load_weight = 0
                load_act = inputs[0]["size_bytes"] if inputs else 0
            else:
                load_weight = inputs[1]["size_bytes"]
                load_act = inputs[0]["size_bytes"]

            store_act = outputs[0]["size_bytes"] if outputs else 0

            result.append({
                "name":              layer["layer_name"],
                "OPs":               OPs,
                "load_weight_bytes": load_weight,
                "load_act_bytes":    load_act,
                "store_act_bytes":   store_act,
            })
        return result


@register_cnn_model("yolov11n")
class YOLOv11nAnalyzer(YOLOv11Analyzer):
    _json_name = "yolov11n.json"


@register_cnn_model("yolov11s")
class YOLOv11sAnalyzer(YOLOv11Analyzer):
    _json_name = "yolov11s.json"


@register_cnn_model("yolov11m")
class YOLOv11mAnalyzer(YOLOv11Analyzer):
    _json_name = "yolov11m.json"
