"""
cnn_models/fastsam_x.py
FastSAM-x roofline 分析子类。
从 fastsam_x.json 读取层信息，构建 FastSAM-x 的 DAG。

网络结构 (PyTorch module indices, YOLOv8x-seg):
  Backbone: model.0-9
    0: stem, 1: down, 2: C2f(3), 3: down, 4: C2f(6)(P3),
    5: down, 6: C2f(6)(P4), 7: down, 8: C2f(3)(P5), 9: SPPF
  Neck FPN+PAN: model.10-21
    10: Upsample, 11: Concat(9+6), 12: C2f(3),
    13: Upsample, 14: Concat(12+4), 15: C2f(3),
    16: Conv(down), 17: Concat(16+12), 18: C2f(3),
    19: Conv(down), 20: Concat(19+9), 21: C2f(3)
  Detect + Segment Head: model.22
    cv2.{0,1,2}: reg branches (P3/P4/P5)
    cv3.{0,1,2}: cls branches
    cv4.{0,1,2}: mask coeff branches
    proto.*: mask prototype head (from P3)
    dfl: DFL
"""

import json
from pathlib import Path
from ..cnn_analyzer import CNNAnalyzer, register_cnn_model

_JSON_PATH = Path(__file__).parent.parent / "cnn_config" / "fastsam_x.json"

_NO_WEIGHT_TYPES = {"Add", "Swish", "Concat", "Interpolate", "MaxPool",
                    "MatMul", "Identity", "ConvTranspose2d"}

# Concat skip connections: concat_module -> skip_source_block
_FASTSAM_CONCATS = {
    "model.11": "model.6",    # SPPF out (via upsample) + P4
    "model.14": "model.4",    # neck1 out (via upsample) + P3
    "model.17": "model.12",   # down(neck2) + neck1
    "model.20": "model.9",    # down(neck3) + SPPF
}

# Detect/segment head: first layer prefix -> source block
_FASTSAM_HEAD = {
    "model.22.cv2.0.": "model.15",    # reg P3
    "model.22.cv3.0.": "model.15",    # cls P3
    "model.22.cv4.0.": "model.15",    # mask coeff P3
    "model.22.proto.": "model.15",    # proto (from P3)
    "model.22.cv2.1.": "model.18",    # reg P4
    "model.22.cv3.1.": "model.18",    # cls P4
    "model.22.cv4.1.": "model.18",    # mask coeff P4
    "model.22.cv2.2.": "model.21",    # reg P5
    "model.22.cv3.2.": "model.21",    # cls P5
    "model.22.cv4.2.": "model.21",    # mask coeff P5
}


@register_cnn_model("fastsam_x")
class FastSAMxAnalyzer(CNNAnalyzer):

    def _load_json_layers(self) -> list[dict]:
        with open(_JSON_PATH, encoding="utf-8") as f:
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
            if name in _FASTSAM_CONCATS:
                skip_block = _FASTSAM_CONCATS[name]
                skip_out = self._block_output(layers, names, skip_block)
                graph[name] = [prev, skip_out]
            elif any(self._is_first_with_prefix(names, i, p) for p in _FASTSAM_HEAD):
                for prefix, src_block in _FASTSAM_HEAD.items():
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
