"""
cnn_models/resnet50.py
ResNet-50 roofline 分析子类。
从 resnet50.json 读取层信息，构建带残差连接的 DAG。

ResNet-50 网络结构：
  Stem   : conv1(7x7) → relu → maxpool
  Layer1 : 3 × Bottleneck(64→256),  56×56
  Layer2 : 4 × Bottleneck(128→512), 28×28
  Layer3 : 6 × Bottleneck(256→1024),14×14
  Layer4 : 3 × Bottleneck(512→2048), 7×7
  Head   : avgpool → fc(1000)

每个 Bottleneck:
  conv1(1×1) → relu → conv2(3×3) → relu → conv3(1×1) → [+shortcut] → relu
  第一个 block 的 shortcut 使用 1×1 downsample 卷积。
"""

import json
from pathlib import Path
from ..cnn_analyzer import CNNAnalyzer, register_cnn_model

_JSON_PATH = Path(__file__).parent.parent / "cnn_config" / "resnet50.json"

_NO_WEIGHT_TYPES = {"Add", "ReLU", "MaxPool", "ReduceMean"}

# 每个 stage 的 block 数
_STAGE_BLOCKS = {
    "layer1": 3,
    "layer2": 4,
    "layer3": 6,
    "layer4": 3,
}


@register_cnn_model("resnet50")
class ResNet50Analyzer(CNNAnalyzer):

    def _load_json_layers(self) -> list[dict]:
        with open(_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)["layers"]

    def get_layer_graph(self) -> dict[str, list[str]]:
        """
        构建 ResNet-50 的 DAG，核心是残差连接：
        每个 Bottleneck 的 Add 节点同时接收 main branch (conv3) 和
        shortcut (identity 或 downsample conv) 两条边。
        """
        layers = self._load_json_layers()
        names = [l["layer_name"] for l in layers]
        name_set = set(names)

        graph: dict[str, list[str]] = {"input": []}

        # ── Stem ───────────────────────────────────────────────
        graph["/conv1/Conv"] = ["input"]
        graph["/relu/Relu"] = ["/conv1/Conv"]
        graph["/maxpool/MaxPool"] = ["/relu/Relu"]

        prev_out = "/maxpool/MaxPool"  # 上一个 stage/block 的输出

        # ── Layer 1-4 ─────────────────────────────────────────
        for stage_name, num_blocks in _STAGE_BLOCKS.items():
            for bi in range(num_blocks):
                prefix = f"/{stage_name}/{stage_name}.{bi}"
                has_ds = f"{prefix}/downsample.0/Conv" in name_set

                # main branch: conv1 → relu1 → conv2 → relu2 → conv3
                graph[f"{prefix}/conv1/Conv"] = [prev_out]
                graph[f"{prefix}/relu1/Relu"] = [f"{prefix}/conv1/Conv"]
                graph[f"{prefix}/conv2/Conv"] = [f"{prefix}/relu1/Relu"]
                graph[f"{prefix}/relu2/Relu"] = [f"{prefix}/conv2/Conv"]
                graph[f"{prefix}/conv3/Conv"] = [f"{prefix}/relu2/Relu"]

                # shortcut
                if has_ds:
                    graph[f"{prefix}/downsample.0/Conv"] = [prev_out]
                    shortcut = f"{prefix}/downsample.0/Conv"
                else:
                    shortcut = prev_out

                # residual add: main + shortcut
                graph[f"{prefix}/Add"] = [
                    f"{prefix}/conv3/Conv",
                    shortcut,
                ]
                graph[f"{prefix}/relu_out/Relu"] = [f"{prefix}/Add"]

                prev_out = f"{prefix}/relu_out/Relu"

        # ── Head ───────────────────────────────────────────────
        graph["/avgpool/ReduceMean"] = [prev_out]
        graph["/fc/Gemm"] = ["/avgpool/ReduceMean"]
        graph["output"] = ["/fc/Gemm"]

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
