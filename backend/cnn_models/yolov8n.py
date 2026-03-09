"""
cnn_models/yolov8n.py
YOLOv8n roofline 分析子类。
从 yolo_v8n.json 读取层信息，构建带分支的 DAG。

YOLOv8n 网络结构（简化）：
  Backbone  : model.0 → model.1 → model.2 → model.3 → model.4
              → model.5 → model.6 → model.7 → model.8 → model.9
  Neck      : model.9 → Resize(model.10) → concat(model.12, from model.6)
              → Resize(model.13) → concat(model.15, from model.4)
  Head P3   : model.15 → model.22/cv2.0, cv3.0
  Head P4   : model.15 → model.16 → concat(model.18, from model.12) → model.22/cv2.1, cv3.1
  Head P5   : model.18 → model.19 → concat(model.21, from model.9)  → model.22/cv2.2, cv3.2
              → model.22/dfl → post-process
"""

import json
from pathlib import Path
from cnn_analyzer import CNNAnalyzer, register_cnn_model

_JSON_PATH = Path(__file__).parent.parent / "cnn_config" / "yolo_v8n.json"

_NO_WEIGHT_TYPES = {"Add", "Subtract", "Multiply", "ReduceMean", "Interpolate"}


@register_cnn_model("yolov8n")
class YOLOv8nAnalyzer(CNNAnalyzer):

    def _load_json_layers(self) -> list[dict]:
        with open(_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)["layers"]

    def get_layer_graph(self) -> dict[str, list[str]]:
        """
        按 YOLOv8n 实际拓扑构建 DAG。
        Concat 节点（Resize 后与 backbone feature 合并）用 layer_name 前缀匹配。
        """
        layers = self._load_json_layers()
        # 先建名称列表，方便查找
        names = [l["layer_name"] for l in layers]

        def find(prefix: str) -> str:
            """找到第一个以 prefix 开头的层名"""
            for n in names:
                if n.startswith(prefix):
                    return n
            return prefix

        graph: dict[str, list[str]] = {"input": []}

        # ── Backbone 线性部分 ──────────────────────────────────
        backbone_seq = [
            "/model.0/conv/Conv/WithoutBiases",
            "/model.0/conv/Conv",
            "/model.1/conv/Conv/WithoutBiases",
            "/model.1/conv/Conv",
            "/model.2/cv1/conv/Conv/WithoutBiases",
            "/model.2/cv1/conv/Conv",
            "/model.2/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.2/m.0/cv1/conv/Conv",
            "/model.2/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.2/m.0/cv2/conv/Conv",
            "/model.2/cv2/conv/Conv/WithoutBiases",
            "/model.2/cv2/conv/Conv",
            "/model.3/conv/Conv/WithoutBiases",
            "/model.3/conv/Conv",
            "/model.4/cv1/conv/Conv/WithoutBiases",
            "/model.4/cv1/conv/Conv",
            "/model.4/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.4/m.0/cv1/conv/Conv",
            "/model.4/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.4/m.0/cv2/conv/Conv",
            "/model.4/m.1/cv1/conv/Conv/WithoutBiases",
            "/model.4/m.1/cv1/conv/Conv",
            "/model.4/m.1/cv2/conv/Conv/WithoutBiases",
            "/model.4/m.1/cv2/conv/Conv",
            "/model.4/cv2/conv/Conv/WithoutBiases",
            "/model.4/cv2/conv/Conv",
            "/model.5/conv/Conv/WithoutBiases",
            "/model.5/conv/Conv",
            "/model.6/cv1/conv/Conv/WithoutBiases",
            "/model.6/cv1/conv/Conv",
            "/model.6/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.6/m.0/cv1/conv/Conv",
            "/model.6/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.6/m.0/cv2/conv/Conv",
            "/model.6/m.1/cv1/conv/Conv/WithoutBiases",
            "/model.6/m.1/cv1/conv/Conv",
            "/model.6/m.1/cv2/conv/Conv/WithoutBiases",
            "/model.6/m.1/cv2/conv/Conv",
            "/model.6/cv2/conv/Conv/WithoutBiases",
            "/model.6/cv2/conv/Conv",
            "/model.7/conv/Conv/WithoutBiases",
            "/model.7/conv/Conv",
            "/model.8/cv1/conv/Conv/WithoutBiases",
            "/model.8/cv1/conv/Conv",
            "/model.8/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.8/m.0/cv1/conv/Conv",
            "/model.8/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.8/m.0/cv2/conv/Conv",
            "/model.8/cv2/conv/Conv/WithoutBiases",
            "/model.8/cv2/conv/Conv",
            "/model.9/cv1/conv/Conv/WithoutBiases",
            "/model.9/cv1/conv/Conv",
            "/model.9/cv2/conv/Conv/WithoutBiases",
            "/model.9/cv2/conv/Conv",
        ]

        prev = "input"
        for name in backbone_seq:
            if name in names:
                graph[name] = [prev]
                prev = name
        p9_out = prev  # model.9 输出，供 Neck 使用

        # ── Neck 上采样分支 1：model.10 Resize → model.12 ──────
        resize10 = "/model.10/Resize"
        if resize10 in names:
            graph[resize10] = [p9_out]

        # model.6 输出（concat 来源）
        m6_out = "/model.6/cv2/conv/Conv"

        neck1_seq = [
            "/model.12/cv1/conv/Conv/WithoutBiases",
            "/model.12/cv1/conv/Conv",
            "/model.12/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.12/m.0/cv1/conv/Conv",
            "/model.12/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.12/m.0/cv2/conv/Conv",
            "/model.12/cv2/conv/Conv/WithoutBiases",
            "/model.12/cv2/conv/Conv",
        ]
        # model.12/cv1 的输入是 Resize10 + model.6 concat
        prev = resize10 if resize10 in names else p9_out
        for i, name in enumerate(neck1_seq):
            if name not in names:
                continue
            if i == 0:
                parents = [prev, m6_out] if m6_out in names else [prev]
            else:
                parents = [prev]
            graph[name] = parents
            prev = name
        p12_out = prev

        # ── Neck 上采样分支 2：model.13 Resize → model.15 ──────
        resize13 = "/model.13/Resize"
        if resize13 in names:
            graph[resize13] = [p12_out]

        m4_out = "/model.4/cv2/conv/Conv"

        neck2_seq = [
            "/model.15/cv1/conv/Conv/WithoutBiases",
            "/model.15/cv1/conv/Conv",
            "/model.15/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.15/m.0/cv1/conv/Conv",
            "/model.15/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.15/m.0/cv2/conv/Conv",
            "/model.15/cv2/conv/Conv/WithoutBiases",
            "/model.15/cv2/conv/Conv",
        ]
        prev = resize13 if resize13 in names else p12_out
        for i, name in enumerate(neck2_seq):
            if name not in names:
                continue
            if i == 0:
                parents = [prev, m4_out] if m4_out in names else [prev]
            else:
                parents = [prev]
            graph[name] = parents
            prev = name
        p15_out = prev  # Head P3 输入

        # ── Head P3（80×80）────────────────────────────────────
        head_p3_seq = [
            "/model.22/cv2.0/cv2.0.0/conv/Conv/WithoutBiases",
            "/model.22/cv2.0/cv2.0.0/conv/Conv",
            "/model.22/cv2.0/cv2.0.1/conv/Conv/WithoutBiases",
            "/model.22/cv2.0/cv2.0.1/conv/Conv",
            "/model.22/cv2.0/cv2.0.2/Conv/WithoutBiases",
            "/model.22/cv2.0/cv2.0.2/Conv",
            "/model.22/cv3.0/cv3.0.0/conv/Conv/WithoutBiases",
            "/model.22/cv3.0/cv3.0.0/conv/Conv",
            "/model.22/cv3.0/cv3.0.1/conv/Conv/WithoutBiases",
            "/model.22/cv3.0/cv3.0.1/conv/Conv",
            "/model.22/cv3.0/cv3.0.2/Conv/WithoutBiases",
            "/model.22/cv3.0/cv3.0.2/Conv",
        ]
        prev = p15_out
        for name in head_p3_seq:
            if name in names:
                graph[name] = [prev]
                prev = name
        p3_out = prev

        # ── Downsample → Head P4（40×40）──────────────────────
        ds1_seq = [
            "/model.16/conv/Conv/WithoutBiases",
            "/model.16/conv/Conv",
        ]
        prev = p15_out
        for name in ds1_seq:
            if name in names:
                graph[name] = [prev]
                prev = name
        ds1_out = prev

        head_p4_seq = [
            "/model.18/cv1/conv/Conv/WithoutBiases",
            "/model.18/cv1/conv/Conv",
            "/model.18/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.18/m.0/cv1/conv/Conv",
            "/model.18/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.18/m.0/cv2/conv/Conv",
            "/model.18/cv2/conv/Conv/WithoutBiases",
            "/model.18/cv2/conv/Conv",
            "/model.22/cv2.1/cv2.1.0/conv/Conv/WithoutBiases",
            "/model.22/cv2.1/cv2.1.0/conv/Conv",
            "/model.22/cv2.1/cv2.1.1/conv/Conv/WithoutBiases",
            "/model.22/cv2.1/cv2.1.1/conv/Conv",
            "/model.22/cv2.1/cv2.1.2/Conv/WithoutBiases",
            "/model.22/cv2.1/cv2.1.2/Conv",
            "/model.22/cv3.1/cv3.1.0/conv/Conv/WithoutBiases",
            "/model.22/cv3.1/cv3.1.0/conv/Conv",
            "/model.22/cv3.1/cv3.1.1/conv/Conv/WithoutBiases",
            "/model.22/cv3.1/cv3.1.1/conv/Conv",
            "/model.22/cv3.1/cv3.1.2/Conv/WithoutBiases",
            "/model.22/cv3.1/cv3.1.2/Conv",
        ]
        prev = ds1_out
        for i, name in enumerate(head_p4_seq):
            if name not in names:
                continue
            if i == 0:
                parents = [prev, p12_out] if p12_out in names else [prev]
            else:
                parents = [prev]
            graph[name] = parents
            prev = name
        p4_out = prev

        # ── Downsample → Head P5（20×20）──────────────────────
        ds2_seq = [
            "/model.19/conv/Conv/WithoutBiases",
            "/model.19/conv/Conv",
        ]
        # 找 model.18 最后一层作为 ds2 输入
        m18_out = "/model.18/cv2/conv/Conv"
        prev = m18_out if m18_out in names else ds1_out
        for name in ds2_seq:
            if name in names:
                graph[name] = [prev]
                prev = name
        ds2_out = prev

        head_p5_seq = [
            "/model.21/cv1/conv/Conv/WithoutBiases",
            "/model.21/cv1/conv/Conv",
            "/model.21/m.0/cv1/conv/Conv/WithoutBiases",
            "/model.21/m.0/cv1/conv/Conv",
            "/model.21/m.0/cv2/conv/Conv/WithoutBiases",
            "/model.21/m.0/cv2/conv/Conv",
            "/model.21/cv2/conv/Conv/WithoutBiases",
            "/model.21/cv2/conv/Conv",
            "/model.22/cv2.2/cv2.2.0/conv/Conv/WithoutBiases",
            "/model.22/cv2.2/cv2.2.0/conv/Conv",
            "/model.22/cv2.2/cv2.2.1/conv/Conv/WithoutBiases",
            "/model.22/cv2.2/cv2.2.1/conv/Conv",
            "/model.22/cv2.2/cv2.2.2/Conv/WithoutBiases",
            "/model.22/cv2.2/cv2.2.2/Conv",
            "/model.22/cv3.2/cv3.2.0/conv/Conv/WithoutBiases",
            "/model.22/cv3.2/cv3.2.0/conv/Conv",
            "/model.22/cv3.2/cv3.2.1/conv/Conv/WithoutBiases",
            "/model.22/cv3.2/cv3.2.1/conv/Conv",
            "/model.22/cv3.2/cv3.2.2/Conv/WithoutBiases",
            "/model.22/cv3.2/cv3.2.2/Conv",
        ]
        prev = ds2_out
        for i, name in enumerate(head_p5_seq):
            if name not in names:
                continue
            if i == 0:
                parents = [prev, p9_out] if p9_out in names else [prev]
            else:
                parents = [prev]
            graph[name] = parents
            prev = name
        p5_out = prev

        # ── DFL + 后处理 ───────────────────────────────────────
        post_seq = [
            "/model.22/dfl/conv/Conv",
            "/model.22/Sub",
            "/model.22/Add_3",
            "/model.22/Div",
            "/model.22/Mul",
        ]
        prev_post = p5_out
        for name in post_seq:
            if name in names:
                graph[name] = [prev_post]
                prev_post = name

        # ── output 汇聚三个 head ───────────────────────────────
        graph["output"] = [p3_out, p4_out, p5_out]

        # 补全 JSON 中存在但未被显式列出的层（保底，线性接在前一层后）
        assigned = set(graph.keys())
        prev_fallback = "input"
        for layer in self._load_json_layers():
            n = layer["layer_name"]
            if n not in assigned:
                graph[n] = [prev_fallback]
            prev_fallback = n

        return graph

    def get_layers(self) -> list[dict]:
        result = []
        for layer in self._load_json_layers():
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
