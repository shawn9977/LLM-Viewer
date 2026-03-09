"""
get_cnn_graph.py
CNN 专用图生成，对应 LLM 侧的 get_model_graph.py。
被 app.py 的 /get_cnn_graph 端点调用。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cnn_models  # 触发所有子类的 @register_cnn_model 注册
from cnn_analyzer import get_cnn_analyzer
from hardwares import get_hardware_info
from utils import str_number


def get_quant_bit(dtype: str) -> int:
    if dtype == "FP16":
        return 16
    elif dtype == "INT8" or dtype == "8-bit":
        return 8
    elif dtype == "INT4" or dtype == "4-bit":
        return 4
    elif dtype == "INT2" or dtype == "2-bit":
        return 2
    elif "bit" in dtype:
        import re
        return int(re.findall(r"\d+", dtype)[0])
    return 16


def get_cnn_graph(model_id: str, hardware: str, cnn_config: dict):
    """
    执行 CNN roofline 分析并返回前端所需的图数据。

    cnn_config 字段：
      w_quant   : "FP16" | "8-bit" | "4-bit"
      a_quant   : "FP16" | "8-bit"
      batchsize : int
    """
    w_bit     = get_quant_bit(cnn_config.get("w_quant", "FP16"))
    a_bit     = get_quant_bit(cnn_config.get("a_quant", "FP16"))
    batchsize = int(cnn_config.get("batchsize", 1))

    analyzer = get_cnn_analyzer(model_id, hardware)
    result   = analyzer.analyze(w_bit=w_bit, a_bit=a_bit, batchsize=batchsize)

    bandwidth, max_OPS, onchip_buffer = get_hardware_info(hardware, w_bit, a_bit, a_bit)
    hardware_info = {
        "bandwidth":    bandwidth,
        "max_OPS":      max_OPS,
        "onchip_buffer": onchip_buffer,
    }

    layer_graph   = analyzer.get_layer_graph()
    layer_results = result["layers"]

    # ONNX 导出时无 bias 的 Conv 会被拆成 WithoutBiases + Conv 两个节点，
    # 对用户无意义，过滤掉并把边透传给其父节点。
    _HIDDEN_SUFFIXES = ("/WithoutBiases",)

    def _is_hidden(n: str) -> bool:
        return any(n.endswith(s) for s in _HIDDEN_SUFFIXES)

    def _resolve(n: str) -> str:
        """沿 DAG 向上跳过所有隐藏节点，返回第一个可见祖先。"""
        while _is_hidden(n):
            parents = layer_graph.get(n, [])
            if not parents:
                break
            n = parents[0]
        return n

    nodes = []
    edges = []

    for name, input_names in layer_graph.items():
        if _is_hidden(name):
            continue  # 跳过 WithoutBiases 节点

        if name in ("input", "output") or name not in layer_results:
            OPs = 0
            mem = 0
            info = {}
        else:
            r   = layer_results[name]
            OPs = r["OPs"]
            mem = r["memory_access"]
            info = r

        nodes.append({
            "id":          name,
            "label":       name.split("/")[-1] or name,
            "description": f"OPs:{str_number(OPs)}, Access:{str_number(mem, 'B')}",
            "info":        info,
        })
        for src in input_names:
            resolved = _resolve(src)
            if resolved != name:  # 避免自环
                edges.append({"source": resolved, "target": name})

    total_results = result["total_results"]
    return nodes, edges, total_results, hardware_info
