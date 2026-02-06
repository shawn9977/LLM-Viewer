import re
import numpy as np
from hardwares import get_hardware_info
from model_analyzer import get_analyzer
from utils import str_number


def get_quant_bit(dtype):
    if dtype == "FP16":
        return 16
    elif dtype == "INT8":
        return 8
    elif dtype == "INT4":
        return 4
    elif "bit" in dtype:
        bitwidth = int(re.findall(r"\d+", dtype)[0])
        return bitwidth
    else:
        raise ValueError(f"Unsupported dtype:{dtype}")


def get_model_graph(model_id, hardware, inference_config):
    # Roofline model
    w_bit = get_quant_bit(inference_config["w_quant"])
    a_bit = get_quant_bit(inference_config["a_quant"])
    kv_bit = get_quant_bit(inference_config["kv_quant"])
    seq_length = int(inference_config["seq_length"])
    batch_size = int(inference_config["batch_size"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["gen_length"])
    tp_size = int(inference_config["tp_size"])
    image_size = inference_config.get("image_size")

    stage = inference_config["stage"]
    if stage == "vision":
        stage = "prefill"
    if stage == "multimodal_ttft":
        stage = "prefill"
    elif stage == "multimodal_tpot":
        stage = "decode"
    input_node_id = "vision_input" if stage == "vision" else "input"
    graph_stage = stage

    analyzer = get_analyzer(model_id, hardware)
    result = analyzer.analyze(
        seqlen=seq_length,
        batchsize=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        tp_size=tp_size,
        image_size=image_size
    )
    bandwidth, max_OPS, onchip_buffer = get_hardware_info(hardware, w_bit, a_bit, kv_bit)
    GQA = analyzer.if_group_qa()
    hardware_info = {
        "bandwidth": bandwidth,
        "max_OPS": max_OPS,
        "onchip_buffer": onchip_buffer,
    }

    nodes = [
        {
            "label": input_node_id,
            "id": input_node_id,
        }
    ]
    edges = []

    def write_to_node(name, OPs, memory_access, info, input_names=[], node_id=None, label=None):
        node = {
            "label": label or name,
            "id": node_id or name,
            "description": f"OPs:{str_number(OPs)}, Access:{str_number(memory_access, 'B')}",
            "info": info,
        }
        if GQA and name in ["qk_matmul", "sv_matmul"]:
            node["label"] += "(GQA)"
        nodes.append(node)
        for input_name in input_names:
            edge = {"source": input_name, "target": node["id"]}
            edges.append(edge)

    def add_layer_graph(layer_graph, result_stage, prefix, label_prefix, root_id=None, root_label=None, root_target=None):
        if root_id and root_label:
            nodes.append({"label": root_label, "id": root_id})
        if root_id and root_target:
            edges.append({"source": root_id, "target": f"{prefix}{root_target}"})
        for name, input_names in layer_graph.items():
            node_id = f"{prefix}{name}"
            node_label = f"{label_prefix}{name}"
            if name in ["input", "output", "vision_input", "vision_output"] or name not in result_stage:
                OPs = 0
                memory_access = 0
                info = {}
            else:
                OPs = result_stage[name]["OPs"]
                memory_access = result_stage[name]["memory_access"]
                info = result_stage[name]
            write_to_node(
                name,
                OPs,
                memory_access,
                info,
                [f"{prefix}{n}" for n in input_names],
                node_id=node_id,
                label=node_label,
            )

    has_vision = hasattr(analyzer.module, "vision_layer_graph")
    if use_flashattention:
        text_layer_graph = analyzer.module.flashattention_transformer_layer_graph
    else:
        text_layer_graph = analyzer.module.transformer_layer_graph

    if stage == "vision":
        if use_flashattention and hasattr(analyzer.module, "vision_flashattention_layer_graph"):
            layer_graph = analyzer.module.vision_flashattention_layer_graph
        elif hasattr(analyzer.module, "vision_layer_graph"):
            layer_graph = analyzer.module.vision_layer_graph
        else:
            # Fallback for non-VLM models when vision stage is selected
            stage = "prefill"
            graph_stage = "prefill"
            input_node_id = "input"
            layer_graph = analyzer.module.transformer_layer_graph
    else:
        layer_graph = text_layer_graph

    total_results = result["total_results"]

    if graph_stage != "chat":
        stage_result = result[graph_stage]
    else:
        stage_result = result["prefill"]

    if stage == "prefill" and has_vision:
        if use_flashattention and hasattr(analyzer.module, "vision_flashattention_layer_graph"):
            vision_layer_graph = analyzer.module.vision_flashattention_layer_graph
        else:
            vision_layer_graph = analyzer.module.vision_layer_graph
        nodes = []
        edges = []
        add_layer_graph(
            vision_layer_graph,
            result["vision"],
            prefix="vision::",
            label_prefix="Vision:",
            root_id="mm_vision_root",
            root_label="Vision Encoder",
            root_target="vision_input",
        )
        add_layer_graph(
            text_layer_graph,
            result["prefill"],
            prefix="text::",
            label_prefix="Text:",
            root_id="mm_text_root",
            root_label="Text Prefill",
            root_target="input",
        )
        if "multimodal_ttft" in total_results:
            total_results["prefill"] = total_results["multimodal_ttft"]
        return nodes, edges, total_results, hardware_info

    for name, input_names in layer_graph.items():
        if name in ["input", "output", "vision_input"] or name not in stage_result:
            OPs = 0
            memory_access = 0
            info = {}
        else:
            OPs = stage_result[name]["OPs"]
            memory_access = stage_result[name]["memory_access"]
            info = stage_result[name]
        write_to_node(name, OPs, memory_access, info, input_names)
    if stage == "chat":
        # seq_length:seq_length+gen_length
        if has_vision and "multimodal_ttft" in total_results:
            total_results["chat"] = total_results["multimodal_ttft"].copy()
        else:
            total_results["chat"] = total_results["prefill"]
        n_divide = min(10, gen_length)
        for lengthi in np.linspace(seq_length + 1, seq_length + gen_length, n_divide):
            gen_result = analyzer.analyze(
                seqlen=lengthi,
                batchsize=batch_size,
                w_bit=w_bit,
                a_bit=a_bit,
                kv_bit=kv_bit,
                use_flashattention=use_flashattention,
                image_size=image_size,
            )
            for k, v in gen_result["total_results"]["decode"].items():
                total_results["chat"][k] += v * gen_length / n_divide
            for name, input_names in layer_graph.items():
                if name in gen_result["decode"]:
                    stage_result[name]["OPs"] += (
                        gen_result["decode"][name]["OPs"] * gen_length / n_divide
                    )
                    stage_result[name]["memory_access"] += (
                        gen_result["decode"][name]["memory_access"]
                        * gen_length
                        / n_divide
                    )
        if has_vision:
            if use_flashattention and hasattr(analyzer.module, "vision_flashattention_layer_graph"):
                vision_layer_graph = analyzer.module.vision_flashattention_layer_graph
            else:
                vision_layer_graph = analyzer.module.vision_layer_graph
            nodes = []
            edges = []
            add_layer_graph(
                vision_layer_graph,
                result["vision"],
                prefix="vision::",
                label_prefix="Vision:",
                root_id="mm_vision_root",
                root_label="Vision Encoder",
                root_target="vision_input",
            )
            add_layer_graph(
                text_layer_graph,
                stage_result,
                prefix="text::",
                label_prefix="Text:",
                root_id="mm_text_root",
                root_label="Text Chat",
                root_target="input",
            )
            return nodes, edges, total_results, hardware_info
        for name, input_names in layer_graph.items():
            if name in ["input", "output"]:
                OPs = 0
                memory_access = 0
                info = {}
            else:
                OPs = stage_result[name]["OPs"]
                memory_access = stage_result[name]["memory_access"]
                info = {}
            write_to_node(name, OPs, memory_access, info, input_names)
    if stage == "decode" and has_vision and "multimodal_tpot" in total_results:
        total_results["decode"] = total_results["multimodal_tpot"]
    return nodes, edges, total_results, hardware_info
