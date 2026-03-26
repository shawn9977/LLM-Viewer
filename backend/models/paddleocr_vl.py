import math


def _text_config(model_params):
    # PaddleOCR-VL config is flat (no text_config sub-key)
    return model_params


def _vision_config(model_params):
    return model_params.get("vision_config", {})


# ===== Text branch =====
def get_num_attention_heads(model_params):
    return _text_config(model_params)["num_attention_heads"]


def get_hidden_size(model_params):
    return _text_config(model_params)["hidden_size"]


def get_head_dim(model_params):
    tc = _text_config(model_params)
    head_dim = tc.get("head_dim")
    if head_dim is None:
        head_dim = tc["hidden_size"] // tc["num_attention_heads"]
    return head_dim


def get_num_key_value_heads(model_params):
    return _text_config(model_params)["num_key_value_heads"]


def get_num_hidden_layers(model_params):
    return _text_config(model_params)["num_hidden_layers"]


def get_intermediate_size(model_params):
    return _text_config(model_params)["intermediate_size"]


def get_vocab_size(model_params):
    return _text_config(model_params)["vocab_size"]


def get_num_active_experts(model_params):
    return 1


def get_norm_layers(model_params):
    # RMSNorm — names must match transformer_layer_graph keys
    return ["attn_norm", "mlp_norm"]


def get_linear_layers(model_params, tp_size=1):
    tc = _text_config(model_params)
    hidden_size = tc["hidden_size"]
    head_dim = get_head_dim(model_params)
    intermediate_size = tc["intermediate_size"]
    kv_heads = tc["num_key_value_heads"]
    attn_heads = tc["num_attention_heads"]
    if tp_size > 1:
        assert attn_heads % tp_size == 0
        assert intermediate_size % tp_size == 0
    return {
        "q_proj":   [hidden_size, attn_heads * head_dim // tp_size],
        "k_proj":   [hidden_size, kv_heads * head_dim // tp_size],
        "v_proj":   [hidden_size, kv_heads * head_dim // tp_size],
        "out_proj": [attn_heads * head_dim // tp_size, hidden_size],
        "gate_proj": [hidden_size, intermediate_size // tp_size],
        "up_proj":   [hidden_size, intermediate_size // tp_size],
        "down_proj": [intermediate_size // tp_size, hidden_size],
    }


def post_process(model_params, args):
    tc = _text_config(model_params)
    hidden_size = tc["hidden_size"]
    vocab_size = tc["vocab_size"]
    bs = args["batchsize"]
    seqlen = args["seqlen"]
    w_byte = args["w_byte"]
    a_byte = args["a_byte"]
    return [
        # embed_tokens: prefill — load embedding table rows for seqlen tokens
        {
            "name": "embed_tokens",
            "stage": "prefill",
            "OPs": bs * seqlen * hidden_size,  # copy ops
            "load_weight": vocab_size * hidden_size * w_byte,
            "load_act": 0,
            "store_act": bs * seqlen * hidden_size * a_byte,
            "load_kv_cache": 0,
            "store_kv_cache": 0,
        },
        # embed_tokens: decode — embedding table 已在 prefill 时加载，不重复计算权重
        {
            "name": "embed_tokens",
            "stage": "decode",
            "OPs": bs * hidden_size,
            "load_weight": 0,
            "load_act": 0,
            "store_act": bs * hidden_size * a_byte,
            "load_kv_cache": 0,
            "store_kv_cache": 0,
        },
        # lm_head: prefill
        {
            "name": "lm_head",
            "stage": "prefill",
            "OPs": bs * seqlen * hidden_size * vocab_size * 2,
            "load_weight": hidden_size * vocab_size * w_byte,
            "load_act": bs * seqlen * hidden_size * a_byte,
            "store_act": bs * seqlen * vocab_size * a_byte,
            "load_kv_cache": 0,
            "store_kv_cache": 0,
        },
        # lm_head: decode
        {
            "name": "lm_head",
            "stage": "decode",
            "OPs": bs * hidden_size * vocab_size * 2,
            "load_weight": hidden_size * vocab_size * w_byte,
            "load_act": bs * hidden_size * a_byte,
            "store_act": bs * vocab_size * a_byte,
            "load_kv_cache": 0,
            "store_kv_cache": 0,
        },
    ]


# ===== Vision branch =====
def get_vision_hidden_size(model_params):
    return _vision_config(model_params)["hidden_size"]


def get_vision_num_heads(model_params):
    return _vision_config(model_params)["num_attention_heads"]


def get_vision_num_hidden_layers(model_params):
    return _vision_config(model_params)["num_hidden_layers"]


def get_vision_patch_size(model_params):
    return _vision_config(model_params)["patch_size"]


def get_vision_in_channels(model_params):
    return _vision_config(model_params).get("num_channels", 3)


def get_vision_intermediate_size(model_params):
    return _vision_config(model_params)["intermediate_size"]


def get_vision_spatial_merge_size(model_params):
    return _vision_config(model_params).get("spatial_merge_size", 1)


def get_vision_temporal_patch_size(model_params):
    return _vision_config(model_params).get("temporal_patch_size", 1)


def get_vision_norm_layers(model_params):
    return ["vision_norm1", "vision_norm2"]


def get_vision_linear_layers(model_params, tp_size=1):
    vc = _vision_config(model_params)
    hidden_size = vc["hidden_size"]
    intermediate_size = vc["intermediate_size"]
    attn_heads = vc["num_attention_heads"]
    head_dim = hidden_size // attn_heads
    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
    # Vision MLP uses fc1+GELU+fc2 (no gate_proj)
    return {
        "vision_q_proj":   [hidden_size, attn_heads * head_dim // tp_size],
        "vision_k_proj":   [hidden_size, attn_heads * head_dim // tp_size],
        "vision_v_proj":   [hidden_size, attn_heads * head_dim // tp_size],
        "vision_out_proj": [attn_heads * head_dim // tp_size, hidden_size],
        "vision_up_proj":  [hidden_size, intermediate_size // tp_size],
        "vision_down_proj": [intermediate_size // tp_size, hidden_size],
    }


def vision_post_process(model_params, args):
    """
    统计 vision encoder 尾部 + mlp_AR projector 的算子：
      1. post_layernorm  LayerNorm(1152)  作用在 num_patches 个 token 上
      2. head.attention.out_proj  Linear(1152->1152)  pooling head，1 个 query token
      3. head.layernorm  LayerNorm(1152)
      4. head.mlp.fc1  Linear(1152->4304)
      5. head.mlp.fc2  Linear(4304->1152)
      6. mlp_AR.pre_norm  LayerNorm(1152)  作用在 num_patches 个 token 上（spatial merge 前）
      7. mlp_AR.linear_1  Linear(4608->4608)  作用在 merged_tokens 上
      8. mlp_AR.act  GELU
      9. mlp_AR.linear_2  Linear(4608->1024)
    args["seqlen"] = merged_tokens（spatial merge 后）
    args["num_patches"] = num_patches（spatial merge 前，由 analyzer 传入）
    """
    vc = _vision_config(model_params)
    vision_hidden = vc["hidden_size"]          # 1152
    vision_intermediate = vc["intermediate_size"]  # 4304
    spatial_merge_size = vc.get("spatial_merge_size", 1)
    text_hidden = _text_config(model_params)["hidden_size"]  # 1024

    merged_tokens = args["seqlen"]
    num_patches = args.get("num_patches", merged_tokens * (spatial_merge_size ** 2))
    bs = args["batchsize"]
    w_byte = args["w_byte"]
    a_byte = args["a_byte"]

    merger_input_size = (spatial_merge_size ** 2) * vision_hidden  # 4 * 1152 = 4608

    layers = []

    # 1. post_layernorm — acts on num_patches tokens, dim=1152
    layers.append({
        "name": "vision_post_layernorm",
        "stage": "vision",
        "OPs": bs * num_patches * vision_hidden * 7,
        "load_weight": vision_hidden * w_byte,
        "load_act": bs * num_patches * vision_hidden * a_byte,
        "store_act": bs * num_patches * vision_hidden * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 2. head.attention.out_proj Linear(1152->1152), 1 query token (pooling)
    layers.append({
        "name": "vision_head_out_proj",
        "stage": "vision",
        "OPs": bs * 1 * vision_hidden * vision_hidden * 2,
        "load_weight": vision_hidden * vision_hidden * w_byte,
        "load_act": bs * 1 * vision_hidden * a_byte,
        "store_act": bs * 1 * vision_hidden * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 3. head.layernorm LayerNorm(1152), 1 token
    layers.append({
        "name": "vision_head_layernorm",
        "stage": "vision",
        "OPs": bs * 1 * vision_hidden * 7,
        "load_weight": vision_hidden * w_byte,
        "load_act": bs * 1 * vision_hidden * a_byte,
        "store_act": bs * 1 * vision_hidden * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 4. head.mlp.fc1 Linear(1152->4304), 1 token
    layers.append({
        "name": "vision_head_fc1",
        "stage": "vision",
        "OPs": bs * 1 * vision_hidden * vision_intermediate * 2,
        "load_weight": vision_hidden * vision_intermediate * w_byte,
        "load_act": bs * 1 * vision_hidden * a_byte,
        "store_act": bs * 1 * vision_intermediate * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 5. head.mlp.fc2 Linear(4304->1152), 1 token
    layers.append({
        "name": "vision_head_fc2",
        "stage": "vision",
        "OPs": bs * 1 * vision_intermediate * vision_hidden * 2,
        "load_weight": vision_intermediate * vision_hidden * w_byte,
        "load_act": bs * 1 * vision_intermediate * a_byte,
        "store_act": bs * 1 * vision_hidden * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 6. mlp_AR.pre_norm LayerNorm(1152), acts on num_patches tokens (before spatial merge)
    layers.append({
        "name": "vision_merger_pre_norm",
        "stage": "vision",
        "OPs": bs * num_patches * vision_hidden * 7,
        "load_weight": vision_hidden * w_byte,
        "load_act": bs * num_patches * vision_hidden * a_byte,
        "store_act": bs * num_patches * vision_hidden * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 7. mlp_AR.linear_1 Linear(4608->4608), merged_tokens
    layers.append({
        "name": "vision_merger_fc1",
        "stage": "vision",
        "OPs": bs * merged_tokens * merger_input_size * merger_input_size * 2,
        "load_weight": merger_input_size * merger_input_size * w_byte,
        "load_act": bs * merged_tokens * merger_input_size * a_byte,
        "store_act": bs * merged_tokens * merger_input_size * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 8. mlp_AR.act GELU
    layers.append({
        "name": "vision_merger_act",
        "stage": "vision",
        "OPs": bs * merged_tokens * merger_input_size * 5,
        "load_weight": 0,
        "load_act": bs * merged_tokens * merger_input_size * a_byte,
        "store_act": bs * merged_tokens * merger_input_size * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    # 9. mlp_AR.linear_2 Linear(4608->1024), merged_tokens
    layers.append({
        "name": "vision_merger_fc2",
        "stage": "vision",
        "OPs": bs * merged_tokens * merger_input_size * text_hidden * 2,
        "load_weight": merger_input_size * text_hidden * w_byte,
        "load_act": bs * merged_tokens * merger_input_size * a_byte,
        "store_act": bs * merged_tokens * text_hidden * a_byte,
        "load_kv_cache": 0, "store_kv_cache": 0,
    })

    return layers


# ===== Layer graphs =====
transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "gate_proj": ["mlp_norm"],
    "up_proj": ["mlp_norm"],
    "mlp_act": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_act"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"],
}

flashattention_transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "fused_attention": ["q_proj", "k_proj", "v_proj"],
    "out_proj": ["fused_attention"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "gate_proj": ["mlp_norm"],
    "up_proj": ["mlp_norm"],
    "mlp_act": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_act"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"],
}

vision_layer_graph = {
    "vision_input": [],
    "vision_patch_embed": ["vision_input"],
    "vision_norm1": ["vision_patch_embed"],
    "vision_q_proj": ["vision_norm1"],
    "vision_k_proj": ["vision_norm1"],
    "vision_v_proj": ["vision_norm1"],
    "vision_qk_matmul": ["vision_q_proj", "vision_k_proj"],
    "vision_softmax": ["vision_qk_matmul"],
    "vision_sv_matmul": ["vision_softmax", "vision_v_proj"],
    "vision_out_proj": ["vision_sv_matmul"],
    "vision_attn_add": ["vision_patch_embed", "vision_out_proj"],
    "vision_norm2": ["vision_attn_add"],
    "vision_up_proj": ["vision_norm2"],
    "vision_mlp_act": ["vision_up_proj"],
    "vision_down_proj": ["vision_mlp_act"],
    "vision_mlp_add": ["vision_attn_add", "vision_down_proj"],
    "vision_proj": ["vision_mlp_add"],
    "vision_output": ["vision_proj"],
}

vision_flashattention_layer_graph = {
    "vision_input": [],
    "vision_patch_embed": ["vision_input"],
    "vision_norm1": ["vision_patch_embed"],
    "vision_q_proj": ["vision_norm1"],
    "vision_k_proj": ["vision_norm1"],
    "vision_v_proj": ["vision_norm1"],
    "vision_fused_attention": ["vision_q_proj", "vision_k_proj", "vision_v_proj"],
    "vision_out_proj": ["vision_fused_attention"],
    "vision_attn_add": ["vision_patch_embed", "vision_out_proj"],
    "vision_norm2": ["vision_attn_add"],
    "vision_up_proj": ["vision_norm2"],
    "vision_mlp_act": ["vision_up_proj"],
    "vision_down_proj": ["vision_mlp_act"],
    "vision_mlp_add": ["vision_attn_add", "vision_down_proj"],
    "vision_proj": ["vision_mlp_add"],
    "vision_output": ["vision_proj"],
}
