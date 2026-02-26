"""
Qwen3-Omni-MoE 模型参数提取和图结构定义
支持 Thinker 部分：视觉编码器 + 音频编码器 + MoE 文本解码器
"""

def _thinker_config(model_params):
    """获取 thinker_config"""
    return model_params.get("thinker_config", {})

def _text_config(model_params):
    """获取文本配置（从 thinker_config.text_config）"""
    return _thinker_config(model_params).get("text_config", {})

def _vision_config(model_params):
    """获取视觉配置（从 thinker_config.vision_config）"""
    return _thinker_config(model_params).get("vision_config", {})

def _audio_config(model_params):
    """获取音频配置（从 thinker_config.audio_config）"""
    return _thinker_config(model_params).get("audio_config", {})


# ===== 文本分支（MoE）=====
def get_num_attention_heads(model_params):
    return _text_config(model_params)["num_attention_heads"]

def get_hidden_size(model_params):
    return _text_config(model_params)["hidden_size"]

def get_head_dim(model_params):
    text_config = _text_config(model_params)
    return text_config.get("head_dim", text_config["hidden_size"] // text_config["num_attention_heads"])

def get_num_key_value_heads(model_params):
    return _text_config(model_params)["num_key_value_heads"]

def get_num_hidden_layers(model_params):
    return _text_config(model_params)["num_hidden_layers"]

def get_intermediate_size(model_params):
    return _text_config(model_params)["moe_intermediate_size"]

def get_vocab_size(model_params):
    return _text_config(model_params)["vocab_size"]

def get_num_active_experts(model_params):
    return _text_config(model_params)["num_experts_per_tok"]

def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]

def get_linear_layers(model_params, tp_size: int):
    text_config = _text_config(model_params)
    hidden_size = text_config["hidden_size"]
    head_dim = get_head_dim(model_params)
    intermediate_size = text_config["moe_intermediate_size"]
    key_value_heads = text_config["num_key_value_heads"]
    attention_heads = text_config["num_attention_heads"]

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
        assert key_value_heads % tp_size == 0

    return {
        "q_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "k_proj": [hidden_size, key_value_heads * head_dim // tp_size],
        "v_proj": [hidden_size, key_value_heads * head_dim // tp_size],
        "out_proj": [attention_heads * head_dim // tp_size, hidden_size],
        "gate_proj": [hidden_size, intermediate_size // tp_size],
        "up_proj": [hidden_size, intermediate_size // tp_size],
        "down_proj": [intermediate_size // tp_size, hidden_size],
    }

def post_process(model_params, args):
    """lm_head 层"""
    text_config = _text_config(model_params)
    hidden_size = text_config["hidden_size"]
    vocab_size = text_config["vocab_size"]
    layers = []
    layers.append({
        "name": "lm_head",
        "stage": "prefill",
        "OPs": args["batchsize"] * args["seqlen"] * hidden_size * vocab_size * 2,
        "load_weight": hidden_size * vocab_size * args["w_byte"],
        "load_act": args["batchsize"] * args["seqlen"] * hidden_size * args["a_byte"],
        "store_act": args["batchsize"] * args["seqlen"] * vocab_size * args["a_byte"],
    })
    layers.append({
        "name": "lm_head",
        "stage": "decode",
        "OPs": args["batchsize"] * hidden_size * vocab_size * 2,
        "load_weight": hidden_size * vocab_size * args["w_byte"],
        "load_act": args["batchsize"] * hidden_size * args["a_byte"],
        "store_act": args["batchsize"] * vocab_size * args["a_byte"],
    })
    return layers


# ===== 视觉分支 =====
def get_vision_num_heads(model_params):
    return _vision_config(model_params)["num_heads"]

def get_vision_hidden_size(model_params):
    return _vision_config(model_params)["hidden_size"]

def get_vision_intermediate_size(model_params):
    return _vision_config(model_params)["intermediate_size"]

def get_vision_num_hidden_layers(model_params):
    return _vision_config(model_params)["depth"]

def get_vision_patch_size(model_params):
    return _vision_config(model_params)["patch_size"]

def get_vision_in_channels(model_params):
    return _vision_config(model_params)["in_channels"]

def get_vision_out_hidden_size(model_params):
    return _vision_config(model_params).get("out_hidden_size")

def get_vision_spatial_merge_size(model_params):
    return _vision_config(model_params).get("spatial_merge_size", 1)

def get_vision_temporal_patch_size(model_params):
    return _vision_config(model_params).get("temporal_patch_size", 1)

def get_vision_norm_layers(model_params):
    return ["vision_norm1", "vision_norm2"]

def get_vision_linear_layers(model_params, tp_size: int):
    vision_config = _vision_config(model_params)
    hidden_size = vision_config["hidden_size"]
    intermediate_size = vision_config["intermediate_size"]
    attention_heads = vision_config["num_heads"]
    head_dim = hidden_size // attention_heads

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0

    return {
        "vision_q_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "vision_k_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "vision_v_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "vision_out_proj": [attention_heads * head_dim // tp_size, hidden_size],
        "vision_gate_proj": [hidden_size, intermediate_size // tp_size],
        "vision_up_proj": [hidden_size, intermediate_size // tp_size],
        "vision_down_proj": [intermediate_size // tp_size, hidden_size],
    }

def vision_post_process(model_params, args):
    """视觉投影层：merger Linear(1152, 2048)"""
    vision_config = _vision_config(model_params)
    hidden_size = vision_config["hidden_size"]
    out_hidden_size = vision_config.get("out_hidden_size")
    if out_hidden_size is None:
        return []
    return [{
        "name": "vision_proj",
        "stage": "vision",
        "OPs": args["batchsize"] * args["seqlen"] * hidden_size * out_hidden_size * 2,
        "load_weight": hidden_size * out_hidden_size * args["w_byte"],
        "load_act": args["batchsize"] * args["seqlen"] * hidden_size * args["a_byte"],
        "store_act": args["batchsize"] * args["seqlen"] * out_hidden_size * args["a_byte"],
    }]


# ===== 音频分支 =====
def get_audio_hidden_size(model_params):
    return _audio_config(model_params)["d_model"]

def get_audio_num_heads(model_params):
    return _audio_config(model_params)["encoder_attention_heads"]

def get_audio_intermediate_size(model_params):
    return _audio_config(model_params)["encoder_ffn_dim"]

def get_audio_num_hidden_layers(model_params):
    return _audio_config(model_params)["encoder_layers"]

def get_audio_output_size(model_params):
    return _audio_config(model_params)["output_dim"]

def get_audio_norm_layers(model_params):
    return ["audio_attn_norm", "audio_mlp_norm"]

def get_audio_linear_layers(model_params, tp_size: int):
    audio_config = _audio_config(model_params)
    hidden_size = audio_config["d_model"]
    intermediate_size = audio_config["encoder_ffn_dim"]
    attention_heads = audio_config["encoder_attention_heads"]
    head_dim = hidden_size // attention_heads

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0

    return {
        "audio_q_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "audio_k_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "audio_v_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "audio_out_proj": [attention_heads * head_dim // tp_size, hidden_size],
        "audio_fc1": [hidden_size, intermediate_size // tp_size],
        "audio_fc2": [intermediate_size // tp_size, hidden_size],
    }

def audio_post_process(model_params, args):
    """音频投影层：proj2 Linear(1280, 2048)"""
    audio_config = _audio_config(model_params)
    hidden_size = audio_config["d_model"]
    output_size = audio_config["output_dim"]
    return [{
        "name": "audio_proj",
        "stage": "audio",
        "OPs": args["batchsize"] * args["seqlen"] * hidden_size * output_size * 2,
        "load_weight": hidden_size * output_size * args["w_byte"],
        "load_act": args["batchsize"] * args["seqlen"] * hidden_size * args["a_byte"],
        "store_act": args["batchsize"] * args["seqlen"] * output_size * args["a_byte"],
    }]


# ===== Layer graphs =====
# 文本分支复用 qwen3_moe 的图结构
from models.qwen3_moe import transformer_layer_graph, flashattention_transformer_layer_graph

# 视觉分支复用 qwen3_vl 的图结构
from models.qwen3_vl import vision_layer_graph, vision_flashattention_layer_graph

# 音频分支的图结构（类似视觉，但用 fc1/fc2 代替 gate_proj/up_proj）
audio_layer_graph = {
    "audio_input": [],
    "audio_attn_norm": ["audio_input"],
    "audio_q_proj": ["audio_attn_norm"],
    "audio_k_proj": ["audio_attn_norm"],
    "audio_v_proj": ["audio_attn_norm"],
    "audio_qk_matmul": ["audio_q_proj", "audio_k_proj"],
    "audio_softmax": ["audio_qk_matmul"],
    "audio_sv_matmul": ["audio_softmax", "audio_v_proj"],
    "audio_out_proj": ["audio_sv_matmul"],
    "audio_attn_add": ["audio_input", "audio_out_proj"],
    "audio_mlp_norm": ["audio_attn_add"],
    "audio_fc1": ["audio_mlp_norm"],
    "audio_mlp_act": ["audio_fc1"],
    "audio_fc2": ["audio_mlp_act"],
    "audio_mlp_add": ["audio_attn_add", "audio_fc2"],
    "audio_output": ["audio_mlp_add"],
}

audio_flashattention_layer_graph = {
    "audio_input": [],
    "audio_attn_norm": ["audio_input"],
    "audio_q_proj": ["audio_attn_norm"],
    "audio_k_proj": ["audio_attn_norm"],
    "audio_v_proj": ["audio_attn_norm"],
    "audio_fused_attention": ["audio_q_proj", "audio_k_proj", "audio_v_proj"],
    "audio_out_proj": ["audio_fused_attention"],
    "audio_attn_add": ["audio_input", "audio_out_proj"],
    "audio_mlp_norm": ["audio_attn_add"],
    "audio_fc1": ["audio_mlp_norm"],
    "audio_mlp_act": ["audio_fc1"],
    "audio_fc2": ["audio_mlp_act"],
    "audio_mlp_add": ["audio_attn_add", "audio_fc2"],
    "audio_output": ["audio_mlp_add"],
}
