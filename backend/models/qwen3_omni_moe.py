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
    """embed_tokens + lm_head（非重复层，仅执行 1 次）"""
    text_config = _text_config(model_params)
    hidden_size = text_config["hidden_size"]
    vocab_size = text_config["vocab_size"]
    layers = []
    # embed_tokens: Embedding(152064, 2048) — 查表操作
    # prefill: 查 seqlen 个 token
    layers.append({
        "name": "embed_tokens",
        "stage": "prefill",
        "OPs": 0,  # Embedding 是查表，无乘加运算
        "load_weight": args["batchsize"] * args["seqlen"] * hidden_size * args["w_byte"],
        "load_act": 0,
        "store_act": args["batchsize"] * args["seqlen"] * hidden_size * args["a_byte"],
    })
    # decode: 查 1 个 token
    layers.append({
        "name": "embed_tokens",
        "stage": "decode",
        "OPs": 0,
        "load_weight": args["batchsize"] * hidden_size * args["w_byte"],
        "load_act": 0,
        "store_act": args["batchsize"] * hidden_size * args["a_byte"],
    })
    # lm_head: Linear(2048, 152064)
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

    # 实际模型: qkv 融合 Linear(1152, 3456) + proj Linear(1152, 1152)
    # 这里拆成 q/k/v 三个，OPs 总量一致
    # 实际模型 MLP: linear_fc1(1152, 4304) + GELU + linear_fc2(4304, 1152)
    # 是标准 2 层 MLP，不是 SwiGLU gated MLP
    return {
        "vision_q_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "vision_k_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "vision_v_proj": [hidden_size, attention_heads * head_dim // tp_size],
        "vision_out_proj": [attention_heads * head_dim // tp_size, hidden_size],
        "vision_fc1": [hidden_size, intermediate_size // tp_size],
        "vision_fc2": [intermediate_size // tp_size, hidden_size],
    }

def vision_post_process(model_params, args):
    """
    视觉 merger 投影层:
      merger.ln_q: LayerNorm(4608)
      merger.mlp[0]: Linear(4608, 4608)
      merger.mlp[1]: GELU
      merger.mlp[2]: Linear(4608, 2048)
    spatial_merge_size=2 → 合并 2x2=4 个 patch → 输入维度 = 4 * hidden_size
    另有 merger_list (3个额外 merger，对应 deepstack_visual_indexes)
    """
    vision_config = _vision_config(model_params)
    hidden_size = vision_config["hidden_size"]
    out_hidden_size = vision_config.get("out_hidden_size")
    spatial_merge_size = vision_config.get("spatial_merge_size", 1)
    if out_hidden_size is None:
        return []

    # merger 输入维度 = spatial_merge_size^2 * hidden_size
    merger_input_size = (spatial_merge_size ** 2) * hidden_size  # 4 * 1152 = 4608

    # deepstack_visual_indexes 决定额外 merger 数量
    deepstack_indexes = vision_config.get("deepstack_visual_indexes", [])
    num_mergers = 1 + len(deepstack_indexes)  # 1 个主 merger + 3 个 merger_list

    layers = []
    for i in range(num_mergers):
        prefix = f"vision_merger_{i}_" if i > 0 else "vision_merger_"
        # merger.ln_q: LayerNorm(4608)
        layers.append({
            "name": f"{prefix}ln_q",
            "stage": "vision",
            "OPs": args["batchsize"] * args["seqlen"] * merger_input_size * 7,
            "load_weight": merger_input_size * args["w_byte"],
            "load_act": args["batchsize"] * args["seqlen"] * merger_input_size * args["a_byte"],
            "store_act": args["batchsize"] * args["seqlen"] * merger_input_size * args["a_byte"],
        })
        # merger.mlp[0]: Linear(4608, 4608)
        layers.append({
            "name": f"{prefix}fc1",
            "stage": "vision",
            "OPs": args["batchsize"] * args["seqlen"] * merger_input_size * merger_input_size * 2,
            "load_weight": merger_input_size * merger_input_size * args["w_byte"],
            "load_act": args["batchsize"] * args["seqlen"] * merger_input_size * args["a_byte"],
            "store_act": args["batchsize"] * args["seqlen"] * merger_input_size * args["a_byte"],
        })
        # merger.mlp[1]: GELU
        layers.append({
            "name": f"{prefix}act",
            "stage": "vision",
            "OPs": args["batchsize"] * args["seqlen"] * merger_input_size * 5,
            "load_weight": 0,
            "load_act": args["batchsize"] * args["seqlen"] * merger_input_size * args["a_byte"],
            "store_act": args["batchsize"] * args["seqlen"] * merger_input_size * args["a_byte"],
        })
        # merger.mlp[2]: Linear(4608, 2048)
        layers.append({
            "name": f"{prefix}fc2",
            "stage": "vision",
            "OPs": args["batchsize"] * args["seqlen"] * merger_input_size * out_hidden_size * 2,
            "load_weight": merger_input_size * out_hidden_size * args["w_byte"],
            "load_act": args["batchsize"] * args["seqlen"] * merger_input_size * args["a_byte"],
            "store_act": args["batchsize"] * args["seqlen"] * out_hidden_size * args["a_byte"],
        })
    return layers


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
    """
    音频前端和投影层（非重复层，仅执行 1 次）:
      conv2d1: Conv2d(1, 480, k=3, s=2, p=1)
      conv2d2: Conv2d(480, 480, k=3, s=2, p=1)
      conv2d3: Conv2d(480, 480, k=3, s=2, p=1)
      conv_out: Linear(7680, 1280)
      ln_post: LayerNorm(1280)
      proj1: Linear(1280, 1280)
      act: GELU
      proj2: Linear(1280, 2048)
    """
    audio_config = _audio_config(model_params)
    hidden_size = audio_config["d_model"]  # 1280
    output_size = audio_config["output_dim"]  # 2048
    downsample_hidden = audio_config.get("downsample_hidden_size", 480)
    num_mel_bins = audio_config.get("num_mel_bins", 128)

    bs = args["batchsize"]
    seq = args["seqlen"]  # audio_tokens
    a_byte = args["a_byte"]
    w_byte = args["w_byte"]

    # 音频输入: mel spectrogram, 频率维度 = num_mel_bins
    # conv2d 每次 stride=2 → 频率维度: 128 → 64 → 32 → 16
    # 时间维度也 stride=2: T → T/2 → T/4 → T/8 (= audio_tokens)
    # n_window=50 → 原始时间帧 = audio_length * 50
    # 经过 3 次 stride=2 → audio_tokens = audio_length * 50 / 8
    freq_dim = num_mel_bins  # 128
    # 原始时间帧数（conv 之前）
    time_frames_0 = seq * 8  # 还原到 conv 之前

    layers = []

    # conv2d1: Conv2d(1, 480, k=3, s=2, p=1) → 输出 (480, freq/2, T/2)
    freq_1 = freq_dim // 2  # 64
    time_1 = time_frames_0 // 2
    conv1_ops = bs * downsample_hidden * 1 * 3 * 3 * freq_1 * time_1 * 2
    conv1_weight = downsample_hidden * 1 * 3 * 3 * w_byte
    layers.append({
        "name": "audio_conv2d1",
        "stage": "audio",
        "OPs": conv1_ops,
        "load_weight": conv1_weight,
        "load_act": bs * 1 * freq_dim * time_frames_0 * a_byte,
        "store_act": bs * downsample_hidden * freq_1 * time_1 * a_byte,
    })

    # conv2d2: Conv2d(480, 480, k=3, s=2, p=1) → 输出 (480, freq/4, T/4)
    freq_2 = freq_1 // 2  # 32
    time_2 = time_1 // 2
    conv2_ops = bs * downsample_hidden * downsample_hidden * 3 * 3 * freq_2 * time_2 * 2
    conv2_weight = downsample_hidden * downsample_hidden * 3 * 3 * w_byte
    layers.append({
        "name": "audio_conv2d2",
        "stage": "audio",
        "OPs": conv2_ops,
        "load_weight": conv2_weight,
        "load_act": bs * downsample_hidden * freq_1 * time_1 * a_byte,
        "store_act": bs * downsample_hidden * freq_2 * time_2 * a_byte,
    })

    # conv2d3: Conv2d(480, 480, k=3, s=2, p=1) → 输出 (480, freq/8, T/8)
    freq_3 = freq_2 // 2  # 16
    time_3 = time_2 // 2  # = seq (audio_tokens)
    conv3_ops = bs * downsample_hidden * downsample_hidden * 3 * 3 * freq_3 * time_3 * 2
    conv3_weight = downsample_hidden * downsample_hidden * 3 * 3 * w_byte
    layers.append({
        "name": "audio_conv2d3",
        "stage": "audio",
        "OPs": conv3_ops,
        "load_weight": conv3_weight,
        "load_act": bs * downsample_hidden * freq_2 * time_2 * a_byte,
        "store_act": bs * downsample_hidden * freq_3 * time_3 * a_byte,
    })

    # conv_out: Linear(7680, 1280) — 480 * 16 = 7680 (频率维度折叠到通道)
    conv_out_ic = downsample_hidden * freq_3  # 480 * 16 = 7680
    layers.append({
        "name": "audio_conv_out",
        "stage": "audio",
        "OPs": bs * seq * conv_out_ic * hidden_size * 2,
        "load_weight": conv_out_ic * hidden_size * w_byte,
        "load_act": bs * seq * conv_out_ic * a_byte,
        "store_act": bs * seq * hidden_size * a_byte,
    })

    # ln_post: LayerNorm(1280) — 最终 LayerNorm，仅 1 次
    layers.append({
        "name": "audio_ln_post",
        "stage": "audio",
        "OPs": bs * seq * hidden_size * 7,
        "load_weight": hidden_size * w_byte,
        "load_act": bs * seq * hidden_size * a_byte,
        "store_act": bs * seq * hidden_size * a_byte,
    })

    # proj1: Linear(1280, 1280)
    layers.append({
        "name": "audio_proj1",
        "stage": "audio",
        "OPs": bs * seq * hidden_size * hidden_size * 2,
        "load_weight": hidden_size * hidden_size * w_byte,
        "load_act": bs * seq * hidden_size * a_byte,
        "store_act": bs * seq * hidden_size * a_byte,
    })

    # act: GELU
    layers.append({
        "name": "audio_proj_act",
        "stage": "audio",
        "OPs": bs * seq * hidden_size * 5,
        "load_weight": 0,
        "load_act": bs * seq * hidden_size * a_byte,
        "store_act": bs * seq * hidden_size * a_byte,
    })

    # proj2: Linear(1280, 2048)
    layers.append({
        "name": "audio_proj2",
        "stage": "audio",
        "OPs": bs * seq * hidden_size * output_size * 2,
        "load_weight": hidden_size * output_size * w_byte,
        "load_act": bs * seq * hidden_size * a_byte,
        "store_act": bs * seq * output_size * a_byte,
    })

    return layers


# ===== Layer graphs =====
# 文本分支复用 qwen3_moe 的图结构
from models.qwen3_moe import transformer_layer_graph, flashattention_transformer_layer_graph

# 视觉分支：Qwen3-Omni 的 ViT 使用标准 2 层 MLP (fc1 + GELU + fc2)，不是 SwiGLU gated MLP
# 因此不能复用 qwen3_vl 的 vision_layer_graph（那个是 gate_proj/up_proj/down_proj）
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
    "vision_fc1": ["vision_norm2"],
    "vision_mlp_act": ["vision_fc1"],
    "vision_fc2": ["vision_mlp_act"],
    "vision_mlp_add": ["vision_attn_add", "vision_fc2"],
    "vision_output": ["vision_mlp_add"],
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
    "vision_fc1": ["vision_norm2"],
    "vision_mlp_act": ["vision_fc1"],
    "vision_fc2": ["vision_mlp_act"],
    "vision_mlp_add": ["vision_attn_add", "vision_fc2"],
    "vision_output": ["vision_mlp_add"],
}

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
