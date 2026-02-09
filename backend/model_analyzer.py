import os
import logging
import importlib
import math
from hardwares import get_hardware_info
from roofline_model import roofline_analyze
from model_params import load_model_params
from utils import str_number, str_number_time

logger = logging.getLogger(__name__)

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
    #"throughtput",
]

MODEL_ANALYZER_REGISTRY = {
    "LLMAnalyzer": ["qwen3", "qwen2", "qwen2_5", "llama", "chatglm"],
    "MoEAnalyzer": ["qwen3_moe", "qwen2_moe", "qwen2_5_moe"],
    "VLMAnalyzer": ["qwen3_vl", "qwen2_vl", "qwen2_5_vl"],
}


class ModelAnalyzer:
    def __init__(self, model_id, hardware, model_params=None):
        self.model_id = model_id
        self.hardware = hardware
   
        self.model_params = model_params if model_params is not None else load_model_params(model_id)
        model_type = self.model_params["model_type"]
        self.module = importlib.import_module(f"models.{model_type.lower()}")
 
        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None

    def analyze(self, seqlen, batchsize, w_bit=16, a_bit=16, kv_bit=None, use_flashattention=False, kv_token_ratio=1, tp_size=1, image_size=None):
        """
        seqlen: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        pass

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1
    ):
        prefill_result = self.analyze(
            prompt_len,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size
        )
        prefill_time = inference_time = prefill_result["total_results"]["prefill"]["inference_time"]

        for i in range(prompt_len, prompt_len + gen_len):
            result = self.analyze(i, batchsize, w_bit, a_bit, kv_bit, use_flashattention=use_flashattention, tp_size=tp_size)
            inference_time += result["total_results"]["decode"]["inference_time"]
        return {"inference_time": inference_time, "prefill_time": prefill_time}

    def if_group_qa(self):
        """
        Returns whether the model uses Grouped Query Attention (GQA).
        """
        return (
            self.module.get_num_attention_heads(self.model_params) !=
            self.module.get_num_key_value_heads(self.model_params)
        )

    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):
        bandwidth, max_OPS, _ = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
        memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
        inference_time = OPs / performance
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }

    def save_csv(self, save_path=None):
        if save_path is None:
            save_path = f"output/{self.model_id[:self.model_id.rfind('/')]}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path += f"{self.model_id[self.model_id.rfind('/'):]}"

        decode_file_name = f"{save_path}_decode.csv"
        prefill_file_name = f"{save_path}_prefill.csv"
        print(f"save to {decode_file_name} and {prefill_file_name}")

        for file_name, stage in [
            (decode_file_name, "decode"),
            (prefill_file_name, "prefill"),
        ]:
            with open(file_name, "a+") as f:

                f.write(
                    f"\n\n=== {self.model_id} {self.hardware} w_bit={self.w_bit} a_bit={self.a_bit} kv_bit={self.kv_bit} batchsize={self.batchsize} seqlen={self.seqlen} tp_size={self.tp_size} ===\n"
                )
                # legend
                f.write(
                    f"layer_name,OPs,Access,arithmetic_intensity,performance,bound,load_weight,load_act,store_act,load_kv_cache,store_kv_cache,inference_time\n"
                )
            with open(file_name, "a+") as f:
                for layer_name, result in self.results[stage].items():
                    f.write(
                        f"{layer_name},{str_number(result['OPs'])},{str_number(result['memory_access'])}B,{str_number(result['arithmetic_intensity'])},{str_number(result['performance'])},"
                        f"{result['bound']},{str_number(result['load_weight'])}B,{str_number(result['load_act'])}B,{str_number(result['store_act'])}B,{str_number(result['load_kv_cache'])}B,"
                        f"{str_number(result['store_kv_cache'])}B,{str_number_time(result['inference_time'])}s\n"
                    )

class LLMAnalyzer(ModelAnalyzer):
    def __init__(self, model_id, hardware, model_params=None):
        super().__init__(model_id, hardware, model_params=model_params)
    def analyze(self, seqlen, batchsize, w_bit=16, a_bit=16, kv_bit=None, use_flashattention=False, kv_token_ratio=1, tp_size=1, image_size=None):
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        model_params = self.model_params
        num_attention_heads = self.module.get_num_attention_heads(model_params)
        hidden_size = self.module.get_hidden_size(model_params)
        num_key_value_heads = self.module.get_num_key_value_heads(model_params)
        num_hidden_layers = self.module.get_num_hidden_layers(model_params)

        for name, (ic, oc) in self.module.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj

            self._analyze_to_results(
                "decode",
                name,
                OPs=ic * oc * batchsize * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * a_byte,
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ic * oc * batchsize * seqlen * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
            )

        # for attention
        head_size = hidden_size // num_attention_heads
        tp_num_attention_heads = max(1, num_attention_heads // tp_size)
        tp_num_key_value_heads = max(1, num_key_value_heads // tp_size)
        # for decode
        qk_matmul_OPs = seqlen * head_size * tp_num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * tp_num_attention_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * tp_num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * tp_num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * tp_num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * tp_num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * tp_num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=(seqlen) * head_size * batchsize * tp_num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * tp_num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * tp_num_key_value_heads) * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * tp_num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * tp_num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in self.module.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            if "rmsnorm" in name:
                norm_OPs = batchsize * (hidden_size / tp_size) * 1 * 4
            else:
                norm_OPs = batchsize * (hidden_size / tp_size) * 1 * 7

            self._analyze_to_results(
                "decode",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * (hidden_size / tp_size) * 1,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * (hidden_size / tp_size) * 1 * 5,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = seqlen * seqlen * head_size * tp_num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * tp_num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * tp_num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_size * batchsize * tp_num_attention_heads * a_byte
            o_numel = seqlen * seqlen * batchsize * tp_num_attention_heads * a_byte
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * tp_num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * tp_num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * tp_num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * tp_num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * tp_num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * tp_num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * tp_num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in self.module.get_norm_layers(model_params):
            if "rmsnorm" in name:
                norm_OPs = batchsize * (hidden_size / tp_size) * seqlen * 4
            else:
                norm_OPs = batchsize * (hidden_size / tp_size) * seqlen * 7

            self._analyze_to_results(
                "prefill",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * (hidden_size / tp_size) * seqlen * 1,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * (hidden_size / tp_size) * seqlen * 1 * 5,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for _, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # memory footprint
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = 0
        for _, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = 0
        for _, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # lm_head
        name = "lm_head"
        args = {"batchsize": batchsize, "seqlen":seqlen, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.module.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]

        self.results["total_results"] = total_results
        return self.results

class MoEAnalyzer(ModelAnalyzer):
    def __init__(self, model_id, hardware, model_params=None):
        super().__init__(model_id, hardware, model_params=model_params)

    def analyze(self, seqlen, batchsize, w_bit=16, a_bit=16, kv_bit=None, use_flashattention=False, kv_token_ratio=1, tp_size = 1, image_size=None):
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        model_params = self.model_params
        num_attention_heads = self.module.get_num_attention_heads(model_params)
        hidden_size = self.module.get_hidden_size(model_params)
        num_key_value_heads = self.module.get_num_key_value_heads(model_params)
        num_hidden_layers = self.module.get_num_hidden_layers(model_params)
        num_active_experts = self.module.get_num_active_experts(model_params)
        for name, (ic, oc) in self.module.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            is_expert_proj = name in ["gate_proj", "up_proj", "down_proj"]

            if is_expert_proj:
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=ic * oc * batchsize * 2 * num_active_experts,
                    load_weight=ic * oc * w_byte * num_active_experts,
                    load_act=ic * batchsize * a_byte * num_active_experts,
                    store_act=oc * batchsize * a_byte * num_active_experts,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
                # for prefill
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=ic * oc * batchsize * seqlen * 2 * num_active_experts,
                    load_weight=ic * oc * w_byte * num_active_experts,
                    load_act=ic * batchsize * seqlen * a_byte * num_active_experts,
                    store_act=oc * batchsize * seqlen * a_byte * num_active_experts,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
            else:
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=ic * oc * batchsize * 2,
                    load_weight=ic * oc * w_byte,
                    load_act=ic * batchsize * a_byte,
                    store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
                )
                # for prefill
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=ic * oc * batchsize * seqlen * 2,
                    load_weight=ic * oc * w_byte,
                    load_act=ic * batchsize * seqlen * a_byte,
                    store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                    load_kv_cache=0,
                    store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
                )

        # for attention
        head_size = hidden_size // num_attention_heads
        tp_num_attention_heads = max(1, num_attention_heads // tp_size)
        tp_num_key_value_heads = max(1, num_key_value_heads // tp_size)
        # for decode
        qk_matmul_OPs = seqlen * head_size * tp_num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * tp_num_attention_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * tp_num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * tp_num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * tp_num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * tp_num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * tp_num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=(seqlen) * head_size * batchsize * tp_num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * tp_num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * tp_num_key_value_heads) * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * tp_num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * tp_num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in self.module.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            if "rmsnorm" in name:
                norm_OPs = batchsize * (hidden_size / tp_size) * 1 * 4
            else:
                norm_OPs = batchsize * (hidden_size / tp_size) * 1 * 7

            self._analyze_to_results(
                "decode",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * (hidden_size / tp_size) * 1,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * (hidden_size / tp_size) * 1 * 5 * num_active_experts,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * 1 * a_byte * num_active_experts,
                store_act=batchsize * (hidden_size / tp_size) * 1 * a_byte * num_active_experts,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = seqlen * seqlen * head_size * tp_num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * tp_num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * tp_num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_size * batchsize * tp_num_attention_heads * a_byte
            o_numel = seqlen * seqlen * batchsize * tp_num_attention_heads * a_byte
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * tp_num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * tp_num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * tp_num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * tp_num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * tp_num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * tp_num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * tp_num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * tp_num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in self.module.get_norm_layers(model_params):
            if "rmsnorm" in name:
                norm_OPs = batchsize * (hidden_size / tp_size) * seqlen * 4
            else:
                norm_OPs = batchsize * (hidden_size / tp_size) * seqlen * 7

            self._analyze_to_results(
                "prefill",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * (hidden_size / tp_size) * seqlen * 1,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                store_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * (hidden_size / tp_size) * seqlen * 1 * 5 * num_active_experts,
                load_weight=0,
                load_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte * num_active_experts,
                store_act=batchsize * (hidden_size / tp_size) * seqlen * a_byte * num_active_experts,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # memory footprint
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # lm_head
        name = "lm_head"
        args = {"batchsize": batchsize, "seqlen":seqlen, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.module.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]

        self.results["total_results"] = total_results
        return self.results

class VLMAnalyzer(ModelAnalyzer):
    """视觉-语言模型分析器，用于分析 VLM 模型（如 Qwen2-VL, Qwen3-VL）的性能"""
    def __init__(self, model_id, hardware, model_params=None):
        super().__init__(model_id, hardware, model_params=model_params)

    def analyze(self, seqlen, batchsize, w_bit=16, a_bit=16, kv_bit=None, use_flashattention=False, kv_token_ratio=1, tp_size=1, image_size=None):
        """
        分析 VLM 模型的性能，包括文本分支和视觉分支

        参数:
            seqlen: 序列长度
            batchsize: 批次大小
            w_bit: 权重位宽
            a_bit: 激活值位宽
            kv_bit: KV 缓存位宽
            use_flashattention: 是否使用 Flash Attention
            kv_token_ratio: KV 压缩比率
            tp_size: 张量并行大小
            image_size: 图像尺寸（宽x高）
        """
        assert seqlen > 0
        assert batchsize > 0
        # 初始化结果字典，包含 decode（解码）、prefill（预填充）和 vision（视觉）三个阶段
        self.results = {"decode": {}, "prefill": {}, "vision": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        # 计算每个数据类型的字节数
        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        model_params = self.model_params

        # ===== 文本分支（与 LLM 相同）=====
        # 获取模型的基本参数
        num_attention_heads = self.module.get_num_attention_heads(model_params)  # 注意力头数量
        hidden_size = self.module.get_hidden_size(model_params)  # 隐藏层大小
        num_key_value_heads = self.module.get_num_key_value_heads(model_params)  # KV 头数量
        num_hidden_layers = self.module.get_num_hidden_layers(model_params)  # 隐藏层数量

        # 遍历所有线性层（如 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）
        # 遍历所有线性层（如 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）
        for name, (ic, oc) in self.module.get_linear_layers(model_params, tp_size).items():
            # 判断是否为 KV 投影层（k_proj 和 v_proj）
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            # 分析 decode 阶段的线性层
            self._analyze_to_results(
                "decode",
                name,
                OPs=ic * oc * batchsize * 2,  # 矩阵乘法的操作数（乘法+加法）
                load_weight=ic * oc * w_byte,  # 加载权重的内存访问量
                load_act=ic * batchsize * a_byte,  # 加载激活值的内存访问量
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,  # KV 投影层不存储激活值
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),  # KV 投影层需要存储到 KV 缓存
            )
            # 分析 prefill 阶段的线性层
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ic * oc * batchsize * seqlen * 2,  # prefill 阶段需要处理整个序列
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
            )

        # 计算注意力机制的参数
        head_size = hidden_size // num_attention_heads  # 每个注意力头的维度
        # decode 阶段的注意力计算量
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2  # Q @ K^T
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2  # Softmax(QK^T) @ V
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5  # Softmax 操作（5步：max, sub, exp, sum, div）
        if use_flashattention:
            # 使用 Flash Attention 融合算子
            name = "fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # Flash Attention-2 的分块大小计算
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)  # 分块数量
            q_numel = 1 * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,  # 融合所有注意力操作
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # 初始化 O 和保存 O
                load_kv_cache=n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            # 不使用 Flash Attention，分别计算 QK、SV 和 Softmax
            self._analyze_to_results(
                "decode",
                "qk_matmul",
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "decode",
                "sv_matmul",
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "decode",
                "softmax",
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # 分析归一化层（如 RMSNorm 或 LayerNorm）
        for name in self.module.get_norm_layers(model_params):
            # RMSNorm 需要 4 步操作，LayerNorm 需要 7 步操作
            norm_OPs = batchsize * hidden_size * 1 * (4 if "rmsnorm" in name else 7)
            self._analyze_to_results(
                "decode",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * hidden_size * a_byte,
                store_act=batchsize * hidden_size * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # 分析残差连接（注意力和 MLP 的加法操作）
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size,  # 逐元素加法
                load_weight=0,
                load_act=batchsize * hidden_size * a_byte,
                store_act=batchsize * hidden_size * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        # 分析 MLP 激活函数（如 SwiGLU）
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 5,  # Swish 激活函数需要 5 步操作
                load_weight=0,
                load_act=batchsize * hidden_size * a_byte,
                store_act=batchsize * hidden_size * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # prefill 阶段的注意力计算（处理整个序列）
        qk_matmul_OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            # prefill 阶段使用 Flash Attention
            name = "fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
            o_numel = seqlen * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,
                load_kv_cache=n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            # prefill 阶段不使用 Flash Attention
            self._analyze_to_results(
                "prefill",
                "qk_matmul",
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "prefill",
                "sv_matmul",
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "prefill",
                "softmax",
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # prefill 阶段的归一化层
        for name in self.module.get_norm_layers(model_params):
            norm_OPs = batchsize * hidden_size * seqlen * (4 if "rmsnorm" in name else 7)
            self._analyze_to_results(
                "prefill",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        # prefill 阶段的残差连接
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        # prefill 阶段的 MLP 激活函数
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 5,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # 计算总结果
        total_results = {"decode": {}, "prefill": {}, "vision": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
            total_results["vision"][data_name] = 0
        # 累加所有层的结果（乘以层数）
        for stage in ["decode", "prefill"]:
            for _, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # 计算内存占用
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        # decode 阶段的临时激活值内存
        decode_tmp_act = sum(result["store_act"] for result in self.results["decode"].values())
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        # prefill 阶段的临时激活值内存
        prefill_tmp_act = sum(result["store_act"] for result in self.results["prefill"].values())
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # 处理 lm_head 层（语言模型头）
        args = {"batchsize": batchsize, "seqlen": seqlen, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.module.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]

        # ===== 视觉分支 =====
        def _parse_image_size(size):
            """解析图像尺寸，支持多种输入格式"""
            if isinstance(size, dict):
                # 字典格式：{"width": 1024, "height": 768} 或 {"w": 1024, "h": 768}
                width = size.get("width") or size.get("w")
                height = size.get("height") or size.get("h")
                if width and height:
                    return int(width), int(height)
            if isinstance(size, (list, tuple)) and len(size) == 2:
                # 列表或元组格式：[1024, 768]
                return int(size[0]), int(size[1])
            if isinstance(size, str) and "x" in size:
                # 字符串格式："1024x768"
                parts = size.lower().split("x")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
            # 默认尺寸
            return 1024, 1024

        # 解析图像尺寸并获取视觉编码器参数
        image_w, image_h = _parse_image_size(image_size)
        patch_size = self.module.get_vision_patch_size(model_params)  # 图像块大小（如 14x14）
        spatial_merge_size = self.module.get_vision_spatial_merge_size(model_params)  # 空间合并大小
        in_channels = self.module.get_vision_in_channels(model_params)  # 输入通道数（通常为 3，RGB）
        vision_hidden_size = self.module.get_vision_hidden_size(model_params)  # 视觉编码器隐藏层大小
        vision_num_heads = self.module.get_vision_num_heads(model_params)  # 视觉编码器注意力头数量
        vision_intermediate_size = self.module.get_vision_intermediate_size(model_params)  # 视觉编码器中间层大小
        vision_num_layers = self.module.get_vision_num_hidden_layers(model_params)  # 视觉编码器层数

        # 计算图像块数量和合并后的 token 数量
        num_patches_w = max(1, math.ceil(image_w / patch_size))  # 宽度方向的图像块数量
        num_patches_h = max(1, math.ceil(image_h / patch_size))  # 高度方向的图像块数量
        num_patches = num_patches_w * num_patches_h  # 总图像块数量
        merged_tokens = max(1, math.ceil(num_patches / max(1, spatial_merge_size) ** 2))  # 空间合并后的 token 数量

        # 分析图像块嵌入层（Patch Embedding）
        patch_ic = in_channels * patch_size * patch_size  # 输入通道数（如 3*14*14=588）
        patch_oc = vision_hidden_size  # 输出通道数
        self._analyze_to_results(
            "vision",
            "vision_patch_embed",
            OPs=patch_ic * patch_oc * batchsize * num_patches * 2,  # 卷积或线性投影的计算量
            load_weight=patch_ic * patch_oc * w_byte,
            load_act=patch_ic * batchsize * num_patches * a_byte,
            store_act=patch_oc * batchsize * num_patches * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
        )

        # 分析视觉编码器的线性层（Q、K、V 投影和 MLP 层）
        for name, (ic, oc) in self.module.get_vision_linear_layers(model_params, tp_size).items():
            self._analyze_to_results(
                "vision",
                name,
                OPs=ic * oc * batchsize * merged_tokens * 2,  # 线性层的计算量
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * merged_tokens * a_byte,
                store_act=oc * batchsize * merged_tokens * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # 分析视觉编码器的注意力机制
        vision_head_size = vision_hidden_size // vision_num_heads  # 每个注意力头的维度
        v_qk_OPs = merged_tokens * merged_tokens * vision_head_size * vision_num_heads * batchsize * 2  # Q @ K^T
        v_sv_OPs = merged_tokens * vision_head_size * merged_tokens * vision_num_heads * batchsize * 2  # Softmax(QK^T) @ V
        v_softmax_OPs = batchsize * vision_num_heads * merged_tokens * merged_tokens * 5  # Softmax 操作

        if use_flashattention:
            # 视觉编码器使用 Flash Attention
            name = "vision_fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            block_size_r = min(math.ceil(onchip_buffer / (a_byte * vision_head_size)), vision_head_size)
            n_blocks_r = math.ceil(merged_tokens / block_size_r)
            q_numel = merged_tokens * vision_head_size * batchsize * vision_num_heads * a_byte
            kv_numel = merged_tokens * vision_head_size * batchsize * vision_num_heads * a_byte * 2  # K 和 V
            o_numel = merged_tokens * merged_tokens * batchsize * vision_num_heads * a_byte
            self._analyze_to_results(
                "vision",
                name,
                OPs=v_qk_OPs + v_sv_OPs + v_softmax_OPs,
                load_weight=0,
                load_act=q_numel + kv_numel,
                store_act=o_numel * 2,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        else:
            # 视觉编码器不使用 Flash Attention
            self._analyze_to_results(
                "vision",
                "vision_qk_matmul",
                OPs=v_qk_OPs,
                load_weight=0,
                load_act=merged_tokens * vision_head_size * batchsize * vision_num_heads * a_byte,
                store_act=merged_tokens * merged_tokens * batchsize * vision_num_heads * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "vision",
                "vision_sv_matmul",
                OPs=v_sv_OPs,
                load_weight=0,
                load_act=merged_tokens * merged_tokens * batchsize * vision_num_heads * a_byte,
                store_act=merged_tokens * vision_head_size * batchsize * vision_num_heads * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "vision",
                "vision_softmax",
                OPs=v_softmax_OPs,
                load_weight=0,
                load_act=batchsize * vision_num_heads * merged_tokens * merged_tokens * a_byte,
                store_act=batchsize * vision_num_heads * merged_tokens * merged_tokens * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # 分析视觉编码器的归一化层
        for name in self.module.get_vision_norm_layers(model_params):
            norm_OPs = batchsize * vision_hidden_size * merged_tokens * 7  # LayerNorm 需要 7 步操作
            self._analyze_to_results(
                "vision",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * vision_hidden_size * merged_tokens * a_byte,
                store_act=batchsize * vision_hidden_size * merged_tokens * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # 分析视觉编码器的残差连接
        for name in ["vision_attn_add", "vision_mlp_add"]:
            self._analyze_to_results(
                "vision",
                name,
                OPs=batchsize * vision_hidden_size * merged_tokens,
                load_weight=0,
                load_act=batchsize * vision_hidden_size * merged_tokens * a_byte,
                store_act=batchsize * vision_hidden_size * merged_tokens * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # 分析视觉编码器的 MLP 激活函数
        self._analyze_to_results(
            "vision",
            "vision_mlp_act",
            OPs=batchsize * vision_hidden_size * merged_tokens * 5,
            load_weight=0,
            load_act=batchsize * vision_hidden_size * merged_tokens * a_byte,
            store_act=batchsize * vision_hidden_size * merged_tokens * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
        )

        # 定义需要重复计算的视觉层（这些层在每个 Transformer 块中都会出现）
        vision_repeat_layers = {
            "vision_q_proj",
            "vision_k_proj",
            "vision_v_proj",
            "vision_out_proj",
            "vision_gate_proj",
            "vision_up_proj",
            "vision_down_proj",
            "vision_qk_matmul",
            "vision_sv_matmul",
            "vision_softmax",
            "vision_norm1",
            "vision_norm2",
            "vision_attn_add",
            "vision_mlp_add",
            "vision_mlp_act",
            "vision_fused_attention",
        }

        # 累加视觉分支的结果（重复层需要乘以层数）
        for name, result in self.results["vision"].items():
            multiplier = vision_num_layers if name in vision_repeat_layers else 1
            for data_name in ALL_DATA_NAMES:
                total_results["vision"][data_name] += result[data_name] * multiplier

        # 处理视觉编码器的后处理层（如投影层）
        vision_args = {"batchsize": batchsize, "seqlen": merged_tokens, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.module.vision_post_process(self.model_params, vision_args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]

        # 计算视觉分支的内存占用
        vision_tmp_act = 0
        for name, result in self.results["vision"].items():
            multiplier = vision_num_layers if name in vision_repeat_layers else 1
            vision_tmp_act += result["store_act"] * multiplier
        vision_weight = total_results["vision"]["load_weight"]
        total_results["vision"]["memory_consumption"] = vision_tmp_act + vision_weight
        total_results["vision"]["memory_consumption_tmp_act"] = vision_tmp_act
        total_results["vision"]["memory_consumption_weight"] = vision_weight
        total_results["vision"]["memory_consumption_kv_cache"] = 0  # 视觉编码器不使用 KV 缓存

        # ===== 多模态总计 =====
        # 计算 TTFT（Time To First Token，首 token 时间）和 TPOT（Time Per Output Token，每 token 时间）
        total_results["multimodal_ttft"] = {}  # TTFT = 视觉编码 + 文本预填充
        total_results["multimodal_tpot"] = {}  # TPOT = 文本解码
        for data_name in ALL_DATA_NAMES:
            # TTFT 包含视觉分支和文本 prefill 分支的计算量
            total_results["multimodal_ttft"][data_name] = (
                total_results["vision"][data_name] + total_results["prefill"][data_name]
            )
            # TPOT 只包含文本 decode 分支的计算量
            total_results["multimodal_tpot"][data_name] = total_results["decode"][data_name]

        # TTFT 阶段的内存占用：权重相加，临时激活值取最大值，KV 缓存来自文本 prefill
        ttft_weight = (
            total_results["vision"]["memory_consumption_weight"] +
            total_results["prefill"]["memory_consumption_weight"]
        )
        # 临时激活值取最大值（因为视觉和文本不会同时占用内存）
        ttft_tmp_act = max(
            total_results["vision"]["memory_consumption_tmp_act"],
            total_results["prefill"]["memory_consumption_tmp_act"],
        )
        ttft_kv = total_results["prefill"]["memory_consumption_kv_cache"]
        total_results["multimodal_ttft"]["memory_consumption_weight"] = ttft_weight
        total_results["multimodal_ttft"]["memory_consumption_tmp_act"] = ttft_tmp_act
        total_results["multimodal_ttft"]["memory_consumption_kv_cache"] = ttft_kv
        total_results["multimodal_ttft"]["memory_consumption"] = ttft_weight + ttft_tmp_act + ttft_kv

        # TPOT 阶段的内存占用（与文本 decode 相同）
        total_results["multimodal_tpot"]["memory_consumption"] = total_results["decode"]["memory_consumption"]
        total_results["multimodal_tpot"]["memory_consumption_weight"] = total_results["decode"]["memory_consumption_weight"]
        total_results["multimodal_tpot"]["memory_consumption_tmp_act"] = total_results["decode"]["memory_consumption_tmp_act"]
        total_results["multimodal_tpot"]["memory_consumption_kv_cache"] = total_results["decode"]["memory_consumption_kv_cache"]

        # 保存总结果并返回
        self.results["total_results"] = total_results
        return self.results

class YOLOAnalyzer(ModelAnalyzer):
    def __init__(self, model_id, hardware, model_params=None):
        super().__init__(model_id, hardware, model_params=model_params)


def get_analyzer(model_id, hardware) -> ModelAnalyzer:
    params = load_model_params(model_id)
    model_type = params["model_type"]
    analyzer_class = None
    for class_name, types in MODEL_ANALYZER_REGISTRY.items():
        if model_type in types:
            analyzer_class = globals()[class_name]
            break
    if analyzer_class is None:
        raise ValueError(f"Unknown model_type: {model_type}")
    ma = analyzer_class(model_id, hardware, model_params=params)
    return ma
