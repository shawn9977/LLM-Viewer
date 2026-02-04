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
        # for decode
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen) * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * num_key_value_heads) * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in self.module.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            if "rmsnorm" in name:
                norm_OPs = batchsize * hidden_size * 1 * 4
            else:
                norm_OPs = batchsize * hidden_size * 1 * 7

            self._analyze_to_results(
                "decode",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 5,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
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
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in self.module.get_norm_layers(model_params):
            if "rmsnorm" in name:
                norm_OPs = batchsize * hidden_size * seqlen * 4
            else:
                norm_OPs = batchsize * hidden_size * seqlen * 7

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
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1 * 5,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
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
        # for decode
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen) * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * num_key_value_heads) * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in self.module.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            if "rmsnorm" in name:
                norm_OPs = batchsize * hidden_size * 1 * 4
            else:
                norm_OPs = batchsize * hidden_size * 1 * 7

            self._analyze_to_results(
                "decode",
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 5 * num_active_experts,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte * num_active_experts,
                store_act=batchsize * hidden_size * 1 * a_byte * num_active_experts,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
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
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in self.module.get_norm_layers(model_params):
            if "rmsnorm" in name:
                norm_OPs = batchsize * hidden_size * seqlen * 4
            else:
                norm_OPs = batchsize * hidden_size * seqlen * 7

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
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]: # swish activation
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1 * 5 * num_active_experts,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte * num_active_experts,
                store_act=batchsize * hidden_size * seqlen * a_byte * num_active_experts,
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
    def __init__(self, model_id, hardware, model_params=None):
        super().__init__(model_id, hardware, model_params=model_params)

    def analyze(self, seqlen, batchsize, w_bit=16, a_bit=16, kv_bit=None, use_flashattention=False, kv_token_ratio=1, tp_size=1, image_size=None):
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}, "vision": {}}
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

        # ===== Text branch (same as LLM) =====
        num_attention_heads = self.module.get_num_attention_heads(model_params)
        hidden_size = self.module.get_hidden_size(model_params)
        num_key_value_heads = self.module.get_num_key_value_heads(model_params)
        num_hidden_layers = self.module.get_num_hidden_layers(model_params)

        for name, (ic, oc) in self.module.get_linear_layers(model_params, tp_size).items():
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

        head_size = hidden_size // num_attention_heads
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = "fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = 1 * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,
                load_kv_cache=n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
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

        for name in self.module.get_norm_layers(model_params):
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

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size,
                load_weight=0,
                load_act=batchsize * hidden_size * a_byte,
                store_act=batchsize * hidden_size * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 5,
                load_weight=0,
                load_act=batchsize * hidden_size * a_byte,
                store_act=batchsize * hidden_size * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        qk_matmul_OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
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

        total_results = {"decode": {}, "prefill": {}, "vision": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
            total_results["vision"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for _, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = sum(result["store_act"] for result in self.results["decode"].values())
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = sum(result["store_act"] for result in self.results["prefill"].values())
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        args = {"batchsize": batchsize, "seqlen": seqlen, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.module.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]

        # ===== Vision branch =====
        def _parse_image_size(size):
            if isinstance(size, dict):
                width = size.get("width") or size.get("w")
                height = size.get("height") or size.get("h")
                if width and height:
                    return int(width), int(height)
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return int(size[0]), int(size[1])
            if isinstance(size, str) and "x" in size:
                parts = size.lower().split("x")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
            return 1024, 1024

        image_w, image_h = _parse_image_size(image_size)
        patch_size = self.module.get_vision_patch_size(model_params)
        spatial_merge_size = self.module.get_vision_spatial_merge_size(model_params)
        in_channels = self.module.get_vision_in_channels(model_params)
        vision_hidden_size = self.module.get_vision_hidden_size(model_params)
        vision_num_heads = self.module.get_vision_num_heads(model_params)
        vision_intermediate_size = self.module.get_vision_intermediate_size(model_params)
        vision_num_layers = self.module.get_vision_num_hidden_layers(model_params)

        num_patches_w = max(1, math.ceil(image_w / patch_size))
        num_patches_h = max(1, math.ceil(image_h / patch_size))
        num_patches = num_patches_w * num_patches_h
        merged_tokens = max(1, math.ceil(num_patches / max(1, spatial_merge_size) ** 2))

        patch_ic = in_channels * patch_size * patch_size
        patch_oc = vision_hidden_size
        self._analyze_to_results(
            "vision",
            "vision_patch_embed",
            OPs=patch_ic * patch_oc * batchsize * num_patches * 2,
            load_weight=patch_ic * patch_oc * w_byte,
            load_act=patch_ic * batchsize * num_patches * a_byte,
            store_act=patch_oc * batchsize * num_patches * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
        )

        for name, (ic, oc) in self.module.get_vision_linear_layers(model_params, tp_size).items():
            self._analyze_to_results(
                "vision",
                name,
                OPs=ic * oc * batchsize * merged_tokens * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * merged_tokens * a_byte,
                store_act=oc * batchsize * merged_tokens * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        vision_head_size = vision_hidden_size // vision_num_heads
        v_qk_OPs = merged_tokens * merged_tokens * vision_head_size * vision_num_heads * batchsize * 2
        v_sv_OPs = merged_tokens * vision_head_size * merged_tokens * vision_num_heads * batchsize * 2
        v_softmax_OPs = batchsize * vision_num_heads * merged_tokens * merged_tokens * 5

        if use_flashattention:
            name = "vision_fused_attention"
            bandwidth, max_OPS, onchip_buffer = get_hardware_info(self.hardware, self.w_bit, self.a_bit, self.kv_bit)
            block_size_r = min(math.ceil(onchip_buffer / (a_byte * vision_head_size)), vision_head_size)
            n_blocks_r = math.ceil(merged_tokens / block_size_r)
            q_numel = merged_tokens * vision_head_size * batchsize * vision_num_heads * a_byte
            kv_numel = merged_tokens * vision_head_size * batchsize * vision_num_heads * a_byte * 2
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

        for name in self.module.get_vision_norm_layers(model_params):
            norm_OPs = batchsize * vision_hidden_size * merged_tokens * 7
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

        for name, result in self.results["vision"].items():
            multiplier = vision_num_layers if name in vision_repeat_layers else 1
            for data_name in ALL_DATA_NAMES:
                total_results["vision"][data_name] += result[data_name] * multiplier

        vision_args = {"batchsize": batchsize, "seqlen": merged_tokens, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.module.vision_post_process(self.model_params, vision_args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]

        vision_tmp_act = 0
        for name, result in self.results["vision"].items():
            multiplier = vision_num_layers if name in vision_repeat_layers else 1
            vision_tmp_act += result["store_act"] * multiplier
        vision_weight = total_results["vision"]["load_weight"]
        total_results["vision"]["memory_consumption"] = vision_tmp_act + vision_weight
        total_results["vision"]["memory_consumption_tmp_act"] = vision_tmp_act
        total_results["vision"]["memory_consumption_weight"] = vision_weight
        total_results["vision"]["memory_consumption_kv_cache"] = 0

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
