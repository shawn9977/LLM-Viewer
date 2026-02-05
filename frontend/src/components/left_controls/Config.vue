<template>
    <h2>Inference Config</h2>
    <div class="config_div">
        Stage:
        <input type="radio" v-model="inference_stage" id="decode" value="decode" checked>
        <label for="decode">Decode</label>
        <input type="radio" v-model="inference_stage" id="prefill" value="prefill">
        <label for="prefill">Prefill</label>
        <input type="radio" v-model="inference_stage" id="chat" value="chat">
        <label for="prefill">Chat</label>
        <input type="radio" v-model="inference_stage" id="vision" value="vision">
        <label for="vision">Vision</label>
    </div>
    <div class="config_div">
        Batchsize:
        <input type="range" min="1" max="256" value="1" v-model.lazy="batch_size">
        <input type="number" v-model.lazy="batch_size" min="1" max="256">
    </div>
    <div class="config_div" v-if="inference_stage=='vision'">
        SeqLength:
        <input type="range" min="1" max="4096" value="1024" v-model.lazy="seq_length">
        <input type="number" v-model.lazy="seq_length" min="1" max="4096">
        <br/>
        Image Size:
        <input type="number" v-model.lazy="image_width" min="1" max="8192">
        <span> x </span>
        <input type="number" v-model.lazy="image_height" min="1" max="8192">
    </div>
    <div class="config_div" v-else-if="inference_stage!='chat'">
        SeqLength:
        <input type="range" min="1" max="4096" value="1024" v-model.lazy="seq_length">
        <input type="number" v-model.lazy="seq_length" min="1" max="4096">
    </div>
    <div class="config_div" v-else>
        PromptLength:
        <input type="range" min="1" max="4096" value="1024" v-model.lazy="seq_length">
        <!-- <span id="seq_length">1024</span> -->
        <input type="number" v-model.lazy="seq_length" min="1" max="4096">
        <br/>
        GenerateLength:
        <input type="range" min="1" max="4096" value="1024" v-model.lazy="gen_length">
        <!-- <span id="seq_length">1024</span> -->
        <input type="number" v-model.lazy="gen_length" min="1" max="4096">
    </div>
    <div class="config_div">
        Tensor parallelism
        <select v-model="tp_size">
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="4">4</option>
            <option value="8">8</option>
        </select>
    </div>
    <!-- <div class="config_div">
        Generation Length:
        <input type="range" min="1" max="4096" value="1024" oninput="gen_length.innerText = this.value">
        <p id="gen_length">1</p>
    </div> -->
    <h2>Optimization Config</h2>
    <div class="config_div">
        Weight Quantization:
        <select v-model="w_quant">
            <option value="FP16">FP16</option>
            <option value="8-bit">8-bit</option>
            <option value="4-bit">4-bit</option>
            <option value="2-bit">2-bit</option>
            <option value="1-bit">1-bit</option>
        </select>
    </div>
    <div class="config_div">
        Activation Quantization
        <select v-model="a_quant">
            <option value="FP16">FP16</option>
            <option value="8-bit">8-bit</option>
            <option value="4-bit">4-bit</option>
            <option value="2-bit">2-bit</option>
            <option value="1-bit">1-bit</option>
        </select>
    </div>
    <div class="config_div">
        KV Cache Quantization
        <select v-model="kv_quant">
            <option value="FP16">FP16</option>
            <option value="8-bit">8-bit</option>
            <option value="4-bit">4-bit</option>
            <option value="2-bit">2-bit</option>
            <option value="1-bit">1-bit</option>
        </select>
    </div>
    <div class="config_div">
        Use FlashAttention
        <input type="checkbox" v-model="use_flashattention">
    </div>
    <h2>Network-wise Analysis</h2>
    <div>
        <h3>{{ inference_stage }}</h3>
        <div v-for="(value, key) in total_results[inference_stage]" :key="key" class="network-wise-info-item">
            <span v-if="['bound'].includes(key)" class="highlight-span">{{ key }}: {{ value }}</span>
            <span v-else-if="['inference_time'].includes(key)" class="highlight-time">{{ key }}: {{ strNumberTime(value) }}</span>
            <span v-else-if="['OPs'].includes(key)" class="highlight-ops">{{ key }}: {{ strNumber(value, 'OPs') }}</span>
            <span v-else>{{ key }}: {{ strNumber(value, 'B') }}</span>
        </div>
        <p>NOTE: The time estimated by the roofline model represents the theoretical performance that the hardware can achieve. 
        The purpose of creating this tool is to help readers gain a clearer understanding of the key factors that influence LLM inference. 
        Only the relative relationships can be referenced. </p>
    </div>
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';
import { strNumber,strNumberTime } from '@/utils.js';

const global_update_trigger = inject('global_update_trigger');


const global_inference_config = inject('global_inference_config');
const total_results = inject('total_results');

const inference_stage = ref('decode');
const batch_size = ref(1);
const seq_length = ref(1024);
const image_width = ref(1024);
const image_height = ref(1024);
const gen_length = ref(1);
const tp_size = ref(1);
const w_quant = ref('8-bit');
const a_quant = ref('8-bit');
const kv_quant = ref('8-bit');
const use_flashattention = ref(false);

watch(inference_stage, (new_stage) => {
    console.log("inference_stage", new_stage)
    global_inference_config.value.stage = new_stage
    global_update_trigger.value += 1
})

watch(batch_size, (n) => {
    console.log("inference_stage", n)
    global_inference_config.value.batch_size = n
    global_update_trigger.value += 1
})

watch(seq_length, (n) => {
    console.log("seq_length", n)
    global_inference_config.value.seq_length = n
    global_update_trigger.value += 1
})

watch(image_width, (n) => {
    global_inference_config.value.image_size = {
        width: n,
        height: image_height.value
    }
    global_update_trigger.value += 1
})

watch(image_height, (n) => {
    global_inference_config.value.image_size = {
        width: image_width.value,
        height: n
    }
    global_update_trigger.value += 1
})

watch(tp_size, (n) => {
    console.log("tp_size", n)
    global_inference_config.value.tp_size = n
    global_update_trigger.value += 1
})

watch(w_quant, (n) => {
    console.log("w_quant", n)
    global_inference_config.value.w_quant = n
    global_update_trigger.value += 1
})

watch(a_quant, (n) => {
    console.log("a_quant", n)
    global_inference_config.value.a_quant = n
    global_update_trigger.value += 1
})

watch(kv_quant, (n) => {
    console.log("kv_quant", n)
    global_inference_config.value.kv_quant = n
    global_update_trigger.value += 1
})

watch(use_flashattention, (n) => {
    console.log("use_flashattention", n)
    global_inference_config.value.use_flashattention = n
    global_update_trigger.value += 1
})

watch(gen_length, (n) => {
    console.log("gen_length", n)
    global_inference_config.value.gen_length = n
    global_update_trigger.value += 1
})

</script>

<style>

.config_div{
    border-top: 1px solid #e2e2e2;
}

.hover_color {
    color: #0000ff;
    cursor: pointer;
}

.network-wise-info-item {
    padding: 3px;
    border-top: 1px solid #e2e2e2;
}

.highlight-span {
    color: #d7263d;
    font-weight: bold;
    background: #fffbe6;
    padding: 2px 6px;
    border-radius: 4px;
}
.highlight-time {
    color: #1e88e5;
}
.highlight-ops {
    color: #43a047;
}

</style>