<template>
    <h2>CNN Config</h2>
    <div class="config_div">
        Batchsize:
        <input type="range" min="1" max="256" value="1" v-model.lazy="batchsize">
        <input type="number" v-model.lazy="batchsize" min="1" max="256">
    </div>
    <h2>Quantization Config</h2>
    <div class="config_div">
        Weight Quantization:
        <select v-model="w_quant">
            <option value="FP16">FP16</option>
            <option value="8-bit">8-bit</option>
            <option value="4-bit">4-bit</option>
            <option value="2-bit">2-bit</option>
        </select>
    </div>
    <div class="config_div">
        Activation Quantization:
        <select v-model="a_quant">
            <option value="FP16">FP16</option>
            <option value="8-bit">8-bit</option>
        </select>
    </div>
    <h2>Network-wise Analysis</h2>
    <div>
        <div v-for="(value, key) in total_results['inference']" :key="key" class="network-wise-info-item">
            <span v-if="['bound'].includes(key)" class="highlight-span">{{ key }}: {{ value }}</span>
            <span v-else-if="['inference_time'].includes(key)" class="highlight-time">{{ key }}: {{ strNumberTime(value) }}</span>
            <span v-else-if="['OPs'].includes(key)" class="highlight-ops">{{ key }}: {{ strNumber(value, 'OPs') }}</span>
            <span v-else>{{ key }}: {{ strNumber(value, 'B') }}</span>
        </div>
        <p>NOTE: The time estimated by the roofline model represents the theoretical performance that the hardware can achieve.</p>
    </div>
</template>

<script setup>
import { inject, ref, watch } from 'vue';
import { strNumber, strNumberTime } from '@/utils.js';

const global_update_trigger = inject('global_update_trigger');
const global_cnn_config = inject('global_cnn_config');
const total_results = inject('total_results');

const batchsize = ref(1);
const w_quant = ref('FP16');
const a_quant = ref('FP16');

watch(batchsize, (n) => {
    global_cnn_config.value.batchsize = n;
    global_update_trigger.value += 1;
});

watch(w_quant, (n) => {
    global_cnn_config.value.w_quant = n;
    global_update_trigger.value += 1;
});

watch(a_quant, (n) => {
    global_cnn_config.value.a_quant = n;
    global_update_trigger.value += 1;
});
</script>

<style>
.config_div {
    border-top: 1px solid #e2e2e2;
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
