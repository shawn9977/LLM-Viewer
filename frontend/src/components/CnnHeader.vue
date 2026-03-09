<template>
    <div class="title">
        <a href="https://github.com/wayfeng/LLM-Viewer" target="_blank" class="hover-bold">LLM-Viewer</a>
        CNN Analyzer
    </div>
    <div class="header_button">
        |
        <span>Model: </span>
        <select v-model="select_model_id">
            <option v-for="mid in available_model_ids" :value="mid">{{ mid }}</option>
        </select>
        <span> | </span>
        <span>Hardware: </span>
        <select v-model="select_hardware">
            <option v-for="hw in available_hardwares" :value="hw">{{ hw }}</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span>Server: </span>
        <select v-model="ip_port">
            <option value="172.16.112.46:5000">172.16.112.46</option>
            <option value="172.16.112.118:5000">172.16.112.118</option>
            <option value="172.16.112.18:5000">172.16.112.18</option>
            <option value="127.0.0.1:5000">localhost</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <a href="/" class="hover-bold">LLM View</a>
    </div>
</template>

<script setup>
import { inject, ref, watch, onMounted } from 'vue';
import axios from 'axios';

const model_id = inject('model_id');
const hardware = inject('hardware');
const global_update_trigger = inject('global_update_trigger');
const ip_port = inject('ip_port');

const available_hardwares = ref([]);
const available_model_ids = ref([]);

function update_available() {
    const url = 'http://' + ip_port.value + '/get_available_cnn';
    axios.get(url).then(function (response) {
        available_hardwares.value = response.data.available_hardwares;
        available_model_ids.value = response.data.available_model_ids;
    }).catch(function (error) {
        console.log("error in get_available_cnn", error);
    });
}

onMounted(() => update_available());

var select_model_id = ref(model_id.value);
watch(select_model_id, (n) => {
    model_id.value = n;
    global_update_trigger.value += 1;
});

var select_hardware = ref(hardware.value);
watch(select_hardware, (n) => {
    hardware.value = n;
    global_update_trigger.value += 1;
});

watch(ip_port, () => update_available());
</script>

<style scoped>
.title {
    font-size: 18px;
    text-align: left;
}

.hover-bold {
    color: inherit;
}

.hover-bold:hover {
    font-weight: bold;
}
</style>
