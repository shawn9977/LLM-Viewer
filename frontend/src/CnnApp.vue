<script setup>
import CnnGraph from "./components/CnnGraph.vue"
import CnnLeftPanel from "./components/CnnLeftPanel.vue"
import CnnHeader from "./components/CnnHeader.vue"

import { ref, provide } from 'vue';

const model_id = ref('mobilenet_v2');
const hardware = ref(import.meta.env.VITE_HARDWARE || 'nvidia_A100');
const global_update_trigger = ref(1);
const total_results = ref({});
const ip_port = ref(import.meta.env.VITE_IP_PORT || '127.0.0.1:5000');

provide("model_id", model_id);
provide("hardware", hardware);
provide("global_update_trigger", global_update_trigger);
provide("total_results", total_results);
provide("ip_port", ip_port);

const global_cnn_config = ref({
  w_quant: "FP16",
  a_quant: "FP16",
  batchsize: 1,
});
provide("global_cnn_config", global_cnn_config);
</script>

<template>
  <div class="app_container">
    <div class="upper_header">
      <CnnHeader></CnnHeader>
    </div>
    <div class="bottom-block">
      <CnnLeftPanel></CnnLeftPanel>
      <CnnGraph></CnnGraph>
    </div>
  </div>
</template>

<style>
body {
  overflow-x: hidden;
  overflow-y: hidden;
}

.app_container {
  width: 100%;
  height: 100vh;
}

.upper_header {
  flex: 1;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 50px;
  background-color: #f0f0f0;
  border-bottom: 3px solid #e2e2e2;
}

.bottom-block {
  display: flex;
  flex-direction: row;
  height: calc(100% - 60px);
}
</style>
