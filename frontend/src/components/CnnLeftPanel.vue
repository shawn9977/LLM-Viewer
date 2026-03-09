<template>
    <div class="left_control">
        <CnnConfig></CnnConfig>
    </div>
</template>

<script setup>
import CnnConfig from "./left_controls/CnnConfig.vue"
</script>

<style>
.left_control {
    align-items: left;
    width: 30%;
    height: 100%;
    background-color: #f0f0f0;
    border: 3px solid #e2e2e2;
    overflow-y: auto;
    margin-bottom: 10px;
}
</style>
