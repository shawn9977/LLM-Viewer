<template>
    <div class="main_graph" ref="graphContainer">
        <div id="info-window" class="float-info-window" v-if="info_window_str.length>0">
            <h3> {{ info_window_str }}</h3>
        </div>
        <div id="cnnGraphContainer" @resize="handleResize"></div>

        <div class="float-search-window">
            <input type="text" v-model.lazy="searchText" placeholder="Search" />
            <div>
                <div v-for="(value) in searchResult" @click="SelectNode(value, true)">
                    {{ value }}
                </div>
            </div>
        </div>
        <div class="float-node-info-window">
            <div v-if="selected_node_id" class="float-node-info-item">
                <strong>{{ selected_node_id }}</strong>
            </div>
            <div v-for="(value, key) in all_node_info[selected_node_id]" :key="key" class="float-node-info-item">
                <span v-if="['bound'].includes(key)">{{ key }}: {{ value }}</span>
                <span v-else-if="['inference_time'].includes(key)">{{ key }}: {{ strNumberTime(value) }}</span>
                <span v-else>{{ key }}: {{ strNumber(value) }}</span>
            </div>
            <div class="float-node-info-item">
                <canvas id="cnnLineChart" width="300" height="200"></canvas>
            </div>
        </div>
    </div>
</template>

<script setup>
import G6 from "@antv/g6"
import { onMounted, watch, inject, ref } from 'vue'
import axios from 'axios'
import { strNumber, strNumberTime } from '@/utils.js';
import { Chart, registerables } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

const model_id = inject('model_id')
const hardware = inject('hardware')
const global_update_trigger = inject('global_update_trigger')
const global_cnn_config = inject('global_cnn_config')
const ip_port = inject('ip_port')
const total_results = inject('total_results')

var hardware_info = {}
var nowFocusNode = null
var nowFocusNodePrevColor = null
var graph = null
var graph_data
const all_node_info = ref({})
Chart.register(...registerables, annotationPlugin)

const searchText = ref('')
var searchResult = []
const selected_node_id = ref("")
var roofline_chart = null
const info_window_str = ref('')

const changeGraphSizeWaitTimer = ref(false)
window.onresize = () => {
    if (!changeGraphSizeWaitTimer.value && graph != null) {
        var leftControlDiv = document.querySelector('.left_control')
        var width = leftControlDiv ? leftControlDiv.offsetWidth : 0
        graph.changeSize(window.innerWidth - width, window.innerHeight)
        changeGraphSizeWaitTimer.value = true
        setTimeout(() => { changeGraphSizeWaitTimer.value = false }, 100)
    }
}

function graphUpdate() {
    const url = 'http://' + ip_port.value + '/get_cnn_graph'
    info_window_str.value = "Loading from server..."
    axios.post(url, {
        model_id: model_id.value,
        hardware: hardware.value,
        cnn_config: global_cnn_config.value
    }).then(function (response) {
        info_window_str.value = ""
        graph_data = response.data
        for (let i = 0; i < graph_data.nodes.length; i++) {
            all_node_info.value[graph_data.nodes[i].id] = graph_data.nodes[i].info
        }
        total_results.value = response.data.total_results
        hardware_info = response.data.hardware_info

        const old_ids = new Set(graph.getNodes().map(node => node.get('id')))
        const new_ids = new Set(graph_data.nodes.map(node => node.id))
        const is_equal = old_ids.size === new_ids.size && [...old_ids].every(key => new_ids.has(key))

        if (is_equal) {
            graph_data.nodes.forEach(function (node) {
                graph.updateItem(node.id, { description: node.description, label: node.label })
            })
        } else {
            nowFocusNode = null
            graph.clear()
            graph.data(graph_data)
            graph.render()
        }
        setTimeout(() => { update_roofline_model() }, 10)
        setTimeout(() => { graph.fitView() }, 10)
    }).catch(function (error) {
        info_window_str.value = "Error in get_cnn_graph"
        console.log("error in graphUpdate", error)
    })
}

watch(() => global_update_trigger.value, () => graphUpdate())

function handleSearch(newText) {
    const nodes = graph.findAll('node', (node) => node.get('id').includes(newText))
    searchResult.length = 0
    for (let i = 0; i < Math.min(nodes.length, 100); i++) {
        searchResult.push(nodes[i].get('id'))
    }
}
watch(searchText, handleSearch)

function SelectNode(nodeId, moveView = false) {
    if (moveView) graph.focusItem(nodeId, true)
    if (nowFocusNode) {
        nowFocusNode.update({ style: { fill: nowFocusNodePrevColor } })
    }
    const node = graph.findById(nodeId)
    if (node) {
        nowFocusNodePrevColor = node.getModel().style.fill || "#ffffff"
        node.update({ style: { fill: "#dffdff" } })
        nowFocusNode = node
    }
    selected_node_id.value = nodeId
}

function update_roofline_model() {
    const ctx = document.getElementById('cnnLineChart')
    if (!ctx) return
    if (roofline_chart) roofline_chart.destroy()

    const bandwidth = hardware_info["bandwidth"]
    const max_OPS = hardware_info["max_OPS"]
    const turningPoint = max_OPS / bandwidth

    var annotation, x_max
    if (selected_node_id.value) {
        const node_ai = all_node_info.value[selected_node_id.value]?.["arithmetic_intensity"]
        x_max = Math.max(turningPoint * 3, (node_ai || 0) + 1)
        annotation = {
            annotations: {
                lineX: {
                    type: 'line',
                    xMin: node_ai, xMax: node_ai,
                    yMin: 0, yMax: max_OPS * 1.1,
                    borderColor: 'blue', borderWidth: 2,
                    borderDash: [5, 5],
                    label: { enabled: true, content: 'Node AI', position: 'top' }
                }
            }
        }
    } else {
        annotation = {}
        x_max = turningPoint * 3
    }

    roofline_chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Roofline',
                data: [{ x: 0, y: 0 }, { x: turningPoint, y: max_OPS }, { x: x_max, y: max_OPS }],
                borderColor: 'black', borderWidth: 2, fill: false, pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Arithmetic Intensity (OPs/byte)' },
                    type: 'linear',
                    ticks: { callback: (v) => v.toFixed(1) },
                    beginAtZero: true, max: x_max
                },
                y: {
                    title: { display: true, text: 'Performance (OPS)' },
                    ticks: { callback: (v) => v.toExponential(1) },
                    beginAtZero: true, max: max_OPS * 1.1
                }
            },
            plugins: {
                title: { display: true, text: 'Roofline Model', position: 'top' },
                legend: { display: false },
                annotation: annotation
            }
        }
    })
}

function release_select() {
    selected_node_id.value = ""
    update_roofline_model()
}

function clickNode(node) {
    SelectNode(node.id)
    setTimeout(() => { update_roofline_model() }, 100)
}

onMounted(() => {
    graph = new G6.Graph({
        container: 'cnnGraphContainer',
        width: window.innerWidth,
        height: window.innerHeight,
        defaultEdge: {
            type: 'polyline',
            sourceAnchor: 1,
            targetAnchor: 0,
            style: {
                endArrow: { path: G6.Arrow.triangle(5, 10), fill: "#aaaaaa", opacity: 50 },
                stroke: "#000000",
            },
        },
        defaultNode: {
            type: 'modelRect',
            size: [190, 60],
            anchorPoints: [[0.5, 0], [0.5, 1]],
            logoIcon: { show: false },
            stateIcon: { show: false },
            labelCfg: { offset: 15, style: { fill: '#000000', fontSize: 20, stroke: '#E7E7E7' } },
            descriptionCfg: { style: { fill: '#656565', fontSize: 14 } },
        },
        modes: { default: ['drag-canvas', 'zoom-canvas', 'lasso-select'] },
        layout: { type: 'dagre', nodesep: 10, ranksep: 20, controlPoints: true },
    })
    graph.on('node:click', (event) => clickNode(event.item.getModel()))
    graph.on('node:touchstart', (event) => clickNode(event.item.getModel()))
    graph.on('canvas:click', () => release_select())
    graphUpdate()
    graph.render()
})
</script>

<style scoped>
.main_graph {
    width: 75%;
    height: 100%;
    position: relative;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    float: right;
    flex-grow: 1;
    background-color: #ffffff;
    border: 0px;
}

.float-search-window {
    position: absolute;
    top: 10px;
    right: 10px;
    height: auto;
    max-height: 50vh;
    background-color: #f1f1f1b7;
    padding: 3px;
    overflow-y: auto;
}

.float-info-window {
    position: absolute;
    top: 10px;
    left: 40%;
    height: auto;
    width: 20%;
    background-color: #f1f1f1b7;
    padding: 5px;
    overflow-y: auto;
}

.float-node-info-window {
    position: absolute;
    top: 10px;
    left: 10px;
    background-color: #f1f1f1b7;
}

.float-node-info-item {
    padding: 3px;
    border-top: 1px solid #e2e2e2;
}
</style>
