# LLM-Viewer

LLM-Viewer 用于可视化大模型/CNN 计算图，并基于 roofline 模型估算推理耗时与显存开销。项目提供：
- Web 界面（交互式查看节点与配置）
- CLI 分析工具（批量或脚本化分析）

## Main 分支主要修改记录（近期）

根据 `main` 分支提交记录（2026-01 ~ 2026-03）整理：
- 后端与前端分目录重构，补充 Docker 化部署流程。
- `ModelAnalyzer` 重构，修复 CLI/路径问题，补充模型文件下载能力。
- 新增与完善 Qwen3、Qwen3-VL、Qwen3-Omni 等模型支持。
- 推理阶段逻辑调整：以 `prefill/decode` 为主，移除 `MMTTFT/MMTPOT` 阶段。
- 增加 Tensor Parallel 相关优化（QK/SV/Softmax、Norm/Add 等算子路径）。
- 更新前端图逻辑与 docker-compose 配置，并补充 CNN 模型与视觉相关文档。

## 快速开始（Docker）

在仓库根目录执行：

```bash
docker compose build
docker compose up --build
```

也可以分两步启动（推荐在首次构建后使用）：

```bash
docker compose up -d
```

- 前端默认地址：`http://localhost:5173`
- 后端默认地址：`http://localhost:5000`

停止/清理容器常用命令：

```bash
# 仅停止容器（保留容器与网络）
docker compose stop

# 停止并移除容器与网络
docker compose down

# 停止并移除容器、网络、数据卷（谨慎使用）
docker compose down -v
```

## 本地开发

### 1) 启动后端

```bash
cd backend
pip install -r requirements.txt
python app.py --local --port 5000
```

### 2) 启动前端

```bash
cd frontend
npm install
VITE_IP_PORT=127.0.0.1:5000 \
VITE_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
VITE_HARDWARE=intel_ARC_B60 \
npm run dev
```

## CLI 用法

```bash
cd backend
python analyze_cli.py Qwen/Qwen3-4B-Instruct-2507 intel_ARC_B60 --batchsize 1 --seqlen 1024
python analyze_cli.py Qwen/Qwen3-VL-8B-Instruct intel_ARC_B60 --batchsize 1 --seqlen 1024 --tp-size 2
```

常用参数：
- `--batchsize`
- `--seqlen`
- `--w_bit --a_bit --kv_bit`
- `--use_flashattention`
- `--tp-size`

## 目录说明

- `backend/`：模型分析、硬件参数、Flask API、CLI
- `frontend/`：Vue + Vite 可视化界面
- `backend/models/`：模型配置与实现入口
- `backend/cnn_models/`：CNN 模型图相关实现
- `patches/`：历史补丁与演进记录

## 说明

本工具给出的是 roofline 视角下的理论估算结果，适合用于趋势分析与方案对比，不等价于端到端真实线上延迟。

论文：LLM Inference Unveiled: Survey and Roofline Model Insights（arXiv:2402.16363）