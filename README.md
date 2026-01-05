# LLM Training Platform

基于 [verl](https://github.com/volcengine/verl) 框架的大语言模型训练平台，提供完整的训练任务管理、计算配置优化、模型手术和实时监控功能。

## 功能特性

### 核心功能
- **计算配置器**: 自动计算最优 GPU 配置、批量大小、ZeRO 阶段
- **训练任务管理**: 创建、启动、暂停、恢复训练任务
- **模型手术台**: 模型合并 (SLERP/TIES/DARE)、检查点选择、SWA 平均
- **实时监控**: Loss/Reward 曲线、GPU 利用率、梯度统计、WebSocket 推送

### 数据管理
- **向量存储**: Milvus 向量库支持，语义相似搜索
- **数据去重**: 基于向量相似度的语义去重
- **数据分布分析**: 字段值统计、占比分析、直方图

### 平台支持
- **Linux (NVIDIA GPU)**: CUDA 12.0+, A100/H100/RTX4090 等
- **macOS (Apple Silicon)**: M1/M2/M3/M4 系列，MPS 后端

## 支持的训练算法

| 算法 | 说明 | 适用场景 |
|------|------|----------|
| SFT | 监督微调 | 基础能力对齐 |
| PPO | 近端策略优化 | RLHF 训练 |
| GRPO | 组相对策略优化 | 无 Critic 的高效 RL |
| DPO | 直接偏好优化 | 偏好对齐 |
| GSPO | 组自博弈偏好优化 | 自我改进 |

## 快速开始

### 环境要求

| 平台 | 要求 |
|------|------|
| **通用** | Python 3.10+, Node.js 18+ (前端) |
| **Linux** | CUDA 12.0+, NVIDIA Driver 525+ |
| **macOS** | macOS 13.0+ (Ventura), Apple Silicon (M1/M2/M3/M4) |

### 安装

#### Linux (NVIDIA GPU)

```bash
# 克隆项目
cd train_platform

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt

# 安装 verl (从源码)
cd verl && pip install -e . && cd ..

# 安装前端依赖
cd frontend && npm install && cd ..
```

#### macOS (Apple Silicon)

```bash
# 克隆项目
cd train_platform

# 创建虚拟环境 (推荐使用 miniforge)
conda create -n train_platform python=3.12
conda activate train_platform

# 安装 PyTorch (MPS 后端会自动启用)
pip install torch torchvision torchaudio

# 安装项目依赖
pip install -r requirements.txt

# 安装 verl (从源码)
cd verl && pip install -e . && cd ..

# 安装前端依赖
cd frontend && npm install && cd ..
```

### 启动开发环境

#### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

#### 2. 启动 Docker 容器（Milvus、Redis 等中间件）

```bash
docker-compose up -d
```

等待服务就绪后，可以访问：
- Milvus UI (Attu): http://localhost:3001
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

#### 3. 启动后端服务（开多个终端窗口）

**终端 1 - Training Platform API：**
```bash
uvicorn training_platform.api.main:app --reload --port 8000
```

**终端 2 - 前端开发服务器：**
```bash
cd frontend && npm run dev
```

#### 4. 停止服务

```bash
# 停止 Docker 容器
docker-compose down

# 后端和前端直接 Ctrl+C 停止
```

### 服务地址

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://localhost:5173 | Vue 开发服务器 |
| Training API | http://localhost:8000 | 主后端 API |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| Milvus | localhost:19530 | 向量数据库 |
| Milvus UI | http://localhost:3002 | Attu 管理界面 |
| Redis | localhost:6381 | 缓存 |

## 文档

- **[训练指南](docs/TRAINING_GUIDE.md)**: 完整的训练教程，包括：
  - 各算法数据格式（SFT/GRPO/PPO/DPO/GSPO）
  - 模型下载方式
  - 训练运行命令
  - 监控与观测（W&B/Prometheus/Grafana）

- **[数据准备](docs/DATA_PREPARATION.md)**: 数据格式详细说明和转换工具

## 项目结构

```
train_platform/
├── training_platform/          # 主后端代码
│   ├── api/
│   │   ├── main.py            # FastAPI 应用入口
│   │   ├── models/            # Pydantic 数据模型
│   │   └── routers/           # API 路由
│   │       ├── compute.py     # 计算配置 API
│   │       ├── jobs.py        # 训练任务 API
│   │       ├── surgery.py     # 模型手术 API
│   │       ├── monitoring.py  # 监控 API (含 Push 模式)
│   │       ├── websocket.py   # WebSocket + MetricsCollector
│   │       └── dataset.py     # 数据集 API (上传/搜索/去重/分析)
│   └── core/
│       ├── memory_estimator.py    # 显存估算 (支持 Apple Silicon)
│       ├── compute_calculator.py  # 配置计算
│       ├── model_merger.py        # 模型合并
│       ├── checkpoint_selector.py # 检查点选择
│       ├── verl_adapter.py        # verl 适配器
│       ├── ray_runner.py          # Ray 任务提交
│       ├── database.py            # SQLModel 数据库
│       └── vector_store.py        # Milvus 向量库
├── frontend/                   # Vue 3 前端
├── tests/                      # 测试文件 (175+ 测试)
├── models/                     # 模型存储目录 (自动检测)
├── datasets/                   # 训练数据目录 (自动检测)
├── docs/                       # 文档
│   ├── TRAINING_GUIDE.md      # 训练指南（数据格式/模型下载/运行/监控）
│   └── DATA_PREPARATION.md    # 数据准备详细文档
├── docker-compose.yml          # Docker 容器编排 (Milvus/Redis/MinIO)
├── .env.example                # 环境变量模板
└── requirements.txt
```

## 评估系统

### 评估模式

平台支持三种评估模式：

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| **API** | OpenAI 兼容 API | 云端模型评估 (DashScope/OpenAI) |
| **本地模型** | vLLM 本地推理 | 评估本地预训练模型 |
| **检查点** | vLLM 加载检查点 | 评估训练过程中的模型 |

### 本地模型推理 (vLLM)

本地模型推理使用 vLLM 进行高效推理，支持：

- 自动模型缓存（避免重复加载）
- 批量推理优化
- ChatML 格式自动转换
- 支持 HuggingFace 格式模型

**使用示例：**

```python
# API 调用
POST /api/v1/evaluation/trigger
{
  "dataset_uuids": ["xxx"],
  "model_type": "local_model",
  "model_path": "/path/to/your/model"
}

# 或检查点评估
{
  "dataset_uuids": ["xxx"],
  "model_type": "checkpoint",
  "checkpoint_id": 123
}
```

**注意事项：**
- 需要有 GPU 可用于 vLLM 推理
- 首次加载模型需要一定时间
- 模型会被缓存以加速后续推理

### 训练数据集管理

支持训练数据集的标签分析和 Loss 计算可视化：

- **标签字段配置**: 选择用于分组的字段（如 tenant, difficulty）
- **分布统计**: 查看各标签的数据分布
- **Loss 高亮**: 查看单条样本时，高亮显示计算 Loss 的部分

### 模型对照

比较训练前后模型的表现差异：

- **整体对比**: 准确率变化、改进/退化数量统计
- **样本级差异**: 查看每个样本的模型输出对比
- **筛选功能**: 按改进/退化/不变筛选样本
