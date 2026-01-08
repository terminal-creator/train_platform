# LLM Training Platform

基于 [verl](https://github.com/volcengine/verl) 框架的大语言模型训练平台，提供完整的训练任务管理、计算配置优化、模型手术和实时监控功能。

**当前版本**: v1.3.0 (Phase 3 - 任务系统升级完成)
**verl 版本**: b12eb3b (v0.7.0-23)

## 🎯 功能特性

### 核心功能
- **计算配置器**: 自动计算最优 GPU 配置、批量大小、ZeRO 阶段
- **训练任务管理**: 创建、启动、暂停、恢复训练任务
- **模型手术台**: 模型合并 (SLERP/TIES/DARE)、检查点选择、SWA 平均
- **实时监控**: Loss/Reward 曲线、GPU 利用率、梯度统计、WebSocket 推送

### 数据管理
- **向量存储**: Milvus 向量库支持，语义相似搜索
- **数据去重**: 基于向量相似度的语义去重
- **数据分布分析**: 字段值统计、占比分析、直方图

### 运行模式
- **本地模式**: 在本机运行训练（需要 NVIDIA GPU）
- **SSH 远程模式**: 通过 SSH 连接远程 GPU 服务器执行训练

### 🔒 安全特性（Phase 0）
- **环境固化**: 固定版本依赖，确保可复现性
- **密码加密**: SSH 密码使用 Fernet 对称加密存储
- **命令注入防护**: 所有命令执行经过安全处理
- **路径验证**: 防止路径遍历和命令注入攻击

### ⚡ 异步任务系统（Phase 3 新增）
- **Celery 分布式任务队列**: 基于 Redis 的异步任务系统
- **Pipeline 工作流编排**: 多阶段训练流水线管理
- **优先级队列**: 5 个优先级队列（训练/评测/默认/预处理/维护）
- **任务监控**: Flower UI 实时监控和管理
- **自动重试**: 任务失败自动重试机制
- **周期任务**: Celery Beat 调度器支持定时任务

### 平台支持
- **Linux (NVIDIA GPU)**: CUDA 12.1+, A100/H100/RTX4090 等
- **macOS (Apple Silicon)**: M1/M2/M3/M4 系列（仅管理节点，训练需使用 SSH 远程模式）

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

#### 方式 1: 自动化安装（推荐）

**本地环境（管理节点或训练节点）:**

```bash
# 克隆项目（包含 verl submodule）
git clone --recursive https://github.com/your-org/train_platform.git
cd train_platform

# 管理节点（macOS/Linux，仅运行平台 API）
bash scripts/setup_local_env.sh manager

# 训练节点（Linux + GPU，可运行训练任务）
bash scripts/setup_local_env.sh training
```

**远程 GPU 服务器:**

```bash
# 在本地推送并自动安装到远程
bash scripts/setup_remote_env.sh user@gpu-server

# 或在远程服务器上直接运行
ssh user@gpu-server
cd /path/to/train_platform
bash scripts/setup_remote_env.sh --remote
```

**验证安装:**

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux
# 或 source venv/bin/activate.fish (fish shell)

# 验证环境
python scripts/verify_env.py
```

#### 方式 2: 手动安装

<details>
<summary>点击展开手动安装步骤</summary>

**Linux (NVIDIA GPU)**

```bash
# 克隆项目
git clone --recursive https://github.com/your-org/train_platform.git
cd train_platform

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖（固定版本）
pip install -r environments/requirements-training.txt

# 安装 verl (从 submodule)
cd environments/verl && pip install -e . && cd ../..

# 安装前端依赖
cd frontend && npm install && cd ..
```

**macOS (Apple Silicon)**

```bash
# 克隆项目
git clone --recursive https://github.com/your-org/train_platform.git
cd train_platform

# 创建虚拟环境 (推荐使用 miniforge)
conda create -n train_platform python=3.12
conda activate train_platform

# 安装 PyTorch (MPS 后端会自动启用)
pip install torch torchvision torchaudio

# 安装项目依赖（管理节点，无 GPU 依赖）
pip install -r environments/requirements-manager.txt

# 安装前端依赖
cd frontend && npm install && cd ..
```

</details>

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
- Milvus UI (Attu): http://localhost:3002
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
- Flower (Celery 监控): http://localhost:5555 (admin/admin123)

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
| Redis | localhost:6381 | 消息队列 & 缓存 |
| Flower | http://localhost:5555 | Celery 任务监控 (admin/admin123) |
| Celery Worker | - | 后台任务执行器 |
| Celery Beat | - | 周期任务调度器 |

## 文档

### 训练和使用指南

- **[训练指南](docs/TRAINING_GUIDE.md)**: 完整的训练教程，包括：
  - 各算法数据格式（SFT/GRPO/PPO/DPO/GSPO）
  - 模型下载方式
  - 训练运行命令
  - 监控与观测（W&B/Prometheus/Grafana）

- **[数据准备](docs/DATA_PREPARATION.md)**: 数据格式详细说明和转换工具

### Phase 文档

- **[Phase 3 进度报告](docs/PHASE3_PROGRESS.md)**: Phase 3 详细实现文档
- **[Phase 3 完成总结](docs/PHASE3_SUMMARY.md)**: Phase 3 功能总结和使用指南

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
│   │       ├── dataset.py     # 数据集 API (上传/搜索/去重/分析)
│   │       ├── run_mode.py    # 运行模式配置 API
│   │       ├── pipelines.py   # ⚡ Pipeline 工作流 API (Phase 3)
│   │       └── celery_tasks_api.py  # ⚡ Celery 任务管理 API (Phase 3)
│   └── core/
│       ├── memory_estimator.py    # 显存估算 (支持 Apple Silicon)
│       ├── compute_calculator.py  # 配置计算
│       ├── model_merger.py        # 模型合并
│       ├── checkpoint_selector.py # 检查点选择
│       ├── verl_adapter.py        # verl 适配器（安全命令执行）
│       ├── ray_runner.py          # Ray 任务提交（本地模式）
│       ├── ssh_runner.py          # SSH 远程执行（SSH 模式，安全加固）
│       ├── run_mode.py            # 运行模式管理（加密存储）
│       ├── crypto_utils.py        # 🔒 加密工具（Fernet 对称加密）
│       ├── command_utils.py       # 🔒 安全命令工具（防注入）
│       ├── database.py            # SQLModel 数据库（包含 Pipeline 模型）
│       ├── vector_store.py        # Milvus 向量库
│       ├── celery_config.py       # ⚡ Celery 配置 (Phase 3)
│       ├── celery_tasks.py        # ⚡ Celery 任务定义 (Phase 3)
│       └── migrate_phase3.py      # ⚡ Phase 3 数据库迁移
├── frontend/                   # Vue 3 前端
├── tests/                      # 测试文件 (175+ 测试)
├── models/                     # 模型存储目录 (自动检测)
├── datasets/                   # 训练数据目录 (自动检测)
├── environments/               # 📦 环境配置（Phase 0 新增）
│   ├── requirements-base.txt       # 基础依赖（21 packages）
│   ├── requirements-training.txt   # 训练节点依赖（29 packages, GPU）
│   ├── requirements-manager.txt    # 管理节点依赖（14 packages, CPU）
│   ├── version.json                # 版本锁定文件
│   ├── README.md                   # 环境说明文档
│   └── verl/                       # verl git submodule
├── scripts/                    # 📜 自动化脚本（Phase 0 新增）
│   ├── setup_local_env.sh          # 本地环境安装脚本
│   ├── setup_remote_env.sh         # 远程环境安装脚本
│   └── verify_env.py               # 环境验证脚本
├── docs/                       # 文档
│   ├── TRAINING_GUIDE.md      # 训练指南（数据格式/模型下载/运行/监控）
│   ├── DATA_PREPARATION.md    # 数据准备详细文档
│   ├── USAGE_GUIDE.md         # 平台使用指南（Phase 0 更新）
│   ├── PHASE3_PROGRESS.md     # ⚡ Phase 3 详细实现文档
│   └── PHASE3_SUMMARY.md      # ⚡ Phase 3 功能总结和使用指南
├── docker-compose.yml          # Docker 容器编排 (Milvus/Redis/Celery/MinIO)
├── Dockerfile.celery           # ⚡ Celery Docker 镜像 (Phase 3)
├── .env.example                # 环境变量模板
├── requirements.txt            # 兼容旧版（推荐使用 environments/）
└── TASKS.md                    # 开发任务清单
```

## GPU 服务器配置指南（SSH 远程模式）

如果你在 Mac 或没有 GPU 的机器上运行平台，需要配置远程 GPU 服务器来执行训练任务。

### 服务器要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA A100/H100/RTX4090 等，显存 >= 24GB |
| CUDA | 12.0+ |
| 内存 | >= 64GB（推荐 128GB+）|
| 存储 | >= 500GB（模型 + 数据集）|
| 网络 | 可访问 HuggingFace/ModelScope |

### Step 1: 快速安装（推荐）

**从本地推送安装:**

```bash
# 在本地执行（会自动推送代码并安装）
bash scripts/setup_remote_env.sh user@gpu-server
```

**或在远程服务器手动安装:**

```bash
# 1. SSH 登录到 GPU 服务器
ssh user@gpu-server

# 2. 克隆项目
git clone --recursive https://github.com/your-org/train_platform.git
cd train_platform

# 3. 运行自动安装脚本
bash scripts/setup_remote_env.sh --remote

# 4. 激活环境并验证
source venv/bin/activate
python scripts/verify_env.py
```

### Step 2: 手动安装（可选）

<details>
<summary>点击展开手动安装步骤</summary>

**服务器基础环境:**

```bash
# SSH 登录到 GPU 服务器
ssh user@gpu-server

# 安装 Miniconda（如果没有）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 验证 GPU
nvidia-smi
```

**安装训练环境:**

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装 PyTorch（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装固定版本依赖
pip install -r environments/requirements-training.txt

# 安装 verl（从 submodule）
cd environments/verl && pip install -e . && cd ../..
```

</details>

### Step 3: 下载模型

**方式 1: HuggingFace（国外服务器）**
```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载 Qwen2.5-7B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /data/models/qwen2.5-7b-instruct

# 下载 Llama-3-8B
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir /data/models/llama3-8b-instruct
```

**方式 2: ModelScope（国内服务器，推荐）**
```bash
# 安装 modelscope
pip install modelscope

# 下载 Qwen2.5-7B
modelscope download --model qwen/Qwen2.5-7B-Instruct \
    --local_dir /data/models/qwen2.5-7b-instruct

# 下载 Qwen2.5-3B（较小，测试用）
modelscope download --model qwen/Qwen2.5-3B-Instruct \
    --local_dir /data/models/qwen2.5-3b-instruct
```

### Step 4: 准备训练数据

```bash
# 下载示例数据集（GSM8K）
python -c "
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main')
ds['train'].to_parquet('/data/datasets/gsm8k_train.parquet')
print(f'Saved {len(ds[\"train\"])} samples')
"

# 或上传自己的数据
# scp local_data.parquet user@gpu-server:/data/datasets/
```

**数据格式要求（Parquet/JSONL）:**
```json
{"prompt": "问题内容", "response": "答案内容"}
{"prompt": "...", "response": "..."}
```

### Step 5: 验证环境

**自动验证（推荐）:**

```bash
# 激活环境
source venv/bin/activate

# 运行验证脚本
python scripts/verify_env.py
```

**手动验证:**

```bash
# 验证 verl 安装
python -c "import verl; print(f'verl version: {verl.__version__}')"

# 验证 GPU 可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 验证模型可加载
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/models/qwen2.5-3b-instruct')
print(f'Tokenizer vocab size: {tokenizer.vocab_size}')
"
```

### Step 6: 平台配置

在本地平台的「设置」页面配置：

| 配置项 | 值 |
|--------|-----|
| 运行模式 | SSH Remote |
| Host | `gpu-server` 或 IP 地址 |
| Port | `22` |
| Username | 你的用户名 |
| Password | SSH 密码（加密存储）|
| Working Directory | `~/train_platform` |
| Python 环境 | `venv` (virtualenv) |

点击「测试连接」验证连接，成功后点击「保存配置」。

**安全说明:**
- SSH 密码使用 Fernet 对称加密存储
- 配置文件位于 `~/.train_platform/run_mode.json`
- 所有命令执行经过路径验证和参数转义

### 目录结构参考

```
GPU 服务器
├── ~/train_platform/           # 平台代码（推荐位置）
│   ├── venv/                   # Python 虚拟环境
│   ├── environments/           # 环境配置和 verl
│   ├── scripts/                # 安装脚本
│   ├── datasets/               # 数据集存储（可选）
│   └── models/                 # 模型存储（可选）
├── /data/models/               # 模型存储（推荐）
│   ├── qwen2.5-3b-instruct/
│   ├── qwen2.5-7b-instruct/
│   └── llama3-8b-instruct/
└── /data/datasets/             # 数据集存储（推荐）
    ├── gsm8k_train.parquet
    └── your_data.jsonl
```

**注意:**
- 平台会在工作目录下自动创建 `jobs/` 和 `logs/` 子目录
- 模型和数据集可以放在任意位置，训练时指定绝对路径即可

### 常见问题

**Q: 连接超时？**
```bash
# 检查 SSH 服务
sudo systemctl status sshd

# 检查防火墙
sudo ufw status
sudo ufw allow 22
```

**Q: GPU 内存不足？**
- 使用更小的模型（如 3B）
- 减小 batch_size
- 启用 LoRA 训练
- 启用梯度检查点

**Q: 模型下载失败？**
```bash
# 设置 HuggingFace 镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用 ModelScope
```

**Q: CUDA 版本不匹配？**
```bash
# 检查 CUDA 版本
nvcc --version
nvidia-smi

# 重新安装对应版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

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
