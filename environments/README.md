# 训练平台环境管理

本目录包含训练平台所有节点的依赖定义和环境管理脚本。

## 📦 环境文件说明

### 1. requirements-base.txt
**基础依赖** - 所有节点都需要安装

包含：
- Web 框架（FastAPI、Uvicorn）
- 数据库（SQLModel、SQLAlchemy）
- SSH 连接（Paramiko）
- 基础工具（NumPy、pytest 等）

### 2. requirements-training.txt
**训练节点依赖** - 需要 GPU 的训练机器

包含：
- base 的所有依赖
- PyTorch + CUDA 12.1
- Transformers、Flash-Attention
- Ray、DeepSpeed
- WandB、TensorBoard

**系统要求**：
- CUDA 12.1+
- NVIDIA GPU
- 64GB+ RAM (推荐)

### 3. requirements-manager.txt
**管理节点依赖** - 运行平台服务的机器

包含：
- base 的所有依赖
- PyTorch CPU 版本
- Transformers（仅用于配置验证）
- Milvus、OpenAI SDK

**系统要求**：
- 无需 GPU
- 16GB+ RAM (推荐)

## 🚀 快速开始

### 安装管理节点环境（本地开发）

```bash
cd /path/to/train_platform
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装管理节点依赖
pip install -r environments/requirements-manager.txt
```

### 安装训练节点环境（GPU 机器）

```bash
cd /path/to/train_platform
python -m venv venv
source venv/bin/activate

# 安装训练节点依赖
pip install -r environments/requirements-training.txt
```

## 🔍 环境验证

安装完成后，使用验证脚本检查环境：

```bash
python scripts/verify_env.py --mode manager  # 管理节点
python scripts/verify_env.py --mode training  # 训练节点
```

## 📝 版本信息

当前版本：**1.0.0**
发布日期：2026-01-08
Python 版本：>=3.9,<3.12
CUDA 版本：12.1

详见 `version.json`

## 🔧 自定义安装

如果需要特殊配置，可以在安装后追加依赖：

```bash
# 例如：安装特定版本的 vLLM
pip install vllm==0.2.7

# 或者：启用 Celery 任务队列
pip install celery==5.3.6 redis==5.0.1
```

## ⚠️ 注意事项

1. **不要修改固定版本**：所有版本都经过兼容性测试，修改可能导致未知问题
2. **CUDA 版本匹配**：确保系统 CUDA 版本与 PyTorch 版本兼容
3. **Flash-Attention**：需要编译，安装时间较长（5-10分钟）
4. **verl 安装**：verl 作为 git submodule 管理，不在 requirements 中

## 🐛 常见问题

### Q: Flash-Attention 安装失败？
A: 确保安装了 CUDA 开发工具：
```bash
# Ubuntu/Debian
sudo apt-get install cuda-toolkit-12-1

# 或者跳过 Flash-Attention
pip install -r requirements-training.txt --no-deps
pip install flash-attn==2.5.0 --no-build-isolation
```

### Q: PyTorch 版本不匹配？
A: 检查 CUDA 版本：
```bash
nvcc --version  # 查看 CUDA 版本
```
根据 CUDA 版本修改 `--extra-index-url`：
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`

### Q: 管理节点也想用 GPU？
A: 安装 training 版本的依赖即可

## 📚 更多信息

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [Ray 文档](https://docs.ray.io/)
- [verl GitHub](https://github.com/volcengine/verl)
