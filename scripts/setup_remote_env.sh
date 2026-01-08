#!/bin/bash
# Training Platform - Remote Environment Setup Script
# 远程环境安装脚本
# 用途：在远程训练服务器上安装环境（通过 SSH）
# 使用方法：
#   1. 本地执行（推送并安装）：
#      bash scripts/setup_remote_env.sh <user@host> [working_dir]
#   2. 远程执行（仅安装）：
#      bash setup_remote_env.sh --remote

set -e

# ==================== 配置 ====================
REMOTE_MODE=false
WORKING_DIR="$HOME/train_platform"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ==================== 函数定义 ====================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==================== 本地模式：推送文件并执行安装 ====================

push_and_install() {
    local REMOTE_HOST=$1
    local REMOTE_DIR=${2:-$HOME/train_platform}

    log_info "=========================================="
    log_info "  远程环境安装 - 本地模式"
    log_info "=========================================="
    log_info "目标主机: $REMOTE_HOST"
    log_info "远程目录: $REMOTE_DIR"
    echo ""

    # 检查 SSH 连接
    log_info "测试 SSH 连接..."
    if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" "echo 'SSH连接成功'"; then
        log_error "SSH 连接失败，请检查："
        log_error "  1. 主机地址是否正确"
        log_error "  2. SSH 密钥是否已配置"
        log_error "  3. 目标主机是否可达"
        exit 1
    fi
    log_info "✓ SSH 连接正常"
    echo ""

    # 创建远程目录
    log_info "创建远程目录..."
    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

    # 推送必要文件
    log_info "推送环境配置文件..."
    rsync -avz --progress \
        --include='environments/' \
        --include='environments/requirements-*.txt' \
        --include='environments/version.json' \
        --include='environments/README.md' \
        --include='scripts/' \
        --include='scripts/setup_remote_env.sh' \
        --include='scripts/verify_env.py' \
        --exclude='*' \
        ./ "$REMOTE_HOST:$REMOTE_DIR/"

    log_info "✓ 文件推送完成"
    echo ""

    # 推送 verl submodule（作为压缩包）
    log_info "打包 verl..."
    if [ -d "environments/verl" ] && [ "$(ls -A environments/verl)" ]; then
        tar -czf /tmp/verl.tar.gz -C environments verl
        log_info "推送 verl..."
        scp /tmp/verl.tar.gz "$REMOTE_HOST:$REMOTE_DIR/environments/"
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR/environments && tar -xzf verl.tar.gz && rm verl.tar.gz"
        rm /tmp/verl.tar.gz
        log_info "✓ verl 推送完成"
    else
        log_warn "verl submodule 未初始化或为空，跳过"
    fi
    echo ""

    # 在远程机器上执行安装
    log_info "在远程机器上执行安装..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && bash scripts/setup_remote_env.sh --remote"

    log_info ""
    log_info "=========================================="
    log_info "  远程环境安装完成！"
    log_info "=========================================="
    log_info ""
    log_info "可以通过 SSH 连接到远程机器并激活环境："
    log_info "  ssh $REMOTE_HOST"
    log_info "  cd $REMOTE_DIR"
    log_info "  source venv/bin/activate"
    echo ""
}

# ==================== 远程模式：在远程机器上安装 ====================

install_on_remote() {
    log_info "=========================================="
    log_info "  远程环境安装 - 远程模式"
    log_info "=========================================="
    log_info "工作目录: $WORKING_DIR"
    echo ""

    # 检查必要文件
    if [ ! -f "environments/requirements-training.txt" ]; then
        log_error "未找到 requirements 文件，请确保文件已从本地推送"
        exit 1
    fi

    # 检查 Python
    log_info "检查 Python 版本..."
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3，请先安装 Python 3.9+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_info "Python 版本: $PYTHON_VERSION"

    # 检查 CUDA（训练节点必须）
    log_info "检查 CUDA..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        log_info "CUDA 版本: $CUDA_VERSION"

        # 检查 GPU
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
            log_info "检测到 $GPU_COUNT 个 GPU"
        else
            log_warn "未找到 nvidia-smi，无法检测 GPU"
        fi
    else
        log_warn "未找到 CUDA，如果这是训练节点，请先安装 CUDA"
    fi
    echo ""

    # 创建虚拟环境
    log_info "创建虚拟环境..."
    if [ -d "venv" ]; then
        log_warn "虚拟环境已存在，将删除重建..."
        rm -rf venv
    fi
    python3 -m venv venv
    source venv/bin/activate

    # 升级 pip
    log_info "升级 pip..."
    pip install --upgrade pip setuptools wheel

    # 安装依赖（训练节点）
    log_info "安装训练节点依赖..."
    log_info "这可能需要 10-30 分钟，请耐心等待..."
    pip install -r environments/requirements-training.txt

    # 安装 verl
    if [ -d "environments/verl" ]; then
        log_info "安装 verl..."
        pip install -e environments/verl
    else
        log_warn "verl 目录不存在，跳过安装"
    fi

    # 验证安装
    log_info "验证安装..."
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

    log_info ""
    log_info "✓ 远程环境安装完成！"
    log_info ""
    log_info "激活环境："
    log_info "  source venv/bin/activate"
    log_info ""
    log_info "运行验证："
    log_info "  python scripts/verify_env.py --mode training"
    echo ""
}

# ==================== 主流程 ====================

main() {
    # 解析参数
    if [ "$1" == "--remote" ]; then
        REMOTE_MODE=true
        WORKING_DIR=$(pwd)
        install_on_remote
    else
        if [ -z "$1" ]; then
            log_error "用法: $0 <user@host> [working_dir]"
            log_error "  或: $0 --remote  (在远程机器上执行)"
            exit 1
        fi
        REMOTE_HOST=$1
        REMOTE_DIR=${2:-$HOME/train_platform}
        push_and_install "$REMOTE_HOST" "$REMOTE_DIR"
    fi
}

main "$@"
