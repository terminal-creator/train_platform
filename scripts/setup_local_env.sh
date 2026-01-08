#!/bin/bash
# Training Platform - Local Environment Setup Script
# 本地环境安装脚本
# 用途：在开发机或训练机上安装固定版本的环境
# 使用方法：bash scripts/setup_local_env.sh [manager|training]

set -e  # 遇到错误立即退出

# ==================== 配置 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_ROOT/environments"
VENV_DIR="$PROJECT_ROOT/venv"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

check_python_version() {
    log_info "检查 Python 版本..."

    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3，请先安装 Python 3.9+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    log_info "当前 Python 版本: $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        log_error "Python 版本过低，需要 3.9+，当前: $PYTHON_VERSION"
        exit 1
    fi

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
        log_warn "Python 版本较新 ($PYTHON_VERSION)，推荐使用 3.9-3.11"
    fi

    log_info "✓ Python 版本检查通过"
}

create_virtualenv() {
    log_info "创建虚拟环境..."

    if [ -d "$VENV_DIR" ]; then
        log_warn "虚拟环境已存在: $VENV_DIR"
        read -p "是否删除并重新创建？ (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "删除旧环境..."
            rm -rf "$VENV_DIR"
        else
            log_info "跳过虚拟环境创建"
            return 0
        fi
    fi

    python3 -m venv "$VENV_DIR"
    log_info "✓ 虚拟环境创建成功: $VENV_DIR"
}

activate_virtualenv() {
    log_info "激活虚拟环境..."

    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        log_error "虚拟环境激活脚本不存在"
        exit 1
    fi

    source "$VENV_DIR/bin/activate"
    log_info "✓ 虚拟环境已激活"

    # 升级 pip
    log_info "升级 pip..."
    pip install --upgrade pip setuptools wheel
}

install_dependencies() {
    local NODE_TYPE=$1
    log_info "安装依赖包 (节点类型: $NODE_TYPE)..."

    if [ "$NODE_TYPE" == "manager" ]; then
        REQUIREMENTS_FILE="$ENV_DIR/requirements-manager.txt"
    elif [ "$NODE_TYPE" == "training" ]; then
        REQUIREMENTS_FILE="$ENV_DIR/requirements-training.txt"
    else
        log_error "未知的节点类型: $NODE_TYPE"
        exit 1
    fi

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        log_error "依赖文件不存在: $REQUIREMENTS_FILE"
        exit 1
    fi

    log_info "使用依赖文件: $REQUIREMENTS_FILE"
    log_info "这可能需要几分钟，请耐心等待..."

    # 安装依赖
    pip install -r "$REQUIREMENTS_FILE"

    log_info "✓ 依赖包安装完成"
}

install_verl() {
    log_info "安装 verl..."

    if [ ! -d "$ENV_DIR/verl" ]; then
        log_error "verl submodule 不存在，请先执行: git submodule update --init --recursive"
        exit 1
    fi

    # 检查 verl 目录是否为空（submodule 未初始化）
    if [ -z "$(ls -A $ENV_DIR/verl)" ]; then
        log_warn "verl submodule 未初始化，正在初始化..."
        cd "$PROJECT_ROOT"
        git submodule update --init --recursive
        cd "$SCRIPT_DIR"
    fi

    log_info "从源码安装 verl..."
    pip install -e "$ENV_DIR/verl"

    log_info "✓ verl 安装完成"
}

verify_installation() {
    log_info "验证安装..."

    # 验证关键包
    python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

    # 验证 verl
    if python3 -c "import verl" 2>/dev/null; then
        log_info "✓ verl 导入成功"
    else
        log_warn "verl 导入失败，这可能是正常的（取决于环境）"
    fi

    log_info "✓ 安装验证完成"
}

print_next_steps() {
    echo ""
    echo "=========================================="
    echo "  环境安装完成！"
    echo "=========================================="
    echo ""
    echo "下一步操作："
    echo "  1. 激活虚拟环境:"
    echo "     source venv/bin/activate"
    echo ""
    echo "  2. 验证环境:"
    echo "     python scripts/verify_env.py --mode $NODE_TYPE"
    echo ""
    echo "  3. 启动平台:"
    echo "     cd training_platform"
    echo "     uvicorn api.main:app --reload"
    echo ""
    echo "=========================================="
}

# ==================== 主流程 ====================

main() {
    log_info "=========================================="
    log_info "  训练平台环境安装"
    log_info "=========================================="
    echo ""

    # 解析参数
    NODE_TYPE=${1:-manager}

    if [ "$NODE_TYPE" != "manager" ] && [ "$NODE_TYPE" != "training" ]; then
        log_error "用法: $0 [manager|training]"
        log_error "  manager  - 管理节点（不需要 GPU）"
        log_error "  training - 训练节点（需要 GPU 和 CUDA）"
        exit 1
    fi

    log_info "节点类型: $NODE_TYPE"
    log_info "项目根目录: $PROJECT_ROOT"
    echo ""

    # 执行安装步骤
    check_python_version
    create_virtualenv
    activate_virtualenv
    install_dependencies "$NODE_TYPE"
    install_verl
    verify_installation

    # 打印后续步骤
    print_next_steps
}

# 运行主流程
main "$@"
