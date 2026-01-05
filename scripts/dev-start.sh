#!/bin/bash

# 开发环境启动脚本
# 用法: ./scripts/dev-start.sh [选项]
#   --docker-only    只启动 Docker 容器
#   --backend-only   只启动后端
#   --frontend-only  只启动前端
#   --all            启动所有服务（默认）
#   --stop           停止所有服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 未安装，请先安装"
        exit 1
    fi
}

# 启动 Docker 容器
start_docker() {
    log_info "启动 Docker 容器..."

    check_command docker
    check_command docker-compose

    docker-compose up -d

    log_info "等待 Milvus 启动..."

    # 等待 Milvus 健康
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
            log_success "Milvus 已就绪"
            break
        fi
        attempt=$((attempt + 1))
        log_info "等待 Milvus... ($attempt/$max_attempts)"
        sleep 2
    done

    if [ $attempt -eq $max_attempts ]; then
        log_warn "Milvus 启动超时，请检查日志: docker logs milvus-standalone"
    fi

    log_success "Docker 容器已启动"
    echo ""
    log_info "服务地址:"
    echo "  - Milvus:     localhost:19530"
    echo "  - Milvus UI:  http://localhost:3001 (Attu)"
    echo "  - MinIO:      http://localhost:9001 (admin: minioadmin/minioadmin)"
    echo "  - Redis:      localhost:6379"
}

# 停止 Docker 容器
stop_docker() {
    log_info "停止 Docker 容器..."
    docker-compose down
    log_success "Docker 容器已停止"
}

# 启动后端
start_backend() {
    log_info "启动后端服务..."

    check_command python3

    # 检查虚拟环境
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # 安装依赖
    if [ -f "requirements.txt" ]; then
        log_info "检查后端依赖..."
        pip install -q -r requirements.txt
    fi

    # 安装 milvus_rag 依赖
    if [ -f "milvus_rag/requirements.txt" ]; then
        log_info "检查 Milvus RAG 依赖..."
        pip install -q -r milvus_rag/requirements.txt
    fi

    # 启动主后端 API
    log_info "启动 Training Platform API (端口 8000)..."
    cd "$PROJECT_ROOT"
    uvicorn training_platform.api.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo $BACKEND_PID > .backend.pid

    # 启动 Milvus RAG API (可选)
    if [ -f "milvus_rag/api_server.py" ]; then
        log_info "启动 Milvus RAG API (端口 8001)..."
        cd "$PROJECT_ROOT/milvus_rag"
        uvicorn api_server:app --reload --host 0.0.0.0 --port 8001 &
        RAG_PID=$!
        echo $RAG_PID > "$PROJECT_ROOT/.rag.pid"
    fi

    cd "$PROJECT_ROOT"

    log_success "后端服务已启动"
    echo ""
    log_info "后端地址:"
    echo "  - Training API:  http://localhost:8000"
    echo "  - API 文档:      http://localhost:8000/docs"
    echo "  - RAG API:       http://localhost:8001"
    echo "  - RAG API 文档:  http://localhost:8001/docs"
}

# 停止后端
stop_backend() {
    log_info "停止后端服务..."

    if [ -f ".backend.pid" ]; then
        kill $(cat .backend.pid) 2>/dev/null || true
        rm -f .backend.pid
    fi

    if [ -f ".rag.pid" ]; then
        kill $(cat .rag.pid) 2>/dev/null || true
        rm -f .rag.pid
    fi

    # 杀死所有 uvicorn 进程
    pkill -f "uvicorn.*training_platform" 2>/dev/null || true
    pkill -f "uvicorn.*api_server" 2>/dev/null || true

    log_success "后端服务已停止"
}

# 启动前端
start_frontend() {
    log_info "启动前端服务..."

    check_command node
    check_command npm

    cd "$PROJECT_ROOT/frontend"

    # 安装依赖
    if [ ! -d "node_modules" ]; then
        log_info "安装前端依赖..."
        npm install
    fi

    # 启动开发服务器
    log_info "启动 Vite 开发服务器..."
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$PROJECT_ROOT/.frontend.pid"

    cd "$PROJECT_ROOT"

    log_success "前端服务已启动"
    echo ""
    log_info "前端地址:"
    echo "  - 开发服务器: http://localhost:5173"
}

# 停止前端
stop_frontend() {
    log_info "停止前端服务..."

    if [ -f ".frontend.pid" ]; then
        kill $(cat .frontend.pid) 2>/dev/null || true
        rm -f .frontend.pid
    fi

    pkill -f "vite" 2>/dev/null || true

    log_success "前端服务已停止"
}

# 显示状态
show_status() {
    echo ""
    echo "========================================="
    echo "        开发环境状态"
    echo "========================================="
    echo ""

    # Docker 状态
    echo "Docker 容器:"
    docker-compose ps 2>/dev/null || echo "  未运行"
    echo ""

    # 后端状态
    echo "后端进程:"
    if pgrep -f "uvicorn.*training_platform" > /dev/null; then
        echo "  Training API: 运行中 (端口 8000)"
    else
        echo "  Training API: 未运行"
    fi

    if pgrep -f "uvicorn.*api_server" > /dev/null; then
        echo "  RAG API:      运行中 (端口 8001)"
    else
        echo "  RAG API:      未运行"
    fi
    echo ""

    # 前端状态
    echo "前端进程:"
    if pgrep -f "vite" > /dev/null; then
        echo "  Vite:         运行中 (端口 5173)"
    else
        echo "  Vite:         未运行"
    fi
    echo ""
}

# 显示帮助
show_help() {
    echo "开发环境启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --docker-only    只启动 Docker 容器 (Milvus, Redis, etc.)"
    echo "  --backend-only   只启动后端服务"
    echo "  --frontend-only  只启动前端服务"
    echo "  --all            启动所有服务（默认）"
    echo "  --stop           停止所有服务"
    echo "  --status         显示服务状态"
    echo "  --help           显示此帮助"
    echo ""
    echo "示例:"
    echo "  $0                    # 启动所有服务"
    echo "  $0 --docker-only      # 只启动 Docker"
    echo "  $0 --stop             # 停止所有服务"
}

# 主函数
main() {
    case "${1:-all}" in
        --docker-only)
            start_docker
            ;;
        --backend-only)
            start_backend
            ;;
        --frontend-only)
            start_frontend
            ;;
        --all|all)
            start_docker
            sleep 2
            start_backend
            sleep 1
            start_frontend
            echo ""
            show_status
            ;;
        --stop|stop)
            stop_frontend
            stop_backend
            stop_docker
            ;;
        --status|status)
            show_status
            ;;
        --help|help|-h)
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
