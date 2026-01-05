# Training Platform 开发环境 Makefile

.PHONY: help install dev docker-up docker-down backend frontend stop status clean logs

# 默认目标
help:
	@echo "Training Platform 开发环境"
	@echo ""
	@echo "使用方法:"
	@echo "  make install      - 安装所有依赖"
	@echo "  make dev          - 启动完整开发环境"
	@echo "  make docker-up    - 只启动 Docker 容器"
	@echo "  make docker-down  - 停止 Docker 容器"
	@echo "  make backend      - 只启动后端"
	@echo "  make frontend     - 只启动前端"
	@echo "  make rag          - 只启动 RAG API"
	@echo "  make stop         - 停止所有服务"
	@echo "  make status       - 查看服务状态"
	@echo "  make logs         - 查看 Docker 日志"
	@echo "  make clean        - 清理临时文件"
	@echo "  make init-data    - 初始化知识库数据"

# 安装依赖
install:
	@echo "安装后端依赖..."
	pip install -r requirements.txt
	pip install -r milvus_rag/requirements.txt
	@echo "安装前端依赖..."
	cd frontend && npm install
	@echo "依赖安装完成"

# 启动完整开发环境
dev: docker-up
	@sleep 3
	@echo "启动后端服务..."
	@make backend &
	@sleep 2
	@make rag &
	@sleep 1
	@echo "启动前端服务..."
	@make frontend

# Docker 相关
docker-up:
	@echo "启动 Docker 容器..."
	docker-compose up -d
	@echo "等待服务就绪..."
	@sleep 5
	@echo "Docker 容器已启动"
	@echo "  - Milvus:     localhost:19530"
	@echo "  - Milvus UI:  http://localhost:3001"
	@echo "  - MinIO:      http://localhost:9001"
	@echo "  - Redis:      localhost:6379"

docker-down:
	@echo "停止 Docker 容器..."
	docker-compose down

docker-restart:
	docker-compose restart

# 后端服务
backend:
	@echo "启动 Training Platform API..."
	uvicorn training_platform.api.main:app --reload --host 0.0.0.0 --port 8000

# RAG API
rag:
	@echo "启动 Milvus RAG API..."
	cd milvus_rag && uvicorn api_server:app --reload --host 0.0.0.0 --port 8001

# 前端服务
frontend:
	@echo "启动前端开发服务器..."
	cd frontend && npm run dev

# 停止所有服务
stop:
	@echo "停止所有服务..."
	-pkill -f "uvicorn.*training_platform" 2>/dev/null || true
	-pkill -f "uvicorn.*api_server" 2>/dev/null || true
	-pkill -f "vite" 2>/dev/null || true
	docker-compose down
	@echo "所有服务已停止"

# 查看状态
status:
	@echo "========================================="
	@echo "        服务状态"
	@echo "========================================="
	@echo ""
	@echo "Docker 容器:"
	@docker-compose ps 2>/dev/null || echo "  未运行"
	@echo ""
	@echo "后端进程:"
	@pgrep -f "uvicorn.*training_platform" > /dev/null && echo "  Training API: 运行中" || echo "  Training API: 未运行"
	@pgrep -f "uvicorn.*api_server" > /dev/null && echo "  RAG API:      运行中" || echo "  RAG API:      未运行"
	@echo ""
	@echo "前端进程:"
	@pgrep -f "vite" > /dev/null && echo "  Vite:         运行中" || echo "  Vite:         未运行"

# 查看日志
logs:
	docker-compose logs -f

logs-milvus:
	docker logs -f milvus-standalone

logs-redis:
	docker logs -f train-redis

# 清理
clean:
	@echo "清理临时文件..."
	-rm -f .backend.pid .rag.pid .frontend.pid
	-rm -rf __pycache__ */__pycache__ */*/__pycache__
	-rm -rf .pytest_cache */.pytest_cache
	-find . -name "*.pyc" -delete
	@echo "清理完成"

# 初始化知识库数据
init-data:
	@echo "初始化知识库数据..."
	cd milvus_rag && python milvus_insert.py --drop
	@echo "数据初始化完成"

# 测试
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=training_platform --cov-report=html

# 构建前端
build-frontend:
	cd frontend && npm run build

# 代码格式化
format:
	black training_platform/ milvus_rag/
	isort training_platform/ milvus_rag/
