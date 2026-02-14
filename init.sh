#!/bin/bash
# LLM Training Platform - Initialization Script
# 初始化开发环境

set -e

PROJECT_ROOT="/Users/weixiaochen/Desktop/Tutor/S4/train_platform"

echo "=== LLM Training Platform - 开发环境初始化 ==="
echo ""

# 1. 检查工作目录
echo "[1/7] 检查工作目录..."
cd "$PROJECT_ROOT"
pwd

# 2. 检查git状态
echo "[2/7] 检查Git状态..."
git status --short | head -20
echo ""

# 3. 检查verl submodule
echo "[3/7] 检查verl submodule..."
if [ -d "environments/verl" ]; then
    cd environments/verl
    VERL_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "verl commit: $VERL_COMMIT"
    cd "$PROJECT_ROOT"
else
    echo "WARNING: verl submodule not found!"
    echo "Run: git submodule update --init --recursive"
fi

# 4. 检查Python环境
echo "[4/7] 检查Python环境..."
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || true
    echo "Python: $(python --version 2>&1)"
    echo "Pip packages: $(pip list 2>/dev/null | wc -l) installed"
else
    echo "WARNING: .venv not found. Run: python -m venv .venv"
fi

# 5. 检查后端服务
echo "[5/7] 检查后端服务..."
if lsof -i :8001 > /dev/null 2>&1; then
    echo "Backend running on port 8001 ✓"
else
    echo "Backend not running. Start with:"
    echo "  uvicorn training_platform.api.main:app --reload --port 8001"
fi

# 6. 检查前端服务
echo "[6/7] 检查前端服务..."
if lsof -i :3000 > /dev/null 2>&1; then
    echo "Frontend running on port 3000 ✓"
else
    echo "Frontend not running. Start with:"
    echo "  cd frontend && npm run dev"
fi

# 7. 显示项目状态
echo "[7/7] 项目状态..."
echo ""
echo "=== 关键文件 ==="
echo "设计文档:    DESIGN.md"
echo "任务清单:    TASKS.md"
echo "进度文件:    claude-progress.txt"
echo "功能列表:    feature_list.json"
echo ""
echo "=== 代码目录 ==="
echo "后端:        training_platform/"
echo "前端:        frontend/"
echo "verl:        environments/verl/"
echo ""
echo "=== 下一步 ==="
echo "1. 读取 claude-progress.txt 了解当前进度"
echo "2. 读取 feature_list.json 选择要实现的功能"
echo "3. 只实现一个功能，测试通过后更新进度"
echo "4. 提交git并更新progress文件"
echo ""
echo "初始化完成！"
