#!/bin/bash

# Celery Workers 启动脚本
#
# 用途：在本地或服务器上启动多个独立的 worker pools
# 架构：long-running (training) + short-lived (other) + beat
#
# 使用方式：
#   ./scripts/start_workers.sh          # 启动所有 workers
#   ./scripts/start_workers.sh training # 只启动 training worker
#   ./scripts/start_workers.sh short    # 只启动 short worker
#   ./scripts/start_workers.sh beat     # 只启动 beat scheduler

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Redis URL (默认)
REDIS_URL=${REDIS_URL:-"redis://localhost:6381/0"}

echo -e "${GREEN}Training Platform Celery Workers${NC}"
echo "=================================="
echo ""

# 检查 Redis 是否运行
echo -e "${YELLOW}Checking Redis connection...${NC}"
if ! redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1; then
    echo -e "${RED}Error: Redis is not running at $REDIS_URL${NC}"
    echo "Please start Redis first:"
    echo "  docker run -d -p 6381:6379 redis:7-alpine"
    exit 1
fi
echo -e "${GREEN}✓ Redis is running${NC}"
echo ""

# 函数：启动 training worker
start_training_worker() {
    echo -e "${GREEN}Starting Long-running Worker (training queue)...${NC}"
    echo "  Queue: training"
    echo "  Concurrency: 1 (avoid GPU contention)"
    echo "  Max tasks per child: 1 (clean restart after each task)"
    echo ""

    celery -A training_platform.core.celery_config worker \
        -Q training \
        -c 1 \
        -l info \
        --max-tasks-per-child 1 \
        -n training@%h &

    TRAINING_PID=$!
    echo -e "${GREEN}✓ Training worker started (PID: $TRAINING_PID)${NC}"
}

# 函数：启动 short-lived worker
start_short_worker() {
    echo -e "${GREEN}Starting Short-lived Worker (fast queues)...${NC}"
    echo "  Queues: default, evaluation, preprocessing, maintenance"
    echo "  Concurrency: 4"
    echo ""

    celery -A training_platform.core.celery_config worker \
        -Q default,evaluation,preprocessing,maintenance \
        -c 4 \
        -l info \
        -n short@%h &

    SHORT_PID=$!
    echo -e "${GREEN}✓ Short worker started (PID: $SHORT_PID)${NC}"
}

# 函数：启动 beat scheduler
start_beat() {
    echo -e "${GREEN}Starting Beat Scheduler (periodic tasks)...${NC}"
    echo "  Tasks:"
    echo "    - update_job_metrics: every 1 minute"
    echo "    - scan_failed_jobs: every 5 minutes"
    echo "    - cleanup_old_checkpoints: every 1 hour"
    echo ""

    celery -A training_platform.core.celery_config beat \
        -l info &

    BEAT_PID=$!
    echo -e "${GREEN}✓ Beat scheduler started (PID: $BEAT_PID)${NC}"
}

# 函数：启动 Flower 监控
start_flower() {
    echo -e "${GREEN}Starting Flower (monitoring dashboard)...${NC}"
    echo "  URL: http://localhost:5555"
    echo ""

    celery -A training_platform.core.celery_config flower \
        --port=5555 &

    FLOWER_PID=$!
    echo -e "${GREEN}✓ Flower started (PID: $FLOWER_PID)${NC}"
}

# 根据参数启动相应的 worker
case "${1:-all}" in
    training)
        start_training_worker
        ;;
    short)
        start_short_worker
        ;;
    beat)
        start_beat
        ;;
    flower)
        start_flower
        ;;
    all)
        start_training_worker
        sleep 2
        start_short_worker
        sleep 2
        start_beat
        sleep 2
        start_flower
        ;;
    *)
        echo -e "${RED}Unknown argument: $1${NC}"
        echo "Usage: $0 [training|short|beat|flower|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=================================="
echo "All workers started successfully!"
echo "==================================${NC}"
echo ""
echo "Monitor workers:"
echo "  - Flower dashboard: http://localhost:5555"
echo "  - View logs: tail -f celery_*.log"
echo ""
echo "Stop workers:"
echo "  - Press Ctrl+C to stop all"
echo "  - Or: pkill -f 'celery.*worker'"
echo ""

# 等待所有后台进程
wait
