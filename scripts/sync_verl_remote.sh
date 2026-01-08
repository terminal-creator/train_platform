#!/bin/bash
# Quick sync verl to remote server
# 快速同步 verl 代码到远程服务器
# 用法: bash scripts/sync_verl_remote.sh <user@host> [remote_dir]

set -e

if [ -z "$1" ]; then
    echo "用法: $0 <user@host> [remote_dir]"
    echo "示例: $0 user@gpu-server ~/train_platform"
    exit 1
fi

REMOTE_HOST=$1
REMOTE_DIR=${2:-$HOME/train_platform}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[INFO]${NC} 同步 verl 到远程服务器..."
echo -e "${GREEN}[INFO]${NC} 目标: $REMOTE_HOST:$REMOTE_DIR"
echo ""

# 检查 verl 目录
if [ ! -d "environments/verl" ]; then
    echo -e "${YELLOW}[ERROR]${NC} environments/verl 目录不存在"
    exit 1
fi

# 打包 verl
echo -e "${GREEN}[INFO]${NC} 打包 verl..."
tar -czf /tmp/verl.tar.gz -C environments verl

# 推送
echo -e "${GREEN}[INFO]${NC} 推送到远程..."
scp /tmp/verl.tar.gz "$REMOTE_HOST:$REMOTE_DIR/environments/"

# 远程解压并重装
echo -e "${GREEN}[INFO]${NC} 远程解压并重装..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR/environments && tar -xzf verl.tar.gz && rm verl.tar.gz && cd .. && source venv/bin/activate && pip install -e environments/verl --no-deps"

# 清理
rm /tmp/verl.tar.gz

echo ""
echo -e "${GREEN}[INFO]${NC} ✓ verl 同步完成！"
echo ""
echo "验证："
echo "  ssh $REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  source venv/bin/activate"
echo "  python -c 'import verl; print(verl.__file__)'"
echo ""
