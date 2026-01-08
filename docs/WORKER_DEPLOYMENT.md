# Worker Pools 部署指南

本文档说明如何部署独立的 Celery worker pools，避免长训练任务阻塞短任务。

---

## 架构设计

### 问题背景

**问题 1: 长任务压制短任务**
```
单个 Worker (concurrency=1):
  ├─ training task (占用 3 小时) ← 阻塞
  └─ update_job_metrics (每分钟) ← 被阻塞！
```

**问题 2: GPU 资源竞争**
```
单个 Worker (concurrency=4):
  ├─ training task 1 (使用 GPU 0-3)
  ├─ training task 2 (尝试使用 GPU 0-3) ← 冲突！
  └─ ...
```

### 解决方案

**独立 Worker Pools 架构：**

```
┌─────────────────────────────────────────────────────────────┐
│                        Redis Queue                          │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ training │   │evaluation│   │preprocess│   │maintenance│
│  queue   │   │  queue   │   │  queue   │   │   queue  │
└─────┬────┘   └─────┬────┘   └─────┬────┘   └─────┬────┘
      │              │              │              │
      ▼              └──────┬───────┴──────┬───────┘
┌──────────┐               ▼              │
│ Worker 1 │         ┌──────────┐         │
│ (training)│        │ Worker 2 │         │
│ c=1      │         │ (short)  │         │
└──────────┘         │ c=4      │         │
                     └──────────┘         │
                                          ▼
                                    ┌──────────┐
                                    │  Beat    │
                                    │Scheduler │
                                    └──────────┘
```

**关键特性：**
- ✅ **队列隔离**：training 独占一个 worker
- ✅ **单并发训练**：避免 GPU 竞争
- ✅ **高并发短任务**：evaluation/preprocessing/maintenance 共享 worker，并发度 4
- ✅ **独立 Beat**：周期任务不受阻塞

---

## 部署方式

### 方式 1: Docker Compose（推荐）

**适用场景：**
- 开发环境
- 测试环境
- 单机部署

**启动命令：**
```bash
# 启动所有服务
docker-compose -f docker-compose.celery.yml up -d

# 查看日志
docker-compose -f docker-compose.celery.yml logs -f

# 停止服务
docker-compose -f docker-compose.celery.yml down
```

**架构：**
```yaml
services:
  celery_worker_training:
    command: celery -A training_platform.core.celery_config worker -Q training -c 1 --max-tasks-per-child 1

  celery_worker_short:
    command: celery -A training_platform.core.celery_config worker -Q default,evaluation,preprocessing,maintenance -c 4

  celery_beat:
    command: celery -A training_platform.core.celery_config beat

  flower:
    command: celery -A training_platform.core.celery_config flower --port=5555
```

**监控：**
- Flower 面板：http://localhost:5555

---

### 方式 2: 本地脚本（开发）

**适用场景：**
- 本地开发
- 调试

**启动命令：**
```bash
# 启动所有 workers
./scripts/start_workers.sh

# 只启动 training worker
./scripts/start_workers.sh training

# 只启动 short worker
./scripts/start_workers.sh short

# 只启动 beat scheduler
./scripts/start_workers.sh beat
```

**停止：**
```bash
# Ctrl+C 或
pkill -f 'celery.*worker'
```

---

### 方式 3: Systemd（生产环境）

**适用场景：**
- 生产服务器
- 需要开机自启
- 需要自动重启

**部署步骤：**

**1. 安装 systemd service 文件**
```bash
# 复制 service 文件
sudo cp scripts/systemd/*.service /etc/systemd/system/

# 重新加载 systemd
sudo systemctl daemon-reload
```

**2. 创建必要的目录**
```bash
# 创建 PID 和日志目录
sudo mkdir -p /var/run/celery
sudo mkdir -p /var/log/celery

# 设置权限
sudo chown -R training:training /var/run/celery
sudo chown -R training:training /var/log/celery
```

**3. 启动服务**
```bash
# 启动 training worker
sudo systemctl start celery-training

# 启动 short worker
sudo systemctl start celery-short

# 启动 beat scheduler
sudo systemctl start celery-beat

# 设置开机自启
sudo systemctl enable celery-training
sudo systemctl enable celery-short
sudo systemctl enable celery-beat
```

**4. 查看状态**
```bash
# 查看服务状态
sudo systemctl status celery-training
sudo systemctl status celery-short
sudo systemctl status celery-beat

# 查看日志
sudo tail -f /var/log/celery/training.log
sudo tail -f /var/log/celery/short.log
sudo tail -f /var/log/celery/beat.log
```

**5. 重启/停止服务**
```bash
# 重启
sudo systemctl restart celery-training

# 停止
sudo systemctl stop celery-training
```

---

## Worker 配置详解

### Training Worker

```bash
celery -A training_platform.core.celery_config worker \
    -Q training \              # 只处理 training 队列
    -c 1 \                     # 单并发（避免 GPU 竞争）
    --max-tasks-per-child 1 \  # 每个任务后重启（避免内存泄漏）
    -n training@%h             # Worker 名称
```

**关键参数：**
- `-c 1`: 单并发，确保同时只有一个训练任务
- `--max-tasks-per-child 1`: 每个任务后重启 worker，避免：
  - 内存泄漏
  - GPU 内存未释放
  - 训练框架状态污染

**适用任务：**
- `train_model`: 主训练任务（3-24 小时）

---

### Short-lived Worker

```bash
celery -A training_platform.core.celery_config worker \
    -Q default,evaluation,preprocessing,maintenance \
    -c 4 \                     # 高并发（快速任务）
    -n short@%h
```

**关键参数：**
- `-Q default,evaluation,preprocessing,maintenance`: 处理多个快速队列
- `-c 4`: 高并发，可同时处理 4 个任务

**适用任务：**
- `run_evaluation`: 评测（5-30 分钟）
- `preprocess_dataset`: 数据预处理（1-10 分钟）
- `cleanup_checkpoints`: 清理检查点（< 1 分钟）
- `update_job_metrics`: 更新指标（< 10 秒）
- `scan_failed_jobs`: 扫描失败任务（< 10 秒）

---

### Beat Scheduler

```bash
celery -A training_platform.core.celery_config beat
```

**周期任务：**
- `update_job_metrics`: 每 1 分钟
- `scan_failed_jobs`: 每 5 分钟
- `cleanup_old_checkpoints`: 每 1 小时

**为什么独立进程：**
- Beat 需要精准的定时
- 不应该被长训练任务阻塞
- 避免多个 beat 实例冲突

---

## 性能对比

### 之前（单 Worker）

| 场景 | 响应时间 | 问题 |
|------|---------|------|
| 训练中，执行 update_metrics | 3 小时 | ❌ 被训练任务阻塞 |
| 训练中，触发评测 | 3 小时 | ❌ 等待训练完成 |
| 多个训练任务排队 | 顺序执行 | ❌ 无法并行（但GPU本来就该串行） |

### 之后（独立 Worker Pools）

| 场景 | 响应时间 | 结果 |
|------|---------|------|
| 训练中，执行 update_metrics | < 10 秒 | ✅ 在 short worker 立即执行 |
| 训练中，触发评测 | < 5 分钟 | ✅ 在 short worker 并行执行 |
| 多个训练任务排队 | 顺序执行 | ✅ 符合预期（避免 GPU 竞争） |

---

## 监控和调试

### Flower 监控面板

**访问：** http://localhost:5555

**功能：**
- 查看所有 workers 的状态
- 查看队列中的任务
- 查看任务执行历史
- 手动终止任务
- 查看 worker 资源使用

**关键指标：**
- **Active tasks**: 当前执行中的任务
- **Processed**: 已完成的任务数
- **Failed**: 失败的任务数
- **Queued**: 队列中等待的任务数

---

### 常见问题

**Q1: Training worker 一直没有任务？**

检查队列配置：
```python
# celery_config.py
task_routes={
    "training_platform.core.celery_tasks.train_model": {
        "queue": "training",  # ✅ 确保路由到 training 队列
    },
}
```

**Q2: 短任务仍然被阻塞？**

确认 short worker 正在运行：
```bash
# Docker Compose
docker-compose -f docker-compose.celery.yml logs celery_worker_short

# Systemd
sudo systemctl status celery-short
```

**Q3: Beat 任务没有触发？**

检查 Beat scheduler：
```bash
# 查看 beat 日志
docker-compose -f docker-compose.celery.yml logs celery_beat

# 确认任务配置
celery -A training_platform.core.celery_config inspect scheduled
```

**Q4: GPU 资源仍然冲突？**

确认 training worker 是单并发：
```bash
# 检查配置
celery -A training_platform.core.celery_config inspect active

# 应该看到：
# - celery_worker_training: concurrency=1
```

---

## 生产环境建议

### 1. 资源分配

**Training Worker:**
- CPU: 8-16 cores
- RAM: 32-64 GB
- GPU: 1-8 GPUs（根据模型大小）
- 并发: 1

**Short Worker:**
- CPU: 4-8 cores
- RAM: 8-16 GB
- GPU: 可选（evaluation 可能需要）
- 并发: 4

### 2. 监控告警

使用 Prometheus + Grafana 监控：
```yaml
# docker-compose.celery.yml 中添加
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
```

**关键指标：**
- Worker 存活状态
- 队列长度（超过阈值告警）
- 任务失败率
- 任务执行时间

### 3. 日志聚合

使用 ELK Stack 或 Loki：
```yaml
# docker-compose.celery.yml
loki:
  image: grafana/loki

promtail:
  image: grafana/promtail
  volumes:
    - /var/log/celery:/var/log/celery
```

### 4. 自动扩缩容

根据队列长度自动扩容 short worker：
```python
# autoscale.py
from celery import Celery

app.conf.worker_autoscaler = 'celery.worker.autoscale:Autoscaler'
app.conf.worker_max_tasks_per_child = 50
app.conf.worker_prefetch_multiplier = 1
```

---

## 总结

✅ **已解决的问题：**
1. ✅ 长训练任务不再阻塞短任务
2. ✅ GPU 资源不再冲突
3. ✅ 周期任务可以精准执行
4. ✅ 系统吞吐量显著提升

✅ **生产就绪：**
- Docker Compose 配置
- Systemd service 文件
- 启动脚本
- 监控方案
- 故障排查指南

**现在的平台已经可以在规模化场景下稳定运行！** 🚀
