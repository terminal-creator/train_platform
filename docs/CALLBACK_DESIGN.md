# verl Callback 接口设计规范

## 1. 设计目标

为 verl 训练器添加 callback 机制，允许外部平台在训练过程中的关键时刻接收实时指标和状态信息。

### 核心需求
- ✅ 实时指标推送：训练过程中的 loss、reward、KL 散度等
- ✅ 异常检测：NaN/Inf、KL 爆炸、Loss 不下降等
- ✅ 本地和远程支持：本地训练和 SSH 远程训练都能监控
- ✅ 低侵入性：不影响 verl 原有训练逻辑

## 2. Callback 接口设计

### 2.1 基础接口

```python
# verl/trainer/callbacks/base.py
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

class TrainerCallback(ABC):
    """
    训练过程的回调接口基类

    所有 callback 方法接收两个参数：
    - trainer: 训练器实例（可访问配置、模型等）
    - metrics: 当前步骤的指标字典
    """

    def on_train_begin(self, trainer: Any, **kwargs):
        """训练开始时调用"""
        pass

    def on_train_end(self, trainer: Any, **kwargs):
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int, **kwargs):
        """每个 epoch 开始时调用"""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, Any], **kwargs):
        """每个 epoch 结束时调用"""
        pass

    def on_step_begin(self, trainer: Any, step: int, **kwargs):
        """每个训练步骤开始时调用"""
        pass

    def on_step_end(self, trainer: Any, step: int, metrics: Dict[str, Any], **kwargs):
        """每个训练步骤结束时调用（核心）"""
        pass

    def on_generate_end(self, trainer: Any, step: int, gen_metrics: Dict[str, Any], **kwargs):
        """生成阶段结束时调用"""
        pass

    def on_reward_end(self, trainer: Any, step: int, reward_metrics: Dict[str, Any], **kwargs):
        """奖励计算结束时调用"""
        pass

    def on_update_end(self, trainer: Any, step: int, update_metrics: Dict[str, Any], **kwargs):
        """模型更新结束时调用"""
        pass

    def on_validation_end(self, trainer: Any, step: int, val_metrics: Dict[str, Any], **kwargs):
        """验证结束时调用"""
        pass

    def on_checkpoint_save(self, trainer: Any, step: int, checkpoint_path: str, **kwargs):
        """保存检查点时调用"""
        pass
```

### 2.2 平台专用 Callback

```python
# verl/trainer/callbacks/platform_callback.py
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from .base import TrainerCallback


class PlatformCallback(TrainerCallback):
    """
    训练平台专用 callback

    功能：
    1. 将训练指标写入 JSON Lines 文件
    2. 检测训练异常（NaN、KL 爆炸等）
    3. 支持本地和 SSH 远程读取
    """

    def __init__(
        self,
        job_id: str,
        output_dir: str = "./platform_metrics",
        enable_anomaly_detection: bool = True,
        nan_check: bool = True,
        kl_threshold: float = 1.0,  # KL 散度阈值
        loss_patience: int = 50,    # Loss 不下降的容忍步数
    ):
        """
        Args:
            job_id: 训练任务 ID
            output_dir: 指标文件输出目录
            enable_anomaly_detection: 是否启用异常检测
            nan_check: 是否检测 NaN/Inf
            kl_threshold: KL 散度异常阈值
            loss_patience: Loss 不下降的容忍步数
        """
        self.job_id = job_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 指标文件：每个 step 一行 JSON
        self.metrics_file = self.output_dir / f"{job_id}_metrics.jsonl"
        self.status_file = self.output_dir / f"{job_id}_status.json"

        # 异常检测配置
        self.enable_anomaly_detection = enable_anomaly_detection
        self.nan_check = nan_check
        self.kl_threshold = kl_threshold
        self.loss_patience = loss_patience

        # 状态追踪
        self.best_loss = float('inf')
        self.no_improve_steps = 0
        self.anomaly_detected = False
        self.anomaly_reason = None

    def on_train_begin(self, trainer: Any, **kwargs):
        """训练开始：初始化状态文件"""
        status = {
            "job_id": self.job_id,
            "status": "running",
            "current_step": 0,
            "total_steps": getattr(trainer, 'total_training_steps', None),
            "anomaly_detected": False,
            "anomaly_reason": None,
        }
        self._write_status(status)

    def on_step_end(self, trainer: Any, step: int, metrics: Dict[str, Any], **kwargs):
        """
        每个训练步骤结束：记录指标并检测异常

        这是最核心的方法，会在每个训练步骤结束时被调用
        """
        # 1. 标准化指标格式
        standard_metrics = self._standardize_metrics(step, metrics)

        # 2. 写入指标文件（追加模式，每行一个 JSON）
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(standard_metrics) + '\n')

        # 3. 异常检测
        if self.enable_anomaly_detection and not self.anomaly_detected:
            self._detect_anomalies(step, standard_metrics)

        # 4. 更新状态文件
        status = {
            "job_id": self.job_id,
            "status": "running",
            "current_step": step,
            "total_steps": getattr(trainer, 'total_training_steps', None),
            "anomaly_detected": self.anomaly_detected,
            "anomaly_reason": self.anomaly_reason,
            "latest_metrics": standard_metrics,
        }
        self._write_status(status)

    def on_train_end(self, trainer: Any, **kwargs):
        """训练结束：标记最终状态"""
        status = {
            "job_id": self.job_id,
            "status": "failed" if self.anomaly_detected else "completed",
            "current_step": getattr(trainer, 'global_steps', 0),
            "total_steps": getattr(trainer, 'total_training_steps', None),
            "anomaly_detected": self.anomaly_detected,
            "anomaly_reason": self.anomaly_reason,
        }
        self._write_status(status)

    def _standardize_metrics(self, step: int, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化指标格式

        将 verl 的原始指标转换为平台统一格式
        """
        return {
            "step": step,
            "timestamp": __import__('time').time(),

            # 核心训练指标
            "loss": {
                "actor_loss": raw_metrics.get("actor/loss"),
                "critic_loss": raw_metrics.get("critic/loss"),
                "total_loss": raw_metrics.get("train/loss"),
            },

            # 奖励指标
            "reward": {
                "mean": raw_metrics.get("reward/mean"),
                "std": raw_metrics.get("reward/std"),
                "max": raw_metrics.get("reward/max"),
                "min": raw_metrics.get("reward/min"),
            },

            # KL 散度
            "kl": {
                "mean": raw_metrics.get("actor/kl_div"),
                "max": raw_metrics.get("actor/kl_div_max"),
            },

            # 梯度信息
            "gradient": {
                "actor_norm": raw_metrics.get("actor/grad_norm"),
                "critic_norm": raw_metrics.get("critic/grad_norm"),
            },

            # 性能指标
            "performance": {
                "tokens_per_second": raw_metrics.get("perf/total_tokens_per_second"),
                "step_time": raw_metrics.get("timing/step"),
                "gpu_memory_allocated": raw_metrics.get("actor/gpu_memory_allocated_GiB"),
            },

            # 原始指标（保留所有）
            "raw": raw_metrics,
        }

    def _detect_anomalies(self, step: int, metrics: Dict[str, Any]):
        """
        检测训练异常

        支持的异常类型：
        1. NaN/Inf 检测
        2. KL 散度爆炸
        3. Loss 长期不下降
        """
        # 1. NaN/Inf 检测
        if self.nan_check:
            actor_loss = metrics["loss"]["actor_loss"]
            if actor_loss is not None and (
                __import__('math').isnan(actor_loss) or
                __import__('math').isinf(actor_loss)
            ):
                self.anomaly_detected = True
                self.anomaly_reason = f"NaN or Inf detected in actor_loss at step {step}"
                return

        # 2. KL 散度爆炸检测
        kl_mean = metrics["kl"]["mean"]
        if kl_mean is not None and kl_mean > self.kl_threshold:
            self.anomaly_detected = True
            self.anomaly_reason = f"KL divergence explosion: {kl_mean:.4f} > {self.kl_threshold} at step {step}"
            return

        # 3. Loss 不下降检测
        actor_loss = metrics["loss"]["actor_loss"]
        if actor_loss is not None:
            if actor_loss < self.best_loss:
                self.best_loss = actor_loss
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1

                if self.no_improve_steps >= self.loss_patience:
                    self.anomaly_detected = True
                    self.anomaly_reason = f"Loss not improving for {self.loss_patience} steps (best: {self.best_loss:.4f}, current: {actor_loss:.4f})"
                    return

    def _write_status(self, status: Dict[str, Any]):
        """写入状态文件（覆盖模式）"""
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
```

## 3. Trainer 集成方式

### 3.1 修改 ray_trainer.py

在 `PPORayTrainer` 类中添加 callback 支持：

```python
# 在 __init__ 方法中
class PPORayTrainer:
    def __init__(self, config):
        # ... 现有代码 ...

        # 添加 callback 列表
        self.callbacks = []

        # 从配置中加载 callback
        if hasattr(config.trainer, 'callbacks'):
            for callback_config in config.trainer.callbacks:
                callback_cls = self._load_callback_class(callback_config['class'])
                callback = callback_cls(**callback_config.get('params', {}))
                self.callbacks.append(callback)

    def _trigger_callbacks(self, event: str, **kwargs):
        """触发所有 callback 的指定事件"""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method and callable(method):
                method(trainer=self, **kwargs)

    def fit(self):
        # 训练开始
        self._trigger_callbacks('on_train_begin')

        for epoch in range(...):
            # Epoch 开始
            self._trigger_callbacks('on_epoch_begin', epoch=epoch)

            for batch in self.train_dataloader:
                # Step 开始
                self._trigger_callbacks('on_step_begin', step=self.global_steps)

                # ... 训练逻辑 ...

                # Step 结束（记录指标）
                self._trigger_callbacks('on_step_end',
                                       step=self.global_steps,
                                       metrics=metrics)

            # Epoch 结束
            self._trigger_callbacks('on_epoch_end', epoch=epoch, metrics=metrics)

        # 训练结束
        self._trigger_callbacks('on_train_end')
```

### 3.2 配置文件示例

在 Hydra 配置中添加 callback 配置：

```yaml
# config.yaml
trainer:
  # ... 其他配置 ...

  callbacks:
    - class: verl.trainer.callbacks.PlatformCallback
      params:
        job_id: ${job_id}  # 从环境变量或参数传入
        output_dir: ./platform_metrics
        enable_anomaly_detection: true
        nan_check: true
        kl_threshold: 1.0
        loss_patience: 50
```

## 4. 指标文件格式

### 4.1 指标文件 (`{job_id}_metrics.jsonl`)

每行一个 JSON 对象，记录一个训练步骤的指标：

```jsonl
{"step": 1, "timestamp": 1704672000.123, "loss": {"actor_loss": 2.5, "critic_loss": 1.2}, "reward": {"mean": 0.5}, "kl": {"mean": 0.1}, ...}
{"step": 2, "timestamp": 1704672010.456, "loss": {"actor_loss": 2.3, "critic_loss": 1.1}, "reward": {"mean": 0.6}, "kl": {"mean": 0.12}, ...}
{"step": 3, "timestamp": 1704672020.789, "loss": {"actor_loss": 2.1, "critic_loss": 1.0}, "reward": {"mean": 0.7}, "kl": {"mean": 0.15}, ...}
```

### 4.2 状态文件 (`{job_id}_status.json`)

实时状态（覆盖更新）：

```json
{
  "job_id": "job_20260108_001",
  "status": "running",
  "current_step": 150,
  "total_steps": 1000,
  "anomaly_detected": false,
  "anomaly_reason": null,
  "latest_metrics": {
    "step": 150,
    "loss": {"actor_loss": 1.5},
    "reward": {"mean": 0.8}
  }
}
```

## 5. 平台读取方式

### 5.1 本地读取

```python
# training_platform/core/local_metrics_reader.py
class LocalMetricsReader:
    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)

    def read_latest_metrics(self, job_id: str, last_n: int = 100):
        """读取最新的 N 条指标"""
        metrics_file = self.metrics_dir / f"{job_id}_metrics.jsonl"

        if not metrics_file.exists():
            return []

        # 读取最后 N 行
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            return [json.loads(line) for line in lines[-last_n:]]

    def read_status(self, job_id: str):
        """读取训练状态"""
        status_file = self.metrics_dir / f"{job_id}_status.json"

        if not status_file.exists():
            return None

        with open(status_file, 'r') as f:
            return json.load(f)
```

### 5.2 SSH 远程读取

```python
# training_platform/core/ssh_metrics_reader.py
class SSHMetricsReader:
    def __init__(self, ssh_client, remote_metrics_dir: str):
        self.ssh_client = ssh_client
        self.remote_metrics_dir = remote_metrics_dir

    def read_latest_metrics(self, job_id: str, last_n: int = 100):
        """通过 SSH 读取远程指标"""
        remote_file = f"{self.remote_metrics_dir}/{job_id}_metrics.jsonl"

        # 使用 tail 读取最后 N 行
        cmd = SafeCommands.tail_file(remote_file, lines=last_n)
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)

        lines = stdout.read().decode('utf-8').strip().split('\n')
        return [json.loads(line) for line in lines if line]

    def read_status(self, job_id: str):
        """读取远程训练状态"""
        remote_file = f"{self.remote_metrics_dir}/{job_id}_status.json"

        cmd = build_command('cat', remote_file)
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)

        content = stdout.read().decode('utf-8')
        return json.loads(content) if content else None
```

## 6. 实现计划

### Phase 1.1: verl 集成（Week 1）
- [ ] 1.1.1 创建 callback 基类和平台 callback
- [ ] 1.1.2 修改 ray_trainer.py 支持 callback
- [ ] 1.1.3 测试 callback 能正常记录指标

### Phase 1.2: 指标存储（Week 2）
- [ ] 1.2.1 实现本地指标读取器
- [ ] 1.2.2 实现 SSH 指标读取器
- [ ] 1.2.3 将指标持久化到数据库

### Phase 1.3: 实时监控（Week 3）
- [ ] 1.3.1 重构 WebSocket 推送真实指标
- [ ] 1.3.2 添加历史指标回放
- [ ] 1.3.3 前端图表展示

### Phase 1.4: 异常检测（Week 4）
- [ ] 1.4.1 完善 NaN/KL/Loss 检测逻辑
- [ ] 1.4.2 异常时自动停止训练
- [ ] 1.4.3 前端异常告警提示

## 7. 优势

✅ **低侵入性**: 只需修改 trainer 添加 callback 调用点
✅ **可扩展**: 支持添加多个 callback
✅ **本地 + 远程**: 文件读取支持本地和 SSH 模式
✅ **实时性**: 每个 step 结束立即写入文件
✅ **容错性**: 异常检测不影响训练主流程
✅ **标准化**: 统一的指标格式，方便解析
