"""
时间线生成器 - 模拟训练过程的时间进度

用于演示流水线的阶段推进
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class StageProgress:
    """阶段进度"""
    stage_id: str
    stage_name: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0  # 0-100
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Dict = field(default_factory=dict)


@dataclass
class PipelineTimeline:
    """流水线时间线"""
    pipeline_id: str
    stages: List[StageProgress] = field(default_factory=list)
    current_stage_index: int = 0
    start_time: datetime = field(default_factory=datetime.now)


class TimelineGenerator:
    """
    时间线生成器

    模拟流水线各阶段的执行进度
    """

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self._timelines: Dict[str, PipelineTimeline] = {}

    def create_timeline(
        self,
        pipeline_id: str,
        stages: List[Dict],
    ) -> PipelineTimeline:
        """创建流水线时间线"""
        stage_progresses = [
            StageProgress(
                stage_id=s.get("id", f"stage-{i}"),
                stage_name=s.get("name", f"Stage {i}"),
            )
            for i, s in enumerate(stages)
        ]

        timeline = PipelineTimeline(
            pipeline_id=pipeline_id,
            stages=stage_progresses,
        )

        self._timelines[pipeline_id] = timeline
        return timeline

    def get_timeline(self, pipeline_id: str) -> Optional[PipelineTimeline]:
        """获取时间线"""
        return self._timelines.get(pipeline_id)

    def start_stage(self, pipeline_id: str, stage_index: int = None) -> Optional[StageProgress]:
        """开始执行阶段"""
        timeline = self._timelines.get(pipeline_id)
        if not timeline:
            return None

        if stage_index is None:
            stage_index = timeline.current_stage_index

        if stage_index >= len(timeline.stages):
            return None

        stage = timeline.stages[stage_index]
        stage.status = "running"
        stage.started_at = datetime.now()
        stage.progress = 0.0

        return stage

    def update_progress(
        self,
        pipeline_id: str,
        stage_index: int,
        progress: float,
        result: Dict = None,
    ) -> Optional[StageProgress]:
        """更新阶段进度"""
        timeline = self._timelines.get(pipeline_id)
        if not timeline or stage_index >= len(timeline.stages):
            return None

        stage = timeline.stages[stage_index]
        stage.progress = min(progress, 100.0)

        if result:
            stage.result.update(result)

        # 如果进度达到100%，标记为完成
        if progress >= 100.0:
            stage.status = "completed"
            stage.completed_at = datetime.now()
            timeline.current_stage_index = stage_index + 1

        return stage

    def complete_stage(
        self,
        pipeline_id: str,
        stage_index: int,
        result: Dict = None,
    ) -> Optional[StageProgress]:
        """完成阶段"""
        return self.update_progress(pipeline_id, stage_index, 100.0, result)

    def fail_stage(
        self,
        pipeline_id: str,
        stage_index: int,
        error: str,
    ) -> Optional[StageProgress]:
        """标记阶段失败"""
        timeline = self._timelines.get(pipeline_id)
        if not timeline or stage_index >= len(timeline.stages):
            return None

        stage = timeline.stages[stage_index]
        stage.status = "failed"
        stage.result["error"] = error

        return stage

    def get_overall_progress(self, pipeline_id: str) -> float:
        """计算总体进度"""
        timeline = self._timelines.get(pipeline_id)
        if not timeline or not timeline.stages:
            return 0.0

        total_progress = 0.0
        for stage in timeline.stages:
            if stage.status == "completed":
                total_progress += 100.0
            elif stage.status == "running":
                total_progress += stage.progress

        return total_progress / len(timeline.stages)

    def simulate_pipeline(
        self,
        pipeline_id: str,
        stage_durations: List[float],  # 每个阶段的模拟时长（秒）
        on_progress: Callable[[str, int, float], None] = None,
    ):
        """
        模拟整个流水线执行

        Args:
            pipeline_id: 流水线ID
            stage_durations: 每个阶段的模拟时长
            on_progress: 进度回调函数 (pipeline_id, stage_index, progress)
        """
        timeline = self._timelines.get(pipeline_id)
        if not timeline:
            return

        for i, stage in enumerate(timeline.stages):
            # 开始阶段
            self.start_stage(pipeline_id, i)

            # 模拟执行
            duration = stage_durations[i] if i < len(stage_durations) else 5.0
            actual_duration = duration / self.speed
            steps = 20
            step_duration = actual_duration / steps

            for step in range(steps):
                progress = (step + 1) / steps * 100
                self.update_progress(pipeline_id, i, progress)

                if on_progress:
                    on_progress(pipeline_id, i, progress)

                time.sleep(step_duration)

            # 完成阶段
            self.complete_stage(pipeline_id, i, {"status": "success"})


# 全局时间线生成器
_timeline_generator: Optional[TimelineGenerator] = None


def get_timeline_generator(speed: float = 1.0) -> TimelineGenerator:
    """获取全局时间线生成器"""
    global _timeline_generator
    if _timeline_generator is None or _timeline_generator.speed != speed:
        _timeline_generator = TimelineGenerator(speed=speed)
    return _timeline_generator
