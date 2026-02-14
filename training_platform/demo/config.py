"""
Demo模式配置
"""
import os
from typing import Optional
from pydantic import BaseModel


class DemoSettings(BaseModel):
    """Demo模式配置"""
    # 是否启用Demo模式
    enabled: bool = False
    # 演示速度倍率 (1.0=正常, 2.0=2倍速)
    speed: float = 1.0
    # 从哪个阶段开始演示 (1-10)
    start_stage: int = 1
    # 是否显示真实数据混合
    show_real_data: bool = False
    # 模拟延迟(ms)，让API响应更真实
    simulated_delay: int = 200

    class Config:
        env_prefix = "TRAIN_PLATFORM_DEMO_"


# 全局配置实例
_demo_settings: Optional[DemoSettings] = None


def get_demo_settings() -> DemoSettings:
    """获取Demo配置"""
    global _demo_settings
    if _demo_settings is None:
        _demo_settings = DemoSettings(
            enabled=os.getenv("TRAIN_PLATFORM_DEMO_MODE", "false").lower() == "true",
            speed=float(os.getenv("TRAIN_PLATFORM_DEMO_SPEED", "1.0")),
            start_stage=int(os.getenv("TRAIN_PLATFORM_DEMO_START_STAGE", "1")),
        )
    return _demo_settings


def set_demo_mode(enabled: bool, speed: float = 1.0, start_stage: int = 1):
    """动态设置Demo模式"""
    global _demo_settings
    _demo_settings = DemoSettings(
        enabled=enabled,
        speed=speed,
        start_stage=start_stage
    )


def is_demo_mode() -> bool:
    """检查是否处于Demo模式"""
    return get_demo_settings().enabled


# 便捷访问
demo_settings = get_demo_settings()
