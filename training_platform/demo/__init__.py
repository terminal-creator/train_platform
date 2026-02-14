"""
Demo Mode Module - 演示模式

为pitch演示提供高质量的Mock数据，展示完整的LLM训练流程。

使用方式:
1. 环境变量: TRAIN_PLATFORM_DEMO_MODE=true
2. 或在设置页面开启Demo模式

演示故事线(10个阶段):
1. 数据准备 - 高质量SFT数据集
2. 计算配置 - 智能算力计算
3. SFT预训练 - 监督微调
4. RM提示词 - 奖励模型配置
5. GRPO训练 - 策略优化
6. DPO对齐 - 直接偏好优化
7. 梯度可视化 - 训练诊断
8. 模型手术 - 检查点选择
9. 模型融合 - 多模型合并
10. 评估部署 - Benchmark评估
"""

from .config import demo_settings, is_demo_mode

__all__ = ['demo_settings', 'is_demo_mode']
