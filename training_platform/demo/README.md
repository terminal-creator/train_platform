# Demo Mode - 演示模式

## 快速启动

```bash
# 一键启动Demo模式
TRAIN_PLATFORM_DEMO_MODE=true DISABLE_AUTH=true uvicorn training_platform.api.main:app --reload --port 8000

# 前端（另一个终端）
cd frontend && npm run dev
```

## 关闭Demo模式

```bash
# 正常启动（无Demo数据）
DISABLE_AUTH=true uvicorn training_platform.api.main:app --reload --port 8000
```

## 隐藏开关

在设置页面底部的 **v1.0.0** 版本号上连点5次，可切换Demo模式。

---

## 演示故事线：Qwen2.5-7B 数学推理能力强化训练

一个完整的从数据准备到模型部署的端到端训练案例。

### 页面导览

| 页面 | 演示内容 | 故事点 |
|-----|---------|--------|
| **数据集** | 3个高质量数据集 | 50K数学指令、20K偏好对、5K评测集 |
| **计算配置** | 自动配置计算器 | 8x A100-80G 最优配置、显存估算 |
| **训练任务** | 5个训练任务 | SFT已完成、GRPO运行中(64%)、DPO待执行 |
| **实时监控** | GRPO训练监控 | Loss/Reward/KL曲线、梯度热力图、GPU利用率 |
| **自定义评估** | Benchmark结果 | GSM8K 58%→82%、MATH 24%→46% |
| **模型手术** | 模型优化 | SLERP融合、SWA权重平均、检查点选择 |
| **流水线** | 自动化流程 | 完整SFT→GRPO→评估→融合流水线 |

### 数据概览

```
📊 数据集
├── math_instruction_50k (SFT训练)
│   ├── 50,000 样本
│   ├── 质量分数: 95%
│   └── 主题: 代数/几何/概率/微积分
├── math_preference_20k (GRPO/DPO训练)
│   ├── 20,000 偏好对
│   ├── 来源: GPT-4 评判
│   └── 胜率差异: 显著
└── math_eval_5k (评估)
    ├── GSM8K: 1,319 样本
    └── MATH: 5,000 样本

🏃 训练任务
├── ✅ Qwen2.5-7B-Math-SFT (已完成)
│   └── GSM8K: 75.2%
├── 🔄 Qwen2.5-7B-Math-GRPO (运行中 64%)
│   ├── 当前步: 3200/5000
│   └── GSM8K: 82.3% (+7.1%)
├── ⏳ Qwen2.5-7B-Math-DPO (待执行)
├── ✅ Qwen2.5-7B-Code-SFT (已完成)
└── ✅ Qwen2.5-7B-Reasoning-GRPO (已完成)

📈 关键指标 (GRPO训练)
├── Reward: -0.3 → 0.82 (↑)
├── KL Divergence: 0.018 (稳定)
├── Policy Loss: 0.089
├── Tokens/sec: 15,000+
└── GPU Memory: 72/80 GB

🔧 模型手术结果
├── SLERP融合: GSM8K 86.8%
├── TIES融合: GSM8K 83.5%
├── DARE融合: GSM8K 84.2%
└── SWA优化: +1.5%提升

🔄 流水线状态
├── 数学推理增强流水线 (运行中)
│   └── 第4阶段/共8阶段 (GRPO训练)
└── 代码能力增强流水线 (已完成)
    └── 总耗时: 32.5小时
```

### 演示要点

1. **数据质量**: 展示OpenAI Messages格式数据、质量检测、分布分析
2. **计算智能**: 自动推荐ZeRO阶段、Batch Size、显存估算
3. **训练监控**: 实时Loss/Reward曲线、梯度健康检测、GPU集群状态
4. **效果验证**: Benchmark评分对比、训练前后提升
5. **模型优化**: 多种融合方法对比、最优检查点选择
6. **流水线**: 端到端自动化、阶段状态跟踪

---

## 技术实现

- **后端中间件**: `DemoModeMiddleware` 无痕拦截API返回Mock数据
- **数据生成**: 基于数学公式生成逼真的训练曲线
- **状态一致**: 所有页面数据相互关联，讲述完整故事
