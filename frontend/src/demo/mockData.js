/**
 * 前端Mock数据
 *
 * 用于在无后端连接时的本地演示
 */

// Demo任务ID
export const DEMO_JOB_IDS = {
  sft: 'demo-sft-qwen7b-math-001',
  grpo: 'demo-grpo-qwen7b-math-002',
  dpo: 'demo-dpo-qwen7b-math-003',
  sftCode: 'demo-sft-qwen7b-code-004',
  grpoReasoning: 'demo-grpo-qwen7b-reasoning-005',
}

// 演示故事线阶段
export const DEMO_STAGES = [
  { id: 1, name: '数据准备', icon: 'mdi-database', description: '高质量数据集准备和质量检测' },
  { id: 2, name: '计算配置', icon: 'mdi-calculator', description: '智能算力优化配置' },
  { id: 3, name: 'SFT预训练', icon: 'mdi-school', description: '监督微调训练' },
  { id: 4, name: 'RM提示词', icon: 'mdi-message-text', description: '奖励模型提示词配置' },
  { id: 5, name: 'GRPO训练', icon: 'mdi-chart-line', description: '策略优化训练' },
  { id: 6, name: 'DPO对齐', icon: 'mdi-tune', description: '直接偏好优化' },
  { id: 7, name: '梯度可视化', icon: 'mdi-gradient-horizontal', description: '训练诊断和可视化' },
  { id: 8, name: '模型手术', icon: 'mdi-medical-bag', description: '检查点选择和优化' },
  { id: 9, name: '模型融合', icon: 'mdi-merge', description: '多模型融合' },
  { id: 10, name: '评估部署', icon: 'mdi-rocket-launch', description: 'Benchmark评估和导出' },
]

// 快速计算配置结果
export const DEMO_COMPUTE_RESULT = {
  config: {
    actor: {
      learning_rate: 5e-7,
      weight_decay: 0.01,
      warmup_ratio: 0.03,
      gradient_accumulation_steps: 8,
      micro_batch_size: 4,
    },
    critic: {
      learning_rate: 1e-6,
    },
    rollout: {
      num_gpus: 4,
      tensor_parallel_size: 2,
    },
    ref: {
      num_gpus: 2,
    },
  },
  yaml: `# Auto-generated VERL configuration
actor:
  learning_rate: 5e-7
  weight_decay: 0.01
  warmup_ratio: 0.03
  gradient_accumulation_steps: 8
  micro_batch_size: 4

critic:
  learning_rate: 1e-6

rollout:
  num_gpus: 4
  tensor_parallel_size: 2

ref:
  num_gpus: 2`,
  memory_estimate: {
    model_weights_gb: 14.0,
    optimizer_states_gb: 42.0,
    gradients_gb: 14.0,
    activations_gb: 8.0,
    total_gb: 78.0,
  },
  zero_stage: 2,
  summary: {
    micro_batch_size: 4,
    global_batch_size: 256,
    gradient_accumulation: 8,
    estimated_memory_per_gpu: '72GB',
    recommended_gpu: 'A100-80G',
    gpu_count: 8,
  },
}

// 演示仪表板数据
export const DEMO_DASHBOARD = {
  summary: {
    total_jobs: 5,
    running_jobs: 1,
    completed_jobs: 3,
    pending_jobs: 1,
    total_pipelines: 2,
    active_pipelines: 1,
  },
  gpu_utilization: {
    total_gpus: 8,
    used_gpus: 8,
    avg_utilization: 92.5,
    avg_memory_used: 74.2,
    gpu_type: 'NVIDIA A100-80G',
  },
  recent_activity: [
    { time: '2分钟前', event: 'GRPO训练 step 3200 完成', type: 'progress' },
    { time: '15分钟前', event: '检查点 checkpoint-3000 保存成功', type: 'checkpoint' },
    { time: '1小时前', event: 'GSM8K评估完成: 82.3%', type: 'evaluation' },
    { time: '2小时前', event: 'SFT训练完成', type: 'completed' },
  ],
}

// 生成实时指标数据
export function generateRealtimeMetrics(step, algorithm = 'grpo') {
  const progress = step / 5000

  if (algorithm === 'grpo') {
    const sigmoid = 1 / (1 + Math.exp(-8 * (progress - 0.4)))
    return {
      step,
      epoch: Math.floor(progress * 3) + 1,
      policy_loss: Math.max(0.25 * (1 - 0.65 * progress) + (Math.random() - 0.5) * 0.02, 0.05),
      value_loss: Math.max(0.15 * (1 - 0.55 * progress) + (Math.random() - 0.5) * 0.01, 0.02),
      reward_mean: Math.min(0.15 + 0.77 * sigmoid + (Math.random() - 0.5) * 0.03, 0.95),
      reward_std: 0.25 * (1 - 0.4 * progress) + (Math.random() - 0.5) * 0.02,
      kl_divergence: 0.025 * progress * (1 + 0.2 * Math.sin(5 * Math.PI * progress)),
      entropy: 2.1 - 0.25 * progress + (Math.random() - 0.5) * 0.05,
      throughput_tokens_per_sec: Math.floor(11000 + Math.random() * 1000),
      gpu_memory_used_gb: 72 + Math.random() * 2,
      timestamp: new Date().toISOString(),
    }
  } else if (algorithm === 'sft') {
    const decay = Math.exp(-3 * progress)
    return {
      step,
      epoch: Math.floor(progress * 2) + 1,
      train_loss: Math.max(0.42 + 2.38 * decay + (Math.random() - 0.5) * 0.04, 0.35),
      eval_loss: Math.max(0.48 + 2.5 * decay + (Math.random() - 0.5) * 0.04, 0.4),
      perplexity: Math.exp(0.42 + 2.38 * decay),
      throughput_tokens_per_sec: Math.floor(11000 + Math.random() * 1000),
      gpu_memory_used_gb: 72 + Math.random() * 2,
      timestamp: new Date().toISOString(),
    }
  }

  return { step, timestamp: new Date().toISOString() }
}

// 生成梯度热力图数据
export function generateGradientHeatmap(numLayers = 32, numSteps = 50) {
  const layers = [
    'embed',
    ...Array.from({ length: numLayers }, (_, i) => `L${i}_attn`),
    ...Array.from({ length: numLayers }, (_, i) => `L${i}_ffn`),
    'head',
  ]

  const steps = Array.from({ length: numSteps }, (_, i) => i * 100)

  const values = layers.map((_, layerIdx) =>
    steps.map((_, stepIdx) => {
      const baseNorm = 0.1 + 0.05 * Math.sin(layerIdx * 0.1)
      const earlyFactor = stepIdx < 10 ? 1.5 : 1.0
      const depthFactor = 0.9 + 0.1 * (1 - layerIdx / layers.length)
      const noise = (Math.random() - 0.5) * 0.04
      const gradNorm = baseNorm * earlyFactor * depthFactor + noise
      return Math.log10(Math.max(gradNorm, 1e-10))
    })
  )

  return {
    layers,
    steps,
    values,
    value_range: { min: -3.0, max: 0.0 },
    colorscale: 'RdYlGn',
  }
}

// 评估对比数据
export const DEMO_EVALUATION_COMPARISON = {
  models: [
    { name: 'Qwen2.5-7B (Base)', color: '#9E9E9E' },
    { name: 'Math-SFT', color: '#2196F3' },
    { name: 'Math-GRPO', color: '#4CAF50' },
    { name: 'Code-SFT', color: '#FF9800' },
    { name: 'Reasoning-GRPO', color: '#9C27B0' },
  ],
  benchmarks: ['GSM8K', 'MATH', 'HumanEval', 'MBPP', 'MMLU'],
  scores: {
    'Qwen2.5-7B (Base)': [58.3, 24.5, 52.4, 58.6, 68.2],
    'Math-SFT': [75.2, 38.5, 54.2, 59.8, 69.5],
    'Math-GRPO': [82.3, 45.6, 55.8, 61.2, 70.8],
    'Code-SFT': [61.5, 26.8, 68.9, 72.4, 68.8],
    'Reasoning-GRPO': [85.6, 41.2, 72.0, 75.6, 71.5],
  },
}
