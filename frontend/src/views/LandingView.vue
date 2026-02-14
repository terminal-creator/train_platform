<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const enterPlatform = () => {
  router.push('/dashboard')
}
</script>

<template>
  <div class="landing-page">
    <!-- Navigation -->
    <nav class="nav">
      <div class="nav-container">
        <a href="#" class="logo">
          <div class="logo-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3 2.5 2.5 0 0 0-1.32 4.24 3 3 0 0 0 .34 5.58 2.5 2.5 0 0 0 2.96 3.08A2.5 2.5 0 0 0 12 19.5a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 12 4.5"/>
            </svg>
          </div>
          <span class="logo-text">TRAIN.AI</span>
        </a>

        <div class="nav-links">
          <a href="#features">核心能力</a>
          <a href="#workflow">工作流程</a>
          <a href="#algorithms">算法支持</a>
          <a href="#metrics">效果展示</a>
        </div>

        <button class="nav-cta" @click="enterPlatform">
          <span>进入平台</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
          </svg>
        </button>
      </div>
    </nav>

    <!-- Hero Section -->
    <main class="hero">
      <div class="glow-blob"></div>
      <div class="glow-blob-2"></div>

      <div class="hero-content">
        <!-- Badge -->
        <div class="badge">
          <span class="badge-dot"></span>
          全球首个专业工业级全流程 Post-Train 平台
        </div>

        <!-- Main Title -->
        <h1 class="hero-title">
          让 LLM Post-Train
          <span class="gradient-text">成为 Science</span>
          <span class="subtitle">而不是 Sense</span>
        </h1>

        <p class="hero-desc">
          从数据准备到模型评估，从 SFT 微调到 RLHF 对齐<br>
          提供完整的工业级解决方案<br>
          让每一次训练都有迹可循，让每一个决策都有据可依
        </p>

        <!-- CTA Buttons -->
        <div class="hero-cta">
          <button class="btn-primary" @click="enterPlatform">
            开始使用
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m9 18 6-6-6-6"/>
            </svg>
          </button>
          <a href="#features" class="btn-secondary">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>
            </svg>
            了解更多
          </a>
        </div>

        <!-- Tech Stack -->
        <div class="tech-stack">
          <span class="tech-label">技术栈</span>
          <div class="tech-tag">PyTorch</div>
          <div class="tech-tag">Ray</div>
          <div class="tech-tag highlight">verl Framework</div>
          <div class="tech-tag">Multi-GPU</div>
        </div>
      </div>
    </main>

    <!-- Workflow Pipeline -->
    <section class="workflow" id="workflow">
      <div class="section-container">
        <div class="section-header">
          <h2>端到端训练流水线</h2>
          <p>一站式完成从数据到部署的全流程</p>
        </div>

        <div class="pipeline">
          <div class="pipeline-step">
            <div class="step-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>
              </svg>
            </div>
            <span class="step-title">数据准备</span>
            <span class="step-desc">质量分析 · 格式转换</span>
          </div>

          <div class="pipeline-arrow">→</div>

          <div class="pipeline-step">
            <div class="step-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v10"/><path d="m4.93 4.93 4.24 4.24m5.66 5.66 4.24 4.24"/><path d="M1 12h6m6 0h10"/><path d="m4.93 19.07 4.24-4.24m5.66-5.66 4.24-4.24"/>
              </svg>
            </div>
            <span class="step-title">计算配置</span>
            <span class="step-desc">显存优化 · 自动调参</span>
          </div>

          <div class="pipeline-arrow">→</div>

          <div class="pipeline-step">
            <div class="step-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>
              </svg>
            </div>
            <span class="step-title">SFT 微调</span>
            <span class="step-desc">指令学习 · 知识注入</span>
          </div>

          <div class="pipeline-arrow">→</div>

          <div class="pipeline-step active">
            <div class="step-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>
              </svg>
            </div>
            <span class="step-title">RLHF 对齐</span>
            <span class="step-desc">PPO · GRPO · DPO</span>
          </div>

          <div class="pipeline-arrow">→</div>

          <div class="pipeline-step">
            <div class="step-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
              </svg>
            </div>
            <span class="step-title">评估对比</span>
            <span class="step-desc">Benchmark · A/B测试</span>
          </div>

          <div class="pipeline-arrow">→</div>

          <div class="pipeline-step">
            <div class="step-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M6 21V9a9 9 0 0 0 9 9"/>
              </svg>
            </div>
            <span class="step-title">模型融合</span>
            <span class="step-desc">SLERP · TIES · SWA</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features" id="features">
      <div class="section-container">
        <div class="section-header">
          <h2>核心能力</h2>
          <p>专为工业级 LLM 训练打造的全栈解决方案</p>
        </div>

        <div class="features-grid">
          <div class="feature-card">
            <div class="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>
              </svg>
            </div>
            <h3>智能显存计算</h3>
            <p>自动估算训练显存需求，智能推荐 Batch Size、梯度累积、ZeRO Stage 等参数</p>
            <div class="feature-tags">
              <span>显存预测</span>
              <span>参数推荐</span>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
            </div>
            <h3>实时训练监控</h3>
            <p>全方位监控 Loss、Reward、KL 散度，梯度热力图可视化，异常自动检测</p>
            <div class="feature-tags">
              <span>实时曲线</span>
              <span>异常告警</span>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10 2v7.31"/><path d="M14 9.3V1.99"/><path d="M8.5 2h7"/><path d="M14 9.3a6.5 6.5 0 1 1-4 0"/><path d="M5.58 16.5h12.85"/>
              </svg>
            </div>
            <h3>Benchmark 评估</h3>
            <p>内置 GSM8K、MATH、HumanEval 等主流评测集，支持自定义评估与对比分析</p>
            <div class="feature-tags">
              <span>标准评测</span>
              <span>对比分析</span>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>
              </svg>
            </div>
            <h3>Pipeline 编排</h3>
            <p>可视化定义多阶段训练流程，自动执行 SFT → RLHF → 评估 → 融合链路</p>
            <div class="feature-tags">
              <span>DAG 编排</span>
              <span>自动触发</span>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>
              </svg>
            </div>
            <h3>模型手术</h3>
            <p>支持 SLERP、TIES、DARE 模型融合，SWA 权重平均，智能 Checkpoint 选择</p>
            <div class="feature-tags">
              <span>模型融合</span>
              <span>权重平均</span>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/><path d="M11 8v6"/><path d="M8 11h6"/>
              </svg>
            </div>
            <h3>数据质量分析</h3>
            <p>自动检测数据格式、Token 分布、重复率，支持 ShareGPT、Alpaca 等格式</p>
            <div class="feature-tags">
              <span>格式检测</span>
              <span>质量评分</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Algorithms Section -->
    <section class="algorithms" id="algorithms">
      <div class="section-container">
        <div class="section-header">
          <h2>全栈算法支持</h2>
          <p>覆盖 Post-Training 全阶段的主流算法</p>
        </div>

        <div class="algo-grid">
          <div class="algo-card">
            <div class="algo-header">
              <div class="algo-badge">SFT</div>
              <div class="algo-title">
                <h3>监督微调</h3>
                <span>Supervised Fine-Tuning</span>
              </div>
            </div>
            <p>在高质量指令数据上进行监督学习，让模型学会遵循指令、理解任务意图</p>
            <div class="algo-use">适用：知识注入 · 指令遵循 · 领域适配</div>
          </div>

          <div class="algo-card">
            <div class="algo-header">
              <div class="algo-badge">PPO</div>
              <div class="algo-title">
                <h3>近端策略优化</h3>
                <span>Proximal Policy Optimization</span>
              </div>
            </div>
            <p>经典的 RLHF 算法，通过 Reward Model 信号优化策略，稳定高效</p>
            <div class="algo-use">适用：对齐人类偏好 · 安全性优化</div>
          </div>

          <div class="algo-card featured">
            <div class="algo-recommend">推荐</div>
            <div class="algo-header">
              <div class="algo-badge">GRPO</div>
              <div class="algo-title">
                <h3>组相对策略优化</h3>
                <span>Group Relative Policy Optimization</span>
              </div>
            </div>
            <p>DeepSeek 提出的高效算法，无需 Critic Model，显存更低，训练更稳定</p>
            <div class="algo-use">适用：推理增强 · 数学能力 · 代码生成</div>
          </div>

          <div class="algo-card">
            <div class="algo-header">
              <div class="algo-badge">DPO</div>
              <div class="algo-title">
                <h3>直接偏好优化</h3>
                <span>Direct Preference Optimization</span>
              </div>
            </div>
            <p>无需 Reward Model 的对齐方法，直接从偏好数据学习，简化训练流程</p>
            <div class="algo-use">适用：偏好对齐 · 风格迁移 · 快速迭代</div>
          </div>

          <div class="algo-card">
            <div class="algo-header">
              <div class="algo-badge">GSPO</div>
              <div class="algo-title">
                <h3>组采样策略优化</h3>
                <span>Group Sampling Policy Optimization</span>
              </div>
            </div>
            <p>结合 GRPO 和采样策略的优化方法，提升样本效率和训练稳定性</p>
            <div class="algo-use">适用：大规模训练 · 高效对齐</div>
          </div>

          <div class="algo-card">
            <div class="algo-header">
              <div class="algo-badge">RM</div>
              <div class="algo-title">
                <h3>Reward Model</h3>
                <span>奖励模型训练</span>
              </div>
            </div>
            <p>训练高质量的奖励模型，为 PPO/GRPO 提供准确的奖励信号</p>
            <div class="algo-use">适用：偏好建模 · 质量评估</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Metrics Section -->
    <section class="metrics" id="metrics">
      <div class="section-container">
        <div class="section-header">
          <h2>效果展示</h2>
          <p>数据驱动的训练决策，看得见的能力提升</p>
        </div>

        <div class="metrics-grid">
          <!-- Metrics Dashboard -->
          <div class="metrics-card">
            <div class="metrics-card-header">
              <h3>训练监控面板</h3>
              <div class="live-indicator">
                <span class="live-dot"></span>
                实时更新
              </div>
            </div>

            <div class="metrics-stats">
              <div class="stat-box highlight">
                <span class="stat-label">Reward Mean</span>
                <span class="stat-value">0.823</span>
                <span class="stat-change positive">↑ 12.5%</span>
              </div>
              <div class="stat-box">
                <span class="stat-label">KL Divergence</span>
                <span class="stat-value">0.018</span>
                <span class="stat-change neutral">稳定</span>
              </div>
              <div class="stat-box">
                <span class="stat-label">Policy Loss</span>
                <span class="stat-value">0.089</span>
                <span class="stat-change positive">↓ 8.2%</span>
              </div>
            </div>

            <div class="chart-placeholder">
              <div class="chart-bar" style="height: 30%"></div>
              <div class="chart-bar" style="height: 45%"></div>
              <div class="chart-bar" style="height: 40%"></div>
              <div class="chart-bar" style="height: 60%"></div>
              <div class="chart-bar" style="height: 55%"></div>
              <div class="chart-bar" style="height: 70%"></div>
              <div class="chart-bar" style="height: 65%"></div>
              <div class="chart-bar" style="height: 80%"></div>
              <div class="chart-bar" style="height: 75%"></div>
              <div class="chart-bar active" style="height: 85%"></div>
            </div>
          </div>

          <!-- Benchmark Results -->
          <div class="metrics-card">
            <div class="metrics-card-header">
              <h3>Benchmark 对比</h3>
              <span class="metrics-subtitle">Qwen2.5-7B 训练前后</span>
            </div>

            <div class="benchmark-list">
              <div class="benchmark-item">
                <div class="benchmark-info">
                  <span class="benchmark-name">GSM8K</span>
                  <div class="benchmark-scores">
                    <span class="score-before">58.3%</span>
                    <span class="score-arrow">→</span>
                    <span class="score-after">82.3%</span>
                    <span class="score-change">+24%</span>
                  </div>
                </div>
                <div class="benchmark-bar">
                  <div class="benchmark-fill" style="width: 82%"></div>
                </div>
              </div>

              <div class="benchmark-item">
                <div class="benchmark-info">
                  <span class="benchmark-name">MATH</span>
                  <div class="benchmark-scores">
                    <span class="score-before">38.5%</span>
                    <span class="score-arrow">→</span>
                    <span class="score-after">48.5%</span>
                    <span class="score-change">+10%</span>
                  </div>
                </div>
                <div class="benchmark-bar">
                  <div class="benchmark-fill" style="width: 48%"></div>
                </div>
              </div>

              <div class="benchmark-item">
                <div class="benchmark-info">
                  <span class="benchmark-name">HumanEval</span>
                  <div class="benchmark-scores">
                    <span class="score-before">68.9%</span>
                    <span class="score-arrow">→</span>
                    <span class="score-after">73.2%</span>
                    <span class="score-change">+4.3%</span>
                  </div>
                </div>
                <div class="benchmark-bar">
                  <div class="benchmark-fill" style="width: 73%"></div>
                </div>
              </div>

              <div class="benchmark-item">
                <div class="benchmark-info">
                  <span class="benchmark-name">MMLU</span>
                  <div class="benchmark-scores">
                    <span class="score-before">70.2%</span>
                    <span class="score-arrow">→</span>
                    <span class="score-after">72.8%</span>
                    <span class="score-change">+2.6%</span>
                  </div>
                </div>
                <div class="benchmark-bar">
                  <div class="benchmark-fill" style="width: 73%"></div>
                </div>
              </div>
            </div>

            <div class="benchmark-summary">
              <span class="summary-label">平均提升</span>
              <span class="summary-value">+10.2%</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="cta">
      <div class="cta-glow"></div>
      <div class="cta-content">
        <h2>
          准备好让训练
          <span class="gradient-text">科学起来</span>
          了吗？
        </h2>
        <p>告别玄学调参，拥抱数据驱动的 LLM 训练新范式</p>

        <div class="cta-buttons">
          <button class="btn-primary large" @click="enterPlatform">
            立即体验
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
            </svg>
          </button>
          <a href="https://github.com/volcengine/verl" target="_blank" class="btn-secondary large">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            GitHub
          </a>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
      <div class="footer-content">
        <div class="footer-brand">
          <div class="logo">
            <div class="logo-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3 2.5 2.5 0 0 0-1.32 4.24 3 3 0 0 0 .34 5.58 2.5 2.5 0 0 0 2.96 3.08A2.5 2.5 0 0 0 12 19.5a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 12 4.5"/>
              </svg>
            </div>
            <span class="logo-text">TRAIN.AI</span>
          </div>
        </div>

        <p class="footer-slogan">让 LLM Post-Train 成为 Science，而不是 Sense</p>

        <div class="footer-status">
          <span class="status-dot"></span>
          基于 <a href="https://github.com/volcengine/verl" target="_blank">verl</a> 框架
        </div>
      </div>

      <div class="footer-bottom">
        <p>© 2024 verl.ai. Built with verl framework.</p>
      </div>
    </footer>
  </div>
</template>

<style scoped>
/* ========== Variables ========== */
:root {
  --primary: #10b981;
  --primary-light: #34d399;
  --primary-dark: #059669;
}

/* ========== Base ========== */
.landing-page {
  background: #050505;
  color: #e5e5e5;
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
}

/* ========== Navigation ========== */
.nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  background: rgba(5, 5, 5, 0.85);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 32px;
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
}

.logo-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.logo-text {
  font-size: 20px;
  font-weight: 600;
  color: white;
  letter-spacing: -0.5px;
}

.nav-links {
  display: flex;
  gap: 40px;
}

.nav-links a {
  color: #737373;
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: color 0.2s;
}

.nav-links a:hover {
  color: #10b981;
}

.nav-cta {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 24px;
  border-radius: 8px;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
  color: #34d399;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.nav-cta:hover {
  background: rgba(16, 185, 129, 0.2);
  border-color: rgba(16, 185, 129, 0.5);
}

/* ========== Hero ========== */
.hero {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 120px 32px 80px;
  overflow: hidden;
}

.glow-blob {
  position: absolute;
  width: 800px;
  height: 800px;
  background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 60%);
  top: -200px;
  left: 50%;
  transform: translateX(-50%);
  filter: blur(100px);
  pointer-events: none;
}

.glow-blob-2 {
  position: absolute;
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, transparent 60%);
  bottom: 0;
  right: -100px;
  filter: blur(80px);
  pointer-events: none;
}

.hero-content {
  position: relative;
  text-align: center;
  max-width: 900px;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 20px;
  border-radius: 100px;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.25);
  font-size: 13px;
  font-weight: 500;
  color: #34d399;
  margin-bottom: 40px;
}

.badge-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.6; transform: scale(0.9); }
}

.hero-title {
  font-size: 64px;
  font-weight: 600;
  color: white;
  line-height: 1.2;
  letter-spacing: -2px;
  margin-bottom: 32px;
}

.gradient-text {
  display: block;
  background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 8px 0;
}

.hero-title .subtitle {
  display: block;
  color: #525252;
  font-size: 48px;
  font-weight: 400;
  margin-top: 8px;
}

.hero-desc {
  font-size: 18px;
  color: #a3a3a3;
  line-height: 1.8;
  margin-bottom: 48px;
}

.hero-cta {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-bottom: 64px;
}

.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 52px;
  padding: 0 32px;
  border-radius: 10px;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
  font-size: 15px;
  font-weight: 600;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 24px rgba(16, 185, 129, 0.3);
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 32px rgba(16, 185, 129, 0.4);
}

.btn-secondary {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 52px;
  padding: 0 32px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 15px;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-secondary:hover {
  background: rgba(255, 255, 255, 0.1);
}

.tech-stack {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 12px;
}

.tech-label {
  color: #525252;
  font-size: 13px;
}

.tech-tag {
  padding: 8px 16px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #a3a3a3;
  font-size: 13px;
  font-weight: 500;
}

.tech-tag.highlight {
  background: rgba(16, 185, 129, 0.1);
  border-color: rgba(16, 185, 129, 0.2);
  color: #34d399;
}

/* ========== Sections ========== */
.section-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 32px;
}

.section-header {
  text-align: center;
  margin-bottom: 64px;
}

.section-header h2 {
  font-size: 40px;
  font-weight: 600;
  color: white;
  letter-spacing: -1px;
  margin-bottom: 16px;
}

.section-header p {
  font-size: 18px;
  color: #737373;
}

/* ========== Workflow ========== */
.workflow {
  padding: 120px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.pipeline {
  display: flex;
  flex-wrap: nowrap;
  justify-content: center;
  align-items: flex-start;
  gap: 12px;
}

.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 20px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  transition: all 0.3s;
  width: 130px;
  flex-shrink: 0;
}

.pipeline-step:hover, .pipeline-step.active {
  background: rgba(16, 185, 129, 0.05);
  border-color: rgba(16, 185, 129, 0.2);
}

.pipeline-step.active .step-icon {
  background: rgba(16, 185, 129, 0.15);
  color: #34d399;
}

.step-icon {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #737373;
  margin-bottom: 16px;
  transition: all 0.3s;
}

.pipeline-step:hover .step-icon {
  color: #34d399;
  background: rgba(16, 185, 129, 0.1);
}

.step-title {
  font-size: 14px;
  font-weight: 600;
  color: #e5e5e5;
  margin-bottom: 4px;
}

.step-desc {
  font-size: 12px;
  color: #525252;
}

.pipeline-arrow {
  color: #404040;
  font-size: 18px;
  padding-top: 38px;
  flex-shrink: 0;
}

/* ========== Features ========== */
.features {
  padding: 120px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

.feature-card {
  padding: 32px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  transition: all 0.3s;
}

.feature-card:hover {
  border-color: rgba(16, 185, 129, 0.3);
  background: rgba(16, 185, 129, 0.03);
}

.feature-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: rgba(16, 185, 129, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #34d399;
  margin-bottom: 20px;
}

.feature-card h3 {
  font-size: 18px;
  font-weight: 600;
  color: white;
  margin-bottom: 12px;
}

.feature-card p {
  font-size: 14px;
  color: #737373;
  line-height: 1.7;
  margin-bottom: 20px;
}

.feature-tags {
  display: flex;
  gap: 8px;
}

.feature-tags span {
  padding: 6px 12px;
  border-radius: 6px;
  background: rgba(16, 185, 129, 0.1);
  color: #34d399;
  font-size: 12px;
  font-weight: 500;
}

/* ========== Algorithms ========== */
.algorithms {
  padding: 120px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  background: linear-gradient(180deg, transparent 0%, rgba(16, 185, 129, 0.02) 50%, transparent 100%);
}

.algo-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

.algo-card {
  padding: 28px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  transition: all 0.3s;
  position: relative;
}

.algo-card:hover {
  border-color: rgba(16, 185, 129, 0.2);
}

.algo-card.featured {
  background: rgba(16, 185, 129, 0.05);
  border-color: rgba(16, 185, 129, 0.2);
}

.algo-recommend {
  position: absolute;
  top: -10px;
  right: 20px;
  padding: 4px 12px;
  border-radius: 100px;
  background: #10b981;
  color: white;
  font-size: 11px;
  font-weight: 600;
}

.algo-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}

.algo-badge {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  background: rgba(16, 185, 129, 0.15);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #34d399;
  font-size: 14px;
  font-weight: 700;
}

.algo-title h3 {
  font-size: 16px;
  font-weight: 600;
  color: white;
  margin-bottom: 2px;
}

.algo-title span {
  font-size: 11px;
  color: #525252;
}

.algo-card > p {
  font-size: 14px;
  color: #737373;
  line-height: 1.6;
  margin-bottom: 16px;
}

.algo-use {
  font-size: 12px;
  color: #525252;
}

/* ========== Metrics ========== */
.metrics {
  padding: 120px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
}

.metrics-card {
  padding: 28px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.metrics-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.metrics-card-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: white;
}

.live-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #34d399;
}

.live-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse 2s infinite;
}

.metrics-subtitle {
  font-size: 12px;
  color: #525252;
}

.metrics-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-box {
  padding: 16px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.03);
}

.stat-box.highlight {
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.stat-label {
  display: block;
  font-size: 11px;
  color: #525252;
  margin-bottom: 8px;
}

.stat-value {
  display: block;
  font-size: 24px;
  font-weight: 600;
  color: white;
}

.stat-change {
  display: block;
  font-size: 12px;
  margin-top: 8px;
}

.stat-change.positive { color: #34d399; }
.stat-change.neutral { color: #737373; }

.chart-placeholder {
  height: 140px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.02);
  display: flex;
  align-items: flex-end;
  padding: 16px;
  gap: 6px;
}

.chart-bar {
  flex: 1;
  border-radius: 4px;
  background: rgba(16, 185, 129, 0.2);
  transition: all 0.3s;
}

.chart-bar:hover { background: rgba(16, 185, 129, 0.4); }
.chart-bar.active { background: linear-gradient(180deg, #10b981, rgba(16, 185, 129, 0.4)); }

.benchmark-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.benchmark-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.benchmark-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.benchmark-name {
  font-size: 14px;
  color: #e5e5e5;
  font-weight: 500;
}

.benchmark-scores {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.score-before { color: #525252; }
.score-arrow { color: #404040; }
.score-after { color: #34d399; font-weight: 600; }
.score-change {
  padding: 2px 8px;
  border-radius: 100px;
  background: rgba(16, 185, 129, 0.15);
  color: #34d399;
  font-size: 12px;
  font-weight: 600;
}

.benchmark-bar {
  height: 6px;
  border-radius: 100px;
  background: rgba(255, 255, 255, 0.06);
}

.benchmark-fill {
  height: 100%;
  border-radius: 100px;
  background: linear-gradient(90deg, #059669, #34d399);
}

.benchmark-summary {
  margin-top: 28px;
  padding-top: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.summary-label { font-size: 14px; color: #737373; }
.summary-value { font-size: 28px; font-weight: 600; color: #34d399; }

/* ========== CTA ========== */
.cta {
  position: relative;
  padding: 160px 32px;
  text-align: center;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  overflow: hidden;
}

.cta-glow {
  position: absolute;
  width: 600px;
  height: 600px;
  background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 60%);
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  filter: blur(80px);
  pointer-events: none;
}

.cta-content {
  position: relative;
  max-width: 700px;
  margin: 0 auto;
}

.cta h2 {
  font-size: 48px;
  font-weight: 600;
  color: white;
  line-height: 1.3;
  letter-spacing: -1px;
  margin-bottom: 20px;
}

.cta p {
  font-size: 18px;
  color: #737373;
  margin-bottom: 40px;
}

.cta-buttons {
  display: flex;
  justify-content: center;
  gap: 16px;
}

.btn-primary.large, .btn-secondary.large {
  height: 56px;
  padding: 0 36px;
  font-size: 16px;
}

/* ========== Footer ========== */
.footer {
  padding: 48px 32px 32px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 24px;
}

.footer-slogan {
  font-size: 14px;
  color: #525252;
}

.footer-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #525252;
}

.footer-status a {
  color: #34d399;
  text-decoration: none;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
}

.footer-bottom {
  max-width: 1200px;
  margin: 32px auto 0;
  padding-top: 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  text-align: center;
}

.footer-bottom p {
  font-size: 13px;
  color: #404040;
}

/* ========== Responsive ========== */
@media (max-width: 1100px) {
  .pipeline {
    overflow-x: auto;
    justify-content: flex-start;
    padding-bottom: 16px;
    margin: 0 -32px;
    padding-left: 32px;
    padding-right: 32px;
  }

  .pipeline::-webkit-scrollbar {
    height: 4px;
  }

  .pipeline::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 2px;
  }

  .pipeline::-webkit-scrollbar-thumb {
    background: rgba(16, 185, 129, 0.3);
    border-radius: 2px;
  }
}

@media (max-width: 1024px) {
  .features-grid, .algo-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .nav-links { display: none; }

  .hero-title {
    font-size: 40px;
    letter-spacing: -1px;
  }

  .hero-title .subtitle {
    font-size: 32px;
  }

  .pipeline {
    flex-direction: column;
    align-items: center;
    overflow-x: visible;
    margin: 0;
    padding-left: 0;
    padding-right: 0;
  }

  .pipeline-step {
    width: 100%;
    max-width: 200px;
  }

  .pipeline-arrow {
    transform: rotate(90deg);
    padding: 8px 0;
  }

  .features-grid, .algo-grid, .metrics-grid {
    grid-template-columns: 1fr;
  }

  .metrics-stats {
    grid-template-columns: 1fr;
  }

  .section-header h2 {
    font-size: 32px;
  }

  .cta h2 {
    font-size: 32px;
  }
}
</style>
