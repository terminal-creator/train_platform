<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useJobsStore } from '@/stores/jobs'
import { useAppStore } from '@/stores/app'
import {
  getJobMetrics,
  getGpuUsage,
  getGradientStats,
  getGradientHeatmap,
  getEvaluations,
  connectMetricsWs,
  getJobAnomalies,
  getJobHealth,
  diagnoseJob,
  syncJobMetrics
} from '@/api'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import {
  Activity,
  Cpu,
  TrendingUp,
  BarChart3,
  RefreshCw,
  Zap,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info,
  Clock,
  Filter
} from 'lucide-vue-next'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const route = useRoute()
const jobsStore = useJobsStore()
const appStore = useAppStore()

// ============ DEMO MODE: Mock Data Generation ============
const DEMO_MODE = false // Set to true to enable mock data

// Generate realistic training metrics
const generateMockMetrics = (numSteps = 500) => {
  const metrics = []
  let baseLoss = 2.5 + Math.random() * 0.5
  let baseReward = -0.5 + Math.random() * 0.2
  let baseKL = 0.001

  for (let i = 0; i < numSteps; i++) {
    const step = i * 10
    const progress = i / numSteps

    // Loss decreases with some noise and occasional spikes
    const lossDecay = Math.exp(-progress * 2)
    const lossNoise = (Math.random() - 0.5) * 0.1 * lossDecay
    const loss = baseLoss * lossDecay + lossNoise + 0.3

    // Reward increases with training
    const rewardGrowth = progress * 2.5
    const rewardNoise = (Math.random() - 0.5) * 0.15
    const reward = baseReward + rewardGrowth + rewardNoise

    // KL divergence increases then stabilizes
    const klGrowth = Math.min(0.02, progress * 0.03)
    const klNoise = Math.random() * 0.002
    const kl = baseKL + klGrowth + klNoise

    // Gradient norms with occasional spikes
    const actorGrad = 0.5 + Math.random() * 0.3 + (Math.random() > 0.95 ? 2 : 0)
    const criticGrad = 0.8 + Math.random() * 0.4 + (Math.random() > 0.95 ? 1.5 : 0)

    // Throughput with warm-up effect
    const warmup = Math.min(1, i / 50)
    const throughput = (15000 + Math.random() * 3000) * warmup

    // GPU memory usage
    const gpuMemory = 65 + Math.random() * 10

    metrics.push({
      step,
      timestamp: new Date(Date.now() - (numSteps - i) * 60000).toISOString(),
      loss: { total_loss: loss, policy_loss: loss * 0.7, value_loss: loss * 0.3 },
      reward: { mean: reward, std: 0.3 + Math.random() * 0.1, min: reward - 1, max: reward + 1.5 },
      kl: { mean: kl, max: kl * 1.5 },
      gradient: { actor_norm: actorGrad, critic_norm: criticGrad },
      performance: { tokens_per_second: throughput, gpu_memory_allocated: gpuMemory },
      total_loss: loss,
      reward_mean: reward,
      kl_divergence: kl,
      grad_norm_actor: actorGrad,
      grad_norm_critic: criticGrad,
      tokens_per_second: throughput,
      gpu_memory_allocated_gib: gpuMemory
    })
  }
  return metrics
}

// Generate GPU usage for a large cluster
const generateMockGpuUsage = (numGpus = 64) => {
  const gpus = []
  for (let i = 0; i < numGpus; i++) {
    // Most GPUs should be highly utilized
    const baseUtil = 85 + Math.random() * 15
    const util = Math.min(100, Math.max(0, baseUtil - (Math.random() > 0.9 ? 20 : 0)))
    gpus.push({
      index: i,
      utilization: Math.round(util),
      memory_used: 70 + Math.random() * 8,
      memory_total: 80,
      name: 'NVIDIA A100-SXM4-80GB'
    })
  }
  return gpus
}

// Generate gradient stats per layer
const generateMockGradientStats = () => {
  const layers = [
    'model.embed_tokens',
    'model.layers.0.self_attn.q_proj',
    'model.layers.0.self_attn.k_proj',
    'model.layers.0.self_attn.v_proj',
    'model.layers.0.self_attn.o_proj',
    'model.layers.0.mlp.gate_proj',
    'model.layers.0.mlp.up_proj',
    'model.layers.0.mlp.down_proj',
    'model.layers.15.self_attn.q_proj',
    'model.layers.15.mlp.gate_proj',
    'model.layers.31.self_attn.q_proj',
    'model.layers.31.mlp.down_proj',
    'model.norm',
    'lm_head'
  ]

  return layers.map(layer => ({
    layer_name: layer,
    mean: (Math.random() - 0.5) * 0.001,
    std: Math.random() * 0.01 + 0.001,
    min: -0.05 - Math.random() * 0.02,
    max: 0.05 + Math.random() * 0.02,
    norm: Math.random() * 0.5 + 0.1
  }))
}

// Generate health status
const generateMockHealth = () => ({
  status: 'healthy',
  health_score: 92 + Math.floor(Math.random() * 6),
  checked_at: new Date().toISOString(),
  metrics_analyzed: 500,
  suggestions: [
    '训练进展顺利，loss 稳定下降',
    '建议在 step 6000 时进行 checkpoint 保存',
    'KL 散度在正常范围内'
  ]
})

// Generate some minor anomalies (to show the detection is working)
const generateMockAnomalies = () => [
  {
    severity: 'low',
    message: 'Step 2340 梯度范数略高于平均值 (2.1x)',
    step: 2340,
    detected_at: new Date(Date.now() - 300000).toISOString()
  },
  {
    severity: 'low',
    message: 'GPU 47 利用率短暂下降至 65%',
    step: 3120,
    detected_at: new Date(Date.now() - 180000).toISOString()
  }
]

// Mock job for demo
const mockJob = {
  id: 'demo-job-001',
  name: 'Qwen2.5-7B-GRPO-Sales',
  status: 'running',
  algorithm: 'grpo',
  current_step: 4980,
  total_steps: 10000,
  model: 'Qwen/Qwen2.5-7B',
  created_at: new Date(Date.now() - 3600000 * 8).toISOString()
}
// ============ END DEMO MODE ============

// Load from route params, or localStorage, or empty
const savedJobId = localStorage.getItem('monitoring_selected_job') || ''
const selectedJobId = ref(route.params.jobId || savedJobId)
const metrics = ref([])
const gpuUsage = ref([])
const gradientStats = ref([])
const gradientHeatmap = ref(null)
const evaluations = ref([])
const loading = ref(false)
const autoRefresh = ref(true)
const wsConnection = ref(null)

// Phase 1.5: Anomaly detection and health
const anomalies = ref([])
const health = ref(null)
const loadingAnomalies = ref(false)
const showAnomalyAlert = ref(true)

// Phase 1.5: Multi-metric chart switching
const availableMetrics = [
  { id: 'loss', label: '损失曲线', icon: TrendingUp },
  { id: 'reward', label: '奖励曲线', icon: Zap },
  { id: 'kl', label: 'KL 散度', icon: BarChart3 },
  { id: 'gradient', label: '梯度范数', icon: Activity },
  { id: 'tokens_per_second', label: '吞吐量', icon: Clock },
  { id: 'gpu_memory', label: 'GPU 显存', icon: Cpu }
]
const selectedChartMetrics = ref(DEMO_MODE ? ['loss', 'reward', 'kl', 'gradient'] : ['loss', 'reward']) // Demo shows more metrics

// Phase 1.5: Historical metrics view
const showHistoryMode = ref(false)
const historyStartStep = ref(0)
const historyEndStep = ref(null)
const historyMetrics = ref([])

const selectedJob = computed(() => {
  if (DEMO_MODE && selectedJobId.value === mockJob.id) {
    return mockJob
  }
  return jobsStore.jobs.find(j => j.id === selectedJobId.value)
})

// Check if job is SFT (not RL algorithm)
const isSFT = computed(() => {
  return selectedJob.value?.algorithm === 'sft'
})

const latestMetrics = computed(() => {
  const data = showHistoryMode.value ? historyMetrics.value : metrics.value
  if (data.length === 0) return null
  return data[data.length - 1]
})

// Get current metrics data (live or history)
const currentMetrics = computed(() => {
  return showHistoryMode.value ? historyMetrics.value : metrics.value
})

// Anomaly severity badge color
const getAnomalySeverityColor = (severity) => {
  switch (severity) {
    case 'critical': return 'bg-red-500 text-white'
    case 'high': return 'bg-orange-500 text-white'
    case 'medium': return 'bg-yellow-500 text-black'
    case 'low': return 'bg-blue-500 text-white'
    default: return 'bg-gray-500 text-white'
  }
}

// Health score color
const getHealthScoreColor = (score) => {
  if (score >= 90) return 'text-green-500'
  if (score >= 70) return 'text-yellow-500'
  if (score >= 50) return 'text-orange-500'
  return 'text-red-500'
}

// Generate chart data based on selected metric
const getChartDataForMetric = (metricId) => {
  const data = currentMetrics.value

  switch (metricId) {
    case 'loss':
      return {
        labels: data.map(m => m.step),
        datasets: [{
          label: '总损失',
          data: data.map(m => m.loss?.total_loss || m.total_loss),
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          tension: 0.4
        }]
      }

    case 'reward':
      return {
        labels: data.map(m => m.step),
        datasets: [{
          label: '平均奖励',
          data: data.map(m => m.reward?.mean || m.reward_mean),
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          fill: true,
          tension: 0.4
        }]
      }

    case 'kl':
      return {
        labels: data.map(m => m.step),
        datasets: [{
          label: 'KL 散度',
          data: data.map(m => m.kl?.mean || m.kl_divergence),
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          fill: true,
          tension: 0.4
        }]
      }

    case 'gradient':
      return {
        labels: data.map(m => m.step),
        datasets: [
          {
            label: 'Actor 梯度范数',
            data: data.map(m => m.gradient?.actor_norm || m.grad_norm_actor),
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            fill: true,
            tension: 0.4
          },
          {
            label: 'Critic 梯度范数',
            data: data.map(m => m.gradient?.critic_norm || m.grad_norm_critic),
            borderColor: '#ec4899',
            backgroundColor: 'rgba(236, 72, 153, 0.1)',
            fill: true,
            tension: 0.4
          }
        ]
      }

    case 'tokens_per_second':
      return {
        labels: data.map(m => m.step),
        datasets: [{
          label: '吞吐量 (tokens/s)',
          data: data.map(m => m.performance?.tokens_per_second || m.tokens_per_second),
          borderColor: '#14b8a6',
          backgroundColor: 'rgba(20, 184, 166, 0.1)',
          fill: true,
          tension: 0.4
        }]
      }

    case 'gpu_memory':
      return {
        labels: data.map(m => m.step),
        datasets: [{
          label: 'GPU 显存 (GiB)',
          data: data.map(m => m.performance?.gpu_memory_allocated || m.gpu_memory_allocated_gib),
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          fill: true,
          tension: 0.4
        }]
      }

    default:
      return { labels: [], datasets: [] }
  }
}

// Get metric label
const getMetricLabel = (metricId) => {
  return availableMetrics.find(m => m.id === metricId)?.label || metricId
}

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: '#6b7280' }
    }
  },
  scales: {
    x: {
      grid: { color: 'rgba(0,0,0,0.05)' },
      ticks: { color: '#6b7280' }
    },
    y: {
      grid: { color: 'rgba(0,0,0,0.05)' },
      ticks: { color: '#6b7280' }
    }
  }
}

const loadData = async () => {
  if (!selectedJobId.value) return
  try {
    loading.value = true
    const [metricsResp, gpuResp, gradientResp, heatmapResp, evalResp] = await Promise.all([
      getJobMetrics(selectedJobId.value).catch(() => ({ metrics: [] })),
      getGpuUsage(selectedJobId.value).catch(() => []),
      getGradientStats(selectedJobId.value).catch(() => ({ stats: [] })),
      getGradientHeatmap(selectedJobId.value).catch(() => null),
      getEvaluations(selectedJobId.value).catch(() => ({ results: [] }))
    ])
    // Extract data from API responses
    metrics.value = metricsResp?.metrics || metricsResp || []
    gpuUsage.value = Array.isArray(gpuResp) ? gpuResp : []
    gradientStats.value = gradientResp?.stats || []
    gradientHeatmap.value = heatmapResp
    evaluations.value = evalResp?.results || []

    console.log('Loaded monitoring data:', {
      metrics: metrics.value.length,
      gpu: gpuUsage.value.length,
      gradient: gradientStats.value.length,
      heatmap: gradientHeatmap.value?.layers?.length || 0,
      eval: evaluations.value.length
    })
  } catch (error) {
    console.error('Error loading monitoring data:', error)
    appStore.showError(error.message)
  } finally {
    loading.value = false
  }
}

// Phase 1.5: Load anomalies and health
const loadAnomaliesAndHealth = async () => {
  if (!selectedJobId.value) return
  try {
    loadingAnomalies.value = true
    const [anomaliesResp, healthResp] = await Promise.all([
      getJobAnomalies(selectedJobId.value).catch(() => ({ anomalies: [] })),
      getJobHealth(selectedJobId.value).catch(() => null)
    ])

    anomalies.value = anomaliesResp?.anomalies || []
    health.value = healthResp

    console.log('Loaded diagnostics:', {
      anomalies: anomalies.value.length,
      health: health.value
    })
  } catch (error) {
    console.error('Error loading diagnostics:', error)
  } finally {
    loadingAnomalies.value = false
  }
}

// Phase 1.5: Load historical metrics
const loadHistoricalMetrics = async () => {
  if (!selectedJobId.value) return
  try {
    loading.value = true
    const params = {}
    if (historyStartStep.value) params.start_step = historyStartStep.value
    if (historyEndStep.value) params.end_step = historyEndStep.value

    const resp = await getJobMetrics(selectedJobId.value, params)
    historyMetrics.value = resp?.metrics || resp || []

    console.log(`Loaded history: ${historyMetrics.value.length} metrics`)
  } catch (error) {
    console.error('Error loading historical metrics:', error)
    appStore.showError(error.message)
  } finally {
    loading.value = false
  }
}

// Phase 1.5: Toggle history mode
const toggleHistoryMode = async () => {
  showHistoryMode.value = !showHistoryMode.value
  if (showHistoryMode.value) {
    // Set default range to all metrics
    if (metrics.value.length > 0) {
      historyStartStep.value = metrics.value[0].step
      historyEndStep.value = metrics.value[metrics.value.length - 1].step
    }
    await loadHistoricalMetrics()
  }
}

// Phase 1.5: Sync and diagnose
const syncAndDiagnose = async () => {
  if (!selectedJobId.value) return
  try {
    loading.value = true
    await syncJobMetrics(selectedJobId.value)
    await diagnoseJob(selectedJobId.value)
    await loadData()
    await loadAnomaliesAndHealth()
    appStore.showSuccess('同步和诊断完成')
  } catch (error) {
    console.error('Error syncing and diagnosing:', error)
    appStore.showError(error.message)
  } finally {
    loading.value = false
  }
}

const connectWebSocket = () => {
  if (!selectedJobId.value || !autoRefresh.value) return

  try {
    wsConnection.value = connectMetricsWs(selectedJobId.value, (data) => {
      if (data.type === 'metrics') {
        metrics.value.push(data.metrics)
        if (metrics.value.length > 1000) {
          metrics.value = metrics.value.slice(-500)
        }
      } else if (data.type === 'gpu') {
        gpuUsage.value = data.usage
      } else if (data.type === 'anomaly') {
        // Real-time anomaly detection
        anomalies.value.push(data.anomaly)
        showAnomalyAlert.value = true
      }
    })
  } catch (error) {
    console.error('WebSocket connection failed:', error)
  }
}

const disconnectWebSocket = () => {
  if (wsConnection.value) {
    wsConnection.value.close()
    wsConnection.value = null
  }
}

const getGpuUtilColor = (util) => {
  if (util >= 90) return 'bg-emerald-500'
  if (util >= 70) return 'bg-yellow-500'
  return 'bg-red-500'
}

const getGradientColor = (norm) => {
  if (norm < 0.1) return 'text-blue-400'
  if (norm < 1) return 'text-green-400'
  if (norm < 10) return 'text-yellow-400'
  return 'text-red-400'
}

// Get heatmap cell color based on log10 gradient norm value
// Input: log10 value, typically in range [-3, 0]
// -3 (very small gradient) -> green
// -1.5 (medium) -> yellow
// 0 (large gradient) -> red
const getHeatmapColor = (value) => {
  // Normalize from [-3, 0] to [0, 1]
  const normalized = Math.min(1, Math.max(0, (value + 3) / 3))

  if (normalized < 0.5) {
    // Green to Yellow
    const g = 200
    const r = Math.round(normalized * 2 * 255)
    return `rgb(${r}, ${g}, 50)`
  } else {
    // Yellow to Red
    const r = 255
    const g = Math.round((1 - (normalized - 0.5) * 2) * 200)
    return `rgb(${r}, ${g}, 50)`
  }
}

watch(selectedJobId, (newId) => {
  // Save to localStorage for persistence across tab switches
  if (newId) {
    localStorage.setItem('monitoring_selected_job', newId)
  } else {
    localStorage.removeItem('monitoring_selected_job')
  }
  disconnectWebSocket()
  loadData()
  loadAnomaliesAndHealth()
  connectWebSocket()
})

watch(autoRefresh, (val) => {
  if (val) {
    connectWebSocket()
  } else {
    disconnectWebSocket()
  }
})

watch([historyStartStep, historyEndStep], () => {
  if (showHistoryMode.value) {
    loadHistoricalMetrics()
  }
})

onMounted(async () => {
  // ============ DEMO MODE: Inject mock data ============
  if (DEMO_MODE) {
    // Add mock job to store if not exists
    if (!jobsStore.jobs.find(j => j.id === mockJob.id)) {
      jobsStore.jobs.unshift(mockJob)
    }
    selectedJobId.value = mockJob.id

    // Generate mock data
    metrics.value = generateMockMetrics(500)
    gpuUsage.value = generateMockGpuUsage(64)
    gradientStats.value = generateMockGradientStats()
    health.value = generateMockHealth()
    anomalies.value = generateMockAnomalies()

    // Simulate live updates
    setInterval(() => {
      if (!DEMO_MODE || !autoRefresh.value) return

      const lastStep = metrics.value.length > 0 ? metrics.value[metrics.value.length - 1].step : 0
      const newStep = lastStep + 10
      const progress = newStep / 10000

      const lossDecay = Math.exp(-progress * 2)
      const loss = 2.5 * lossDecay + (Math.random() - 0.5) * 0.05 + 0.3
      const reward = -0.5 + progress * 2.5 + (Math.random() - 0.5) * 0.1
      const kl = 0.001 + Math.min(0.02, progress * 0.03) + Math.random() * 0.001

      metrics.value.push({
        step: newStep,
        timestamp: new Date().toISOString(),
        loss: { total_loss: loss },
        reward: { mean: reward },
        kl: { mean: kl },
        gradient: { actor_norm: 0.5 + Math.random() * 0.3, critic_norm: 0.8 + Math.random() * 0.4 },
        performance: { tokens_per_second: 15000 + Math.random() * 3000, gpu_memory_allocated: 65 + Math.random() * 10 },
        total_loss: loss,
        reward_mean: reward,
        kl_divergence: kl
      })

      // Keep only last 500 points
      if (metrics.value.length > 500) {
        metrics.value = metrics.value.slice(-500)
      }

      // Randomly update some GPU utilizations
      gpuUsage.value = gpuUsage.value.map(gpu => ({
        ...gpu,
        utilization: Math.min(100, Math.max(70, gpu.utilization + (Math.random() - 0.5) * 5))
      }))

      // Update mock job progress
      mockJob.current_step = newStep
    }, 2000)

    return
  }
  // ============ END DEMO MODE ============

  await jobsStore.fetchJobs()

  // If saved job ID doesn't exist in jobs list, clear it
  if (selectedJobId.value && !jobsStore.jobs.find(j => j.id === selectedJobId.value)) {
    selectedJobId.value = ''
    localStorage.removeItem('monitoring_selected_job')
  }

  // Auto-select first running job, or first job if none running
  if (!selectedJobId.value && jobsStore.jobs.length > 0) {
    const runningJob = jobsStore.jobs.find(j => j.status === 'running')
    selectedJobId.value = runningJob?.id || jobsStore.jobs[0].id
  }

  if (selectedJobId.value) {
    await loadData()
    await loadAnomaliesAndHealth()
    connectWebSocket()
  }
})

onUnmounted(() => {
  disconnectWebSocket()
})
</script>

<template>
  <div>
    <!-- Demo Mode Banner -->
    <div v-if="DEMO_MODE" class="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-2 rounded-lg mb-4 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <span class="animate-pulse w-2 h-2 bg-green-400 rounded-full"></span>
        <span class="font-medium">🚀 DEMO MODE - Qwen2.5-7B GRPO 训练实时监控</span>
      </div>
      <div class="text-sm opacity-80">
        64x A100 · 15K+ tokens/s · 8小时已运行
      </div>
    </div>

    <!-- Header Controls -->
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-4">
        <select v-model="selectedJobId" class="input-light min-w-64">
          <option value="">选择任务...</option>
          <option v-for="job in jobsStore.jobs" :key="job.id" :value="job.id">
            {{ job.name }} ({{ job.status }})
          </option>
        </select>
        <button
          @click="loadData"
          :disabled="!selectedJobId || loading"
          class="btn-secondary flex items-center gap-2"
        >
          <RefreshCw :class="['w-4 h-4', loading && 'animate-spin']" />
          刷新
        </button>
        <button
          @click="syncAndDiagnose"
          :disabled="!selectedJobId || loading"
          class="btn-primary flex items-center gap-2"
        >
          <Activity :class="['w-4 h-4', loading && 'animate-spin']" />
          同步诊断
        </button>
      </div>
      <div class="flex items-center gap-4">
        <button
          @click="toggleHistoryMode"
          :class="[
            'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
            showHistoryMode ? 'bg-primary-500 text-white' : 'bg-gray-200 text-gray-700'
          ]"
        >
          <Clock class="w-4 h-4" />
          {{ showHistoryMode ? '退出历史' : '历史回看' }}
        </button>
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" v-model="autoRefresh" class="w-4 h-4 accent-primary-500" :disabled="showHistoryMode">
          <span class="text-sm text-gray-400">自动刷新</span>
        </label>
      </div>
    </div>

    <!-- Phase 1.5: Historical Range Selector -->
    <div v-if="showHistoryMode" class="glass-card rounded-lg p-4 mb-4">
      <div class="flex items-center gap-4">
        <Filter class="w-5 h-5 text-gray-400" />
        <div class="flex items-center gap-2 flex-1">
          <label class="text-sm text-gray-400">步数范围:</label>
          <input
            v-model.number="historyStartStep"
            type="number"
            placeholder="起始步数"
            class="input-light w-32"
            min="0"
          />
          <span class="text-gray-400">-</span>
          <input
            v-model.number="historyEndStep"
            type="number"
            placeholder="结束步数"
            class="input-light w-32"
            min="0"
          />
          <span class="text-sm text-gray-500 ml-2">
            (共 {{ historyMetrics.length }} 条记录)
          </span>
        </div>
      </div>
    </div>

    <!-- Phase 1.5: Anomaly Alert Banner -->
    <div
      v-if="showAnomalyAlert && anomalies.length > 0"
      class="glass-card rounded-lg p-4 mb-4 border-l-4 border-red-500"
    >
      <div class="flex items-start gap-3">
        <AlertTriangle class="w-5 h-5 text-red-500 mt-0.5" />
        <div class="flex-1">
          <div class="flex items-center justify-between">
            <h4 class="font-medium text-red-700">检测到训练异常</h4>
            <button @click="showAnomalyAlert = false" class="text-gray-400 hover:text-gray-600">
              <XCircle class="w-4 h-4" />
            </button>
          </div>
          <div class="mt-2 space-y-2">
            <div v-for="(anomaly, idx) in anomalies.slice(0, 3)" :key="idx" class="text-sm">
              <span :class="['px-2 py-0.5 rounded text-xs font-medium mr-2', getAnomalySeverityColor(anomaly.severity)]">
                {{ anomaly.severity.toUpperCase() }}
              </span>
              <span class="text-gray-700">{{ anomaly.message }}</span>
              <span class="text-gray-400 ml-2">(Step {{ anomaly.step }})</span>
            </div>
            <div v-if="anomalies.length > 3" class="text-sm text-gray-500">
              ... 还有 {{ anomalies.length - 3 }} 个异常
            </div>
          </div>
          <div class="mt-3 flex gap-2">
            <button class="btn-sm btn-secondary">查看详情</button>
            <button @click="loadAnomaliesAndHealth" class="btn-sm btn-secondary">刷新诊断</button>
          </div>
        </div>
      </div>
    </div>

    <div v-if="!selectedJobId" class="glass-card rounded-lg p-8 text-center">
      <Activity class="w-10 h-10 text-gray-400 mx-auto mb-4" />
      <p class="text-gray-400">选择任务以查看监控数据</p>
    </div>

    <div v-else class="space-y-6">
      <!-- Health Score Card -->
      <div v-if="health" class="glass-card rounded-lg p-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <component
              :is="health.status === 'healthy' ? CheckCircle : (health.status === 'warning' ? AlertTriangle : XCircle)"
              :class="['w-6 h-6', getHealthScoreColor(health.health_score)]"
            />
            <div>
              <h3 class="font-medium">训练健康状态</h3>
              <p class="text-sm text-gray-400">最后检查: {{ new Date(health.checked_at).toLocaleString() }}</p>
            </div>
          </div>
          <div class="text-right">
            <div class="text-3xl font-bold" :class="getHealthScoreColor(health.health_score)">
              {{ health.health_score }}
            </div>
            <div class="text-sm text-gray-400">健康评分</div>
          </div>
        </div>
        <div v-if="health.suggestions && health.suggestions.length > 0" class="mt-4 pt-4 border-t border-gray-200">
          <h4 class="text-sm font-medium text-gray-700 mb-2">建议:</h4>
          <ul class="space-y-1">
            <li v-for="(suggestion, idx) in health.suggestions" :key="idx" class="text-sm text-gray-600 flex items-start gap-2">
              <Info class="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
              <span>{{ suggestion }}</span>
            </li>
          </ul>
        </div>
      </div>

      <!-- Live Metrics Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="glass-card rounded-lg p-3">
          <div class="flex items-center gap-2 text-gray-400 text-sm mb-2">
            <Activity class="w-4 h-4" />
            当前步数
          </div>
          <p class="text-lg font-semibold">
            {{ latestMetrics?.step || selectedJob?.current_step || 0 }}
            <span class="text-sm text-gray-500">/ {{ selectedJob?.total_steps || '?' }}</span>
          </p>
        </div>
        <div class="glass-card rounded-lg p-3">
          <div class="flex items-center gap-2 text-gray-400 text-sm mb-2">
            <TrendingUp class="w-4 h-4" />
            总损失
          </div>
          <p class="text-lg font-semibold">
            {{ (latestMetrics?.loss?.total_loss || latestMetrics?.total_loss)?.toFixed(4) || 'N/A' }}
          </p>
        </div>
        <div class="glass-card rounded-lg p-3">
          <div class="flex items-center gap-2 text-gray-400 text-sm mb-2">
            <Zap class="w-4 h-4" />
            平均奖励
          </div>
          <p class="text-lg font-semibold accent-text">
            {{ (latestMetrics?.reward?.mean || latestMetrics?.reward_mean)?.toFixed(3) || 'N/A' }}
          </p>
        </div>
        <div class="glass-card rounded-lg p-3">
          <div class="flex items-center gap-2 text-gray-400 text-sm mb-2">
            <BarChart3 class="w-4 h-4" />
            KL 散度
          </div>
          <p class="text-lg font-semibold">
            {{ (latestMetrics?.kl?.mean || latestMetrics?.kl_divergence)?.toFixed(4) || 'N/A' }}
          </p>
        </div>
      </div>

      <!-- Phase 1.5: Metric Selector -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-3">选择要显示的指标</h3>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="metric in availableMetrics"
            :key="metric.id"
            @click="() => {
              const idx = selectedChartMetrics.indexOf(metric.id)
              if (idx >= 0) {
                selectedChartMetrics.splice(idx, 1)
              } else {
                selectedChartMetrics.push(metric.id)
              }
            }"
            :class="[
              'flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors',
              selectedChartMetrics.includes(metric.id)
                ? 'bg-primary-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            ]"
          >
            <component :is="metric.icon" class="w-4 h-4" />
            {{ metric.label }}
          </button>
        </div>
      </div>

      <!-- Phase 1.5: Dynamic Charts -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div
          v-for="metricId in selectedChartMetrics"
          :key="metricId"
          class="glass-card rounded-lg p-4"
        >
          <h3 class="font-medium mb-4">{{ getMetricLabel(metricId) }}</h3>
          <div class="h-64">
            <Line
              v-if="currentMetrics.length > 0"
              :data="getChartDataForMetric(metricId)"
              :options="chartOptions"
            />
            <div v-else class="flex items-center justify-center h-full text-gray-500">
              暂无指标数据
            </div>
          </div>
        </div>
      </div>

      <!-- GPU Usage Grid -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-4 flex items-center gap-2">
          <Cpu class="w-4 h-4 accent-text" />
          GPU 利用率
        </h3>
        <div v-if="gpuUsage.length === 0" class="text-center py-8 text-gray-500">
          暂无 GPU 数据
        </div>
        <div v-else class="grid grid-cols-4 md:grid-cols-8 lg:grid-cols-16 gap-2">
          <div
            v-for="gpu in gpuUsage"
            :key="gpu.index"
            class="relative group"
          >
            <div
              :class="[
                'aspect-square rounded-lg flex items-center justify-center text-xs font-medium',
                getGpuUtilColor(gpu.utilization)
              ]"
              :style="{ opacity: 0.3 + (gpu.utilization / 100) * 0.7 }"
            >
              {{ gpu.index }}
            </div>
            <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-10">
              GPU {{ gpu.index }}: {{ gpu.utilization }}%
              <br>
              显存: {{ gpu.memory_used?.toFixed(1) || 0 }}/{{ gpu.memory_total?.toFixed(1) || 80 }}GB
            </div>
          </div>
        </div>
        <div class="flex items-center gap-4 mt-4 text-sm text-gray-400">
          <span class="flex items-center gap-1">
            <span class="w-3 h-3 rounded bg-emerald-500"></span> &gt;90%
          </span>
          <span class="flex items-center gap-1">
            <span class="w-3 h-3 rounded bg-yellow-500"></span> 70-90%
          </span>
          <span class="flex items-center gap-1">
            <span class="w-3 h-3 rounded bg-red-500"></span> &lt;70%
          </span>
        </div>
      </div>

      <!-- Gradient Heatmap -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-4">梯度热力图</h3>
        <div v-if="!gradientHeatmap || !gradientHeatmap.layers" class="text-center py-8 text-gray-500">
          暂无热力图数据
        </div>
        <div v-else class="overflow-x-auto">
          <!-- Heatmap Legend -->
          <div class="flex items-center justify-end gap-4 mb-3 text-xs text-gray-500">
            <span>梯度范数 (log10):</span>
            <div class="flex items-center gap-1">
              <span class="w-4 h-4 rounded" style="background: rgb(50, 200, 50)"></span>
              <span>低</span>
            </div>
            <div class="flex items-center gap-1">
              <span class="w-4 h-4 rounded" style="background: rgb(255, 200, 50)"></span>
              <span>中</span>
            </div>
            <div class="flex items-center gap-1">
              <span class="w-4 h-4 rounded" style="background: rgb(255, 50, 50)"></span>
              <span>高</span>
            </div>
          </div>
          <!-- Heatmap Grid -->
          <div class="relative">
            <!-- Y-axis labels (layers) -->
            <div class="flex">
              <div class="w-24 flex-shrink-0"></div>
              <div class="flex-1 flex justify-between text-2xs text-gray-400 mb-1 px-1">
                <span>Step {{ gradientHeatmap.steps[0] }}</span>
                <span>Step {{ gradientHeatmap.steps[Math.floor(gradientHeatmap.steps.length / 2)] }}</span>
                <span>Step {{ gradientHeatmap.steps[gradientHeatmap.steps.length - 1] }}</span>
              </div>
            </div>
            <!-- Heatmap rows -->
            <div class="space-y-px max-h-80 overflow-y-auto">
              <div
                v-for="(row, layerIdx) in gradientHeatmap.values.slice(0, 20)"
                :key="layerIdx"
                class="flex items-center"
              >
                <div class="w-24 flex-shrink-0 text-2xs text-gray-500 truncate pr-2">
                  {{ gradientHeatmap.layers[layerIdx] }}
                </div>
                <div class="flex-1 flex gap-px">
                  <div
                    v-for="(value, stepIdx) in row"
                    :key="stepIdx"
                    class="flex-1 h-3 rounded-sm cursor-pointer hover:opacity-80"
                    :style="{ backgroundColor: getHeatmapColor(value) }"
                    :title="`${gradientHeatmap.layers[layerIdx]} @ Step ${gradientHeatmap.steps[stepIdx]}: ${value.toFixed(4)}`"
                  ></div>
                </div>
              </div>
            </div>
            <div v-if="gradientHeatmap.layers.length > 20" class="text-center text-2xs text-gray-400 mt-2">
              显示前20层，共 {{ gradientHeatmap.layers.length }} 层
            </div>
          </div>
        </div>
      </div>

      <!-- Gradient Stats -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-4">各层梯度统计</h3>
        <div v-if="gradientStats.length === 0" class="text-center py-8 text-gray-500">
          暂无梯度数据
        </div>
        <div v-else class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-gray-400 border-b border-gray-200">
                <th class="pb-3">层名称</th>
                <th class="pb-3 text-right">梯度范数</th>
                <th class="pb-3 text-right">均值</th>
                <th class="pb-3 text-right">标准差</th>
                <th class="pb-3 text-right">最大值</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="stat in gradientStats"
                :key="stat.layer_name"
                class="border-b border-gray-100"
              >
                <td class="py-2 font-mono text-xs text-gray-700">{{ stat.layer_name || 'Unknown' }}</td>
                <td :class="['py-2 text-right font-medium', getGradientColor(Math.abs(stat.mean || 0))]">
                  {{ stat.std?.toExponential(2) || 'N/A' }}
                </td>
                <td class="py-2 text-right text-gray-400">
                  {{ stat.mean?.toExponential(2) || 'N/A' }}
                </td>
                <td class="py-2 text-right text-gray-400">
                  {{ stat.std?.toExponential(2) || 'N/A' }}
                </td>
                <td class="py-2 text-right text-gray-400">
                  {{ stat.max?.toExponential(2) || 'N/A' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>
