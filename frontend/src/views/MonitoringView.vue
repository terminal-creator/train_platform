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
  connectMetricsWs
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
  Zap
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

const selectedJob = computed(() => {
  return jobsStore.jobs.find(j => j.id === selectedJobId.value)
})

// Check if job is SFT (not RL algorithm)
const isSFT = computed(() => {
  return selectedJob.value?.algorithm === 'sft'
})

const latestMetrics = computed(() => {
  if (metrics.value.length === 0) return null
  return metrics.value[metrics.value.length - 1]
})

const lossChartData = computed(() => {
  // For SFT, show training loss (use policy_loss as total loss, or total_loss if available)
  if (isSFT.value) {
    return {
      labels: metrics.value.map(m => m.step),
      datasets: [
        {
          label: '训练损失',
          data: metrics.value.map(m => m.total_loss || m.policy_loss),
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          tension: 0.4
        }
      ]
    }
  }
  // For RL algorithms, show policy and value loss
  return {
    labels: metrics.value.map(m => m.step),
    datasets: [
      {
        label: '策略损失',
        data: metrics.value.map(m => m.policy_loss),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: '价值损失',
        data: metrics.value.map(m => m.value_loss),
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  }
})

const rewardChartData = computed(() => {
  // For SFT, show perplexity or entropy instead of reward
  if (isSFT.value) {
    return {
      labels: metrics.value.map(m => m.step),
      datasets: [
        {
          label: '困惑度 (Perplexity)',
          data: metrics.value.map(m => m.entropy ? Math.exp(m.entropy) : null),
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          fill: true,
          tension: 0.4
        }
      ]
    }
  }
  return {
    labels: metrics.value.map(m => m.step),
    datasets: [
      {
        label: '平均奖励',
        data: metrics.value.map(m => m.reward_mean),
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  }
})

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

// Get heatmap cell color based on normalized value (0-1)
const getHeatmapColor = (value) => {
  // Green to yellow to red gradient
  const normalized = Math.min(1, Math.max(0, value * 10)) // Scale for better visibility
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
  connectWebSocket()
})

watch(autoRefresh, (val) => {
  if (val) {
    connectWebSocket()
  } else {
    disconnectWebSocket()
  }
})

onMounted(async () => {
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
    connectWebSocket()
  }
})

onUnmounted(() => {
  disconnectWebSocket()
})
</script>

<template>
  <div>
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
      </div>
      <label class="flex items-center gap-2 cursor-pointer">
        <input type="checkbox" v-model="autoRefresh" class="w-4 h-4 accent-primary-500">
        <span class="text-sm text-gray-400">自动刷新</span>
      </label>
    </div>

    <div v-if="!selectedJobId" class="glass-card rounded-lg p-8 text-center">
      <Activity class="w-10 h-10 text-gray-400 mx-auto mb-4" />
      <p class="text-gray-400">选择任务以查看监控数据</p>
    </div>

    <div v-else class="space-y-6">
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
            {{ isSFT ? '训练损失' : '策略损失' }}
          </div>
          <p class="text-lg font-semibold">
            {{ (isSFT ? (latestMetrics?.total_loss || latestMetrics?.policy_loss) : latestMetrics?.policy_loss)?.toFixed(4) || 'N/A' }}
          </p>
        </div>
        <div class="glass-card rounded-lg p-3">
          <div class="flex items-center gap-2 text-gray-400 text-sm mb-2">
            <Zap class="w-4 h-4" />
            {{ isSFT ? '困惑度' : '平均奖励' }}
          </div>
          <p class="text-lg font-semibold accent-text">
            {{ isSFT
              ? (latestMetrics?.entropy ? Math.exp(latestMetrics.entropy).toFixed(2) : 'N/A')
              : (latestMetrics?.reward_mean?.toFixed(3) || 'N/A')
            }}
          </p>
        </div>
        <div class="glass-card rounded-lg p-3">
          <div class="flex items-center gap-2 text-gray-400 text-sm mb-2">
            <BarChart3 class="w-4 h-4" />
            {{ isSFT ? '学习率' : 'KL 散度' }}
          </div>
          <p class="text-lg font-semibold">
            {{ isSFT
              ? (latestMetrics?.learning_rate?.toExponential(2) || 'N/A')
              : (latestMetrics?.kl_divergence?.toFixed(4) || 'N/A')
            }}
          </p>
        </div>
      </div>

      <!-- Charts -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div class="glass-card rounded-lg p-4">
          <h3 class="font-medium mb-4">损失曲线</h3>
          <div class="h-64">
            <Line v-if="metrics.length > 0" :data="lossChartData" :options="chartOptions" />
            <div v-else class="flex items-center justify-center h-full text-gray-500">
              暂无指标数据
            </div>
          </div>
        </div>
        <div class="glass-card rounded-lg p-4">
          <h3 class="font-medium mb-4">奖励曲线</h3>
          <div class="h-64">
            <Line v-if="metrics.length > 0" :data="rewardChartData" :options="chartOptions" />
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

      <!-- Layer Parameter Change Heatmap -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-4">层参数变化热力图</h3>
        <p class="text-sm text-gray-500 mb-4">显示各层在训练过程中的参数变化幅度，颜色越深表示变化越大</p>
        <div v-if="!gradientHeatmap || !gradientHeatmap.layers" class="text-center py-8 text-gray-500">
          暂无热力图数据
        </div>
        <div v-else class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-gray-400 border-b border-gray-200">
                <th class="pb-3 sticky left-0 bg-white/90">层名称</th>
                <th
                  v-for="step in gradientHeatmap.steps"
                  :key="step"
                  class="pb-3 text-center px-2 min-w-12"
                >
                  {{ step }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(layer, layerIdx) in gradientHeatmap.layers"
                :key="layer"
                class="border-b border-gray-100"
              >
                <td class="py-2 font-mono text-xs text-gray-700 sticky left-0 bg-white/90 whitespace-nowrap pr-4">
                  {{ layer }}
                </td>
                <td
                  v-for="(value, stepIdx) in gradientHeatmap.data[layerIdx]"
                  :key="stepIdx"
                  class="py-1 px-1"
                >
                  <div
                    class="w-8 h-8 rounded flex items-center justify-center text-xs font-medium text-white cursor-pointer transition-transform hover:scale-110"
                    :style="{ backgroundColor: getHeatmapColor(value) }"
                    :title="`${layer} @ Step ${gradientHeatmap.steps[stepIdx]}: ${value.toFixed(4)}`"
                  >
                    {{ (value * 100).toFixed(0) }}
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="flex items-center justify-center gap-4 mt-4 text-sm text-gray-400">
          <span class="flex items-center gap-1">
            <span class="w-4 h-4 rounded" style="background: rgb(0, 200, 50)"></span> 小变化
          </span>
          <span class="flex items-center gap-1">
            <span class="w-4 h-4 rounded" style="background: rgb(255, 200, 50)"></span> 中等变化
          </span>
          <span class="flex items-center gap-1">
            <span class="w-4 h-4 rounded" style="background: rgb(255, 50, 50)"></span> 大变化
          </span>
        </div>
      </div>

      <!-- Evaluation Results -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-4">评估结果</h3>
        <div v-if="evaluations.length === 0" class="text-center py-8 text-gray-500">
          暂无评估结果
        </div>
        <div v-else class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-gray-400 border-b border-gray-200">
                <th class="pb-3">步数</th>
                <th class="pb-3">基准测试</th>
                <th class="pb-3 text-right">得分</th>
                <th class="pb-3 text-right">样本数</th>
                <th class="pb-3">评估时间</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(eval_, idx) in evaluations"
                :key="idx"
                class="border-b border-gray-100"
              >
                <td class="py-2 font-medium text-gray-700">{{ eval_.checkpoint_step }}</td>
                <td class="py-2 uppercase text-gray-700">{{ eval_.benchmark }}</td>
                <td class="py-2 text-right">
                  <span class="accent-text font-medium">{{ eval_.score?.toFixed(1) }}%</span>
                </td>
                <td class="py-2 text-right text-gray-400">{{ eval_.num_samples }}</td>
                <td class="py-2 text-gray-400 text-xs">
                  {{ new Date(eval_.evaluated_at).toLocaleString() }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>
