<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getDashboard } from '@/api'
import { useAppStore } from '@/stores/app'
import StatCard from '@/components/StatCard.vue'
import {
  Play,
  Cpu,
  CheckCircle,
  Hash,
  Calculator,
  PlusCircle,
  GitMerge,
  AlertTriangle
} from 'lucide-vue-next'

const router = useRouter()
const appStore = useAppStore()

const dashboard = ref({
  active_jobs: 0,
  queued_jobs: 0,
  completed_jobs: 0,
  failed_jobs: 0,
  total_gpu_hours: 0,
  total_training_tokens: 0,
  recent_alerts: [],
  top_performing_experiments: []
})

const formatTokens = (n) => {
  if (!n) return '0'
  if (n >= 1e12) return (n / 1e12).toFixed(1) + 'T'
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  return n
}

const loadDashboard = async () => {
  try {
    appStore.loading = true
    dashboard.value = await getDashboard()
  } catch (error) {
    appStore.showError(error.message)
  } finally {
    appStore.loading = false
  }
}

onMounted(loadDashboard)
</script>

<template>
  <div>
    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <StatCard title="运行中任务" :subtitle="`${dashboard.queued_jobs} 个排队中`" gradient>
        <template #icon><Play class="w-4 h-4 accent-text" /></template>
        {{ dashboard.active_jobs }}
      </StatCard>

      <StatCard title="GPU 时长" subtitle="本月累计">
        <template #icon><Cpu class="w-4 h-4 text-cyan-500" /></template>
        {{ dashboard.total_gpu_hours?.toFixed(1) || 0 }}
      </StatCard>

      <StatCard title="已完成" :subtitle="`${dashboard.failed_jobs} 个失败`">
        <template #icon><CheckCircle class="w-4 h-4 text-green-500" /></template>
        {{ dashboard.completed_jobs }}
      </StatCard>

      <StatCard title="训练 Token 数" subtitle="总计">
        <template #icon><Hash class="w-4 h-4 text-purple-500" /></template>
        {{ formatTokens(dashboard.total_training_tokens) }}
      </StatCard>
    </div>

    <!-- Quick Actions -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <button
        @click="router.push('/compute')"
        class="glass-card rounded-lg p-4 text-left hover:border-primary-300 transition-colors group"
      >
        <div class="w-10 h-10 rounded-lg bg-primary-50 flex items-center justify-center mb-3 group-hover:bg-primary-100">
          <Calculator class="w-5 h-5 accent-text" />
        </div>
        <h3 class="font-medium text-sm text-gray-800 mb-1">计算配置器</h3>
        <p class="text-xs text-gray-500">计算最优 GPU 和批量大小配置</p>
      </button>

      <button
        @click="router.push('/jobs')"
        class="glass-card rounded-lg p-4 text-left hover:border-cyan-300 transition-colors group"
      >
        <div class="w-10 h-10 rounded-lg bg-cyan-50 flex items-center justify-center mb-3 group-hover:bg-cyan-100">
          <PlusCircle class="w-5 h-5 text-cyan-500" />
        </div>
        <h3 class="font-medium text-sm text-gray-800 mb-1">新建训练任务</h3>
        <p class="text-xs text-gray-500">创建 SFT、PPO、GRPO 或 DPO 训练</p>
      </button>

      <button
        @click="router.push('/surgery')"
        class="glass-card rounded-lg p-4 text-left hover:border-purple-300 transition-colors group"
      >
        <div class="w-10 h-10 rounded-lg bg-purple-50 flex items-center justify-center mb-3 group-hover:bg-purple-100">
          <GitMerge class="w-5 h-5 text-purple-500" />
        </div>
        <h3 class="font-medium text-sm text-gray-800 mb-1">模型手术</h3>
        <p class="text-xs text-gray-500">模型合并或选择最佳检查点</p>
      </button>
    </div>

    <!-- Top Experiments & Alerts -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium text-sm text-gray-800 mb-3">表现最佳实验</h3>
        <div class="space-y-2">
          <div
            v-for="exp in dashboard.top_performing_experiments"
            :key="exp.name"
            class="flex items-center justify-between py-1.5 border-b border-gray-100"
          >
            <span class="font-medium text-xs text-gray-700">{{ exp.name }}</span>
            <div class="flex gap-3 text-xs">
              <span v-if="exp.gsm8k" class="text-gray-500">
                GSM8K: <span class="accent-text font-medium">{{ exp.gsm8k }}%</span>
              </span>
              <span v-if="exp.math" class="text-gray-500">
                MATH: <span class="accent-text font-medium">{{ exp.math }}%</span>
              </span>
              <span v-if="exp.humaneval" class="text-gray-500">
                HumanEval: <span class="accent-text font-medium">{{ exp.humaneval }}%</span>
              </span>
            </div>
          </div>
          <p v-if="!dashboard.top_performing_experiments?.length" class="text-gray-400 text-xs">
            暂无实验数据
          </p>
        </div>
      </div>

      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium text-sm text-gray-800 mb-3">最近告警</h3>
        <div class="space-y-2">
          <div
            v-for="alert in dashboard.recent_alerts"
            :key="alert.job_id"
            :class="[
              'flex items-center gap-2 py-1.5 px-2 rounded-md',
              alert.severity === 'warning' ? 'bg-yellow-50' : 'bg-red-50'
            ]"
          >
            <AlertTriangle
              :class="['w-3.5 h-3.5', alert.severity === 'warning' ? 'text-yellow-500' : 'text-red-500']"
            />
            <div class="flex-1">
              <p class="text-xs text-gray-700">{{ alert.message }}</p>
              <p class="text-2xs text-gray-400">任务: {{ alert.job_id }}</p>
            </div>
          </div>
          <p v-if="!dashboard.recent_alerts?.length" class="text-gray-400 text-xs">
            暂无告警
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
