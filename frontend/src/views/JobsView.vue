<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useJobsStore } from '@/stores/jobs'
import { useComputeStore } from '@/stores/compute'
import Modal from '@/components/Modal.vue'
import {
  Plus,
  Play,
  Pause,
  Square,
  ExternalLink,
  Trash2,
  Inbox,
  FolderOpen,
  Pencil,
  Monitor,
  Server,
  RefreshCw,
  Settings
} from 'lucide-vue-next'

const router = useRouter()
const jobsStore = useJobsStore()
const computeStore = useComputeStore()

// Run mode switch handler
const handleModeSwitch = async (mode) => {
  await jobsStore.switchRunMode(mode)
}

// Refresh models and datasets
const refreshResources = async () => {
  await Promise.all([
    jobsStore.fetchAvailableModels(),
    jobsStore.fetchAvailableDatasets()
  ])
}

const filter = ref('all')
const showNewJobModal = ref(false)
const showEditJobModal = ref(false)
const editingJob = ref(null)
const useCustomModelPath = ref(false)
const useCustomDatasetPath = ref(false)
const editUseCustomModelPath = ref(false)
const editUseCustomDatasetPath = ref(false)

const filters = [
  { key: 'all', label: '全部' },
  { key: 'running', label: '运行中' },
  { key: 'completed', label: '已完成' },
  { key: 'failed', label: '失败' },
  { key: 'pending', label: '待处理' }
]

// Unified reward script mode
const useUnifiedRewardScript = ref(false)

const newJob = ref({
  name: '',
  algorithm: 'grpo',
  model_path: '',
  train_data_path: '',
  num_gpus: 8,
  gpu_type: 'A100-80G',
  batch_size: 256,
  learning_rate: 1e-6,
  num_epochs: 3,
  context_length: 4096,
  lora_enabled: false,
  lora_rank: 8,
  description: '',
  // GRPO reward function config (legacy)
  reward_fn_type: 'math_verify',
  reward_fn_extract_answer: 'boxed',
  reward_fn_compare_method: 'exact',
  reward_fn_answer_key: 'solution',
  reward_fn_custom_path: '',
  // PPO reward model config (legacy)
  reward_model_path: '',
  reward_model_enable_gc: true,
  reward_model_offload: false,
  reward_model_micro_batch: 4,
  // Unified reward script config
  reward_script_path: '',
  reward_script_type: 'rule',
  reward_script_metadata: {}
})

const rewardFnTypes = [
  { value: 'math_verify', label: '数学验证', desc: '验证数学答案正确性' },
  { value: 'format_check', label: '格式检查', desc: '检查输出格式规范' },
  { value: 'custom', label: '自定义函数', desc: '使用自定义Python脚本' }
]

const extractAnswerMethods = [
  { value: 'boxed', label: '\\boxed{}', desc: '从LaTeX盒子提取' },
  { value: 'last_number', label: '最后数字', desc: '提取最后出现的数字' },
  { value: 'json', label: 'JSON字段', desc: '从JSON响应提取' }
]

const compareMethods = [
  { value: 'exact', label: '精确匹配', desc: '字符串完全相等' },
  { value: 'numeric', label: '数值比较', desc: '数值相等（忽略格式）' },
  { value: 'fuzzy', label: '模糊匹配', desc: '允许小误差' }
]

const algorithms = [
  { value: 'grpo', label: 'GRPO' },
  { value: 'ppo', label: 'PPO' },
  { value: 'dpo', label: 'DPO' },
  { value: 'sft', label: 'SFT' },
  { value: 'gspo', label: 'GSPO' }
]

const filteredJobs = computed(() => {
  if (filter.value === 'all') return jobsStore.jobs
  return jobsStore.jobs.filter(j => j.status === filter.value)
})

const getStatusClass = (status) => {
  const classes = {
    running: 'status-running',
    completed: 'status-completed',
    failed: 'status-failed',
    pending: 'status-pending',
    queued: 'status-queued',
    paused: 'status-paused',
    cancelled: 'status-paused'
  }
  return classes[status] || 'status-paused'
}

const getStatusText = (status) => {
  const texts = {
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    pending: '待处理',
    queued: '排队中',
    paused: '已暂停',
    cancelled: '已取消'
  }
  return texts[status] || status
}

const getProgress = (job) => {
  if (!job.total_steps || job.total_steps === 0) return 0
  return Math.round((job.current_step / job.total_steps) * 100)
}

const formatDate = (date) => {
  if (!date) return '未知'
  const d = new Date(date)
  const now = new Date()
  const diff = (now - d) / 1000
  if (diff < 60) return '刚刚'
  if (diff < 3600) return Math.floor(diff / 60) + ' 分钟前'
  if (diff < 86400) return Math.floor(diff / 3600) + ' 小时前'
  return Math.floor(diff / 86400) + ' 天前'
}

const resetNewJob = () => {
  newJob.value = {
    name: '',
    algorithm: 'grpo',
    model_path: '',
    train_data_path: '',
    num_gpus: 8,
    gpu_type: 'A100-80G',
    batch_size: 256,
    learning_rate: 1e-6,
    num_epochs: 3,
    context_length: 4096,
    lora_enabled: false,
    lora_rank: 8,
    description: '',
    // GRPO reward function (legacy)
    reward_fn_type: 'math_verify',
    reward_fn_extract_answer: 'boxed',
    reward_fn_compare_method: 'exact',
    reward_fn_answer_key: 'solution',
    reward_fn_custom_path: '',
    // PPO reward model (legacy)
    reward_model_path: '',
    reward_model_enable_gc: true,
    reward_model_offload: false,
    reward_model_micro_batch: 4,
    // Unified reward script
    reward_script_path: '',
    reward_script_type: 'rule',
    reward_script_metadata: {}
  }
  useCustomModelPath.value = false
  useCustomDatasetPath.value = false
  useUnifiedRewardScript.value = false
}

const createJob = async () => {
  await jobsStore.createJob(newJob.value)
  showNewJobModal.value = false
  resetNewJob()
}

const openEditModal = (job) => {
  editingJob.value = {
    id: job.id,
    name: job.name,
    description: job.description || '',
    algorithm: job.algorithm,
    model_path: job.model_path || '',
    train_data_path: job.train_data_path || '',
    num_gpus: job.num_gpus || 1,
    gpu_type: job.gpu_type || 'A100-80G',
    batch_size: job.batch_size || 256,
    learning_rate: job.learning_rate || 1e-6,
    num_epochs: job.total_epochs || 3,
    context_length: job.context_length || 4096,
    lora_enabled: job.lora_enabled || false,
    lora_rank: job.lora_rank || 8
  }
  // Check if model/dataset paths are in the available lists
  const modelInList = jobsStore.availableModels.some(m => m.path === job.model_path)
  const datasetInList = jobsStore.availableDatasets.some(d => d.path === job.train_data_path)
  editUseCustomModelPath.value = !modelInList
  editUseCustomDatasetPath.value = !datasetInList
  showEditJobModal.value = true
}

const updateJob = async () => {
  if (!editingJob.value) return
  await jobsStore.updateJob(editingJob.value.id, {
    name: editingJob.value.name,
    description: editingJob.value.description,
    algorithm: editingJob.value.algorithm,
    model_path: editingJob.value.model_path,
    train_data_path: editingJob.value.train_data_path,
    num_gpus: editingJob.value.num_gpus,
    gpu_type: editingJob.value.gpu_type,
    batch_size: editingJob.value.batch_size,
    learning_rate: editingJob.value.learning_rate,
    num_epochs: editingJob.value.num_epochs,
    context_length: editingJob.value.context_length,
    lora_enabled: editingJob.value.lora_enabled,
    lora_rank: editingJob.value.lora_rank
  })
  showEditJobModal.value = false
  editingJob.value = null
}

const restartJob = async (job) => {
  // Reset status to pending first, then start
  if (job.status !== 'pending') {
    await jobsStore.updateJob(job.id, { status: 'pending' })
  }
  await jobsStore.startJob(job.id)
}

const deleteJob = async (id) => {
  if (confirm('确定要删除这个任务吗？')) {
    await jobsStore.deleteJob(id)
  }
}

const viewJob = (job) => {
  router.push(`/monitoring/${job.id}`)
}

onMounted(async () => {
  // First fetch run mode config to know which source to use
  await jobsStore.fetchRunModeConfig()

  // Then fetch everything else
  await Promise.all([
    jobsStore.fetchJobs(),
    computeStore.fetchGpuTypes(),
    jobsStore.fetchAvailableModels(),
    jobsStore.fetchAvailableDatasets(),
    jobsStore.fetchAvailableRewardScripts()
  ])
})
</script>

<template>
  <div>
    <!-- Run Mode Switch -->
    <div class="glass-card rounded-lg p-3 mb-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <span class="text-xs text-gray-500">运行模式:</span>
          <div class="flex bg-gray-100 rounded-lg p-0.5">
            <button
              @click="handleModeSwitch('local')"
              :disabled="jobsStore.runModeLoading"
              :class="[
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition-all',
                jobsStore.runMode === 'local'
                  ? 'bg-white shadow text-gray-800 font-medium'
                  : 'text-gray-500 hover:text-gray-700'
              ]"
            >
              <Monitor class="w-3.5 h-3.5" />
              本地
            </button>
            <button
              @click="handleModeSwitch('ssh')"
              :disabled="jobsStore.runModeLoading || !jobsStore.runModeConfig?.ssh_configured"
              :class="[
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition-all',
                jobsStore.runMode === 'ssh'
                  ? 'bg-white shadow text-gray-800 font-medium'
                  : 'text-gray-500 hover:text-gray-700',
                !jobsStore.runModeConfig?.ssh_configured && 'opacity-50 cursor-not-allowed'
              ]"
              :title="!jobsStore.runModeConfig?.ssh_configured ? '请先在设置页面配置SSH' : ''"
            >
              <Server class="w-3.5 h-3.5" />
              SSH远程
            </button>
          </div>
          <RefreshCw
            v-if="jobsStore.runModeLoading"
            class="w-4 h-4 text-gray-400 animate-spin"
          />
        </div>

        <div class="flex items-center gap-3">
          <!-- Current mode info -->
          <div v-if="jobsStore.runMode === 'ssh'" class="flex items-center gap-2 text-xs">
            <span class="px-2 py-0.5 rounded bg-green-100 text-green-700">
              {{ jobsStore.runModeConfig?.ssh_host }}
            </span>
          </div>

          <!-- Refresh button -->
          <button
            @click="refreshResources"
            class="p-1.5 rounded-md hover:bg-gray-100"
            title="刷新模型和数据集列表"
          >
            <RefreshCw class="w-4 h-4 text-gray-500" />
          </button>

          <!-- Settings link -->
          <router-link
            to="/settings"
            class="p-1.5 rounded-md hover:bg-gray-100"
            title="SSH设置"
          >
            <Settings class="w-4 h-4 text-gray-500" />
          </router-link>
        </div>
      </div>
    </div>

    <!-- Header -->
    <div class="flex justify-between items-center mb-4">
      <div class="flex gap-1.5">
        <button
          v-for="f in filters"
          :key="f.key"
          @click="filter = f.key"
          :class="[
            'px-3 py-1.5 rounded-md text-xs',
            filter === f.key ? 'bg-primary-100 accent-text font-medium' : 'bg-gray-100 text-gray-500 hover:text-gray-700'
          ]"
        >
          {{ f.label }}
        </button>
      </div>
      <button @click="showNewJobModal = true" class="btn-primary flex items-center gap-1.5 text-xs">
        <Plus class="w-3.5 h-3.5" />
        新建任务
      </button>
    </div>

    <!-- Jobs List -->
    <div class="space-y-3">
      <div v-if="filteredJobs.length === 0" class="glass-card rounded-lg p-8 text-center">
        <Inbox class="w-10 h-10 text-gray-400 mx-auto mb-3" />
        <p class="text-gray-500 text-sm">暂无训练任务，创建你的第一个任务吧！</p>
      </div>

      <div v-for="job in filteredJobs" :key="job.id" class="glass-card rounded-lg p-4">
        <div class="flex items-start justify-between">
          <div class="flex-1">
            <div class="flex items-center gap-2 mb-1.5">
              <h4 class="font-medium text-sm text-gray-800">{{ job.name }}</h4>
              <span :class="['px-1.5 py-0.5 rounded text-2xs', getStatusClass(job.status)]">
                {{ getStatusText(job.status) }}
              </span>
              <span class="px-1.5 py-0.5 rounded bg-gray-100 text-2xs uppercase text-gray-600">
                {{ job.algorithm }}
              </span>
              <span v-if="job.lora_enabled" class="px-1.5 py-0.5 rounded bg-purple-100 text-purple-600 text-2xs">
                LoRA
              </span>
            </div>
            <p class="text-xs text-gray-500 mb-3">
              {{ job.model_path?.split('/').pop() }} |
              {{ job.num_gpus }} 个 GPU |
              创建于 {{ formatDate(job.created_at) }}
            </p>

            <div class="flex items-center gap-5">
              <div>
                <span class="text-2xs text-gray-400">进度</span>
                <div class="flex items-center gap-1.5">
                  <div class="w-24 bg-gray-200 rounded-full h-1.5">
                    <div
                      class="accent-gradient h-1.5 rounded-full"
                      :style="{ width: getProgress(job) + '%' }"
                    />
                  </div>
                  <span class="text-xs text-gray-700">{{ getProgress(job) }}%</span>
                </div>
              </div>
              <div>
                <span class="text-2xs text-gray-400">步数</span>
                <p class="text-xs font-medium text-gray-700">
                  {{ job.current_step || 0 }} / {{ job.total_steps || '?' }}
                </p>
              </div>
              <div v-if="job.latest_metrics">
                <span class="text-2xs text-gray-400">奖励</span>
                <p class="text-xs font-medium accent-text">
                  {{ job.latest_metrics.reward_mean?.toFixed(3) || 'N/A' }}
                </p>
              </div>
            </div>
          </div>

          <div class="flex gap-1">
            <button
              v-if="['pending', 'cancelled', 'failed'].includes(job.status)"
              @click="restartJob(job)"
              class="p-1.5 rounded-md hover:bg-green-50"
              title="启动"
            >
              <Play class="w-4 h-4 text-green-500" />
            </button>
            <button
              v-if="job.status === 'running'"
              @click="jobsStore.pauseJob(job.id)"
              class="p-1.5 rounded-md hover:bg-yellow-50"
              title="暂停"
            >
              <Pause class="w-4 h-4 text-yellow-500" />
            </button>
            <button
              v-if="job.status === 'paused'"
              @click="jobsStore.resumeJob(job.id)"
              class="p-1.5 rounded-md hover:bg-green-50"
              title="恢复"
            >
              <Play class="w-4 h-4 text-green-500" />
            </button>
            <button
              v-if="['running', 'queued'].includes(job.status)"
              @click="jobsStore.stopJob(job.id)"
              class="p-1.5 rounded-md hover:bg-red-50"
              title="停止"
            >
              <Square class="w-4 h-4 text-red-500" />
            </button>
            <button
              v-if="job.status === 'pending'"
              @click="openEditModal(job)"
              class="p-1.5 rounded-md hover:bg-blue-50"
              title="编辑"
            >
              <Pencil class="w-4 h-4 text-blue-500" />
            </button>
            <button
              @click="viewJob(job)"
              class="p-1.5 rounded-md hover:bg-gray-100"
              title="查看详情"
            >
              <ExternalLink class="w-4 h-4 text-gray-500" />
            </button>
            <button
              @click="deleteJob(job.id)"
              class="p-1.5 rounded-md hover:bg-red-50"
              title="删除"
            >
              <Trash2 class="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- New Job Modal -->
    <Modal :show="showNewJobModal" title="创建新训练任务" @close="showNewJobModal = false">
      <div class="space-y-4">
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">任务名称 *</label>
            <input type="text" v-model="newJob.name" placeholder="my-training-job" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">训练算法 *</label>
            <select v-model="newJob.algorithm" class="w-full input-light">
              <option v-for="algo in algorithms" :key="algo.value" :value="algo.value">
                {{ algo.label }}
              </option>
            </select>
          </div>
        </div>

        <div>
          <div class="flex items-center justify-between mb-1.5">
            <label class="text-xs text-gray-500">模型路径 *</label>
            <button
              type="button"
              @click="useCustomModelPath = !useCustomModelPath"
              class="text-2xs text-primary-500 hover:underline"
            >
              {{ useCustomModelPath ? '选择已有模型' : '输入自定义路径' }}
            </button>
          </div>
          <div v-if="!useCustomModelPath && jobsStore.availableModels.length > 0">
            <select v-model="newJob.model_path" class="w-full input-light">
              <option value="">请选择模型...</option>
              <option v-for="model in jobsStore.availableModels" :key="model.path" :value="model.path">
                {{ model.name }} ({{ model.size_gb }} GB)
              </option>
            </select>
          </div>
          <div v-else>
            <input type="text" v-model="newJob.model_path" placeholder="/path/to/model" class="w-full input-light">
            <p v-if="jobsStore.availableModels.length === 0" class="text-2xs text-gray-400 mt-1">
              <FolderOpen class="w-3 h-3 inline mr-0.5" />
              将模型放入 ./models/ 目录可自动检测
            </p>
          </div>
        </div>

        <div>
          <div class="flex items-center justify-between mb-1.5">
            <label class="text-xs text-gray-500">训练数据路径 *</label>
            <button
              type="button"
              @click="useCustomDatasetPath = !useCustomDatasetPath"
              class="text-2xs text-primary-500 hover:underline"
            >
              {{ useCustomDatasetPath ? '选择已有数据集' : '输入自定义路径' }}
            </button>
          </div>
          <div v-if="!useCustomDatasetPath && jobsStore.availableDatasets.length > 0">
            <select v-model="newJob.train_data_path" class="w-full input-light">
              <option value="">请选择数据集...</option>
              <option v-for="ds in jobsStore.availableDatasets" :key="ds.path" :value="ds.path">
                {{ ds.name }} ({{ ds.format.toUpperCase() }}, {{ ds.size_mb }} MB)
              </option>
            </select>
          </div>
          <div v-else>
            <input
              type="text"
              v-model="newJob.train_data_path"
              placeholder="/path/to/train_data.jsonl"
              class="w-full input-light"
            >
            <p v-if="jobsStore.availableDatasets.length === 0" class="text-2xs text-gray-400 mt-1">
              <FolderOpen class="w-3 h-3 inline mr-0.5" />
              将数据文件放入 ./datasets/ 目录可自动检测
            </p>
          </div>
        </div>

        <div class="grid grid-cols-3 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">GPU 数量</label>
            <input type="number" v-model.number="newJob.num_gpus" min="1" max="1024" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">GPU 类型</label>
            <select v-model="newJob.gpu_type" class="w-full input-light">
              <option v-for="gpu in computeStore.gpuTypes" :key="gpu.id" :value="gpu.id">
                {{ gpu.name }}
              </option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">批量大小</label>
            <input type="number" v-model.number="newJob.batch_size" min="1" class="w-full input-light">
          </div>
        </div>

        <div class="grid grid-cols-3 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">学习率</label>
            <input type="number" v-model.number="newJob.learning_rate" step="0.000001" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">训练轮数</label>
            <input type="number" v-model.number="newJob.num_epochs" min="1" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">上下文长度</label>
            <input type="number" v-model.number="newJob.context_length" class="w-full input-light">
          </div>
        </div>

        <div class="flex items-center gap-3">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" v-model="newJob.lora_enabled" class="w-3.5 h-3.5 accent-primary-500">
            <span class="text-xs text-gray-700">启用 LoRA</span>
          </label>
          <input
            v-if="newJob.lora_enabled"
            type="number"
            v-model.number="newJob.lora_rank"
            min="1"
            max="256"
            placeholder="Rank"
            class="w-20 input-light !px-2 !py-1.5 text-xs"
          >
        </div>

        <!-- GRPO Reward Function Configuration -->
        <div v-if="newJob.algorithm === 'grpo'" class="border border-blue-200 rounded-lg p-3 bg-blue-50/50">
          <h5 class="text-xs font-medium text-blue-700 mb-3 flex items-center gap-1.5">
            <span class="w-4 h-4 rounded bg-blue-500 text-white flex items-center justify-center text-2xs">f</span>
            奖励函数配置 (GRPO)
          </h5>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">函数类型</label>
              <select v-model="newJob.reward_fn_type" class="w-full input-light text-xs">
                <option v-for="fn in rewardFnTypes" :key="fn.value" :value="fn.value">
                  {{ fn.label }} - {{ fn.desc }}
                </option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">答案字段</label>
              <input
                type="text"
                v-model="newJob.reward_fn_answer_key"
                placeholder="solution"
                class="w-full input-light text-xs"
              >
            </div>
          </div>
          <div v-if="newJob.reward_fn_type === 'math_verify'" class="grid grid-cols-2 gap-3 mt-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">答案提取方式</label>
              <select v-model="newJob.reward_fn_extract_answer" class="w-full input-light text-xs">
                <option v-for="m in extractAnswerMethods" :key="m.value" :value="m.value">
                  {{ m.label }} - {{ m.desc }}
                </option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">比较方法</label>
              <select v-model="newJob.reward_fn_compare_method" class="w-full input-light text-xs">
                <option v-for="m in compareMethods" :key="m.value" :value="m.value">
                  {{ m.label }} - {{ m.desc }}
                </option>
              </select>
            </div>
          </div>
          <div v-if="newJob.reward_fn_type === 'custom'" class="mt-3">
            <label class="block text-xs text-gray-500 mb-1.5">自定义函数路径</label>
            <input
              type="text"
              v-model="newJob.reward_fn_custom_path"
              placeholder="/path/to/reward_fn.py"
              class="w-full input-light text-xs"
            >
            <p class="text-2xs text-gray-400 mt-1">Python脚本需实现 reward_fn(prompt, response, solution) -> float</p>
          </div>
        </div>

        <!-- PPO Reward Model Configuration (Legacy) -->
        <div v-if="newJob.algorithm === 'ppo' && !useUnifiedRewardScript" class="border border-purple-200 rounded-lg p-3 bg-purple-50/50">
          <div class="flex items-center justify-between mb-3">
            <h5 class="text-xs font-medium text-purple-700 flex items-center gap-1.5">
              <span class="w-4 h-4 rounded bg-purple-500 text-white flex items-center justify-center text-2xs">M</span>
              奖励模型配置 (PPO)
            </h5>
            <button
              type="button"
              @click="useUnifiedRewardScript = true"
              class="text-2xs text-purple-500 hover:underline"
            >
              使用奖励脚本
            </button>
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">奖励模型路径 *</label>
            <input
              type="text"
              v-model="newJob.reward_model_path"
              placeholder="/path/to/reward_model"
              class="w-full input-light text-xs"
            >
            <p class="text-2xs text-gray-400 mt-1">需要预训练的奖励模型来评估回答质量</p>
          </div>
          <div class="grid grid-cols-3 gap-3 mt-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">Micro Batch</label>
              <input
                type="number"
                v-model.number="newJob.reward_model_micro_batch"
                min="1"
                class="w-full input-light text-xs"
              >
            </div>
            <div class="flex items-center">
              <label class="flex items-center gap-2 cursor-pointer mt-4">
                <input type="checkbox" v-model="newJob.reward_model_enable_gc" class="w-3.5 h-3.5 accent-purple-500">
                <span class="text-xs text-gray-700">梯度检查点</span>
              </label>
            </div>
            <div class="flex items-center">
              <label class="flex items-center gap-2 cursor-pointer mt-4">
                <input type="checkbox" v-model="newJob.reward_model_offload" class="w-3.5 h-3.5 accent-purple-500">
                <span class="text-xs text-gray-700">参数卸载</span>
              </label>
            </div>
          </div>
        </div>

        <!-- Unified Reward Script Configuration (for PPO/GRPO/GSPO) -->
        <div v-if="['ppo', 'grpo', 'gspo'].includes(newJob.algorithm) && useUnifiedRewardScript" class="border border-green-200 rounded-lg p-3 bg-green-50/50">
          <div class="flex items-center justify-between mb-3">
            <h5 class="text-xs font-medium text-green-700 flex items-center gap-1.5">
              <span class="w-4 h-4 rounded bg-green-500 text-white flex items-center justify-center text-2xs">S</span>
              奖励脚本配置
            </h5>
            <button
              type="button"
              @click="useUnifiedRewardScript = false"
              class="text-2xs text-green-500 hover:underline"
            >
              使用{{ newJob.algorithm === 'ppo' ? '奖励模型' : '内置函数' }}
            </button>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">选择脚本</label>
              <select v-model="newJob.reward_script_path" class="w-full input-light text-xs">
                <option value="">请选择脚本...</option>
                <option v-for="script in jobsStore.availableRewardScripts" :key="script.path" :value="script.path">
                  {{ script.name }} ({{ script.type }})
                </option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">脚本类型</label>
              <select v-model="newJob.reward_script_type" class="w-full input-light text-xs">
                <option value="rule">规则函数 (Rule)</option>
                <option value="api">API调用 (API)</option>
                <option value="model">本地模型 (Model)</option>
              </select>
            </div>
          </div>

          <!-- Show selected script info -->
          <div v-if="newJob.reward_script_path" class="mt-3 p-2 bg-white rounded border border-green-100">
            <p class="text-xs text-gray-600">
              {{ jobsStore.availableRewardScripts.find(s => s.path === newJob.reward_script_path)?.description || '无描述' }}
            </p>
          </div>

          <!-- Script-specific metadata configuration -->
          <div v-if="newJob.reward_script_type === 'api'" class="mt-3">
            <label class="block text-xs text-gray-500 mb-1.5">API 配置 (JSON)</label>
            <textarea
              v-model="newJob.reward_script_metadata.api_config"
              rows="2"
              placeholder='{"api_type": "openai", "model": "gpt-4"}'
              class="w-full input-light text-xs font-mono"
            />
          </div>

          <div v-if="newJob.reward_script_type === 'model'" class="mt-3">
            <label class="block text-xs text-gray-500 mb-1.5">模型路径</label>
            <input
              type="text"
              v-model="newJob.reward_script_metadata.model_path"
              placeholder="/path/to/reward_model"
              class="w-full input-light text-xs"
            >
          </div>

          <p class="text-2xs text-gray-400 mt-3">
            奖励脚本需实现标准接口: stdin(JSON) → stdout(JSON) —
            <a href="https://github.com/..." target="_blank" class="text-green-500 hover:underline">查看文档</a>
          </p>
        </div>

        <!-- Toggle to use unified script for GRPO -->
        <div v-if="newJob.algorithm === 'grpo' && !useUnifiedRewardScript" class="text-right">
          <button
            type="button"
            @click="useUnifiedRewardScript = true"
            class="text-2xs text-green-500 hover:underline"
          >
            切换到奖励脚本模式
          </button>
        </div>

        <!-- Toggle for GSPO (always uses unified script) -->
        <div v-if="newJob.algorithm === 'gspo' && !useUnifiedRewardScript" class="border border-green-200 rounded-lg p-3 bg-green-50/50">
          <h5 class="text-xs font-medium text-green-700 mb-3 flex items-center gap-1.5">
            <span class="w-4 h-4 rounded bg-green-500 text-white flex items-center justify-center text-2xs">S</span>
            奖励配置 (GSPO)
          </h5>
          <p class="text-xs text-gray-500 mb-3">GSPO 需要配置奖励脚本来评估生成结果</p>
          <button
            type="button"
            @click="useUnifiedRewardScript = true"
            class="btn-secondary text-xs"
          >
            配置奖励脚本
          </button>
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">描述</label>
          <textarea
            v-model="newJob.description"
            rows="2"
            placeholder="可选描述..."
            class="w-full input-light"
          />
        </div>

        <div class="flex justify-end gap-3">
          <button @click="showNewJobModal = false" class="btn-secondary">
            取消
          </button>
          <button
            @click="createJob"
            :disabled="!newJob.name || !newJob.model_path || !newJob.train_data_path"
            class="btn-primary"
          >
            创建任务
          </button>
        </div>
      </div>
    </Modal>

    <!-- Edit Job Modal -->
    <Modal :show="showEditJobModal" title="编辑训练任务" @close="showEditJobModal = false">
      <div v-if="editingJob" class="space-y-4">
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">任务名称 *</label>
            <input type="text" v-model="editingJob.name" placeholder="my-training-job" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">训练算法 *</label>
            <select v-model="editingJob.algorithm" class="w-full input-light">
              <option v-for="algo in algorithms" :key="algo.value" :value="algo.value">
                {{ algo.label }}
              </option>
            </select>
          </div>
        </div>

        <div>
          <div class="flex items-center justify-between mb-1.5">
            <label class="text-xs text-gray-500">模型路径 *</label>
            <button
              type="button"
              @click="editUseCustomModelPath = !editUseCustomModelPath"
              class="text-2xs text-primary-500 hover:underline"
            >
              {{ editUseCustomModelPath ? '选择已有模型' : '输入自定义路径' }}
            </button>
          </div>
          <div v-if="!editUseCustomModelPath && jobsStore.availableModels.length > 0">
            <select v-model="editingJob.model_path" class="w-full input-light">
              <option value="">请选择模型...</option>
              <option v-for="model in jobsStore.availableModels" :key="model.path" :value="model.path">
                {{ model.name }} ({{ model.size_gb }} GB)
              </option>
            </select>
          </div>
          <div v-else>
            <input type="text" v-model="editingJob.model_path" placeholder="/path/to/model" class="w-full input-light">
          </div>
        </div>

        <div>
          <div class="flex items-center justify-between mb-1.5">
            <label class="text-xs text-gray-500">训练数据路径 *</label>
            <button
              type="button"
              @click="editUseCustomDatasetPath = !editUseCustomDatasetPath"
              class="text-2xs text-primary-500 hover:underline"
            >
              {{ editUseCustomDatasetPath ? '选择已有数据集' : '输入自定义路径' }}
            </button>
          </div>
          <div v-if="!editUseCustomDatasetPath && jobsStore.availableDatasets.length > 0">
            <select v-model="editingJob.train_data_path" class="w-full input-light">
              <option value="">请选择数据集...</option>
              <option v-for="ds in jobsStore.availableDatasets" :key="ds.path" :value="ds.path">
                {{ ds.name }} ({{ ds.format.toUpperCase() }}, {{ ds.size_mb }} MB)
              </option>
            </select>
          </div>
          <div v-else>
            <input type="text" v-model="editingJob.train_data_path" placeholder="/path/to/train_data.jsonl" class="w-full input-light">
          </div>
        </div>

        <div class="grid grid-cols-3 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">GPU 数量</label>
            <input type="number" v-model.number="editingJob.num_gpus" min="1" max="1024" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">GPU 类型</label>
            <select v-model="editingJob.gpu_type" class="w-full input-light">
              <option v-for="gpu in computeStore.gpuTypes" :key="gpu.id" :value="gpu.id">
                {{ gpu.name }}
              </option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">批量大小</label>
            <input type="number" v-model.number="editingJob.batch_size" min="1" class="w-full input-light">
          </div>
        </div>

        <div class="grid grid-cols-3 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">学习率</label>
            <input type="number" v-model.number="editingJob.learning_rate" step="0.000001" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">训练轮数</label>
            <input type="number" v-model.number="editingJob.num_epochs" min="1" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">上下文长度</label>
            <input type="number" v-model.number="editingJob.context_length" class="w-full input-light">
          </div>
        </div>

        <div class="flex items-center gap-3">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" v-model="editingJob.lora_enabled" class="w-3.5 h-3.5 accent-primary-500">
            <span class="text-xs text-gray-700">启用 LoRA</span>
          </label>
          <input
            v-if="editingJob.lora_enabled"
            type="number"
            v-model.number="editingJob.lora_rank"
            min="1"
            max="256"
            placeholder="Rank"
            class="w-20 input-light !px-2 !py-1.5 text-xs"
          >
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">描述</label>
          <textarea
            v-model="editingJob.description"
            rows="2"
            placeholder="可选描述..."
            class="w-full input-light"
          />
        </div>

        <div class="flex justify-end gap-3">
          <button @click="showEditJobModal = false" class="btn-secondary">
            取消
          </button>
          <button
            @click="updateJob"
            :disabled="!editingJob.name || !editingJob.model_path || !editingJob.train_data_path"
            class="btn-primary"
          >
            保存修改
          </button>
        </div>
      </div>
    </Modal>
  </div>
</template>
