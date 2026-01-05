<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useEvaluationStore } from '@/stores/evaluation'
import { useJobsStore } from '@/stores/jobs'
import * as api from '@/api'
import Modal from '@/components/Modal.vue'
import { Radar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js'
import {
  Database,
  Upload,
  Play,
  Trash2,
  Eye,
  RefreshCw,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  GitCompare,
  ArrowUp,
  ArrowDown,
  Minus
} from 'lucide-vue-next'

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
)

const evaluationStore = useEvaluationStore()
const jobsStore = useJobsStore()

// Tabs
const activeTab = ref('datasets')

// Dataset management
const showUploadModal = ref(false)
const showPreviewModal = ref(false)
const previewData = ref(null)
const uploadForm = ref({
  name: '',
  description: '',
  format: 'qa',
  capability: 'math',
  eval_method: 'exact_match',
  file: null,
})

// Evaluation trigger
const selectedJobId = ref('')
const selectedCheckpointId = ref(null)
const selectedDatasetUuids = ref([])
const checkpoints = ref([])

// Model type selection
const modelType = ref('api')  // api, local_model, checkpoint
const modelConfig = ref({
  // API mode
  api_base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  api_model: 'qwen-plus',
  api_key_env: 'DASHSCOPE_API_KEY',
  // Local model mode
  model_path: '',
})

const modelTypes = [
  { value: 'api', label: 'API 模型', description: 'OpenAI 兼容 API' },
  { value: 'local_model', label: '本地模型', description: '训练前的模型' },
  { value: 'checkpoint', label: '检查点', description: '训练后的检查点' },
]

// Results view
const resultsJobId = ref('')
const jobResults = ref(null)

// Comparison view
const comparisons = ref([])
const showCreateComparisonModal = ref(false)
const comparisonForm = ref({
  name: '',
  description: '',
  dataset_uuid: '',
  model_a_task_uuid: '',
  model_a_name: '',
  model_b_task_uuid: '',
  model_b_name: '',
})
const tasksForDataset = ref([])
const selectedComparison = ref(null)
const comparisonDiffs = ref([])
const diffFilter = ref('all') // all, improved, degraded
const loadingComparison = ref(false)

// Polling for task updates
const pollingInterval = ref(null)

const formats = [
  { value: 'qa', label: 'QA (问答格式)' },
  { value: 'dialogue', label: '对话格式' },
]

const capabilities = [
  { value: 'math', label: '数学' },
  { value: 'code', label: '代码' },
  { value: 'reasoning', label: '推理' },
  { value: 'language', label: '语言' },
  { value: 'custom', label: '自定义' },
]

const evalMethods = [
  { value: 'exact_match', label: '精确匹配' },
  { value: 'contains', label: '包含关键词' },
  { value: 'llm_judge', label: 'LLM 评判' },
]

const getStatusIcon = (status) => {
  const icons = {
    pending: Clock,
    running: RefreshCw,
    completed: CheckCircle,
    failed: XCircle,
  }
  return icons[status] || Clock
}

const getStatusClass = (status) => {
  const classes = {
    pending: 'text-gray-400',
    running: 'text-blue-500 animate-spin',
    completed: 'text-green-500',
    failed: 'text-red-500',
  }
  return classes[status] || 'text-gray-400'
}

const getCapabilityColor = (cap) => {
  const colors = {
    math: 'bg-blue-100 text-blue-600',
    code: 'bg-green-100 text-green-600',
    reasoning: 'bg-purple-100 text-purple-600',
    language: 'bg-yellow-100 text-yellow-600',
    custom: 'bg-gray-100 text-gray-600',
  }
  return colors[cap] || 'bg-gray-100 text-gray-600'
}

// Radar chart for multi-dimensional comparison
const radarChartData = computed(() => {
  if (!jobResults.value || !jobResults.value.results_by_capability) return null

  const labels = Object.keys(jobResults.value.results_by_capability)
  if (labels.length === 0) return null

  // Get latest score for each capability
  const data = labels.map(cap => {
    const results = jobResults.value.results_by_capability[cap]
    if (!results || results.length === 0) return 0
    return results[results.length - 1].score
  })

  return {
    labels: labels.map(l => capabilities.find(c => c.value === l)?.label || l),
    datasets: [{
      label: '当前得分',
      data,
      backgroundColor: 'rgba(16, 185, 129, 0.2)',
      borderColor: '#10b981',
      pointBackgroundColor: '#10b981',
    }]
  }
})

const radarOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    r: {
      min: 0,
      max: 100,
      ticks: { stepSize: 20 }
    }
  }
}

// Handle file selection
const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    uploadForm.value.file = file
    if (!uploadForm.value.name) {
      uploadForm.value.name = file.name.replace(/\.[^/.]+$/, '')
    }
  }
}

// Upload dataset
const submitUpload = async () => {
  if (!uploadForm.value.file || !uploadForm.value.name) return

  await evaluationStore.uploadDataset(uploadForm.value.file, {
    name: uploadForm.value.name,
    description: uploadForm.value.description,
    format: uploadForm.value.format,
    capability: uploadForm.value.capability,
    eval_method: uploadForm.value.eval_method,
  })

  showUploadModal.value = false
  uploadForm.value = {
    name: '',
    description: '',
    format: 'qa',
    capability: 'math',
    eval_method: 'exact_match',
    file: null,
  }
}

// Preview dataset
const openPreview = async (dataset) => {
  previewData.value = await evaluationStore.previewDataset(dataset.uuid)
  showPreviewModal.value = true
}

// Delete dataset
const deleteDataset = async (uuid) => {
  if (confirm('确定要删除这个评估数据集吗？')) {
    await evaluationStore.deleteDataset(uuid)
  }
}

// Load checkpoints when job is selected
const loadCheckpoints = async () => {
  if (!selectedJobId.value) {
    checkpoints.value = []
    return
  }
  try {
    const data = await api.getCheckpoints(selectedJobId.value)
    checkpoints.value = data || []
  } catch (e) {
    checkpoints.value = []
  }
}

// Trigger evaluation
const triggerEvaluation = async () => {
  if (selectedDatasetUuids.value.length === 0) return

  // Build params based on model type
  const params = {
    dataset_uuids: selectedDatasetUuids.value,
    model_type: modelType.value,
  }

  if (modelType.value === 'checkpoint') {
    if (!selectedCheckpointId.value) return
    params.checkpoint_id = selectedCheckpointId.value
  } else if (modelType.value === 'local_model') {
    if (!modelConfig.value.model_path) return
    params.model_path = modelConfig.value.model_path
  } else if (modelType.value === 'api') {
    params.api_base_url = modelConfig.value.api_base_url
    params.api_model = modelConfig.value.api_model
    params.api_key_env = modelConfig.value.api_key_env
  }

  await evaluationStore.triggerEvaluation(params)

  // Start polling for updates
  startPolling()

  // Refresh tasks for checkpoint mode
  if (modelType.value === 'checkpoint' && selectedJobId.value) {
    await evaluationStore.fetchJobTasks(selectedJobId.value)
  }
}

// Validate trigger button
const canTrigger = computed(() => {
  if (selectedDatasetUuids.value.length === 0) return false

  if (modelType.value === 'checkpoint') {
    return !!selectedCheckpointId.value
  } else if (modelType.value === 'local_model') {
    return !!modelConfig.value.model_path
  } else if (modelType.value === 'api') {
    return !!modelConfig.value.api_model
  }
  return false
})

// Start polling for task updates
const startPolling = () => {
  if (pollingInterval.value) return

  pollingInterval.value = setInterval(async () => {
    const pendingTasks = evaluationStore.evalTasks.filter(
      t => t.status === 'pending' || t.status === 'running'
    )

    if (pendingTasks.length === 0) {
      stopPolling()
      return
    }

    for (const task of pendingTasks) {
      await evaluationStore.refreshTask(task.uuid)
    }
  }, 2000)
}

const stopPolling = () => {
  if (pollingInterval.value) {
    clearInterval(pollingInterval.value)
    pollingInterval.value = null
  }
}

// Load results for job
const loadJobResults = async () => {
  if (!resultsJobId.value) {
    jobResults.value = null
    return
  }
  jobResults.value = await evaluationStore.fetchJobResults(resultsJobId.value)
  await evaluationStore.fetchJobTasks(resultsJobId.value)
}

// ===== Comparison Functions =====

const fetchComparisons = async () => {
  try {
    comparisons.value = await api.getComparisons()
  } catch (e) {
    console.error('Failed to fetch comparisons:', e)
  }
}

const loadTasksForDataset = async () => {
  if (!comparisonForm.value.dataset_uuid) {
    tasksForDataset.value = []
    return
  }
  try {
    tasksForDataset.value = await api.getEvalTasks({
      dataset_uuid: comparisonForm.value.dataset_uuid,
      status: 'completed'
    })
  } catch (e) {
    tasksForDataset.value = []
  }
}

const createComparison = async () => {
  if (!comparisonForm.value.name || !comparisonForm.value.model_a_task_uuid || !comparisonForm.value.model_b_task_uuid) return

  try {
    await api.createComparison(comparisonForm.value)
    showCreateComparisonModal.value = false
    comparisonForm.value = {
      name: '',
      description: '',
      dataset_uuid: '',
      model_a_task_uuid: '',
      model_a_name: '',
      model_b_task_uuid: '',
      model_b_name: '',
    }
    await fetchComparisons()
  } catch (e) {
    console.error('Failed to create comparison:', e)
    alert('Failed to create comparison: ' + e.message)
  }
}

const selectComparison = async (comp) => {
  selectedComparison.value = comp
  loadingComparison.value = true
  try {
    const [details, diffs] = await Promise.all([
      api.getComparison(comp.uuid),
      api.getComparisonDiffs(comp.uuid, { limit: 100 })
    ])
    selectedComparison.value = details
    comparisonDiffs.value = diffs
  } catch (e) {
    console.error('Failed to load comparison:', e)
  } finally {
    loadingComparison.value = false
  }
}

const filteredDiffs = computed(() => {
  if (diffFilter.value === 'all') return comparisonDiffs.value
  return comparisonDiffs.value.filter(d => d.change === diffFilter.value)
})

const deleteComparison = async (uuid) => {
  if (!confirm('Delete this comparison?')) return
  try {
    await api.deleteComparison(uuid)
    if (selectedComparison.value?.uuid === uuid) {
      selectedComparison.value = null
    }
    await fetchComparisons()
  } catch (e) {
    console.error('Failed to delete comparison:', e)
  }
}

// Watch for job selection changes in evaluate tab
watch(selectedJobId, () => {
  loadCheckpoints()
  if (selectedJobId.value) {
    evaluationStore.fetchJobTasks(selectedJobId.value)
  }
})

onMounted(async () => {
  await Promise.all([
    evaluationStore.fetchDatasets(),
    jobsStore.fetchJobs(),
    fetchComparisons(),
  ])

  // Check for pending tasks and start polling
  const hasPending = evaluationStore.evalTasks.some(
    t => t.status === 'pending' || t.status === 'running'
  )
  if (hasPending) {
    startPolling()
  }
})

// Cleanup
import { onUnmounted } from 'vue'
onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <div>
    <!-- Header -->
    <div class="flex justify-between items-center mb-4">
      <div>
        <h2 class="text-lg font-semibold text-gray-800">自定义评估</h2>
        <p class="text-xs text-gray-500">管理评估数据集，执行检查点评估，追踪能力变化</p>
      </div>
    </div>

    <!-- Tabs -->
    <div class="flex border-b border-gray-200 mb-4">
      <button
        @click="activeTab = 'datasets'"
        :class="[
          'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
          activeTab === 'datasets'
            ? 'border-primary-500 text-primary-600'
            : 'border-transparent text-gray-500 hover:text-gray-700'
        ]"
      >
        <Database class="w-4 h-4 inline mr-1.5" />
        评估数据集
      </button>
      <button
        @click="activeTab = 'evaluate'"
        :class="[
          'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
          activeTab === 'evaluate'
            ? 'border-primary-500 text-primary-600'
            : 'border-transparent text-gray-500 hover:text-gray-700'
        ]"
      >
        <Play class="w-4 h-4 inline mr-1.5" />
        执行评估
      </button>
      <button
        @click="activeTab = 'results'"
        :class="[
          'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
          activeTab === 'results'
            ? 'border-primary-500 text-primary-600'
            : 'border-transparent text-gray-500 hover:text-gray-700'
        ]"
      >
        <FileText class="w-4 h-4 inline mr-1.5" />
        评估结果
      </button>
      <button
        @click="activeTab = 'compare'"
        :class="[
          'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
          activeTab === 'compare'
            ? 'border-primary-500 text-primary-600'
            : 'border-transparent text-gray-500 hover:text-gray-700'
        ]"
      >
        <GitCompare class="w-4 h-4 inline mr-1.5" />
        模型对照
      </button>
    </div>

    <!-- Datasets Tab -->
    <div v-if="activeTab === 'datasets'">
      <div class="flex justify-end mb-4">
        <button @click="showUploadModal = true" class="btn-primary flex items-center gap-1.5 text-xs">
          <Upload class="w-3.5 h-3.5" />
          上传数据集
        </button>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div
          v-for="dataset in evaluationStore.datasets"
          :key="dataset.uuid"
          class="glass-card rounded-lg p-4"
        >
          <div class="flex items-start justify-between mb-2">
            <div>
              <h4 class="font-medium text-sm text-gray-800">{{ dataset.name }}</h4>
              <p v-if="dataset.description" class="text-2xs text-gray-500 mt-0.5">
                {{ dataset.description }}
              </p>
            </div>
            <div class="flex gap-1">
              <button @click="openPreview(dataset)" class="p-1 hover:bg-gray-100 rounded" title="预览">
                <Eye class="w-4 h-4 text-gray-400" />
              </button>
              <button @click="deleteDataset(dataset.uuid)" class="p-1 hover:bg-red-50 rounded" title="删除">
                <Trash2 class="w-4 h-4 text-gray-400" />
              </button>
            </div>
          </div>

          <div class="flex flex-wrap gap-1.5 mb-2">
            <span :class="['px-1.5 py-0.5 rounded text-2xs', getCapabilityColor(dataset.capability)]">
              {{ capabilities.find(c => c.value === dataset.capability)?.label }}
            </span>
            <span class="px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 text-2xs uppercase">
              {{ dataset.format }}
            </span>
            <span class="px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 text-2xs">
              {{ evalMethods.find(m => m.value === dataset.eval_method)?.label }}
            </span>
          </div>

          <p class="text-2xs text-gray-500">
            {{ dataset.sample_count }} 条样本
          </p>
        </div>
      </div>

      <div v-if="evaluationStore.datasets.length === 0" class="glass-card rounded-lg p-8 text-center">
        <Database class="w-10 h-10 text-gray-300 mx-auto mb-3" />
        <p class="text-gray-500 text-sm">暂无评估数据集</p>
        <p class="text-2xs text-gray-400 mt-1">上传 JSONL 格式的评估数据开始使用</p>
      </div>
    </div>

    <!-- Evaluate Tab -->
    <div v-if="activeTab === 'evaluate'">
      <div class="grid grid-cols-12 gap-4">
        <!-- Left: Selection -->
        <div class="col-span-5">
          <div class="glass-card rounded-lg p-4 space-y-4">
            <!-- Model Type Selection -->
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">推理模型类型</label>
              <div class="grid grid-cols-3 gap-2">
                <label
                  v-for="mt in modelTypes"
                  :key="mt.value"
                  :class="[
                    'flex flex-col items-center p-2 rounded-lg border cursor-pointer transition-all',
                    modelType === mt.value
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  ]"
                >
                  <input
                    type="radio"
                    :value="mt.value"
                    v-model="modelType"
                    class="hidden"
                  >
                  <span :class="['text-xs font-medium', modelType === mt.value ? 'text-primary-600' : 'text-gray-700']">
                    {{ mt.label }}
                  </span>
                  <span class="text-2xs text-gray-400">{{ mt.description }}</span>
                </label>
              </div>
            </div>

            <!-- API Mode Settings -->
            <div v-if="modelType === 'api'" class="space-y-3 p-3 bg-gray-50 rounded-lg">
              <div>
                <label class="block text-xs text-gray-500 mb-1">API Base URL</label>
                <input
                  type="text"
                  v-model="modelConfig.api_base_url"
                  class="w-full input-light text-xs"
                  placeholder="https://api.openai.com/v1"
                >
              </div>
              <div>
                <label class="block text-xs text-gray-500 mb-1">模型名称</label>
                <input
                  type="text"
                  v-model="modelConfig.api_model"
                  class="w-full input-light text-xs"
                  placeholder="gpt-4, qwen-plus 等"
                >
              </div>
              <div>
                <label class="block text-xs text-gray-500 mb-1">API Key 环境变量</label>
                <input
                  type="text"
                  v-model="modelConfig.api_key_env"
                  class="w-full input-light text-xs"
                  placeholder="DASHSCOPE_API_KEY"
                >
              </div>
            </div>

            <!-- Local Model Mode Settings -->
            <div v-if="modelType === 'local_model'" class="space-y-3 p-3 bg-gray-50 rounded-lg">
              <div>
                <label class="block text-xs text-gray-500 mb-1">模型路径</label>
                <input
                  type="text"
                  v-model="modelConfig.model_path"
                  class="w-full input-light text-xs"
                  placeholder="/path/to/model"
                >
                <p class="text-2xs text-gray-400 mt-1">本地模型推理暂未实现，需要配置 vLLM</p>
              </div>
            </div>

            <!-- Checkpoint Mode Settings -->
            <div v-if="modelType === 'checkpoint'" class="space-y-3">
              <div>
                <label class="block text-xs text-gray-500 mb-1.5">选择训练任务</label>
                <select v-model="selectedJobId" class="w-full input-light">
                  <option value="">请选择...</option>
                  <option v-for="job in jobsStore.jobs" :key="job.id" :value="job.id">
                    {{ job.name }}
                  </option>
                </select>
              </div>

              <div v-if="checkpoints.length > 0">
                <label class="block text-xs text-gray-500 mb-1.5">选择检查点</label>
                <select v-model="selectedCheckpointId" class="w-full input-light">
                  <option :value="null">请选择...</option>
                  <option v-for="cp in checkpoints" :key="cp.id" :value="cp.id">
                    Step {{ cp.step }}
                  </option>
                </select>
              </div>

              <div v-if="checkpoints.length === 0 && selectedJobId" class="text-center py-4 text-gray-400 text-xs">
                该任务暂无检查点
              </div>

              <p class="text-2xs text-gray-400">检查点推理暂未实现，需要配置模型加载</p>
            </div>

            <!-- Dataset Selection -->
            <div v-if="evaluationStore.datasets.length > 0">
              <label class="block text-xs text-gray-500 mb-1.5">选择评估数据集</label>
              <div class="space-y-2 max-h-48 overflow-y-auto">
                <label
                  v-for="dataset in evaluationStore.datasets"
                  :key="dataset.uuid"
                  class="flex items-center gap-2 p-2 rounded hover:bg-gray-50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    :value="dataset.uuid"
                    v-model="selectedDatasetUuids"
                    class="w-4 h-4 accent-primary-500"
                  >
                  <span class="flex-1 text-sm text-gray-700">{{ dataset.name }}</span>
                  <span :class="['px-1.5 py-0.5 rounded text-2xs', getCapabilityColor(dataset.capability)]">
                    {{ dataset.capability }}
                  </span>
                </label>
              </div>
            </div>

            <button
              @click="triggerEvaluation"
              :disabled="!canTrigger"
              class="btn-primary w-full flex items-center justify-center gap-2"
            >
              <Play class="w-4 h-4" />
              开始评估
            </button>
          </div>
        </div>

        <!-- Right: Task List -->
        <div class="col-span-7">
          <div class="glass-card rounded-lg p-4">
            <h3 class="text-sm font-medium text-gray-700 mb-3">评估任务</h3>

            <div v-if="evaluationStore.evalTasks.length === 0" class="text-center py-8">
              <Clock class="w-8 h-8 text-gray-300 mx-auto mb-2" />
              <p class="text-sm text-gray-500">暂无评估任务</p>
            </div>

            <div v-else class="space-y-2 max-h-96 overflow-y-auto">
              <div
                v-for="task in evaluationStore.evalTasks"
                :key="task.uuid"
                class="p-3 rounded-lg bg-gray-50"
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm font-medium text-gray-700">{{ task.dataset_name }}</span>
                  <component
                    :is="getStatusIcon(task.status)"
                    :class="['w-4 h-4', getStatusClass(task.status)]"
                  />
                </div>
                <div class="flex items-center gap-3 text-2xs text-gray-500">
                  <span v-if="task.model_type === 'checkpoint' && task.checkpoint_step">
                    Step {{ task.checkpoint_step }}
                  </span>
                  <span v-else-if="task.model_type === 'api'" class="px-1.5 py-0.5 rounded bg-blue-50 text-blue-600">
                    API
                  </span>
                  <span v-else-if="task.model_type === 'local_model'" class="px-1.5 py-0.5 rounded bg-orange-50 text-orange-600">
                    本地模型
                  </span>
                  <span :class="['px-1.5 py-0.5 rounded', getCapabilityColor(task.capability)]">
                    {{ task.capability }}
                  </span>
                  <span v-if="task.score !== null" class="font-medium text-green-600">
                    {{ task.score.toFixed(1) }}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Results Tab -->
    <div v-if="activeTab === 'results'">
      <div class="mb-4">
        <select v-model="resultsJobId" @change="loadJobResults" class="input-light min-w-64">
          <option value="">选择任务查看结果...</option>
          <option v-for="job in jobsStore.jobs" :key="job.id" :value="job.id">
            {{ job.name }}
          </option>
        </select>
      </div>

      <div v-if="jobResults" class="grid grid-cols-12 gap-4">
        <!-- Radar Chart -->
        <div class="col-span-5">
          <div class="glass-card rounded-lg p-4">
            <h3 class="text-sm font-medium text-gray-700 mb-4">能力雷达图</h3>
            <div class="h-64">
              <Radar v-if="radarChartData" :data="radarChartData" :options="radarOptions" />
              <div v-else class="flex items-center justify-center h-full text-gray-500 text-sm">
                暂无数据
              </div>
            </div>
          </div>
        </div>

        <!-- Results Table -->
        <div class="col-span-7">
          <div class="glass-card rounded-lg p-4">
            <h3 class="text-sm font-medium text-gray-700 mb-4">分项得分</h3>
            <div class="space-y-4 max-h-96 overflow-y-auto">
              <div
                v-for="(results, capability) in jobResults.results_by_capability"
                :key="capability"
              >
                <div class="flex items-center gap-2 mb-2">
                  <span :class="['px-2 py-0.5 rounded text-xs', getCapabilityColor(capability)]">
                    {{ capabilities.find(c => c.value === capability)?.label || capability }}
                  </span>
                </div>
                <table class="w-full text-xs">
                  <thead>
                    <tr class="text-left text-gray-400 border-b">
                      <th class="pb-2">步数</th>
                      <th class="pb-2">数据集</th>
                      <th class="pb-2 text-right">得分</th>
                      <th class="pb-2 text-right">变化</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(result, idx) in results"
                      :key="idx"
                      class="border-b border-gray-100"
                    >
                      <td class="py-2 font-medium text-gray-700">{{ result.step }}</td>
                      <td class="py-2 text-gray-600">{{ result.dataset_name }}</td>
                      <td class="py-2 text-right font-medium text-green-600">
                        {{ result.score?.toFixed(1) }}%
                      </td>
                      <td class="py-2 text-right">
                        <span
                          v-if="idx > 0"
                          :class="[
                            'text-xs',
                            result.score > results[idx - 1].score ? 'text-green-500' : 'text-red-500'
                          ]"
                        >
                          {{ result.score > results[idx - 1].score ? '+' : '' }}
                          {{ (result.score - results[idx - 1].score).toFixed(1) }}
                        </span>
                        <span v-else class="text-gray-400">-</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-else class="glass-card rounded-lg p-8 text-center">
        <FileText class="w-10 h-10 text-gray-300 mx-auto mb-3" />
        <p class="text-gray-500 text-sm">选择任务查看评估结果</p>
      </div>
    </div>

    <!-- Compare Tab -->
    <div v-if="activeTab === 'compare'">
      <div class="grid grid-cols-12 gap-4">
        <!-- Left: Comparison List -->
        <div class="col-span-4">
          <div class="glass-card rounded-lg p-4">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-sm font-medium text-gray-700">对照列表</h3>
              <button @click="showCreateComparisonModal = true" class="btn-primary text-xs px-2 py-1">
                + 新建对照
              </button>
            </div>

            <div v-if="comparisons.length === 0" class="text-center py-8">
              <GitCompare class="w-8 h-8 text-gray-300 mx-auto mb-2" />
              <p class="text-sm text-gray-500">暂无模型对照</p>
              <p class="text-2xs text-gray-400 mt-1">创建对照以比较模型表现</p>
            </div>

            <div v-else class="space-y-2 max-h-[500px] overflow-y-auto">
              <div
                v-for="comp in comparisons"
                :key="comp.uuid"
                :class="[
                  'p-3 rounded-lg cursor-pointer transition-all border',
                  selectedComparison?.uuid === comp.uuid
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-transparent bg-gray-50 hover:bg-gray-100'
                ]"
                @click="selectComparison(comp)"
              >
                <div class="flex items-start justify-between">
                  <div class="flex-1 min-w-0">
                    <h4 class="text-sm font-medium text-gray-700 truncate">{{ comp.name }}</h4>
                    <p class="text-2xs text-gray-500 mt-1">
                      {{ comp.model_a_name }} → {{ comp.model_b_name }}
                    </p>
                  </div>
                  <div class="flex items-center gap-1 ml-2">
                    <component
                      :is="getStatusIcon(comp.status)"
                      :class="['w-4 h-4', getStatusClass(comp.status)]"
                    />
                    <button
                      @click.stop="deleteComparison(comp.uuid)"
                      class="p-1 hover:bg-red-100 rounded"
                    >
                      <Trash2 class="w-3.5 h-3.5 text-gray-400" />
                    </button>
                  </div>
                </div>
                <div v-if="comp.status === 'completed' && comp.comparison_results" class="mt-2">
                  <div class="flex items-center gap-2 text-xs">
                    <span class="text-gray-500">准确率变化:</span>
                    <span
                      :class="[
                        'font-medium',
                        comp.comparison_results.accuracy_delta > 0 ? 'text-green-600' :
                        comp.comparison_results.accuracy_delta < 0 ? 'text-red-600' : 'text-gray-600'
                      ]"
                    >
                      {{ comp.comparison_results.accuracy_delta > 0 ? '+' : '' }}{{ comp.comparison_results.accuracy_delta?.toFixed(1) }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right: Comparison Details -->
        <div class="col-span-8">
          <div v-if="!selectedComparison" class="glass-card rounded-lg p-8 text-center">
            <GitCompare class="w-10 h-10 text-gray-300 mx-auto mb-3" />
            <p class="text-gray-500 text-sm">选择或创建对照以查看详情</p>
          </div>

          <div v-else-if="loadingComparison" class="glass-card rounded-lg p-8 text-center">
            <RefreshCw class="w-8 h-8 text-primary-500 animate-spin mx-auto mb-3" />
            <p class="text-gray-500 text-sm">加载对照数据...</p>
          </div>

          <div v-else class="space-y-4">
            <!-- Overall Stats -->
            <div class="glass-card rounded-lg p-4">
              <h3 class="text-sm font-medium text-gray-700 mb-4">整体对比</h3>
              <div class="grid grid-cols-3 gap-4">
                <!-- Model A Score -->
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                  <p class="text-2xs text-gray-500 mb-1">{{ selectedComparison.model_a_name }}</p>
                  <p class="text-2xl font-bold text-gray-700">
                    {{ selectedComparison.comparison_results?.model_a_accuracy?.toFixed(1) || '-' }}%
                  </p>
                </div>

                <!-- Delta -->
                <div class="text-center p-4 rounded-lg flex flex-col items-center justify-center"
                  :class="[
                    selectedComparison.comparison_results?.accuracy_delta > 0 ? 'bg-green-50' :
                    selectedComparison.comparison_results?.accuracy_delta < 0 ? 'bg-red-50' : 'bg-gray-50'
                  ]"
                >
                  <div class="flex items-center gap-1">
                    <ArrowUp v-if="selectedComparison.comparison_results?.accuracy_delta > 0" class="w-5 h-5 text-green-600" />
                    <ArrowDown v-else-if="selectedComparison.comparison_results?.accuracy_delta < 0" class="w-5 h-5 text-red-600" />
                    <Minus v-else class="w-5 h-5 text-gray-500" />
                    <span
                      :class="[
                        'text-2xl font-bold',
                        selectedComparison.comparison_results?.accuracy_delta > 0 ? 'text-green-600' :
                        selectedComparison.comparison_results?.accuracy_delta < 0 ? 'text-red-600' : 'text-gray-600'
                      ]"
                    >
                      {{ selectedComparison.comparison_results?.accuracy_delta > 0 ? '+' : '' }}{{ selectedComparison.comparison_results?.accuracy_delta?.toFixed(1) || '0' }}%
                    </span>
                  </div>
                  <p class="text-2xs text-gray-500 mt-1">准确率变化</p>
                </div>

                <!-- Model B Score -->
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                  <p class="text-2xs text-gray-500 mb-1">{{ selectedComparison.model_b_name }}</p>
                  <p class="text-2xl font-bold text-gray-700">
                    {{ selectedComparison.comparison_results?.model_b_accuracy?.toFixed(1) || '-' }}%
                  </p>
                </div>
              </div>

              <!-- Sample Counts -->
              <div class="flex justify-center gap-6 mt-4 pt-4 border-t border-gray-100">
                <div class="flex items-center gap-1.5">
                  <ArrowUp class="w-4 h-4 text-green-500" />
                  <span class="text-sm text-gray-600">
                    <span class="font-medium text-green-600">{{ selectedComparison.comparison_results?.improved_count || 0 }}</span> 改进
                  </span>
                </div>
                <div class="flex items-center gap-1.5">
                  <ArrowDown class="w-4 h-4 text-red-500" />
                  <span class="text-sm text-gray-600">
                    <span class="font-medium text-red-600">{{ selectedComparison.comparison_results?.degraded_count || 0 }}</span> 退化
                  </span>
                </div>
                <div class="flex items-center gap-1.5">
                  <Minus class="w-4 h-4 text-gray-400" />
                  <span class="text-sm text-gray-600">
                    <span class="font-medium text-gray-600">{{ selectedComparison.comparison_results?.unchanged_count || 0 }}</span> 不变
                  </span>
                </div>
              </div>
            </div>

            <!-- Sample Diffs -->
            <div class="glass-card rounded-lg p-4">
              <div class="flex items-center justify-between mb-4">
                <h3 class="text-sm font-medium text-gray-700">样本级对比</h3>
                <div class="flex gap-1">
                  <button
                    @click="diffFilter = 'all'"
                    :class="[
                      'px-2 py-1 text-xs rounded',
                      diffFilter === 'all' ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    ]"
                  >
                    全部 ({{ comparisonDiffs.length }})
                  </button>
                  <button
                    @click="diffFilter = 'improved'"
                    :class="[
                      'px-2 py-1 text-xs rounded',
                      diffFilter === 'improved' ? 'bg-green-600 text-white' : 'bg-green-50 text-green-600 hover:bg-green-100'
                    ]"
                  >
                    改进
                  </button>
                  <button
                    @click="diffFilter = 'degraded'"
                    :class="[
                      'px-2 py-1 text-xs rounded',
                      diffFilter === 'degraded' ? 'bg-red-600 text-white' : 'bg-red-50 text-red-600 hover:bg-red-100'
                    ]"
                  >
                    退化
                  </button>
                </div>
              </div>

              <div v-if="filteredDiffs.length === 0" class="text-center py-8 text-gray-400 text-sm">
                暂无样本数据
              </div>

              <div v-else class="space-y-3 max-h-96 overflow-y-auto">
                <div
                  v-for="(diff, idx) in filteredDiffs"
                  :key="idx"
                  class="p-3 rounded-lg border"
                  :class="[
                    diff.change === 'improved' ? 'border-green-200 bg-green-50/50' :
                    diff.change === 'degraded' ? 'border-red-200 bg-red-50/50' : 'border-gray-200 bg-gray-50'
                  ]"
                >
                  <!-- Question -->
                  <div class="mb-2">
                    <span class="text-2xs text-gray-400 uppercase">问题</span>
                    <p class="text-sm text-gray-700 mt-0.5">{{ diff.input }}</p>
                  </div>

                  <!-- Expected Answer -->
                  <div class="mb-2">
                    <span class="text-2xs text-gray-400 uppercase">期望答案</span>
                    <p class="text-sm text-gray-600 mt-0.5 font-mono bg-white/50 px-2 py-1 rounded">{{ diff.expected }}</p>
                  </div>

                  <!-- Responses Comparison -->
                  <div class="grid grid-cols-2 gap-3 mt-3 pt-3 border-t border-gray-200">
                    <div>
                      <div class="flex items-center gap-1 mb-1">
                        <span class="text-2xs text-gray-500">{{ selectedComparison.model_a_name }}</span>
                        <component
                          :is="diff.model_a_correct ? CheckCircle : XCircle"
                          :class="['w-3 h-3', diff.model_a_correct ? 'text-green-500' : 'text-red-500']"
                        />
                      </div>
                      <p class="text-xs text-gray-700 bg-white/50 px-2 py-1.5 rounded break-words">
                        {{ diff.model_a_response || '(无响应)' }}
                      </p>
                    </div>
                    <div>
                      <div class="flex items-center gap-1 mb-1">
                        <span class="text-2xs text-gray-500">{{ selectedComparison.model_b_name }}</span>
                        <component
                          :is="diff.model_b_correct ? CheckCircle : XCircle"
                          :class="['w-3 h-3', diff.model_b_correct ? 'text-green-500' : 'text-red-500']"
                        />
                      </div>
                      <p class="text-xs text-gray-700 bg-white/50 px-2 py-1.5 rounded break-words">
                        {{ diff.model_b_response || '(无响应)' }}
                      </p>
                    </div>
                  </div>

                  <!-- Change indicator -->
                  <div class="flex justify-end mt-2">
                    <span
                      :class="[
                        'px-2 py-0.5 text-2xs rounded-full',
                        diff.change === 'improved' ? 'bg-green-100 text-green-700' :
                        diff.change === 'degraded' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'
                      ]"
                    >
                      {{ diff.change === 'improved' ? '改进' : diff.change === 'degraded' ? '退化' : '不变' }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Create Comparison Modal -->
    <Modal :show="showCreateComparisonModal" title="创建模型对照" @close="showCreateComparisonModal = false">
      <div class="space-y-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">对照名称 *</label>
          <input type="text" v-model="comparisonForm.name" class="w-full input-light" placeholder="如: Step 1000 vs 训练前">
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">描述</label>
          <textarea v-model="comparisonForm.description" rows="2" class="w-full input-light" placeholder="对照说明..." />
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">选择评估数据集 *</label>
          <select v-model="comparisonForm.dataset_uuid" @change="loadTasksForDataset" class="w-full input-light">
            <option value="">请选择...</option>
            <option v-for="ds in evaluationStore.datasets" :key="ds.uuid" :value="ds.uuid">
              {{ ds.name }}
            </option>
          </select>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <!-- Model A -->
          <div class="p-3 bg-gray-50 rounded-lg space-y-3">
            <h4 class="text-xs font-medium text-gray-700">基线模型 (A)</h4>
            <div>
              <label class="block text-2xs text-gray-500 mb-1">名称</label>
              <input type="text" v-model="comparisonForm.model_a_name" class="w-full input-light text-xs" placeholder="如: 训练前">
            </div>
            <div>
              <label class="block text-2xs text-gray-500 mb-1">评估任务</label>
              <select v-model="comparisonForm.model_a_task_uuid" class="w-full input-light text-xs">
                <option value="">请选择...</option>
                <option v-for="task in tasksForDataset" :key="task.uuid" :value="task.uuid">
                  {{ task.name || task.dataset_name }} ({{ task.score?.toFixed(1) }}%)
                </option>
              </select>
            </div>
          </div>

          <!-- Model B -->
          <div class="p-3 bg-gray-50 rounded-lg space-y-3">
            <h4 class="text-xs font-medium text-gray-700">目标模型 (B)</h4>
            <div>
              <label class="block text-2xs text-gray-500 mb-1">名称</label>
              <input type="text" v-model="comparisonForm.model_b_name" class="w-full input-light text-xs" placeholder="如: Step 1000">
            </div>
            <div>
              <label class="block text-2xs text-gray-500 mb-1">评估任务</label>
              <select v-model="comparisonForm.model_b_task_uuid" class="w-full input-light text-xs">
                <option value="">请选择...</option>
                <option v-for="task in tasksForDataset" :key="task.uuid" :value="task.uuid">
                  {{ task.name || task.dataset_name }} ({{ task.score?.toFixed(1) }}%)
                </option>
              </select>
            </div>
          </div>
        </div>

        <div class="flex justify-end gap-3 pt-2">
          <button @click="showCreateComparisonModal = false" class="btn-secondary">取消</button>
          <button
            @click="createComparison"
            :disabled="!comparisonForm.name || !comparisonForm.model_a_task_uuid || !comparisonForm.model_b_task_uuid"
            class="btn-primary"
          >
            创建对照
          </button>
        </div>
      </div>
    </Modal>

    <!-- Upload Modal -->
    <Modal :show="showUploadModal" title="上传评估数据集" @close="showUploadModal = false">
      <div class="space-y-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">选择文件 *</label>
          <input
            type="file"
            accept=".jsonl,.json,.ndjson"
            @change="handleFileSelect"
            class="w-full input-light"
          >
          <p class="text-2xs text-gray-400 mt-1">支持 .jsonl, .json, .ndjson 格式</p>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">数据集名称 *</label>
            <input type="text" v-model="uploadForm.name" class="w-full input-light">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">数据格式 *</label>
            <select v-model="uploadForm.format" class="w-full input-light">
              <option v-for="f in formats" :key="f.value" :value="f.value">
                {{ f.label }}
              </option>
            </select>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">能力类别 *</label>
            <select v-model="uploadForm.capability" class="w-full input-light">
              <option v-for="c in capabilities" :key="c.value" :value="c.value">
                {{ c.label }}
              </option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">评估方法 *</label>
            <select v-model="uploadForm.eval_method" class="w-full input-light">
              <option v-for="m in evalMethods" :key="m.value" :value="m.value">
                {{ m.label }}
              </option>
            </select>
          </div>
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">描述</label>
          <textarea v-model="uploadForm.description" rows="2" class="w-full input-light" />
        </div>

        <div class="flex justify-end gap-3">
          <button @click="showUploadModal = false" class="btn-secondary">取消</button>
          <button
            @click="submitUpload"
            :disabled="!uploadForm.file || !uploadForm.name"
            class="btn-primary"
          >
            上传
          </button>
        </div>
      </div>
    </Modal>

    <!-- Preview Modal -->
    <Modal :show="showPreviewModal" title="数据预览" @close="showPreviewModal = false" maxWidth="max-w-4xl">
      <div v-if="previewData">
        <p class="text-sm text-gray-500 mb-3">
          共 {{ previewData.sample_count }} 条，显示前 {{ previewData.samples.length }} 条
        </p>
        <div class="space-y-3 max-h-96 overflow-y-auto">
          <div
            v-for="(sample, idx) in previewData.samples"
            :key="idx"
            class="p-3 bg-gray-50 rounded-lg text-sm"
          >
            <template v-if="previewData.format === 'qa'">
              <p class="font-medium text-gray-700 mb-1">Q: {{ sample.question }}</p>
              <p class="text-gray-600">A: {{ sample.answer }}</p>
            </template>
            <template v-else>
              <div v-for="(msg, i) in sample.messages" :key="i" class="mb-1">
                <span :class="msg.role === 'user' ? 'text-blue-600' : 'text-green-600'">
                  {{ msg.role }}:
                </span>
                <span class="text-gray-700 ml-1">{{ msg.content }}</span>
              </div>
            </template>
          </div>
        </div>
      </div>
    </Modal>
  </div>
</template>
