<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useJobsStore } from '@/stores/jobs'
import * as api from '@/api'
import {
  Database,
  FileText,
  Eye,
  BarChart3,
  RefreshCw,
  Upload,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Settings,
  PieChart,
  Tag,
  Inbox,
  Cloud,
  CloudOff,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Activity,
  MessageSquare,
  Hash,
  AlertTriangle,
  TrendingUp
} from 'lucide-vue-next'

const jobsStore = useJobsStore()
const loading = ref(false)
const trainingDatasets = ref([])
const selectedDataset = ref(null)
const currentSample = ref(null)
const sampleIndex = ref(0)
const distribution = ref([])
const previewLoading = ref(false)
const activeTab = ref('overview')  // 默认显示概览
const showUploadModal = ref(false)

// Stats and Quality
const datasetStats = ref(null)
const qualityCheck = ref(null)
const statsLoading = ref(false)

// Upload form
const uploadForm = ref({
  name: '',
  description: '',
  labelFields: '',
  promptField: 'prompt',
  responseField: 'response'
})
const uploadFile = ref(null)
const uploading = ref(false)
const detectedFormat = ref(null)  // 'messages' or 'prompt_response' or null
const detectedColumns = ref([])  // Detected columns from file
const selectedLabelFields = ref([])  // Selected label fields
const syncing = ref({})  // Track syncing status per dataset uuid

const getSyncStatusInfo = (status) => {
  const statusMap = {
    'synced': { icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-100', text: '已同步' },
    'syncing': { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100', text: '同步中', spin: true },
    'failed': { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-100', text: '同步失败' },
    'not_synced': { icon: CloudOff, color: 'text-gray-400', bg: 'bg-gray-100', text: '未同步' },
  }
  return statusMap[status] || statusMap['not_synced']
}

const syncDataset = async (dataset) => {
  syncing.value[dataset.uuid] = true
  try {
    const result = await api.syncTrainingDataset(dataset.uuid, true)
    // Update the dataset in the list
    const idx = trainingDatasets.value.findIndex(d => d.uuid === dataset.uuid)
    if (idx !== -1) {
      trainingDatasets.value[idx] = result
    }
    if (selectedDataset.value?.uuid === dataset.uuid) {
      selectedDataset.value = result
    }
  } catch (error) {
    console.error('Sync failed:', error)
    alert('Sync failed: ' + error.message)
  } finally {
    syncing.value[dataset.uuid] = false
  }
}

const formatFileSize = (mb) => {
  if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`
  if (mb < 1024) return `${mb.toFixed(2)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}

const getFormatColor = (format) => {
  const colors = {
    parquet: 'bg-blue-100 text-blue-600',
    jsonl: 'bg-green-100 text-green-600',
    json: 'bg-yellow-100 text-yellow-600',
    csv: 'bg-purple-100 text-purple-600'
  }
  return colors[format] || 'bg-gray-100 text-gray-600'
}

const fetchTrainingDatasets = async () => {
  loading.value = true
  try {
    trainingDatasets.value = await api.getTrainingDatasets()
  } catch (error) {
    console.error('Failed to fetch training datasets:', error)
  } finally {
    loading.value = false
  }
}

const selectDataset = async (dataset) => {
  selectedDataset.value = dataset
  activeTab.value = 'overview'
  sampleIndex.value = 0
  distribution.value = []
  currentSample.value = null
  datasetStats.value = null
  qualityCheck.value = null

  // Load stats and quality in parallel
  await Promise.all([
    loadStats(dataset),
    loadQuality(dataset),
    loadDistribution(dataset),
  ])
}

const loadStats = async (dataset) => {
  statsLoading.value = true
  try {
    datasetStats.value = await api.getTrainingDatasetStats(dataset.uuid)
  } catch (error) {
    console.error('Failed to load stats:', error)
  } finally {
    statsLoading.value = false
  }
}

const loadQuality = async (dataset) => {
  try {
    qualityCheck.value = await api.getTrainingDatasetQuality(dataset.uuid)
  } catch (error) {
    console.error('Failed to load quality:', error)
  }
}

// Helper to get bar width percentage
const getBarWidth = (value, distribution) => {
  const max = Math.max(...Object.values(distribution))
  return max > 0 ? (value / max) * 100 : 0
}

// Sort distribution keys in order
const sortedDistKeys = (dist) => {
  const order = ['0-50', '50-100', '100-200', '200-500', '500-1k', '1k-2k', '2k+', '1轮', '2轮', '3轮', '4-5轮', '6+轮']
  return Object.keys(dist).sort((a, b) => {
    const ia = order.indexOf(a)
    const ib = order.indexOf(b)
    if (ia !== -1 && ib !== -1) return ia - ib
    return a.localeCompare(b)
  })
}

// Quality score color
const getQualityColor = (score) => {
  if (score >= 90) return 'text-green-600'
  if (score >= 70) return 'text-yellow-600'
  return 'text-red-600'
}

const loadDistribution = async (dataset) => {
  try {
    distribution.value = await api.getTrainingDatasetDistribution(dataset.uuid)
  } catch (error) {
    console.error('Failed to load distribution:', error)
  }
}

const loadSample = async (dataset, index) => {
  previewLoading.value = true
  try {
    currentSample.value = await api.getTrainingDatasetSample(dataset.uuid, index)
    sampleIndex.value = index
  } catch (error) {
    console.error('Failed to load sample:', error)
  } finally {
    previewLoading.value = false
  }
}

const prevSample = () => {
  if (sampleIndex.value > 0) {
    loadSample(selectedDataset.value, sampleIndex.value - 1)
  }
}

const nextSample = () => {
  if (sampleIndex.value < selectedDataset.value.total_rows - 1) {
    loadSample(selectedDataset.value, sampleIndex.value + 1)
  }
}

const openUploadModal = () => {
  // Reset form when opening modal
  uploadForm.value = {
    name: '',
    description: '',
    labelFields: '',
    promptField: 'prompt',
    responseField: 'response'
  }
  uploadFile.value = null
  detectedFormat.value = null
  detectedColumns.value = []
  selectedLabelFields.value = []
  showUploadModal.value = true
}

const handleFileSelect = async (e) => {
  const file = e.target.files[0]
  if (!file) return

  uploadFile.value = file
  if (!uploadForm.value.name) {
    uploadForm.value.name = file.name.replace(/\.[^/.]+$/, '')
  }

  // Auto-detect format and columns by reading first line
  detectedFormat.value = null
  detectedColumns.value = []
  selectedLabelFields.value = []
  try {
    const text = await file.slice(0, 10000).text()  // Read first 10KB
    const firstLine = text.split('\n')[0]
    if (firstLine) {
      const parsed = JSON.parse(firstLine)

      // Detect all columns
      const columns = Object.keys(parsed)

      if (parsed.messages && Array.isArray(parsed.messages)) {
        detectedFormat.value = 'messages'
        // For messages format, exclude 'messages' from label field options
        detectedColumns.value = columns.filter(c => c !== 'messages')
      } else if (parsed.prompt !== undefined || parsed.response !== undefined) {
        detectedFormat.value = 'prompt_response'
        // Exclude prompt/response fields from label options
        detectedColumns.value = columns.filter(c =>
          c !== uploadForm.value.promptField && c !== uploadForm.value.responseField
        )
      } else {
        // Unknown format, show all columns
        detectedColumns.value = columns
      }

      // Auto-select common label fields
      const commonLabels = ['domain', 'intent', 'category', 'type', 'difficulty', 'product', 'scenario']
      selectedLabelFields.value = detectedColumns.value.filter(c =>
        commonLabels.includes(c.toLowerCase())
      )
    }
  } catch (err) {
    console.warn('Could not detect format:', err)
  }
}

const handleUpload = async () => {
  if (!uploadFile.value || !uploadForm.value.name) return

  uploading.value = true
  try {
    const formData = new FormData()
    formData.append('file', uploadFile.value)

    const params = {
      name: uploadForm.value.name,
      description: uploadForm.value.description || '',
      prompt_field: uploadForm.value.promptField,
      response_field: uploadForm.value.responseField
    }
    // Use selected label fields
    if (selectedLabelFields.value.length > 0) {
      params.label_fields = selectedLabelFields.value.join(',')
    }

    await api.uploadTrainingDataset(formData, params)
    showUploadModal.value = false
    uploadForm.value = { name: '', description: '', labelFields: '', promptField: 'prompt', responseField: 'response' }
    uploadFile.value = null
    detectedColumns.value = []
    selectedLabelFields.value = []
    await fetchTrainingDatasets()
  } catch (error) {
    console.error('Upload failed:', error)
    alert('Upload failed: ' + error.message)
  } finally {
    uploading.value = false
  }
}

const deleteDataset = async (dataset) => {
  if (!confirm(`Delete dataset "${dataset.name}"?`)) return

  try {
    await api.deleteTrainingDataset(dataset.uuid)
    if (selectedDataset.value?.uuid === dataset.uuid) {
      selectedDataset.value = null
    }
    await fetchTrainingDatasets()
  } catch (error) {
    console.error('Delete failed:', error)
  }
}

const getDistributionChartData = (dist) => {
  if (!dist.distribution) return []
  return Object.entries(dist.distribution).map(([name, value]) => ({
    name,
    value
  }))
}

onMounted(async () => {
  await fetchTrainingDatasets()
})
</script>

<template>
  <div>
    <!-- Header -->
    <div class="flex justify-between items-center mb-4">
      <div>
        <h2 class="text-lg font-semibold text-gray-800">训练数据集</h2>
        <p class="text-xs text-gray-500">管理训练数据集，分析标签分布和数据质量</p>
      </div>
      <div class="flex gap-2">
        <button @click="fetchTrainingDatasets" :disabled="loading" class="btn-secondary flex items-center gap-1.5 text-xs">
          <RefreshCw :class="['w-3.5 h-3.5', loading && 'animate-spin']" />
          刷新
        </button>
        <button @click="openUploadModal" class="btn-primary flex items-center gap-1.5 text-xs">
          <Upload class="w-3.5 h-3.5" />
          上传
        </button>
      </div>
    </div>

    <div class="grid grid-cols-12 gap-4" style="height: calc(100vh - 180px);">
      <!-- Dataset List - Fixed -->
      <div class="col-span-4 h-full">
        <div class="glass-card rounded-lg p-4 h-full flex flex-col">
          <h3 class="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
            <Database class="w-4 h-4" />
            数据集列表
            <span class="text-2xs text-gray-400">({{ trainingDatasets.length }})</span>
          </h3>

          <div v-if="trainingDatasets.length === 0" class="text-center py-8">
            <Inbox class="w-10 h-10 text-gray-300 mx-auto mb-3" />
            <p class="text-sm text-gray-500">暂无数据集</p>
            <p class="text-2xs text-gray-400 mt-1">上传训练数据集开始使用</p>
          </div>

          <div v-else class="space-y-2 flex-1 overflow-y-auto">
            <div
              v-for="ds in trainingDatasets"
              :key="ds.uuid"
              @click="selectDataset(ds)"
              :class="[
                'p-3 rounded-lg cursor-pointer transition-all relative group',
                selectedDataset?.uuid === ds.uuid
                  ? 'bg-primary-50 border border-primary-200'
                  : 'bg-gray-50 hover:bg-gray-100 border border-transparent'
              ]"
            >
              <button
                @click.stop="deleteDataset(ds)"
                class="absolute top-2 right-2 p-1 rounded hover:bg-red-100 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <Trash2 class="w-3.5 h-3.5" />
              </button>
              <div class="flex items-center justify-between mb-1 pr-6">
                <span class="text-sm font-medium text-gray-700 truncate flex-1">
                  {{ ds.name }}
                </span>
                <span :class="['px-1.5 py-0.5 rounded text-2xs uppercase', getFormatColor(ds.file_format)]">
                  {{ ds.file_format }}
                </span>
              </div>
              <div class="flex gap-3 text-2xs text-gray-500 items-center">
                <span>{{ ds.total_rows.toLocaleString() }} rows</span>
                <span>{{ formatFileSize(ds.file_size_mb) }}</span>
                <span
                  v-if="ds.sync_status"
                  :class="['flex items-center gap-0.5', getSyncStatusInfo(ds.sync_status).color]"
                  :title="ds.sync_status === 'synced' ? ds.remote_path : (ds.sync_error || '')"
                >
                  <component
                    :is="getSyncStatusInfo(ds.sync_status).icon"
                    :class="['w-3 h-3', getSyncStatusInfo(ds.sync_status).spin && 'animate-spin']"
                  />
                  <span class="text-2xs">{{ getSyncStatusInfo(ds.sync_status).text }}</span>
                </span>
              </div>
              <div v-if="ds.label_fields?.length > 0" class="flex gap-1 mt-1.5 flex-wrap">
                <span
                  v-for="field in ds.label_fields"
                  :key="field"
                  class="px-1.5 py-0.5 bg-gray-200 rounded text-2xs text-gray-600"
                >
                  <Tag class="w-2.5 h-2.5 inline mr-0.5" />
                  {{ field }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Dataset Details - Scrollable -->
      <div class="col-span-8 h-full overflow-y-auto">
        <div v-if="!selectedDataset" class="glass-card rounded-lg p-8 text-center">
          <FileText class="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p class="text-gray-500">选择数据集查看详情</p>
        </div>

        <div v-else class="glass-card rounded-lg">
          <!-- Dataset Header -->
          <div class="p-4 border-b border-gray-100">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-sm font-medium text-gray-800">{{ selectedDataset.name }}</h3>
                <p class="text-2xs text-gray-500 mt-0.5">{{ selectedDataset.total_rows.toLocaleString() }} 条样本</p>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-gray-500">
                  损失字段: <code class="bg-gray-100 px-1 rounded">{{ selectedDataset.response_field }}</code>
                </span>
              </div>
            </div>
          </div>

          <!-- Tabs -->
          <div class="flex border-b border-gray-100 overflow-x-auto">
            <button
              @click="activeTab = 'overview'"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors whitespace-nowrap',
                activeTab === 'overview'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <Activity class="w-3.5 h-3.5 inline mr-1" />
              概览
            </button>
            <button
              @click="activeTab = 'distribution'"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors whitespace-nowrap',
                activeTab === 'distribution'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <PieChart class="w-3.5 h-3.5 inline mr-1" />
              标签分布
            </button>
            <button
              @click="activeTab = 'quality'"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors whitespace-nowrap',
                activeTab === 'quality'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <AlertTriangle class="w-3.5 h-3.5 inline mr-1" />
              质量检测
              <span v-if="qualityCheck?.issues_found" class="ml-1 px-1.5 py-0.5 bg-red-100 text-red-600 rounded text-2xs">
                {{ qualityCheck.issues_found }}
              </span>
            </button>
            <button
              @click="activeTab = 'preview'; loadSample(selectedDataset, 0)"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors whitespace-nowrap',
                activeTab === 'preview'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <Eye class="w-3.5 h-3.5 inline mr-1" />
              样本预览
            </button>
            <button
              @click="activeTab = 'settings'"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors whitespace-nowrap',
                activeTab === 'settings'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <Settings class="w-3.5 h-3.5 inline mr-1" />
              设置
            </button>
          </div>

          <!-- Content -->
          <div class="p-4">
            <!-- Overview Tab -->
            <div v-if="activeTab === 'overview'">
              <div v-if="statsLoading" class="text-center py-8">
                <RefreshCw class="w-6 h-6 text-gray-400 mx-auto mb-2 animate-spin" />
                <p class="text-sm text-gray-500">加载统计数据...</p>
              </div>

              <div v-else-if="datasetStats" class="space-y-6">
                <!-- Stats Cards -->
                <div class="grid grid-cols-4 gap-3">
                  <div class="bg-blue-50 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <Hash class="w-4 h-4 text-blue-500" />
                      <span class="text-2xs text-blue-600">样本数</span>
                    </div>
                    <div class="text-xl font-semibold text-blue-700">{{ datasetStats.total_samples.toLocaleString() }}</div>
                  </div>
                  <div class="bg-purple-50 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <MessageSquare class="w-4 h-4 text-purple-500" />
                      <span class="text-2xs text-purple-600">平均轮次</span>
                    </div>
                    <div class="text-xl font-semibold text-purple-700">{{ datasetStats.avg_turns }}</div>
                  </div>
                  <div class="bg-green-50 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <TrendingUp class="w-4 h-4 text-green-500" />
                      <span class="text-2xs text-green-600">平均回复长度</span>
                    </div>
                    <div class="text-xl font-semibold text-green-700">{{ datasetStats.avg_response_chars }} 字</div>
                  </div>
                  <div class="bg-orange-50 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <CheckCircle2 class="w-4 h-4 text-orange-500" />
                      <span class="text-2xs text-orange-600">质量得分</span>
                    </div>
                    <div :class="['text-xl font-semibold', getQualityColor(qualityCheck?.quality_score || 0)]">
                      {{ qualityCheck?.quality_score || '-' }}
                    </div>
                  </div>
                </div>

                <!-- Format Info -->
                <div class="bg-gray-50 rounded-lg p-3">
                  <div class="flex items-center gap-4 text-xs">
                    <span class="text-gray-500">格式:</span>
                    <span class="px-2 py-0.5 bg-purple-100 text-purple-700 rounded">
                      {{ datasetStats.format_type === 'messages' ? 'OpenAI Messages' : 'Prompt/Response' }}
                    </span>
                    <span class="text-gray-500">System Prompt:</span>
                    <span class="text-gray-700">{{ datasetStats.has_system_prompt }}%</span>
                    <span class="text-gray-500">平均总长度:</span>
                    <span class="text-gray-700">{{ datasetStats.avg_total_chars }} 字</span>
                  </div>
                </div>

                <!-- Length Distribution -->
                <div class="grid grid-cols-2 gap-4">
                  <!-- Prompt Length -->
                  <div class="bg-gray-50 rounded-lg p-4">
                    <h4 class="text-xs font-medium text-gray-700 mb-3">Prompt 长度分布</h4>
                    <div class="space-y-2">
                      <div v-for="key in sortedDistKeys(datasetStats.prompt_length_distribution)" :key="key" class="flex items-center gap-2">
                        <span class="text-2xs text-gray-500 w-12">{{ key }}</span>
                        <div class="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            class="h-full bg-blue-500 rounded-full"
                            :style="{ width: getBarWidth(datasetStats.prompt_length_distribution[key], datasetStats.prompt_length_distribution) + '%' }"
                          />
                        </div>
                        <span class="text-2xs text-gray-600 w-10 text-right">{{ datasetStats.prompt_length_distribution[key] }}</span>
                      </div>
                    </div>
                  </div>

                  <!-- Response Length -->
                  <div class="bg-gray-50 rounded-lg p-4">
                    <h4 class="text-xs font-medium text-gray-700 mb-3">Response 长度分布</h4>
                    <div class="space-y-2">
                      <div v-for="key in sortedDistKeys(datasetStats.response_length_distribution)" :key="key" class="flex items-center gap-2">
                        <span class="text-2xs text-gray-500 w-12">{{ key }}</span>
                        <div class="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            class="h-full bg-green-500 rounded-full"
                            :style="{ width: getBarWidth(datasetStats.response_length_distribution[key], datasetStats.response_length_distribution) + '%' }"
                          />
                        </div>
                        <span class="text-2xs text-gray-600 w-10 text-right">{{ datasetStats.response_length_distribution[key] }}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Turns Distribution -->
                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-xs font-medium text-gray-700 mb-3">对话轮次分布</h4>
                  <div class="flex gap-4">
                    <div v-for="key in sortedDistKeys(datasetStats.turns_distribution)" :key="key" class="flex-1">
                      <div class="text-center">
                        <div class="h-20 flex items-end justify-center mb-1">
                          <div
                            class="w-8 bg-purple-500 rounded-t"
                            :style="{ height: getBarWidth(datasetStats.turns_distribution[key], datasetStats.turns_distribution) + '%' }"
                          />
                        </div>
                        <div class="text-2xs text-gray-600">{{ key }}</div>
                        <div class="text-2xs text-gray-500">{{ datasetStats.turns_distribution[key] }}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Quality Tab -->
            <div v-else-if="activeTab === 'quality'">
              <div v-if="!qualityCheck" class="text-center py-8">
                <RefreshCw class="w-6 h-6 text-gray-400 mx-auto mb-2 animate-spin" />
                <p class="text-sm text-gray-500">检测中...</p>
              </div>

              <div v-else class="space-y-4">
                <!-- Quality Score Card -->
                <div class="bg-gray-50 rounded-lg p-4 flex items-center justify-between">
                  <div>
                    <h4 class="text-sm font-medium text-gray-700">数据质量评分</h4>
                    <p class="text-2xs text-gray-500 mt-0.5">基于空回复、短回复、重复等指标计算</p>
                  </div>
                  <div :class="['text-3xl font-bold', getQualityColor(qualityCheck.quality_score)]">
                    {{ qualityCheck.quality_score }}
                    <span class="text-sm font-normal text-gray-400">/ 100</span>
                  </div>
                </div>

                <!-- Issues Summary -->
                <div v-if="qualityCheck.issues.length === 0" class="bg-green-50 rounded-lg p-4 text-center">
                  <CheckCircle2 class="w-8 h-8 text-green-500 mx-auto mb-2" />
                  <p class="text-sm text-green-700">未发现质量问题</p>
                  <p class="text-2xs text-green-600 mt-1">数据集质量良好，可以用于训练</p>
                </div>

                <div v-else class="space-y-3">
                  <div
                    v-for="issue in qualityCheck.issues"
                    :key="issue.issue_type"
                    class="bg-red-50 rounded-lg p-4 border border-red-100"
                  >
                    <div class="flex items-center justify-between mb-2">
                      <div class="flex items-center gap-2">
                        <AlertTriangle class="w-4 h-4 text-red-500" />
                        <span class="text-sm font-medium text-red-700">{{ issue.issue_type }}</span>
                      </div>
                      <div class="text-right">
                        <span class="text-sm font-semibold text-red-600">{{ issue.count }}</span>
                        <span class="text-2xs text-red-500 ml-1">({{ issue.percentage }}%)</span>
                      </div>
                    </div>
                    <div class="text-2xs text-red-600">
                      问题样本索引: {{ issue.sample_indices.slice(0, 3).join(', ') }}{{ issue.sample_indices.length > 3 ? '...' : '' }}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Sample Viewer Tab -->
            <div v-else-if="activeTab === 'preview'">
              <!-- Navigation -->
              <div class="flex items-center justify-between mb-4">
                <div class="flex items-center gap-2">
                  <button
                    @click="prevSample"
                    :disabled="sampleIndex === 0 || previewLoading"
                    class="p-1.5 rounded hover:bg-gray-100 disabled:opacity-50 text-gray-600"
                  >
                    <ChevronLeft class="w-4 h-4" />
                  </button>
                  <span class="text-xs text-gray-600">
                    第 {{ sampleIndex + 1 }} / {{ selectedDataset.total_rows.toLocaleString() }} 条
                  </span>
                  <button
                    @click="nextSample"
                    :disabled="sampleIndex >= selectedDataset.total_rows - 1 || previewLoading"
                    class="p-1.5 rounded hover:bg-gray-100 disabled:opacity-50 text-gray-600"
                  >
                    <ChevronRight class="w-4 h-4" />
                  </button>
                </div>
                <div v-if="currentSample?.labels && Object.keys(currentSample.labels).length > 0" class="flex gap-2">
                  <span
                    v-for="(value, key) in currentSample.labels"
                    :key="key"
                    class="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-2xs"
                  >
                    {{ key }}: {{ value }}
                  </span>
                </div>
              </div>

              <!-- Loading -->
              <div v-if="previewLoading" class="text-center py-8">
                <RefreshCw class="w-6 h-6 text-gray-400 mx-auto mb-2 animate-spin" />
                <p class="text-sm text-gray-500">加载中...</p>
              </div>

              <!-- Sample Content with Loss Highlighting -->
              <div v-else-if="currentSample" class="space-y-4">
                <div
                  v-for="(segment, idx) in currentSample.loss_segments"
                  :key="idx"
                  :class="[
                    'p-3 rounded-lg border',
                    segment.field === 'system' ? 'bg-purple-50 border-purple-200' :
                    segment.field === 'user' ? 'bg-blue-50 border-blue-200' :
                    segment.field === 'assistant' ? 'bg-green-50 border-green-200' :
                    segment.computes_loss ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                  ]"
                >
                  <div class="flex items-center justify-between mb-2">
                    <span :class="[
                      'text-xs font-medium',
                      segment.field === 'system' ? 'text-purple-700' :
                      segment.field === 'user' ? 'text-blue-700' :
                      segment.field === 'assistant' ? 'text-green-700' : 'text-gray-700'
                    ]">
                      {{ segment.field === 'system' ? '🔧 System' :
                         segment.field === 'user' ? '👤 User' :
                         segment.field === 'assistant' ? '🤖 Assistant' : segment.field }}
                    </span>
                    <span
                      :class="[
                        'px-1.5 py-0.5 rounded text-2xs',
                        segment.computes_loss
                          ? 'bg-green-100 text-green-700'
                          : 'bg-gray-200 text-gray-600'
                      ]"
                    >
                      {{ segment.computes_loss ? '计算损失' : '不计算损失' }}
                    </span>
                  </div>
                  <pre class="text-xs text-gray-700 whitespace-pre-wrap font-mono">{{ segment.text }}</pre>
                </div>
              </div>
            </div>

            <!-- Distribution Tab -->
            <div v-else-if="activeTab === 'distribution'">
              <div v-if="distribution.length === 0" class="text-center py-8">
                <PieChart class="w-10 h-10 text-gray-300 mx-auto mb-3" />
                <p class="text-sm text-gray-500">未配置标签字段</p>
                <p class="text-2xs text-gray-400 mt-1">在设置中配置标签字段</p>
              </div>

              <div v-else class="grid grid-cols-2 gap-4">
                <div
                  v-for="dist in distribution"
                  :key="dist.field"
                  class="bg-gray-50 rounded-lg p-4"
                >
                  <h4 class="text-sm font-medium text-gray-700 mb-3">{{ dist.field }}</h4>
                  <div class="space-y-2">
                    <div
                      v-for="item in getDistributionChartData(dist)"
                      :key="item.name"
                      class="flex items-center gap-2"
                    >
                      <div class="flex-1">
                        <div class="flex justify-between text-xs mb-1">
                          <span class="text-gray-700">{{ item.name }}</span>
                          <span class="text-gray-500">{{ item.value }} ({{ ((item.value / dist.total) * 100).toFixed(1) }}%)</span>
                        </div>
                        <div class="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            class="h-full bg-primary-500 rounded-full"
                            :style="{ width: `${(item.value / dist.total) * 100}%` }"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                  <div class="mt-3 pt-3 border-t border-gray-200 text-2xs text-gray-500">
                    共 {{ dist.total.toLocaleString() }} 条 · {{ Object.keys(dist.distribution).length }} 个类别
                  </div>
                </div>
              </div>
            </div>

            <!-- Settings Tab -->
            <div v-else-if="activeTab === 'settings'">
              <div class="space-y-4">
                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">数据集信息</h4>
                  <div class="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span class="text-gray-500">文件：</span>
                      <span class="ml-2 text-gray-700">{{ selectedDataset.file_path?.split('/').pop() }}</span>
                    </div>
                    <div>
                      <span class="text-gray-500">格式：</span>
                      <span class="ml-2 text-gray-700 uppercase">{{ selectedDataset.file_format }}</span>
                    </div>
                    <div>
                      <span class="text-gray-500">大小：</span>
                      <span class="ml-2 text-gray-700">{{ formatFileSize(selectedDataset.file_size_mb) }}</span>
                    </div>
                    <div>
                      <span class="text-gray-500">样本数：</span>
                      <span class="ml-2 text-gray-700">{{ selectedDataset.total_rows.toLocaleString() }}</span>
                    </div>
                  </div>
                </div>

                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">字段列表</h4>
                  <div class="flex flex-wrap gap-2">
                    <span
                      v-for="col in selectedDataset.columns"
                      :key="col"
                      :class="[
                        'px-2 py-1 rounded text-xs',
                        col === selectedDataset.prompt_field ? 'bg-gray-200 text-gray-700' :
                        col === selectedDataset.response_field ? 'bg-green-100 text-green-700' :
                        selectedDataset.label_fields?.includes(col) ? 'bg-blue-100 text-blue-700' :
                        'bg-gray-100 text-gray-600'
                      ]"
                    >
                      {{ col }}
                      <span v-if="col === selectedDataset.prompt_field" class="text-2xs">(输入)</span>
                      <span v-if="col === selectedDataset.response_field" class="text-2xs">(输出/损失)</span>
                      <span v-if="selectedDataset.label_fields?.includes(col)" class="text-2xs">(标签)</span>
                    </span>
                  </div>
                </div>

                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">损失计算配置</h4>
                  <div class="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span class="text-gray-500">输入字段：</span>
                      <code class="ml-2 px-1.5 py-0.5 bg-gray-200 rounded">{{ selectedDataset.prompt_field }}</code>
                    </div>
                    <div>
                      <span class="text-gray-500">输出字段：</span>
                      <code class="ml-2 px-1.5 py-0.5 bg-green-100 rounded text-green-700">{{ selectedDataset.response_field }}</code>
                    </div>
                  </div>
                  <p class="text-2xs text-gray-500 mt-2">
                    训练时仅在输出字段上计算损失
                  </p>
                </div>

                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">标签字段</h4>
                  <div v-if="selectedDataset.label_fields?.length > 0" class="flex flex-wrap gap-2">
                    <span
                      v-for="field in selectedDataset.label_fields"
                      :key="field"
                      class="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs"
                    >
                      {{ field }}
                    </span>
                  </div>
                  <p v-else class="text-xs text-gray-500">未配置标签字段</p>
                </div>

                <!-- Remote Sync Status -->
                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
                    <Cloud class="w-4 h-4" />
                    远程同步状态
                  </h4>
                  <div class="space-y-3">
                    <div class="flex items-center gap-2">
                      <span class="text-xs text-gray-500">状态：</span>
                      <span
                        :class="[
                          'flex items-center gap-1 px-2 py-0.5 rounded text-xs',
                          getSyncStatusInfo(selectedDataset.sync_status).bg,
                          getSyncStatusInfo(selectedDataset.sync_status).color
                        ]"
                      >
                        <component
                          :is="getSyncStatusInfo(selectedDataset.sync_status).icon"
                          :class="['w-3.5 h-3.5', getSyncStatusInfo(selectedDataset.sync_status).spin && 'animate-spin']"
                        />
                        {{ getSyncStatusInfo(selectedDataset.sync_status).text }}
                      </span>
                    </div>

                    <div v-if="selectedDataset.remote_path" class="text-xs">
                      <span class="text-gray-500">远程路径：</span>
                      <code class="ml-2 px-1.5 py-0.5 bg-gray-200 rounded text-gray-700 break-all">
                        {{ selectedDataset.remote_path }}
                      </code>
                    </div>

                    <div v-if="selectedDataset.sync_error" class="text-xs">
                      <span class="text-gray-500">错误：</span>
                      <span class="ml-2 text-red-600">{{ selectedDataset.sync_error }}</span>
                    </div>

                    <div v-if="selectedDataset.synced_at" class="text-xs">
                      <span class="text-gray-500">上次同步：</span>
                      <span class="ml-2 text-gray-700">{{ new Date(selectedDataset.synced_at).toLocaleString() }}</span>
                    </div>

                    <div class="pt-2 border-t border-gray-200">
                      <button
                        @click="syncDataset(selectedDataset)"
                        :disabled="syncing[selectedDataset.uuid]"
                        class="btn-secondary text-xs flex items-center gap-1.5"
                      >
                        <Loader2 v-if="syncing[selectedDataset.uuid]" class="w-3.5 h-3.5 animate-spin" />
                        <Cloud v-else class="w-3.5 h-3.5" />
                        {{ syncing[selectedDataset.uuid] ? '同步中...' : (selectedDataset.sync_status === 'synced' ? '重新同步' : '同步到远程') }}
                      </button>
                      <p class="text-2xs text-gray-400 mt-1.5">
                        将数据集同步到远程 SSH 服务器用于训练
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Upload Modal -->
    <div v-if="showUploadModal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="bg-white rounded-xl shadow-xl w-full max-w-md p-6">
        <h3 class="text-lg font-semibold text-gray-800 mb-4">上传训练数据集</h3>

        <div class="space-y-4">
          <div>
            <label class="block text-xs text-gray-600 mb-1">文件 (JSONL, JSON, Parquet)</label>
            <div class="flex items-center gap-2">
              <label class="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded cursor-pointer text-xs text-gray-700 border border-gray-300">
                选择文件
                <input
                  type="file"
                  accept=".jsonl,.json,.parquet,.ndjson"
                  @change="handleFileSelect"
                  class="hidden"
                />
              </label>
              <span v-if="uploadFile" class="text-xs text-green-600 flex items-center gap-1">
                <CheckCircle2 class="w-3.5 h-3.5" />
                {{ uploadFile.name }}
              </span>
              <span v-else class="text-xs text-gray-400">未选择文件</span>
            </div>
          </div>

          <div>
            <label class="block text-xs text-gray-600 mb-1">名称 *</label>
            <input
              v-model="uploadForm.name"
              type="text"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="数据集名称"
            />
          </div>

          <div>
            <label class="block text-xs text-gray-600 mb-1">描述</label>
            <textarea
              v-model="uploadForm.description"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-primary-500"
              rows="2"
              placeholder="可选描述"
            />
          </div>

          <!-- Format Detection Notice -->
          <div v-if="detectedFormat" class="p-3 rounded-lg" :class="detectedFormat === 'messages' ? 'bg-purple-50 border border-purple-200' : 'bg-blue-50 border border-blue-200'">
            <div class="flex items-center gap-2">
              <span class="text-xs font-medium" :class="detectedFormat === 'messages' ? 'text-purple-700' : 'text-blue-700'">
                {{ detectedFormat === 'messages' ? '✅ 检测到 OpenAI Messages 格式' : '✅ 检测到 Prompt/Response 格式' }}
              </span>
            </div>
            <p class="text-2xs mt-1" :class="detectedFormat === 'messages' ? 'text-purple-600' : 'text-blue-600'">
              {{ detectedFormat === 'messages' ? '支持多轮对话，自动识别 system/user/assistant 角色' : '单轮对话格式' }}
            </p>
          </div>

          <!-- Prompt/Response fields - only show for non-messages format -->
          <div v-if="detectedFormat !== 'messages'" class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-600 mb-1">输入字段名</label>
              <input
                v-model="uploadForm.promptField"
                type="text"
                class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <p class="text-2xs text-gray-400 mt-0.5">数据中 prompt 字段名</p>
            </div>
            <div>
              <label class="block text-xs text-gray-600 mb-1">输出字段名</label>
              <input
                v-model="uploadForm.responseField"
                type="text"
                class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <p class="text-2xs text-gray-400 mt-0.5">数据中 response 字段名</p>
            </div>
          </div>

          <div>
            <label class="block text-xs text-gray-600 mb-1">标签字段（用于分布分析）</label>
            <div v-if="detectedColumns.length > 0" class="flex flex-wrap gap-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
              <label
                v-for="col in detectedColumns"
                :key="col"
                class="flex items-center gap-1.5 px-2 py-1 rounded cursor-pointer transition-colors"
                :class="selectedLabelFields.includes(col) ? 'bg-primary-100 text-primary-700' : 'bg-white text-gray-600 hover:bg-gray-100'"
              >
                <input
                  type="checkbox"
                  :value="col"
                  v-model="selectedLabelFields"
                  class="w-3.5 h-3.5 text-primary-500 rounded border-gray-300 focus:ring-primary-500"
                />
                <span class="text-xs">{{ col }}</span>
              </label>
            </div>
            <p v-else class="text-2xs text-gray-400 p-3 bg-gray-50 rounded-lg border border-gray-200">
              请先选择文件，将自动检测可用的标签字段
            </p>
          </div>
        </div>

        <div class="flex justify-end gap-3 mt-6">
          <button
            @click="showUploadModal = false"
            class="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            @click="handleUpload"
            :disabled="uploading || !uploadFile || !uploadForm.name"
            class="btn-primary text-sm"
          >
            {{ uploading ? '上传中...' : '上传' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
