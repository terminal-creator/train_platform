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
  CheckCircle2
} from 'lucide-vue-next'

const jobsStore = useJobsStore()
const loading = ref(false)
const trainingDatasets = ref([])
const selectedDataset = ref(null)
const currentSample = ref(null)
const sampleIndex = ref(0)
const distribution = ref([])
const previewLoading = ref(false)
const activeTab = ref('preview')
const showUploadModal = ref(false)

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
const syncing = ref({})  // Track syncing status per dataset uuid

const getSyncStatusInfo = (status) => {
  const statusMap = {
    'synced': { icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-100', text: 'Synced' },
    'syncing': { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100', text: 'Syncing', spin: true },
    'failed': { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-100', text: 'Failed' },
    'not_synced': { icon: CloudOff, color: 'text-gray-400', bg: 'bg-gray-100', text: 'Not Synced' },
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
  activeTab.value = 'preview'
  sampleIndex.value = 0
  distribution.value = []
  currentSample.value = null
  await loadDistribution(dataset)
  await loadSample(dataset, 0)
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

const handleFileSelect = (e) => {
  uploadFile.value = e.target.files[0]
  if (uploadFile.value && !uploadForm.value.name) {
    uploadForm.value.name = uploadFile.value.name.replace(/\.[^/.]+$/, '')
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
    if (uploadForm.value.labelFields) {
      params.label_fields = uploadForm.value.labelFields
    }

    await api.uploadTrainingDataset(formData, params)
    showUploadModal.value = false
    uploadForm.value = { name: '', description: '', labelFields: '', promptField: 'prompt', responseField: 'response' }
    uploadFile.value = null
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
        <h2 class="text-lg font-semibold text-gray-800">Training Datasets</h2>
        <p class="text-xs text-gray-500">Manage training datasets with label distribution analysis</p>
      </div>
      <div class="flex gap-2">
        <button @click="fetchTrainingDatasets" :disabled="loading" class="btn-secondary flex items-center gap-1.5 text-xs">
          <RefreshCw :class="['w-3.5 h-3.5', loading && 'animate-spin']" />
          Refresh
        </button>
        <button @click="showUploadModal = true" class="btn-primary flex items-center gap-1.5 text-xs">
          <Upload class="w-3.5 h-3.5" />
          Upload
        </button>
      </div>
    </div>

    <div class="grid grid-cols-12 gap-4">
      <!-- Dataset List -->
      <div class="col-span-4">
        <div class="glass-card rounded-lg p-4">
          <h3 class="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
            <Database class="w-4 h-4" />
            Datasets
            <span class="text-2xs text-gray-400">({{ trainingDatasets.length }})</span>
          </h3>

          <div v-if="trainingDatasets.length === 0" class="text-center py-8">
            <Inbox class="w-10 h-10 text-gray-300 mx-auto mb-3" />
            <p class="text-sm text-gray-500">No datasets</p>
            <p class="text-2xs text-gray-400 mt-1">Upload a training dataset to get started</p>
          </div>

          <div v-else class="space-y-2 max-h-[500px] overflow-y-auto">
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

      <!-- Dataset Details -->
      <div class="col-span-8">
        <div v-if="!selectedDataset" class="glass-card rounded-lg p-8 text-center">
          <FileText class="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p class="text-gray-500">Select a dataset to view details</p>
        </div>

        <div v-else class="glass-card rounded-lg">
          <!-- Dataset Header -->
          <div class="p-4 border-b border-gray-100">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-sm font-medium text-gray-800">{{ selectedDataset.name }}</h3>
                <p class="text-2xs text-gray-500 mt-0.5">{{ selectedDataset.total_rows.toLocaleString() }} samples</p>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-gray-500">
                  Loss: <code class="bg-gray-100 px-1 rounded">{{ selectedDataset.response_field }}</code>
                </span>
              </div>
            </div>
          </div>

          <!-- Tabs -->
          <div class="flex border-b border-gray-100">
            <button
              @click="activeTab = 'preview'"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors',
                activeTab === 'preview'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <Eye class="w-3.5 h-3.5 inline mr-1" />
              Sample Viewer
            </button>
            <button
              @click="activeTab = 'distribution'"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors',
                activeTab === 'distribution'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <PieChart class="w-3.5 h-3.5 inline mr-1" />
              Distribution
            </button>
            <button
              @click="activeTab = 'settings'"
              :class="[
                'px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors',
                activeTab === 'settings'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              ]"
            >
              <Settings class="w-3.5 h-3.5 inline mr-1" />
              Settings
            </button>
          </div>

          <!-- Content -->
          <div class="p-4">
            <!-- Sample Viewer Tab -->
            <div v-if="activeTab === 'preview'">
              <!-- Navigation -->
              <div class="flex items-center justify-between mb-4">
                <div class="flex items-center gap-2">
                  <button
                    @click="prevSample"
                    :disabled="sampleIndex === 0 || previewLoading"
                    class="p-1.5 rounded hover:bg-gray-100 disabled:opacity-50"
                  >
                    <ChevronLeft class="w-4 h-4" />
                  </button>
                  <span class="text-xs text-gray-600">
                    Sample {{ sampleIndex + 1 }} of {{ selectedDataset.total_rows.toLocaleString() }}
                  </span>
                  <button
                    @click="nextSample"
                    :disabled="sampleIndex >= selectedDataset.total_rows - 1 || previewLoading"
                    class="p-1.5 rounded hover:bg-gray-100 disabled:opacity-50"
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
                <p class="text-sm text-gray-500">Loading...</p>
              </div>

              <!-- Sample Content with Loss Highlighting -->
              <div v-else-if="currentSample" class="space-y-4">
                <div
                  v-for="segment in currentSample.loss_segments"
                  :key="segment.field"
                  :class="[
                    'p-3 rounded-lg border',
                    segment.computes_loss
                      ? 'bg-green-50 border-green-200'
                      : 'bg-gray-50 border-gray-200'
                  ]"
                >
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-medium text-gray-700">{{ segment.field }}</span>
                    <span
                      :class="[
                        'px-1.5 py-0.5 rounded text-2xs',
                        segment.computes_loss
                          ? 'bg-green-100 text-green-700'
                          : 'bg-gray-200 text-gray-600'
                      ]"
                    >
                      {{ segment.computes_loss ? 'Computes Loss' : 'No Loss' }}
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
                <p class="text-sm text-gray-500">No label fields configured</p>
                <p class="text-2xs text-gray-400 mt-1">Configure label fields in Settings tab</p>
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
                    Total: {{ dist.total.toLocaleString() }} samples
                  </div>
                </div>
              </div>
            </div>

            <!-- Settings Tab -->
            <div v-else-if="activeTab === 'settings'">
              <div class="space-y-4">
                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">Dataset Information</h4>
                  <div class="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span class="text-gray-500">File:</span>
                      <span class="ml-2 text-gray-700">{{ selectedDataset.file_path?.split('/').pop() }}</span>
                    </div>
                    <div>
                      <span class="text-gray-500">Format:</span>
                      <span class="ml-2 text-gray-700 uppercase">{{ selectedDataset.file_format }}</span>
                    </div>
                    <div>
                      <span class="text-gray-500">Size:</span>
                      <span class="ml-2 text-gray-700">{{ formatFileSize(selectedDataset.file_size_mb) }}</span>
                    </div>
                    <div>
                      <span class="text-gray-500">Rows:</span>
                      <span class="ml-2 text-gray-700">{{ selectedDataset.total_rows.toLocaleString() }}</span>
                    </div>
                  </div>
                </div>

                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">Columns</h4>
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
                      <span v-if="col === selectedDataset.prompt_field" class="text-2xs">(prompt)</span>
                      <span v-if="col === selectedDataset.response_field" class="text-2xs">(response/loss)</span>
                      <span v-if="selectedDataset.label_fields?.includes(col)" class="text-2xs">(label)</span>
                    </span>
                  </div>
                </div>

                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">Loss Computation</h4>
                  <div class="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span class="text-gray-500">Prompt Field:</span>
                      <code class="ml-2 px-1.5 py-0.5 bg-gray-200 rounded">{{ selectedDataset.prompt_field }}</code>
                    </div>
                    <div>
                      <span class="text-gray-500">Response Field:</span>
                      <code class="ml-2 px-1.5 py-0.5 bg-green-100 rounded text-green-700">{{ selectedDataset.response_field }}</code>
                    </div>
                  </div>
                  <p class="text-2xs text-gray-500 mt-2">
                    Loss is computed only on the response field during training.
                  </p>
                </div>

                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3">Label Fields</h4>
                  <div v-if="selectedDataset.label_fields?.length > 0" class="flex flex-wrap gap-2">
                    <span
                      v-for="field in selectedDataset.label_fields"
                      :key="field"
                      class="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs"
                    >
                      {{ field }}
                    </span>
                  </div>
                  <p v-else class="text-xs text-gray-500">No label fields configured</p>
                </div>

                <!-- Remote Sync Status -->
                <div class="bg-gray-50 rounded-lg p-4">
                  <h4 class="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
                    <Cloud class="w-4 h-4" />
                    Remote Sync Status
                  </h4>
                  <div class="space-y-3">
                    <div class="flex items-center gap-2">
                      <span class="text-xs text-gray-500">Status:</span>
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
                      <span class="text-gray-500">Remote Path:</span>
                      <code class="ml-2 px-1.5 py-0.5 bg-gray-200 rounded text-gray-700 break-all">
                        {{ selectedDataset.remote_path }}
                      </code>
                    </div>

                    <div v-if="selectedDataset.sync_error" class="text-xs">
                      <span class="text-gray-500">Error:</span>
                      <span class="ml-2 text-red-600">{{ selectedDataset.sync_error }}</span>
                    </div>

                    <div v-if="selectedDataset.synced_at" class="text-xs">
                      <span class="text-gray-500">Last Synced:</span>
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
                        {{ syncing[selectedDataset.uuid] ? 'Syncing...' : (selectedDataset.sync_status === 'synced' ? 'Re-sync to Remote' : 'Sync to Remote') }}
                      </button>
                      <p class="text-2xs text-gray-400 mt-1.5">
                        Sync this dataset to the remote SSH server for training.
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
        <h3 class="text-lg font-semibold text-gray-800 mb-4">Upload Training Dataset</h3>

        <div class="space-y-4">
          <div>
            <label class="block text-xs text-gray-600 mb-1">File (JSONL, JSON, Parquet)</label>
            <input
              type="file"
              accept=".jsonl,.json,.parquet,.ndjson"
              @change="handleFileSelect"
              class="w-full text-xs"
            />
          </div>

          <div>
            <label class="block text-xs text-gray-600 mb-1">Name *</label>
            <input
              v-model="uploadForm.name"
              type="text"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="Dataset name"
            />
          </div>

          <div>
            <label class="block text-xs text-gray-600 mb-1">Description</label>
            <textarea
              v-model="uploadForm.description"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              rows="2"
              placeholder="Optional description"
            />
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-600 mb-1">Prompt Field</label>
              <input
                v-model="uploadForm.promptField"
                type="text"
                class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="prompt"
              />
            </div>
            <div>
              <label class="block text-xs text-gray-600 mb-1">Response Field</label>
              <input
                v-model="uploadForm.responseField"
                type="text"
                class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="response"
              />
            </div>
          </div>

          <div>
            <label class="block text-xs text-gray-600 mb-1">Label Fields (comma-separated, optional)</label>
            <input
              v-model="uploadForm.labelFields"
              type="text"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="e.g., tenant, category"
            />
            <p class="text-2xs text-gray-400 mt-1">Leave empty to auto-detect</p>
          </div>
        </div>

        <div class="flex justify-end gap-3 mt-6">
          <button
            @click="showUploadModal = false"
            class="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            @click="handleUpload"
            :disabled="uploading || !uploadFile || !uploadForm.name"
            class="btn-primary text-sm"
          >
            {{ uploading ? 'Uploading...' : 'Upload' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
