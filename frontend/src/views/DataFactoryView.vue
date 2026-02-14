<script setup>
import { ref, computed, onMounted } from 'vue'
import * as api from '@/api'
import {
  Factory,
  FileText,
  Trash2,
  RefreshCw,
  Settings,
  Scissors,
  BarChart3,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  ArrowRight,
  Download,
  Search,
  Sparkles,
  Copy,
  SplitSquareHorizontal,
  ClipboardCheck,
  Play
} from 'lucide-vue-next'

// Active tab
const activeTab = ref('templates')

// ===== Config Templates =====
const templates = ref([])
const selectedTemplate = ref(null)
const templatesLoading = ref(false)

const loadTemplates = async () => {
  templatesLoading.value = true
  try {
    templates.value = await api.getConfigTemplates()
  } catch (e) {
    console.error('Failed to load templates:', e)
  } finally {
    templatesLoading.value = false
  }
}

const selectTemplate = async (algo) => {
  try {
    selectedTemplate.value = await api.getConfigTemplate(algo)
  } catch (e) {
    console.error('Failed to load template:', e)
  }
}

// ===== Data Cleaning =====
const cleaningForm = ref({
  input_path: '',
  output_path: '',
  min_prompt_length: 1,
  max_prompt_length: 10000,
  min_response_length: 1,
  max_response_length: 50000,
  remove_empty: true,
  remove_duplicates: false,
  strip_whitespace: true,
  remove_html_tags: false,
  min_unique_chars_ratio: 0.0,
  max_repetition_ratio: 1.0,
})
const cleaningResult = ref(null)
const cleaningLoading = ref(false)

const runCleaning = async () => {
  cleaningLoading.value = true
  cleaningResult.value = null
  try {
    cleaningResult.value = await api.cleanData(cleaningForm.value)
  } catch (e) {
    alert('Cleaning failed: ' + e.message)
  } finally {
    cleaningLoading.value = false
  }
}

// ===== Deduplication =====
const dedupForm = ref({
  input_path: '',
  output_path: '',
  method: 'minhash',
  threshold: 0.8,
  num_perm: 128,
  ngram_size: 3,
  text_fields: ['prompt', 'response'],
})
const dedupResult = ref(null)
const dedupLoading = ref(false)

const runDedup = async () => {
  dedupLoading.value = true
  dedupResult.value = null
  try {
    dedupResult.value = await api.deduplicateData(dedupForm.value)
  } catch (e) {
    alert('Deduplication failed: ' + e.message)
  } finally {
    dedupLoading.value = false
  }
}

// ===== Quality Assessment =====
const qualityForm = ref({
  input_path: '',
  prompt_field: 'prompt',
  response_field: 'response',
})
const qualityResult = ref(null)
const qualityLoading = ref(false)

const runQuality = async () => {
  qualityLoading.value = true
  qualityResult.value = null
  try {
    qualityResult.value = await api.assessQuality(qualityForm.value)
  } catch (e) {
    alert('Quality assessment failed: ' + e.message)
  } finally {
    qualityLoading.value = false
  }
}

// ===== Format Conversion =====
const convertForm = ref({
  input_path: '',
  output_path: '',
  target_format: 'sft',
  source_format: '',
})
const convertResult = ref(null)
const convertLoading = ref(false)
const detectedFormat = ref(null)

const detectFormatFn = async () => {
  if (!convertForm.value.input_path) return
  try {
    detectedFormat.value = await api.detectFormat(convertForm.value.input_path)
  } catch (e) {
    console.error('Format detection failed:', e)
  }
}

const runConvert = async () => {
  convertLoading.value = true
  convertResult.value = null
  try {
    const data = { ...convertForm.value }
    if (!data.source_format) delete data.source_format
    convertResult.value = await api.convertFormat(data)
  } catch (e) {
    alert('Conversion failed: ' + e.message)
  } finally {
    convertLoading.value = false
  }
}

// ===== Data Split =====
const splitForm = ref({
  input_path: '',
  output_dir: '',
  method: 'random',
  train_ratio: 0.8,
  val_ratio: 0.1,
  test_ratio: 0.1,
  seed: 42,
  output_format: 'jsonl',
})
const splitResult = ref(null)
const splitLoading = ref(false)

const runSplit = async () => {
  splitLoading.value = true
  splitResult.value = null
  try {
    splitResult.value = await api.splitData(splitForm.value)
  } catch (e) {
    alert('Split failed: ' + e.message)
  } finally {
    splitLoading.value = false
  }
}

// ===== Benchmarks =====
const benchmarks = ref([])
const benchmarksLoading = ref(false)

const loadBenchmarks = async () => {
  benchmarksLoading.value = true
  try {
    benchmarks.value = await api.getBenchmarks()
  } catch (e) {
    console.error('Failed to load benchmarks:', e)
  } finally {
    benchmarksLoading.value = false
  }
}

onMounted(() => {
  loadTemplates()
  loadBenchmarks()
})

const tabs = [
  { key: 'templates', label: '配置模板', icon: Settings },
  { key: 'cleaning', label: '数据清洗', icon: Sparkles },
  { key: 'dedup', label: '去重检测', icon: Copy },
  { key: 'quality', label: '质量评估', icon: BarChart3 },
  { key: 'convert', label: '格式转换', icon: ArrowRight },
  { key: 'split', label: '数据切分', icon: SplitSquareHorizontal },
  { key: 'benchmarks', label: 'Benchmarks', icon: ClipboardCheck },
]
</script>

<template>
  <div class="p-6 max-w-7xl mx-auto">
    <!-- Page Header -->
    <div class="mb-6">
      <h1 class="text-2xl font-bold text-gray-800 flex items-center gap-2">
        <Factory class="w-6 h-6 text-indigo-600" />
        Data Factory
      </h1>
      <p class="text-sm text-gray-500 mt-1">
        Data cleaning, deduplication, quality assessment, format conversion, splitting & benchmarks
      </p>
    </div>

    <!-- Tabs -->
    <div class="border-b border-gray-200 mb-6">
      <nav class="flex gap-6">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          @click="activeTab = tab.key"
          :class="[
            'flex items-center gap-1.5 pb-3 px-1 text-sm font-medium border-b-2 transition-colors',
            activeTab === tab.key
              ? 'border-indigo-600 text-indigo-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          ]"
        >
          <component :is="tab.icon" class="w-4 h-4" />
          {{ tab.label }}
        </button>
      </nav>
    </div>

    <!-- Config Templates Tab -->
    <div v-if="activeTab === 'templates'">
      <div class="grid grid-cols-4 gap-4 mb-6">
        <div
          v-for="t in templates"
          :key="t.algorithm"
          @click="selectTemplate(t.algorithm)"
          :class="[
            'p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md',
            selectedTemplate?.algorithm === t.algorithm
              ? 'border-indigo-600 bg-indigo-50'
              : 'border-gray-200 bg-white'
          ]"
        >
          <h3 class="font-semibold text-sm text-gray-800">{{ t.algorithm?.toUpperCase() }}</h3>
          <p class="text-xs text-gray-500 mt-1">{{ t.name }}</p>
        </div>
      </div>

      <div v-if="selectedTemplate" class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-2">{{ selectedTemplate.name }}</h2>
        <p class="text-sm text-gray-600 mb-4">{{ selectedTemplate.description }}</p>

        <div class="mb-4">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Required Fields</h3>
          <div class="flex gap-2">
            <span
              v-for="f in selectedTemplate.required_fields"
              :key="f"
              class="px-2 py-1 bg-red-50 text-red-700 text-xs rounded"
            >{{ f }}</span>
          </div>
        </div>

        <div v-if="selectedTemplate.tips" class="mb-4">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Tips</h3>
          <ul class="list-disc list-inside text-sm text-gray-600 space-y-1">
            <li v-for="tip in selectedTemplate.tips" :key="tip">{{ tip }}</li>
          </ul>
        </div>

        <div class="mb-4">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Default Config</h3>
          <pre class="bg-gray-50 rounded p-4 text-xs overflow-auto max-h-96">{{ JSON.stringify(selectedTemplate.defaults, null, 2) }}</pre>
        </div>
      </div>
    </div>

    <!-- Data Cleaning Tab -->
    <div v-if="activeTab === 'cleaning'" class="grid grid-cols-2 gap-6">
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4">Data Cleaning Configuration</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Input Path</label>
            <input v-model="cleaningForm.input_path" class="input-field" placeholder="./datasets/data.jsonl" />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Output Path (optional)</label>
            <input v-model="cleaningForm.output_path" class="input-field" placeholder="Auto-generated if empty" />
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1">Min Prompt Length</label>
              <input v-model.number="cleaningForm.min_prompt_length" type="number" class="input-field" />
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Max Prompt Length</label>
              <input v-model.number="cleaningForm.max_prompt_length" type="number" class="input-field" />
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Min Response Length</label>
              <input v-model.number="cleaningForm.min_response_length" type="number" class="input-field" />
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Max Response Length</label>
              <input v-model.number="cleaningForm.max_response_length" type="number" class="input-field" />
            </div>
          </div>
          <div class="space-y-2">
            <label class="flex items-center gap-2 text-sm">
              <input type="checkbox" v-model="cleaningForm.remove_empty" class="rounded" />
              Remove empty entries
            </label>
            <label class="flex items-center gap-2 text-sm">
              <input type="checkbox" v-model="cleaningForm.remove_duplicates" class="rounded" />
              Remove exact duplicates
            </label>
            <label class="flex items-center gap-2 text-sm">
              <input type="checkbox" v-model="cleaningForm.strip_whitespace" class="rounded" />
              Strip whitespace
            </label>
            <label class="flex items-center gap-2 text-sm">
              <input type="checkbox" v-model="cleaningForm.remove_html_tags" class="rounded" />
              Remove HTML tags
            </label>
          </div>
          <button @click="runCleaning" :disabled="cleaningLoading || !cleaningForm.input_path" class="btn-primary w-full">
            <Loader2 v-if="cleaningLoading" class="w-4 h-4 animate-spin" />
            <Sparkles v-else class="w-4 h-4" />
            {{ cleaningLoading ? 'Cleaning...' : 'Run Cleaning' }}
          </button>
        </div>
      </div>

      <div v-if="cleaningResult" class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <CheckCircle2 class="w-5 h-5 text-green-500" />
          Cleaning Results
        </h2>
        <div class="space-y-3">
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Output</span>
            <span class="text-sm font-medium">{{ cleaningResult.output_path }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Original Count</span>
            <span class="text-sm font-medium">{{ cleaningResult.original_count }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Cleaned Count</span>
            <span class="text-sm font-medium text-green-600">{{ cleaningResult.cleaned_count }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Removed</span>
            <span class="text-sm font-medium text-red-600">{{ cleaningResult.removed_count }}</span>
          </div>
          <div v-if="cleaningResult.removal_reasons" class="mt-4">
            <h3 class="text-sm font-medium text-gray-700 mb-2">Removal Reasons</h3>
            <div v-for="(count, reason) in cleaningResult.removal_reasons" :key="reason" class="flex justify-between items-center p-2 text-sm">
              <span class="text-gray-600">{{ reason }}</span>
              <span class="font-medium">{{ count }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Deduplication Tab -->
    <div v-if="activeTab === 'dedup'" class="grid grid-cols-2 gap-6">
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4">Deduplication Configuration</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Input Path</label>
            <input v-model="dedupForm.input_path" class="input-field" placeholder="./datasets/data.jsonl" />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Method</label>
            <select v-model="dedupForm.method" class="input-field">
              <option value="minhash">MinHash (fuzzy)</option>
              <option value="simhash">SimHash (fuzzy)</option>
              <option value="exact">Exact Match</option>
            </select>
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1">Threshold</label>
              <input v-model.number="dedupForm.threshold" type="number" step="0.05" min="0" max="1" class="input-field" />
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">N-gram Size</label>
              <input v-model.number="dedupForm.ngram_size" type="number" class="input-field" />
            </div>
          </div>
          <button @click="runDedup" :disabled="dedupLoading || !dedupForm.input_path" class="btn-primary w-full">
            <Loader2 v-if="dedupLoading" class="w-4 h-4 animate-spin" />
            <Copy v-else class="w-4 h-4" />
            {{ dedupLoading ? 'Deduplicating...' : 'Run Deduplication' }}
          </button>
        </div>
      </div>

      <div v-if="dedupResult" class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <CheckCircle2 class="w-5 h-5 text-green-500" />
          Deduplication Results
        </h2>
        <div class="space-y-3">
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Original Count</span>
            <span class="text-sm font-medium">{{ dedupResult.original_count }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Deduplicated Count</span>
            <span class="text-sm font-medium text-green-600">{{ dedupResult.deduplicated_count }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Duplicates Found</span>
            <span class="text-sm font-medium text-orange-600">{{ dedupResult.duplicates_found }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Duplicate Rate</span>
            <span class="text-sm font-medium">{{ (dedupResult.duplicate_rate * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Quality Assessment Tab -->
    <div v-if="activeTab === 'quality'" class="grid grid-cols-2 gap-6">
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4">Quality Assessment</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Input Path</label>
            <input v-model="qualityForm.input_path" class="input-field" placeholder="./datasets/data.jsonl" />
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1">Prompt Field</label>
              <input v-model="qualityForm.prompt_field" class="input-field" />
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Response Field</label>
              <input v-model="qualityForm.response_field" class="input-field" />
            </div>
          </div>
          <button @click="runQuality" :disabled="qualityLoading || !qualityForm.input_path" class="btn-primary w-full">
            <Loader2 v-if="qualityLoading" class="w-4 h-4 animate-spin" />
            <BarChart3 v-else class="w-4 h-4" />
            {{ qualityLoading ? 'Assessing...' : 'Run Quality Assessment' }}
          </button>
        </div>
      </div>

      <div v-if="qualityResult" class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <CheckCircle2 class="w-5 h-5 text-green-500" />
          Quality Report
        </h2>
        <div class="space-y-4">
          <div class="flex justify-between items-center p-3 bg-indigo-50 rounded">
            <span class="text-sm font-medium text-indigo-700">Overall Score</span>
            <span class="text-lg font-bold text-indigo-700">{{ (qualityResult.overall_score * 100).toFixed(1) }}%</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Total Samples</span>
            <span class="text-sm font-medium">{{ qualityResult.total_samples }}</span>
          </div>
          <div v-if="qualityResult.dimension_scores" class="mt-4">
            <h3 class="text-sm font-medium text-gray-700 mb-2">Dimension Scores</h3>
            <div v-for="(score, dim) in qualityResult.dimension_scores" :key="dim" class="mb-2">
              <div class="flex justify-between text-sm mb-1">
                <span class="text-gray-600 capitalize">{{ dim }}</span>
                <span class="font-medium">{{ (score * 100).toFixed(1) }}%</span>
              </div>
              <div class="w-full bg-gray-200 rounded-full h-2">
                <div class="bg-indigo-500 rounded-full h-2 transition-all" :style="{ width: (score * 100) + '%' }"></div>
              </div>
            </div>
          </div>
          <div v-if="qualityResult.common_issues?.length" class="mt-4">
            <h3 class="text-sm font-medium text-gray-700 mb-2">Common Issues</h3>
            <div v-for="issue in qualityResult.common_issues" :key="issue" class="flex items-center gap-2 p-2 text-sm text-orange-700 bg-orange-50 rounded mb-1">
              <AlertTriangle class="w-3 h-3 flex-shrink-0" />
              {{ issue }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Format Conversion Tab -->
    <div v-if="activeTab === 'convert'" class="grid grid-cols-2 gap-6">
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4">Format Conversion</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Input Path</label>
            <div class="flex gap-2">
              <input v-model="convertForm.input_path" class="input-field flex-1" placeholder="./datasets/data.jsonl" />
              <button @click="detectFormatFn" class="btn-secondary text-xs">
                <Search class="w-3 h-3" /> Detect
              </button>
            </div>
            <p v-if="detectedFormat" class="text-xs text-green-600 mt-1">
              Detected: {{ detectedFormat.format }} (confidence: {{ (detectedFormat.confidence * 100).toFixed(0) }}%)
            </p>
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Output Path</label>
            <input v-model="convertForm.output_path" class="input-field" placeholder="./datasets/output.jsonl" />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Target Format</label>
            <select v-model="convertForm.target_format" class="input-field">
              <option value="sft">SFT (prompt-response)</option>
              <option value="dpo">DPO (prompt-chosen-rejected)</option>
              <option value="grpo">GRPO (prompt + solution)</option>
              <option value="openai_messages">OpenAI Messages</option>
            </select>
          </div>
          <button @click="runConvert" :disabled="convertLoading || !convertForm.input_path || !convertForm.output_path" class="btn-primary w-full">
            <Loader2 v-if="convertLoading" class="w-4 h-4 animate-spin" />
            <ArrowRight v-else class="w-4 h-4" />
            {{ convertLoading ? 'Converting...' : 'Convert' }}
          </button>
        </div>
      </div>

      <div v-if="convertResult" class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <CheckCircle2 class="w-5 h-5 text-green-500" />
          Conversion Result
        </h2>
        <div class="space-y-3">
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Output Path</span>
            <span class="text-sm font-medium">{{ convertResult.output_path }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Total Records</span>
            <span class="text-sm font-medium">{{ convertResult.total }}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Converted</span>
            <span class="text-sm font-medium text-green-600">{{ convertResult.converted }}</span>
          </div>
          <div v-if="convertResult.skipped" class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Skipped</span>
            <span class="text-sm font-medium text-orange-600">{{ convertResult.skipped }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Data Split Tab -->
    <div v-if="activeTab === 'split'" class="grid grid-cols-2 gap-6">
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4">Data Splitting</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Input Path</label>
            <input v-model="splitForm.input_path" class="input-field" placeholder="./datasets/data.jsonl" />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Output Directory</label>
            <input v-model="splitForm.output_dir" class="input-field" placeholder="./datasets/split_output" />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Method</label>
            <select v-model="splitForm.method" class="input-field">
              <option value="random">Random</option>
              <option value="stratified">Stratified</option>
              <option value="temporal">Temporal</option>
            </select>
          </div>
          <div class="grid grid-cols-3 gap-3">
            <div>
              <label class="block text-xs text-gray-500 mb-1">Train Ratio</label>
              <input v-model.number="splitForm.train_ratio" type="number" step="0.05" class="input-field" />
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Val Ratio</label>
              <input v-model.number="splitForm.val_ratio" type="number" step="0.05" class="input-field" />
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Test Ratio</label>
              <input v-model.number="splitForm.test_ratio" type="number" step="0.05" class="input-field" />
            </div>
          </div>
          <button @click="runSplit" :disabled="splitLoading || !splitForm.input_path || !splitForm.output_dir" class="btn-primary w-full">
            <Loader2 v-if="splitLoading" class="w-4 h-4 animate-spin" />
            <SplitSquareHorizontal v-else class="w-4 h-4" />
            {{ splitLoading ? 'Splitting...' : 'Split Data' }}
          </button>
        </div>
      </div>

      <div v-if="splitResult" class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <CheckCircle2 class="w-5 h-5 text-green-500" />
          Split Result
        </h2>
        <div class="space-y-3">
          <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600">Total</span>
            <span class="text-sm font-medium">{{ splitResult.total }}</span>
          </div>
          <div v-for="(info, split) in splitResult.splits" :key="split" class="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span class="text-sm text-gray-600 capitalize">{{ split }}</span>
            <span class="text-sm font-medium">{{ info.count }} ({{ info.path }})</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Benchmarks Tab -->
    <div v-if="activeTab === 'benchmarks'">
      <div class="grid grid-cols-2 gap-4">
        <div
          v-for="bm in benchmarks"
          :key="bm.name"
          class="bg-white rounded-lg border border-gray-200 p-6"
        >
          <div class="flex items-start justify-between mb-3">
            <div>
              <h3 class="font-semibold text-gray-800">{{ bm.name }}</h3>
              <p class="text-sm text-gray-500 mt-1">{{ bm.description }}</p>
            </div>
            <span class="px-2 py-1 bg-indigo-50 text-indigo-700 text-xs rounded font-medium">
              {{ bm.category || 'benchmark' }}
            </span>
          </div>
          <div v-if="bm.metrics" class="flex gap-2 mt-3">
            <span v-for="m in bm.metrics" :key="m" class="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded">
              {{ m }}
            </span>
          </div>
        </div>
      </div>
      <div v-if="!benchmarks.length && !benchmarksLoading" class="text-center py-12 text-gray-500">
        No benchmarks available
      </div>
    </div>
  </div>
</template>

<style scoped>
.input-field {
  @apply w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent;
}
.btn-primary {
  @apply flex items-center justify-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors;
}
.btn-secondary {
  @apply flex items-center gap-1 px-3 py-2 border border-gray-300 text-gray-700 text-sm rounded-lg hover:bg-gray-50 transition-colors;
}
</style>
