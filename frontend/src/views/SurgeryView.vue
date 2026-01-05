<script setup>
import { ref, computed, onMounted } from 'vue'
import {
  getMergeMethods,
  mergeModels,
  getCheckpoints,
  selectBestCheckpoint,
  createSwaMerge
} from '@/api'
import { useAppStore } from '@/stores/app'
import {
  GitMerge,
  Package,
  Layers,
  Play,
  CheckCircle,
  AlertCircle
} from 'lucide-vue-next'

const appStore = useAppStore()

const activeTab = ref('merge')

// Merge state
const mergeMethods = ref([])
const mergeConfig = ref({
  method: 'slerp',
  models: ['', ''],
  output_path: '',
  weights: [0.5, 0.5],
  slerp_t: 0.5,
  density: 0.5,
  majority_sign_method: 'total'
})
const mergeResult = ref(null)
const merging = ref(false)

// Checkpoint state
const checkpoints = ref([])
const checkpointConfig = ref({
  job_id: '',
  metric: 'eval_loss',
  mode: 'min',
  top_k: 5
})
const selectedCheckpoints = ref([])
const loadingCheckpoints = ref(false)

// SWA state
const swaConfig = ref({
  checkpoints: [],
  output_path: '',
  weights: []
})
const swaResult = ref(null)
const swaMerging = ref(false)

const currentMethod = computed(() => {
  return mergeMethods.value.find(m => m.id === mergeConfig.value.method)
})

const addModel = () => {
  mergeConfig.value.models.push('')
  mergeConfig.value.weights.push(1 / mergeConfig.value.models.length)
  normalizeWeights()
}

const removeModel = (index) => {
  if (mergeConfig.value.models.length > 2) {
    mergeConfig.value.models.splice(index, 1)
    mergeConfig.value.weights.splice(index, 1)
    normalizeWeights()
  }
}

const normalizeWeights = () => {
  const total = mergeConfig.value.weights.reduce((a, b) => a + b, 0)
  if (total > 0) {
    mergeConfig.value.weights = mergeConfig.value.weights.map(w => w / total)
  }
}

const runMerge = async () => {
  try {
    merging.value = true
    mergeResult.value = null
    const config = { ...mergeConfig.value }
    if (config.method === 'slerp') {
      config.t = config.slerp_t
    }
    mergeResult.value = await mergeModels(config)
    appStore.showSuccess('模型合并完成')
  } catch (error) {
    appStore.showError(error.message)
  } finally {
    merging.value = false
  }
}

const loadCheckpoints = async () => {
  if (!checkpointConfig.value.job_id) return
  try {
    loadingCheckpoints.value = true
    checkpoints.value = await getCheckpoints(checkpointConfig.value.job_id)
  } catch (error) {
    appStore.showError(error.message)
  } finally {
    loadingCheckpoints.value = false
  }
}

const selectBest = async () => {
  try {
    loadingCheckpoints.value = true
    selectedCheckpoints.value = await selectBestCheckpoint(checkpointConfig.value)
    appStore.showSuccess(`找到 ${selectedCheckpoints.value.length} 个最佳检查点`)
  } catch (error) {
    appStore.showError(error.message)
  } finally {
    loadingCheckpoints.value = false
  }
}

const addToSwa = (checkpoint) => {
  if (!swaConfig.value.checkpoints.includes(checkpoint.path)) {
    swaConfig.value.checkpoints.push(checkpoint.path)
    swaConfig.value.weights.push(1 / swaConfig.value.checkpoints.length)
    normalizeSwaWeights()
  }
}

const removeFromSwa = (index) => {
  swaConfig.value.checkpoints.splice(index, 1)
  swaConfig.value.weights.splice(index, 1)
  normalizeSwaWeights()
}

const normalizeSwaWeights = () => {
  const len = swaConfig.value.checkpoints.length
  if (len > 0) {
    swaConfig.value.weights = swaConfig.value.checkpoints.map(() => 1 / len)
  }
}

const runSwa = async () => {
  try {
    swaMerging.value = true
    swaResult.value = null
    swaResult.value = await createSwaMerge(swaConfig.value)
    appStore.showSuccess('SWA 合并完成')
  } catch (error) {
    appStore.showError(error.message)
  } finally {
    swaMerging.value = false
  }
}

onMounted(async () => {
  try {
    mergeMethods.value = await getMergeMethods()
  } catch (error) {
    appStore.showError(error.message)
  }
})
</script>

<template>
  <div>
    <!-- Tab Navigation -->
    <div class="flex gap-1.5 mb-4">
      <button
        @click="activeTab = 'merge'"
        :class="[
          'px-3 py-1.5 rounded-md text-xs flex items-center gap-1.5',
          activeTab === 'merge' ? 'bg-primary-100 accent-text font-medium' : 'bg-gray-100 text-gray-500 hover:text-gray-700'
        ]"
      >
        <GitMerge class="w-3.5 h-3.5" />
        模型合并
      </button>
      <button
        @click="activeTab = 'checkpoint'"
        :class="[
          'px-3 py-1.5 rounded-md text-xs flex items-center gap-1.5',
          activeTab === 'checkpoint' ? 'bg-primary-100 accent-text font-medium' : 'bg-gray-100 text-gray-500 hover:text-gray-700'
        ]"
      >
        <Package class="w-3.5 h-3.5" />
        检查点选择
      </button>
      <button
        @click="activeTab = 'swa'"
        :class="[
          'px-3 py-1.5 rounded-md text-xs flex items-center gap-1.5',
          activeTab === 'swa' ? 'bg-primary-100 accent-text font-medium' : 'bg-gray-100 text-gray-500 hover:text-gray-700'
        ]"
      >
        <Layers class="w-3.5 h-3.5" />
        SWA 平均
      </button>
    </div>

    <!-- Model Merging Tab -->
    <div v-if="activeTab === 'merge'" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium text-sm text-gray-800 mb-4 flex items-center gap-2">
          <GitMerge class="w-4 h-4 accent-text" />
          合并配置
        </h3>

        <div class="space-y-4">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">合并方法</label>
            <select v-model="mergeConfig.method" class="w-full input-light">
              <option v-for="method in mergeMethods" :key="method.id" :value="method.id">
                {{ method.name }}
              </option>
            </select>
            <p v-if="currentMethod" class="text-xs text-gray-500 mt-1">
              {{ currentMethod.description }}
            </p>
          </div>

          <div>
            <label class="block text-xs text-gray-500 mb-1.5">待合并模型</label>
            <div class="space-y-2">
              <div v-for="(model, index) in mergeConfig.models" :key="index" class="flex gap-2">
                <input
                  type="text"
                  v-model="mergeConfig.models[index]"
                  :placeholder="`模型 ${index + 1} 路径`"
                  class="flex-1 input-light"
                >
                <input
                  v-if="mergeConfig.method === 'linear'"
                  type="number"
                  v-model.number="mergeConfig.weights[index]"
                  step="0.1"
                  min="0"
                  max="1"
                  class="w-20 input-light text-center"
                >
                <button
                  v-if="mergeConfig.models.length > 2"
                  @click="removeModel(index)"
                  class="px-3 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-50"
                >
                  ×
                </button>
              </div>
            </div>
            <button
              @click="addModel"
              class="mt-2 text-sm accent-text hover:underline"
            >
              + 添加更多模型
            </button>
          </div>

          <div v-if="mergeConfig.method === 'slerp'">
            <label class="block text-xs text-gray-500 mb-1.5">
              插值因子 (t): {{ mergeConfig.slerp_t.toFixed(2) }}
            </label>
            <input
              type="range"
              v-model.number="mergeConfig.slerp_t"
              min="0"
              max="1"
              step="0.01"
              class="w-full"
            >
            <div class="flex justify-between text-xs text-gray-500">
              <span>模型 1</span>
              <span>模型 2</span>
            </div>
          </div>

          <div v-if="['ties', 'dare'].includes(mergeConfig.method)">
            <label class="block text-xs text-gray-500 mb-1.5">
              密度: {{ mergeConfig.density.toFixed(2) }}
            </label>
            <input
              type="range"
              v-model.number="mergeConfig.density"
              min="0.1"
              max="1"
              step="0.05"
              class="w-full"
            >
          </div>

          <div>
            <label class="block text-xs text-gray-500 mb-1.5">输出路径</label>
            <input
              type="text"
              v-model="mergeConfig.output_path"
              placeholder="/path/to/merged_model"
              class="w-full input-light"
            >
          </div>

          <button
            @click="runMerge"
            :disabled="merging || !mergeConfig.models[0] || !mergeConfig.models[1] || !mergeConfig.output_path"
            class="w-full btn-primary flex items-center justify-center gap-2"
          >
            <Play v-if="!merging" class="w-4 h-4" />
            <span v-if="merging" class="loading">合并中...</span>
            <span v-else>开始合并</span>
          </button>
        </div>
      </div>

      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-6">合并结果</h3>
        <div v-if="!mergeResult" class="text-center py-12 text-gray-500">
          <GitMerge class="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>配置并运行合并以查看结果</p>
        </div>
        <div v-else class="space-y-4">
          <div class="flex items-center gap-2 text-green-400">
            <CheckCircle class="w-5 h-5" />
            <span>模型合并成功</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-400">输出路径</span>
              <span class="font-mono">{{ mergeResult.output_path }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">合并方法</span>
              <span>{{ mergeResult.method }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">合并模型数</span>
              <span>{{ mergeResult.source_models?.length || 2 }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Checkpoint Selection Tab -->
    <div v-if="activeTab === 'checkpoint'" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium text-sm text-gray-800 mb-4 flex items-center gap-2">
          <Package class="w-4 h-4 accent-text" />
          检查点选择
        </h3>

        <div class="space-y-4">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">任务 ID</label>
            <div class="flex gap-2">
              <input
                type="text"
                v-model="checkpointConfig.job_id"
                placeholder="输入任务 ID"
                class="flex-1 input-light"
              >
              <button
                @click="loadCheckpoints"
                :disabled="!checkpointConfig.job_id || loadingCheckpoints"
                class="btn-secondary"
              >
                加载
              </button>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">选择指标</label>
              <select v-model="checkpointConfig.metric" class="w-full input-light">
                <option value="eval_loss">验证损失</option>
                <option value="reward_mean">平均奖励</option>
                <option value="gsm8k">GSM8K 分数</option>
                <option value="math">MATH 分数</option>
                <option value="humaneval">HumanEval 分数</option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1.5">优化方向</label>
              <select v-model="checkpointConfig.mode" class="w-full input-light">
                <option value="min">最小化</option>
                <option value="max">最大化</option>
              </select>
            </div>
          </div>

          <div>
            <label class="block text-xs text-gray-500 mb-1.5">Top K</label>
            <input
              type="number"
              v-model.number="checkpointConfig.top_k"
              min="1"
              max="20"
              class="w-full input-light"
            >
          </div>

          <button
            @click="selectBest"
            :disabled="!checkpointConfig.job_id || loadingCheckpoints"
            class="w-full btn-primary"
          >
            <span v-if="loadingCheckpoints" class="loading">分析中...</span>
            <span v-else>查找最佳检查点</span>
          </button>
        </div>
      </div>

      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-6">可用检查点</h3>
        <div v-if="checkpoints.length === 0" class="text-center py-12 text-gray-500">
          <Package class="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>从任务加载检查点以在此查看</p>
        </div>
        <div v-else class="space-y-2 max-h-96 overflow-y-auto">
          <div
            v-for="cp in checkpoints"
            :key="cp.path"
            :class="[
              'flex items-center justify-between p-3 rounded-lg',
              selectedCheckpoints.some(s => s.path === cp.path) ? 'bg-primary-50 border border-primary-200' : 'bg-gray-50'
            ]"
          >
            <div>
              <p class="font-medium text-sm">步数 {{ cp.step }}</p>
              <p class="text-xs text-gray-500">{{ cp.path }}</p>
            </div>
            <div class="flex items-center gap-3">
              <div v-if="cp.metrics" class="text-right text-xs">
                <p v-if="cp.metrics.eval_loss" class="text-gray-400">
                  损失: {{ cp.metrics.eval_loss.toFixed(4) }}
                </p>
                <p v-if="cp.metrics.reward_mean" class="accent-text">
                  奖励: {{ cp.metrics.reward_mean.toFixed(3) }}
                </p>
              </div>
              <button
                @click="addToSwa(cp)"
                class="p-2 hover:bg-gray-100 rounded-lg"
                title="添加到 SWA"
              >
                <Layers class="w-4 h-4 text-gray-400" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- SWA Tab -->
    <div v-if="activeTab === 'swa'" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium text-sm text-gray-800 mb-4 flex items-center gap-2">
          <Layers class="w-4 h-4 accent-text" />
          随机权重平均 (SWA)
        </h3>

        <div class="space-y-4">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">
              已选检查点 ({{ swaConfig.checkpoints.length }})
            </label>
            <div v-if="swaConfig.checkpoints.length === 0" class="text-center py-8 bg-gray-50 rounded-lg">
              <p class="text-gray-500 text-sm">
                从检查点选择标签页选择检查点
              </p>
            </div>
            <div v-else class="space-y-2 max-h-64 overflow-y-auto">
              <div
                v-for="(path, index) in swaConfig.checkpoints"
                :key="path"
                class="flex items-center gap-2 p-2 bg-gray-50 rounded-lg"
              >
                <span class="flex-1 text-sm font-mono truncate">{{ path }}</span>
                <input
                  type="number"
                  v-model.number="swaConfig.weights[index]"
                  step="0.1"
                  min="0"
                  max="1"
                  class="w-16 input-light text-center text-sm"
                >
                <button
                  @click="removeFromSwa(index)"
                  class="p-1 hover:bg-red-50 rounded text-red-400"
                >
                  ×
                </button>
              </div>
            </div>
          </div>

          <div>
            <label class="block text-xs text-gray-500 mb-1.5">输出路径</label>
            <input
              type="text"
              v-model="swaConfig.output_path"
              placeholder="/path/to/swa_model"
              class="w-full input-light"
            >
          </div>

          <button
            @click="runSwa"
            :disabled="swaMerging || swaConfig.checkpoints.length < 2 || !swaConfig.output_path"
            class="w-full btn-primary flex items-center justify-center gap-2"
          >
            <Play v-if="!swaMerging" class="w-4 h-4" />
            <span v-if="swaMerging" class="loading">合并中...</span>
            <span v-else>运行 SWA 合并</span>
          </button>
        </div>
      </div>

      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium mb-6">SWA 结果</h3>
        <div v-if="!swaResult" class="text-center py-12 text-gray-500">
          <Layers class="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>选择检查点并运行 SWA 以查看结果</p>
        </div>
        <div v-else class="space-y-4">
          <div class="flex items-center gap-2 text-green-400">
            <CheckCircle class="w-5 h-5" />
            <span>SWA 合并成功</span>
          </div>
          <div class="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-400">输出路径</span>
              <span class="font-mono">{{ swaResult.output_path }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">合并检查点数</span>
              <span>{{ swaResult.num_checkpoints }}</span>
            </div>
          </div>
          <div class="bg-blue-50 text-blue-600 text-sm p-3 rounded-lg">
            <AlertCircle class="w-4 h-4 inline mr-2" />
            记得对合并后的模型运行评估以验证性能
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
