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

    // Demo mode: pre-fill with demo data
    const isDemoMode = localStorage.getItem('demo_mode') === 'true'
    if (isDemoMode || mergeMethods.value.length > 0) {
      // Pre-fill merge config
      mergeConfig.value = {
        method: 'slerp',
        models: ['/models/Qwen2.5-7B-Math-SFT', '/models/Qwen2.5-7B-Math-GRPO'],
        output_path: '/models/Qwen2.5-7B-Math-Merged',
        weights: [0.5, 0.5],
        slerp_t: 0.6,
        density: 0.5,
        majority_sign_method: 'total'
      }

      // Pre-fill merge result (show completed merge with comparison)
      mergeResult.value = {
        status: 'success',
        method: 'SLERP',
        output_path: '/models/Qwen2.5-7B-Math-Merged',
        source_models: [
          { name: 'Qwen2.5-7B-Math-SFT', path: '/models/Qwen2.5-7B-Math-SFT' },
          { name: 'Qwen2.5-7B-Math-GRPO', path: '/models/Qwen2.5-7B-Math-GRPO' }
        ],
        merge_time_seconds: 125.3,
        output_size_gb: 13.5,
        // 对比数据：合并前 vs 合并后
        comparison: {
          'GSM8K': {
            model_1: { name: 'SFT', score: 75.2 },
            model_2: { name: 'GRPO', score: 82.3 },
            merged: 86.8,
            improvement: '+4.5%'
          },
          'MATH': {
            model_1: { name: 'SFT', score: 38.5 },
            model_2: { name: 'GRPO', score: 45.6 },
            merged: 48.5,
            improvement: '+2.9%'
          },
          'HumanEval': {
            model_1: { name: 'SFT', score: 54.2 },
            model_2: { name: 'GRPO', score: 55.8 },
            merged: 58.5,
            improvement: '+2.7%'
          }
        }
      }

      // Pre-fill checkpoint config
      checkpointConfig.value = {
        job_id: 'demo-grpo-qwen7b-math-002',
        metric: 'reward_mean',
        mode: 'max',
        top_k: 5
      }

      // Pre-fill checkpoints list (available checkpoints)
      checkpoints.value = [
        { step: 3200, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-3200', metrics: { eval_loss: 0.325, reward_mean: 0.82 } },
        { step: 3000, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-3000', metrics: { eval_loss: 0.342, reward_mean: 0.81 } },
        { step: 2800, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-2800', metrics: { eval_loss: 0.358, reward_mean: 0.79 } },
        { step: 2500, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-2500', metrics: { eval_loss: 0.385, reward_mean: 0.78 } },
        { step: 2000, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-2000', metrics: { eval_loss: 0.412, reward_mean: 0.72 } },
        { step: 1500, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-1500', metrics: { eval_loss: 0.456, reward_mean: 0.65 } },
        { step: 1000, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-1000', metrics: { eval_loss: 0.512, reward_mean: 0.55 } },
      ]

      // Pre-fill selected checkpoints (best 3)
      selectedCheckpoints.value = [
        { step: 3200, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-3200', reward_mean: 0.82, selected: true },
        { step: 3000, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-3000', reward_mean: 0.81, selected: true },
        { step: 2800, path: '/checkpoints/demo-grpo-qwen7b-math-002/step-2800', reward_mean: 0.79, selected: true },
      ]

      // Pre-fill SWA config
      swaConfig.value = {
        checkpoints: [
          '/checkpoints/demo-grpo-qwen7b-math-002/step-2500',
          '/checkpoints/demo-grpo-qwen7b-math-002/step-2800',
          '/checkpoints/demo-grpo-qwen7b-math-002/step-3000'
        ],
        output_path: '/models/Qwen2.5-7B-Math-SWA',
        weights: [0.33, 0.33, 0.34]
      }

      // Pre-fill SWA result with comparison
      swaResult.value = {
        status: 'success',
        output_path: '/models/Qwen2.5-7B-Math-SWA',
        source_checkpoints: swaConfig.value.checkpoints,
        merge_time_seconds: 45.2,
        // SWA 前后对比
        comparison: {
          'GSM8K': { before: 82.3, after: 84.2, improvement: '+1.9%' },
          'MATH': { before: 45.6, after: 47.8, improvement: '+2.2%' },
          'HumanEval': { before: 55.8, after: 56.5, improvement: '+0.7%' }
        },
        summary: {
          avg_improvement: '+1.6%',
          best_checkpoint_used: 'step-3000',
          stability_score: 95
        }
      }
    }
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
        <h3 class="font-medium mb-4">合并结果</h3>
        <div v-if="!mergeResult" class="text-center py-12 text-gray-500">
          <GitMerge class="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>配置并运行合并以查看结果</p>
        </div>
        <div v-else class="space-y-4">
          <div class="flex items-center gap-2 text-green-500">
            <CheckCircle class="w-5 h-5" />
            <span class="font-medium">模型合并成功</span>
          </div>

          <!-- 基本信息 -->
          <div class="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-500">输出路径</span>
              <span class="font-mono text-gray-700">{{ mergeResult.output_path }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500">合并方法</span>
              <span class="text-gray-700">{{ mergeResult.method }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500">源模型数量</span>
              <span class="text-gray-700">{{ mergeResult.source_models?.length || 2 }} 个模型</span>
            </div>
            <div v-if="mergeResult.merge_time_seconds" class="flex justify-between">
              <span class="text-gray-500">合并耗时</span>
              <span class="text-gray-700">{{ mergeResult.merge_time_seconds.toFixed(1) }} 秒</span>
            </div>
            <div v-if="mergeResult.output_size_gb" class="flex justify-between">
              <span class="text-gray-500">模型大小</span>
              <span class="text-gray-700">{{ mergeResult.output_size_gb }} GB</span>
            </div>
          </div>

          <!-- 评估对比表格 -->
          <div v-if="mergeResult.comparison" class="bg-gray-50 rounded-lg p-4">
            <h4 class="text-xs text-gray-600 mb-3 font-medium">合并效果对比</h4>
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-gray-500 border-b">
                  <th class="pb-2">Benchmark</th>
                  <th class="pb-2 text-center">模型 1</th>
                  <th class="pb-2 text-center">模型 2</th>
                  <th class="pb-2 text-center">合并后</th>
                  <th class="pb-2 text-right">提升</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(data, benchmark) in mergeResult.comparison" :key="benchmark" class="border-b border-gray-100">
                  <td class="py-2 font-medium text-gray-700">{{ benchmark }}</td>
                  <td class="py-2 text-center text-gray-500">
                    <span class="text-xs text-gray-400">{{ data.model_1.name }}</span><br>
                    {{ data.model_1.score }}%
                  </td>
                  <td class="py-2 text-center text-gray-500">
                    <span class="text-xs text-gray-400">{{ data.model_2.name }}</span><br>
                    {{ data.model_2.score }}%
                  </td>
                  <td class="py-2 text-center font-bold text-green-600">{{ data.merged }}%</td>
                  <td class="py-2 text-right text-green-500 font-medium">{{ data.improvement }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 源模型列表 -->
          <div v-if="mergeResult.source_models?.length" class="bg-gray-50 rounded-lg p-4">
            <h4 class="text-xs text-gray-500 mb-2">源模型路径</h4>
            <div class="space-y-1">
              <div v-for="(model, idx) in mergeResult.source_models" :key="idx" class="text-sm font-mono text-gray-600">
                {{ idx + 1 }}. {{ model.path || model }}
              </div>
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
        <h3 class="font-medium mb-4">SWA 结果</h3>
        <div v-if="!swaResult" class="text-center py-12 text-gray-500">
          <Layers class="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>选择检查点并运行 SWA 以查看结果</p>
        </div>
        <div v-else class="space-y-4">
          <div class="flex items-center gap-2 text-green-500">
            <CheckCircle class="w-5 h-5" />
            <span class="font-medium">SWA 权重平均成功</span>
          </div>

          <!-- 基本信息 -->
          <div class="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-500">输出路径</span>
              <span class="font-mono text-gray-700">{{ swaResult.output_path }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500">合并检查点数</span>
              <span class="text-gray-700">{{ swaResult.source_checkpoints?.length || swaResult.num_checkpoints || 3 }} 个</span>
            </div>
            <div v-if="swaResult.merge_time_seconds" class="flex justify-between">
              <span class="text-gray-500">合并耗时</span>
              <span class="text-gray-700">{{ swaResult.merge_time_seconds.toFixed(1) }} 秒</span>
            </div>
            <div v-if="swaResult.improvement" class="flex justify-between">
              <span class="text-gray-500">性能提升</span>
              <span class="text-green-600 font-medium">{{ swaResult.improvement }}</span>
            </div>
          </div>

          <!-- 源检查点列表 -->
          <div v-if="swaResult.source_checkpoints?.length" class="bg-gray-50 rounded-lg p-4">
            <h4 class="text-xs text-gray-500 mb-2">源检查点</h4>
            <div class="space-y-1">
              <div v-for="(cp, idx) in swaResult.source_checkpoints" :key="idx" class="text-sm font-mono text-gray-600">
                {{ idx + 1 }}. {{ cp }}
              </div>
            </div>
          </div>

          <!-- SWA 前后对比 -->
          <div v-if="swaResult.comparison" class="bg-gray-50 rounded-lg p-4">
            <h4 class="text-xs text-gray-600 mb-3 font-medium">SWA 效果对比</h4>
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-gray-500 border-b">
                  <th class="pb-2">Benchmark</th>
                  <th class="pb-2 text-center">SWA 前</th>
                  <th class="pb-2 text-center">SWA 后</th>
                  <th class="pb-2 text-right">提升</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(data, benchmark) in swaResult.comparison" :key="benchmark" class="border-b border-gray-100">
                  <td class="py-2 font-medium text-gray-700">{{ benchmark }}</td>
                  <td class="py-2 text-center text-gray-500">{{ data.before }}%</td>
                  <td class="py-2 text-center font-bold text-green-600">{{ data.after }}%</td>
                  <td class="py-2 text-right text-green-500 font-medium">{{ data.improvement }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 汇总信息 -->
          <div v-if="swaResult.summary" class="bg-green-50 rounded-lg p-4">
            <div class="grid grid-cols-3 gap-3 text-center">
              <div>
                <div class="text-lg font-bold text-green-600">{{ swaResult.summary.avg_improvement }}</div>
                <div class="text-xs text-gray-500">平均提升</div>
              </div>
              <div>
                <div class="text-lg font-bold text-blue-600">{{ swaResult.summary.stability_score }}</div>
                <div class="text-xs text-gray-500">稳定性评分</div>
              </div>
              <div>
                <div class="text-sm font-medium text-gray-700">{{ swaResult.summary.best_checkpoint_used }}</div>
                <div class="text-xs text-gray-500">最佳检查点</div>
              </div>
            </div>
          </div>

          <div class="bg-blue-50 text-blue-600 text-sm p-3 rounded-lg">
            <AlertCircle class="w-4 h-4 inline mr-2" />
            SWA 通过平均多个检查点的权重来提高模型稳定性和泛化能力
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
