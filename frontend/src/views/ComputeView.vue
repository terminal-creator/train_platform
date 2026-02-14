<script setup>
import { ref, onMounted, computed } from 'vue'
import { useComputeStore } from '@/stores/compute'
import { Settings, HardDrive, Zap } from 'lucide-vue-next'

const computeStore = useComputeStore()

const config = ref({
  model_size: '7B',
  gpu_type: 'A100-80G',
  num_gpus: 8,
  context_length: 4096,
  training_type: 'grpo',
  lora_enabled: false,
  lora_rank: 8
})

const contextOptions = [1024, 2048, 4096, 8192, 16384, 32768]

const trainingTypes = [
  { value: 'sft', label: 'SFT (监督微调)' },
  { value: 'ppo', label: 'PPO (近端策略优化)' },
  { value: 'grpo', label: 'GRPO (组相对策略优化)' },
  { value: 'dpo', label: 'DPO (直接偏好优化)' },
  { value: 'gspo', label: 'GSPO (组自博弈偏好优化)' }
]

const gpuMemory = computed(() => {
  const gpu = computeStore.gpuTypes.find(g => g.id === config.value.gpu_type)
  return gpu?.memory_gb || 80
})

const memory = computed(() => {
  const r = computeStore.result
  if (!r?.memory_estimate) return null
  return {
    modelWeights: r.memory_estimate.model_weights_gb,
    optimizer: r.memory_estimate.optimizer_states_gb,
    gradients: r.memory_estimate.gradients_gb,
    activations: r.memory_estimate.activations_gb,
    perGpu: r.memory_estimate.per_gpu_gb,
    utilization: r.memory_estimate.utilization_percent
  }
})

const formatLearningRate = (lr) => {
  if (!lr) return '1e-6'
  if (lr >= 1) return lr.toString()
  const exp = Math.floor(Math.log10(lr))
  const mantissa = lr / Math.pow(10, exp)
  if (Math.abs(mantissa - 1) < 0.01) {
    return `1e${exp}`
  }
  return `${mantissa.toFixed(1)}e${exp}`
}

const configResult = computed(() => {
  const r = computeStore.result
  if (!r?.config) return null
  return {
    zeroStage: r.zero_stage,
    microBatchSize: r.config.actor?.micro_batch_size_per_gpu,
    gradientAccum: r.config.trainer?.gradient_accumulation_steps,
    globalBatchSize: r.config.computed?.global_batch_size,
    tensorParallel: r.config.rollout?.tensor_parallel_size,
    learningRate: formatLearningRate(r.config.actor?.optim?.lr)
  }
})

const calculate = async () => {
  await computeStore.calculateConfig(config.value)
}

onMounted(async () => {
  await Promise.all([
    computeStore.fetchGpuTypes(),
    computeStore.fetchModelSizes()
  ])
})
</script>

<template>
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- Input Panel -->
    <div class="glass-card rounded-lg p-4">
      <h3 class="font-medium text-sm text-gray-800 mb-4 flex items-center gap-2">
        <Settings class="w-4 h-4 accent-text" />
        配置参数
      </h3>

      <div class="space-y-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">模型规模</label>
          <select v-model="config.model_size" class="w-full input-light">
            <option v-for="size in computeStore.modelSizes" :key="size.id" :value="size.id">
              {{ size.name }} ({{ size.params_billion }}B 参数)
            </option>
          </select>
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">GPU 类型</label>
          <select v-model="config.gpu_type" class="w-full input-light">
            <option v-for="gpu in computeStore.gpuTypes" :key="gpu.id" :value="gpu.id">
              {{ gpu.name }} ({{ gpu.memory_gb }}GB)
            </option>
          </select>
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">GPU 数量</label>
          <input type="number" v-model.number="config.num_gpus" min="1" max="1024" class="w-full input-light">
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">上下文长度</label>
          <select v-model="config.context_length" class="w-full input-light">
            <option v-for="len in contextOptions" :key="len" :value="len">
              {{ len >= 1024 ? (len / 1024) + 'K' : len }}
            </option>
          </select>
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1.5">训练类型</label>
          <select v-model="config.training_type" class="w-full input-light">
            <option v-for="type in trainingTypes" :key="type.value" :value="type.value">
              {{ type.label }}
            </option>
          </select>
        </div>

        <div class="flex items-center gap-3">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" v-model="config.lora_enabled" class="w-3.5 h-3.5 accent-primary-500">
            <span class="text-xs text-gray-700">启用 LoRA</span>
          </label>
          <input
            v-if="config.lora_enabled"
            type="number"
            v-model.number="config.lora_rank"
            min="1"
            max="256"
            placeholder="Rank"
            class="w-20 input-light !px-2 !py-1.5 text-xs"
          >
        </div>

        <button
          @click="calculate"
          :disabled="computeStore.loading"
          class="w-full btn-primary"
        >
          <span v-if="computeStore.loading" class="loading">计算中...</span>
          <span v-else>计算配置</span>
        </button>
      </div>
    </div>

    <!-- Results Panel -->
    <div class="space-y-4">
      <!-- Memory Breakdown -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium text-sm text-gray-800 mb-4 flex items-center gap-2">
          <HardDrive class="w-4 h-4 accent-text" />
          显存占用分析
        </h3>

        <div class="mb-4">
          <div class="flex justify-between text-xs mb-1.5">
            <span class="text-gray-500">GPU 显存使用</span>
            <span :class="memory?.utilization > 90 ? 'text-red-500' : 'accent-text'">
              {{ memory?.perGpu?.toFixed(1) || 0 }} / {{ gpuMemory }} GB
              ({{ memory?.utilization?.toFixed(1) || 0 }}%)
            </span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2.5">
            <div
              :class="['h-2.5 rounded-full transition-all', memory?.utilization > 90 ? 'bg-red-500' : 'accent-gradient']"
              :style="{ width: Math.min(memory?.utilization || 0, 100) + '%' }"
            />
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3 text-xs">
          <div class="bg-gray-50 rounded-md p-2.5">
            <span class="text-gray-500">Model Weights</span>
            <p class="font-medium text-gray-800">{{ memory?.modelWeights?.toFixed(2) || 0 }} GB</p>
          </div>
          <div class="bg-gray-50 rounded-md p-2.5">
            <span class="text-gray-500">Optimizer States</span>
            <p class="font-medium text-gray-800">{{ memory?.optimizer?.toFixed(2) || 0 }} GB</p>
          </div>
          <div class="bg-gray-50 rounded-md p-2.5">
            <span class="text-gray-500">Gradients</span>
            <p class="font-medium text-gray-800">{{ memory?.gradients?.toFixed(2) || 0 }} GB</p>
          </div>
          <div class="bg-gray-50 rounded-md p-2.5">
            <span class="text-gray-500">Activations</span>
            <p class="font-medium text-gray-800">{{ memory?.activations?.toFixed(2) || 0 }} GB</p>
          </div>
        </div>
      </div>

      <!-- Recommended Config -->
      <div class="glass-card rounded-lg p-4">
        <h3 class="font-medium text-sm text-gray-800 mb-4 flex items-center gap-2">
          <Zap class="w-4 h-4 accent-text" />
          推荐配置
        </h3>

        <div class="space-y-2 text-xs">
          <div class="flex justify-between py-1.5 border-b border-gray-100">
            <span class="text-gray-500">ZeRO Stage</span>
            <span class="font-medium text-gray-800">ZeRO-{{ configResult?.zeroStage || 2 }}</span>
          </div>
          <div class="flex justify-between py-1.5 border-b border-gray-100">
            <span class="text-gray-500">Micro Batch Size</span>
            <span class="font-medium text-gray-800">{{ configResult?.microBatchSize || 4 }}</span>
          </div>
          <div class="flex justify-between py-1.5 border-b border-gray-100">
            <span class="text-gray-500">Gradient Accumulation</span>
            <span class="font-medium text-gray-800">{{ configResult?.gradientAccum || 8 }}</span>
          </div>
          <div class="flex justify-between py-1.5 border-b border-gray-100">
            <span class="text-gray-500">Global Batch Size</span>
            <span class="font-medium text-gray-800">{{ configResult?.globalBatchSize || 256 }}</span>
          </div>
          <div class="flex justify-between py-1.5 border-b border-gray-100">
            <span class="text-gray-500">Tensor Parallel</span>
            <span class="font-medium text-gray-800">{{ configResult?.tensorParallel || 1 }}</span>
          </div>
          <div class="flex justify-between py-1.5">
            <span class="text-gray-500">Learning Rate</span>
            <span class="font-medium text-gray-800">{{ configResult?.learningRate || '1e-6' }}</span>
          </div>
        </div>

        <div v-if="computeStore.result?.warnings?.length" class="mt-3 space-y-1.5">
          <div
            v-for="warning in computeStore.result.warnings"
            :key="warning.message"
            :class="[
              'text-xs px-2.5 py-1.5 rounded-md',
              warning.level === 'error' ? 'bg-red-50 text-red-600' : 'bg-yellow-50 text-yellow-600'
            ]"
          >
            {{ warning.message }}
          </div>
        </div>

        <div v-if="computeStore.result?.recommendations?.length" class="mt-3 space-y-1.5">
          <div
            v-for="rec in computeStore.result.recommendations"
            :key="rec"
            class="text-xs px-2.5 py-1.5 rounded-md bg-blue-50 text-blue-600"
          >
            {{ rec }}
          </div>
        </div>

        <button
          v-if="computeStore.result?.yaml"
          @click="computeStore.exportYaml"
          class="w-full mt-4 btn-secondary"
        >
          导出 YAML 配置
        </button>
      </div>
    </div>
  </div>
</template>
