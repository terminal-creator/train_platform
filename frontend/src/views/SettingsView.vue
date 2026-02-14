<template>
  <div class="p-6">
    <h1 class="text-2xl font-bold text-gray-900 mb-6">设置</h1>

    <!-- Run Mode Configuration -->
    <div class="bg-white rounded-lg shadow p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">运行模式配置</h2>
      <p class="text-sm text-gray-600 mb-4">
        配置训练任务的执行方式。选择在本地执行或通过SSH在GPU服务器上远程执行。
      </p>

      <!-- Mode Selection -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-2">执行模式</label>
        <div class="flex gap-4">
          <label class="flex items-center">
            <input
              type="radio"
              :checked="store.runMode === 'local'"
              @change="store.switchMode('local')"
              class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
            />
            <span class="ml-2 text-sm text-gray-700">本地</span>
          </label>
          <label class="flex items-center">
            <input
              type="radio"
              :checked="store.runMode === 'ssh'"
              @change="store.switchMode('ssh')"
              class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
            />
            <span class="ml-2 text-sm text-gray-700">SSH远程</span>
          </label>
        </div>
      </div>

      <!-- Local Mode Info -->
      <div v-if="store.runMode === 'local'" class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <div class="flex items-start">
          <svg class="h-5 w-5 text-blue-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
          </svg>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-blue-800">本地执行</h3>
            <p class="mt-1 text-sm text-blue-700">
              训练任务将使用Ray在本机上运行。请确保已安装所需的NVIDIA GPU和CUDA。
            </p>
          </div>
        </div>
      </div>

      <!-- SSH Configuration -->
      <div v-if="store.runMode === 'ssh'" class="space-y-4">
        <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
          <div class="flex items-start">
            <svg class="h-5 w-5 text-yellow-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-yellow-800">SSH远程执行</h3>
              <p class="mt-1 text-sm text-yellow-700">
                训练任务将通过SSH在远程GPU服务器上执行。请确保服务器已安装verl及所有依赖项。
              </p>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">主机地址</label>
            <input
              v-model="store.sshConfig.host"
              type="text"
              placeholder="例如：gpu-server.example.com"
              class="w-full input-light"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">端口</label>
            <input
              v-model.number="store.sshConfig.port"
              type="number"
              placeholder="22"
              class="w-full input-light"
            />
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">用户名</label>
            <input
              v-model="store.sshConfig.username"
              type="text"
              placeholder="请输入用户名"
              class="w-full input-light"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">认证方式</label>
            <select
              v-model="store.authMethod"
              class="w-full input-light"
            >
              <option value="password">密码</option>
              <option value="key">SSH密钥</option>
            </select>
          </div>
        </div>

        <div v-if="store.authMethod === 'password'">
          <label class="block text-sm font-medium text-gray-700 mb-1">密码</label>
          <input
            v-model="store.sshConfig.password"
            type="password"
            placeholder="请输入密码"
            class="w-full input-light"
          />
        </div>

        <div v-if="store.authMethod === 'key'">
          <label class="block text-sm font-medium text-gray-700 mb-1">SSH密钥路径</label>
          <input
            v-model="store.sshConfig.key_path"
            type="text"
            placeholder="~/.ssh/id_rsa"
            class="w-full input-light"
          />
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">工作目录</label>
            <input
              v-model="store.sshConfig.working_dir"
              type="text"
              placeholder="~/verl_jobs"
              class="w-full input-light"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Conda环境（可选）</label>
            <input
              v-model="store.sshConfig.conda_env"
              type="text"
              placeholder="例如：verl"
              class="w-full input-light"
            />
          </div>
        </div>

        <!-- Test Connection Button -->
        <div class="flex items-center gap-4 mt-4">
          <button
            @click="store.testSSHConnection"
            :disabled="store.testingConnection || !store.sshConfig.host || !store.sshConfig.username"
            class="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <svg v-if="store.testingConnection" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            测试连接
          </button>
          <span v-if="store.connectionStatus" :class="store.connectionStatus.success ? 'text-green-600' : 'text-red-600'" class="text-sm">
            {{ store.connectionStatus.message || store.connectionStatus.error }}
          </span>
        </div>
      </div>

      <!-- GPU Info -->
      <div v-if="store.gpuInfo && store.gpuInfo.gpus && store.gpuInfo.gpus.length > 0" class="mt-6">
        <h3 class="text-sm font-medium text-gray-700 mb-2">可用GPU</h3>
        <div class="bg-gray-50 rounded-lg p-4">
          <div class="grid gap-2">
            <div
              v-for="gpu in store.gpuInfo.gpus"
              :key="gpu.index"
              class="flex items-center justify-between text-sm"
            >
              <span class="font-medium">GPU {{ gpu.index }}: {{ gpu.name }}</span>
              <span class="text-gray-500">
                {{ gpu.memory_used }}MB / {{ gpu.memory_total }}MB ({{ gpu.utilization }}% 利用率)
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Save Button -->
      <div class="mt-6 flex justify-end">
        <button
          @click="store.saveConfig"
          :disabled="store.saving"
          class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <svg v-if="store.saving" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          保存配置
        </button>
      </div>
    </div>

    <!-- Current Status -->
    <div class="bg-white rounded-lg shadow p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">当前状态</h2>
      <div class="grid grid-cols-2 gap-4 text-sm">
        <div class="flex">
          <span class="text-gray-500 w-24">模式：</span>
          <span class="font-medium">
            {{ store.runMode === 'ssh' ? 'SSH远程' : '本地' }}
          </span>
        </div>
        <div class="flex">
          <span class="text-gray-500 w-24">连接状态：</span>
          <span
            :class="store.connectionStatus?.success ? 'text-green-600' : 'text-gray-500'"
            class="font-medium"
          >
            {{ store.connectionStatus?.success ? '已连接' : '未连接' }}
          </span>
        </div>
        <div class="flex" v-if="store.runMode === 'ssh'">
          <span class="text-gray-500 w-24">SSH主机：</span>
          <span class="font-medium">{{ store.sshConfig.host || '-' }}</span>
        </div>
        <div class="flex">
          <span class="text-gray-500 w-24">GPU数量：</span>
          <span class="font-medium">{{ store.gpuInfo?.gpu_count || 0 }}</span>
        </div>
        <div class="flex" v-if="store.runMode === 'ssh'">
          <span class="text-gray-500 w-24">工作目录：</span>
          <span class="font-medium">{{ store.sshConfig.working_dir || '-' }}</span>
        </div>
        <div class="flex" v-if="store.runMode === 'ssh' && store.sshConfig.conda_env">
          <span class="text-gray-500 w-24">Conda环境：</span>
          <span class="font-medium">{{ store.sshConfig.conda_env }}</span>
        </div>
      </div>
    </div>

    <!-- Hidden: Version info, click 5 times to toggle demo mode -->
    <div class="mt-8 text-center">
      <span
        class="text-xs text-gray-400 cursor-default select-none"
        @click="handleVersionClick"
      >
        v1.0.0
      </span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { useDemoStore } from '@/demo/demoStore'

const store = useSettingsStore()
const demoStore = useDemoStore()

const demoSpeed = ref(1)
const clickCount = ref(0)
let clickTimer = null

function handleVersionClick() {
  clickCount.value++
  if (clickTimer) clearTimeout(clickTimer)
  clickTimer = setTimeout(() => { clickCount.value = 0 }, 2000)
  if (clickCount.value >= 5) {
    clickCount.value = 0
    toggleDemoMode()
  }
}

async function toggleDemoMode() {
  try {
    await demoStore.toggleDemoMode()
  } catch (error) {
    console.error('Failed to toggle demo mode:', error)
  }
}

async function updateDemoSpeed() {
  if (demoStore.enabled) {
    await demoStore.setSpeed(demoSpeed.value)
  }
}

onMounted(() => {
  store.loadConfig()
  demoStore.init()
  demoSpeed.value = demoStore.speed
})
</script>
