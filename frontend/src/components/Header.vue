<script setup>
import { computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { useDemoStore } from '@/demo/demoStore'
import { Settings, Play, Pause } from 'lucide-vue-next'

const route = useRoute()
const router = useRouter()
const appStore = useAppStore()
const demoStore = useDemoStore()

const pageTitle = computed(() => {
  const titles = {
    '/': '仪表盘',
    '/settings': '设置',
    '/datasets': '数据集',
    '/compute': '计算配置',
    '/jobs': '训练任务',
    '/pipelines': 'Pipeline 管理',
    '/tasks': '任务监控',
    '/monitoring': '实时监控',
    '/evaluation': '自定义评估',
    '/surgery': '模型手术'
  }
  return titles[route.path] || route.name || '训练平台'
})

function goToSettings() {
  router.push('/settings')
}

onMounted(() => {
  demoStore.init()
})
</script>

<template>
  <header class="h-12 border-b border-gray-200 flex items-center justify-between px-6 bg-white/90 backdrop-blur sticky top-0 z-10">
    <div class="flex items-center gap-3">
      <h2 class="text-base font-semibold text-gray-800">{{ pageTitle }}</h2>
      <span v-if="appStore.loading" class="loading text-xs text-gray-400">Loading...</span>
    </div>
    <div class="flex items-center gap-3">
      <button
        @click="goToSettings"
        class="p-1.5 rounded-md hover:bg-gray-100 text-gray-500"
        title="设置"
      >
        <Settings class="w-4 h-4" />
      </button>
    </div>
  </header>
</template>
