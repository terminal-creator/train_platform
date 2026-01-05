<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { Settings } from 'lucide-vue-next'

const route = useRoute()
const appStore = useAppStore()

const pageTitle = computed(() => {
  const titles = {
    '/': 'Dashboard',
    '/compute': 'Compute Calculator',
    '/jobs': 'Training Jobs',
    '/surgery': 'Model Surgery',
    '/monitoring': 'Monitoring'
  }
  return titles[route.path] || 'Monitoring'
})
</script>

<template>
  <header class="h-12 border-b border-gray-200 flex items-center justify-between px-6 bg-white/90 backdrop-blur sticky top-0 z-10">
    <div class="flex items-center gap-3">
      <h2 class="text-base font-semibold text-gray-800">{{ pageTitle }}</h2>
      <span v-if="appStore.loading" class="loading text-xs text-gray-400">Loading...</span>
    </div>
    <div class="flex items-center gap-3">
      <button class="p-1.5 rounded-md hover:bg-gray-100 text-gray-500">
        <Settings class="w-4 h-4" />
      </button>
    </div>
  </header>
</template>
