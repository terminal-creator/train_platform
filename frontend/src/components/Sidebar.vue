<script setup>
import { RouterLink, useRoute } from 'vue-router'
import {
  Brain,
  LayoutDashboard,
  Calculator,
  Database,
  PlayCircle,
  GitMerge,
  Activity,
  ClipboardCheck
} from 'lucide-vue-next'

const route = useRoute()

const navItems = [
  { path: '/', name: '仪表盘', icon: LayoutDashboard },
  { path: '/compute', name: '计算配置器', icon: Calculator },
  { path: '/datasets', name: '数据集', icon: Database },
  { path: '/jobs', name: '训练任务', icon: PlayCircle },
  { path: '/surgery', name: '模型手术', icon: GitMerge },
  { path: '/monitoring', name: '实时监控', icon: Activity },
  { path: '/evaluation', name: '自定义评估', icon: ClipboardCheck },
]
</script>

<template>
  <aside class="w-56 bg-white border-r border-gray-200 flex flex-col">
    <!-- Logo -->
    <div class="p-4 border-b border-gray-200">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg accent-gradient flex items-center justify-center">
          <Brain class="w-4 h-4 text-white" />
        </div>
        <div>
          <h1 class="font-semibold text-sm text-gray-800">训练平台</h1>
          <p class="text-2xs text-gray-400">基于 verl</p>
        </div>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 p-3 space-y-0.5">
      <RouterLink
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        :class="[
          'sidebar-item',
          route.path === item.path || (item.path !== '/' && route.path.startsWith(item.path))
            ? 'active'
            : 'text-gray-500 hover:text-gray-700'
        ]"
      >
        <component :is="item.icon" class="w-4 h-4" />
        {{ item.name }}
      </RouterLink>
    </nav>

    <!-- API Status -->
    <div class="p-3 border-t border-gray-200">
      <div class="flex items-center gap-2 px-3 py-1.5">
        <div class="w-1.5 h-1.5 rounded-full bg-green-500"></div>
        <div class="flex-1">
          <p class="text-xs font-medium text-gray-700">API 已连接</p>
          <p class="text-2xs text-gray-400">localhost:8000</p>
        </div>
      </div>
    </div>
  </aside>
</template>
