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
  ClipboardCheck,
  Settings,
  Workflow,
  ListChecks,
  BookOpen,
  FileText,
  Factory
} from 'lucide-vue-next'

const route = useRoute()

const navItems = [
  { path: '/settings', name: '1. 设置', icon: Settings },
  { path: '/datasets', name: '2. 数据集', icon: Database },
  { path: '/compute', name: '3. 计算配置', icon: Calculator },
  { path: '/jobs', name: '4. 训练任务', icon: PlayCircle },
  { path: '/pipelines', name: '5. Pipeline 管理', icon: Workflow },
  { path: '/tasks', name: '6. 任务监控', icon: ListChecks },
  { path: '/monitoring', name: '7. 实时监控', icon: Activity },
  { path: '/data-factory', name: '8. 数据工厂', icon: Factory },
  { path: '/evaluation', name: '9. 自定义评估', icon: ClipboardCheck },
  { path: '/surgery', name: '10. 模型手术', icon: GitMerge },
  { path: '/dashboard', name: '11. 仪表盘', icon: LayoutDashboard },
]

const extraItems = [
  { path: '/blog', name: '技术Blog', icon: BookOpen },
  { path: '/resume', name: '简历修改', icon: FileText },
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
    <nav class="flex-1 p-3 space-y-0.5 overflow-y-auto">
      <RouterLink
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        :class="[
          'sidebar-item',
          route.path === item.path || route.path.startsWith(item.path + '/')
            ? 'active'
            : 'text-gray-500 hover:text-gray-700'
        ]"
      >
        <component :is="item.icon" class="w-4 h-4" />
        {{ item.name }}
      </RouterLink>

      <!-- Divider -->
      <div class="my-3 border-t border-gray-200"></div>

      <!-- Extra Items -->
      <RouterLink
        v-for="item in extraItems"
        :key="item.path"
        :to="item.path"
        :class="[
          'sidebar-item',
          route.path === item.path || route.path.startsWith(item.path + '/')
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
