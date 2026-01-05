<script setup>
import { useAppStore } from '@/stores/app'
import { AlertCircle, CheckCircle, Info } from 'lucide-vue-next'

const appStore = useAppStore()

const getIcon = (type) => {
  const icons = {
    error: AlertCircle,
    success: CheckCircle,
    info: Info
  }
  return icons[type] || Info
}

const getColorClass = (type) => {
  const colors = {
    error: 'border-red-200 bg-red-50 text-red-600',
    success: 'border-green-200 bg-green-50 text-green-600',
    info: 'border-blue-200 bg-blue-50 text-blue-600'
  }
  return colors[type] || colors.info
}
</script>

<template>
  <Teleport to="body">
    <div class="fixed top-4 right-4 z-50 space-y-2">
      <TransitionGroup name="toast">
        <div
          v-for="toast in appStore.toasts"
          :key="toast.id"
          :class="[
            'glass-card rounded-lg p-4 flex items-center gap-3 min-w-80 animate-slide-in',
            getColorClass(toast.type)
          ]"
        >
          <component :is="getIcon(toast.type)" class="w-4 h-4 flex-shrink-0" />
          <span class="text-xs text-gray-700">{{ toast.message }}</span>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100%);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100%);
}

.animate-slide-in {
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}
</style>
