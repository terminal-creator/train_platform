<script setup>
import { X } from 'lucide-vue-next'

defineProps({
  title: String,
  show: Boolean,
  maxWidth: {
    type: String,
    default: 'max-w-2xl'
  }
})

const emit = defineEmits(['close'])
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div
        v-if="show"
        class="fixed inset-0 z-50 flex items-center justify-center"
        @click.self="emit('close')"
      >
        <div class="absolute inset-0 bg-black/40 backdrop-blur-sm" />
        <div :class="['glass-card rounded-xl p-6 w-full relative max-h-[90vh] overflow-y-auto', maxWidth]">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-base font-semibold text-gray-800">{{ title }}</h3>
            <button @click="emit('close')" class="p-1.5 hover:bg-gray-100 rounded-md text-gray-500">
              <X class="w-4 h-4" />
            </button>
          </div>
          <slot />
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-enter-active,
.modal-leave-active {
  transition: all 0.3s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-from .glass-card,
.modal-leave-to .glass-card {
  transform: scale(0.95);
}
</style>
