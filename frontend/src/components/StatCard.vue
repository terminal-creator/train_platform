<script setup>
defineProps({
  title: String,
  value: [String, Number],
  subtitle: String,
  gradient: Boolean
})
</script>

<template>
  <div :class="['glass-card rounded-lg p-4', gradient && 'stat-card']">
    <div class="flex items-center justify-between mb-2">
      <span class="text-gray-500 text-xs">{{ title }}</span>
      <slot name="icon" />
    </div>
    <p class="text-xl font-semibold text-gray-800">
      <slot>{{ value }}</slot>
    </p>
    <p v-if="subtitle" class="text-2xs text-gray-400 mt-1">{{ subtitle }}</p>
  </div>
</template>
