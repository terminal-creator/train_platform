<script setup>
import { computed } from 'vue'
import { RouterView, useRoute } from 'vue-router'
import Sidebar from '@/components/Sidebar.vue'
import Header from '@/components/Header.vue'
import ToastContainer from '@/components/ToastContainer.vue'

const route = useRoute()

// Check if current route uses blank layout (no sidebar/header)
const isBlankLayout = computed(() => route.meta?.layout === 'blank')
</script>

<template>
  <!-- Blank layout for landing page -->
  <div v-if="isBlankLayout" class="min-h-screen">
    <RouterView />
  </div>

  <!-- Default layout with sidebar and header -->
  <div v-else class="flex min-h-screen bg-light-100">
    <ToastContainer />
    <Sidebar />
    <main class="flex-1 overflow-auto bg-light-100">
      <Header />
      <div class="p-6">
        <RouterView />
      </div>
    </main>
  </div>
</template>
