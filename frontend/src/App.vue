<script setup>
import { RouterView } from 'vue-router'
import Sidebar from '@/components/Sidebar.vue'
import Header from '@/components/Header.vue'
import ToastContainer from '@/components/ToastContainer.vue'
</script>

<template>
  <div class="flex min-h-screen bg-light-100">
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
