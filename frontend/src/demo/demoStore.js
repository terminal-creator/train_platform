/**
 * Demo模式状态管理
 *
 * 提供Demo模式的全局状态和控制
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api'

export const useDemoStore = defineStore('demo', () => {
  // 状态
  const enabled = ref(false)
  const speed = ref(1.0)
  const startStage = ref(1)
  const loading = ref(false)
  const error = ref(null)

  // 计算属性
  const isActive = computed(() => enabled.value)

  // 从后端获取Demo状态
  async function fetchStatus() {
    try {
      loading.value = true
      const response = await api.get('/demo/status')
      enabled.value = response.data.enabled
      speed.value = response.data.speed
      startStage.value = response.data.start_stage
      error.value = null
    } catch (err) {
      console.error('Failed to fetch demo status:', err)
      error.value = err.message
    } finally {
      loading.value = false
    }
  }

  // 启用Demo模式
  async function enableDemoMode(config = {}) {
    try {
      loading.value = true
      const response = await api.post('/demo/enable', {
        enabled: true,
        speed: config.speed || 1.0,
        start_stage: config.startStage || 1,
      })
      enabled.value = true
      speed.value = response.data.speed
      startStage.value = response.data.start_stage
      error.value = null

      // 保存到localStorage
      localStorage.setItem('demo_mode', 'true')
      localStorage.setItem('demo_speed', String(speed.value))

      return response.data
    } catch (err) {
      console.error('Failed to enable demo mode:', err)
      error.value = err.message
      throw err
    } finally {
      loading.value = false
    }
  }

  // 禁用Demo模式
  async function disableDemoMode() {
    try {
      loading.value = true
      await api.post('/demo/disable')
      enabled.value = false
      error.value = null

      // 清除localStorage
      localStorage.removeItem('demo_mode')
      localStorage.removeItem('demo_speed')
    } catch (err) {
      console.error('Failed to disable demo mode:', err)
      error.value = err.message
      throw err
    } finally {
      loading.value = false
    }
  }

  // 切换Demo模式
  async function toggleDemoMode() {
    if (enabled.value) {
      await disableDemoMode()
    } else {
      await enableDemoMode()
    }
  }

  // 设置演示速度
  async function setSpeed(newSpeed) {
    if (enabled.value) {
      await enableDemoMode({ speed: newSpeed, startStage: startStage.value })
    } else {
      speed.value = newSpeed
    }
  }

  // 初始化 - 从localStorage恢复状态
  function init() {
    const savedMode = localStorage.getItem('demo_mode')
    const savedSpeed = localStorage.getItem('demo_speed')

    if (savedMode === 'true') {
      enabled.value = true
    }
    if (savedSpeed) {
      speed.value = parseFloat(savedSpeed)
    }

    // 同步后端状态
    fetchStatus()
  }

  return {
    // 状态
    enabled,
    speed,
    startStage,
    loading,
    error,

    // 计算属性
    isActive,

    // 方法
    fetchStatus,
    enableDemoMode,
    disableDemoMode,
    toggleDemoMode,
    setSpeed,
    init,
  }
})
