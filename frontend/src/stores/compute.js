import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as api from '@/api'
import { useAppStore } from './app'

export const useComputeStore = defineStore('compute', () => {
  const gpuTypes = ref([])
  const modelSizes = ref([])
  const result = ref(null)
  const loading = ref(false)

  const appStore = useAppStore()

  const fetchGpuTypes = async () => {
    try {
      const data = await api.getGpuTypes()
      gpuTypes.value = data.gpu_types
      return data.gpu_types
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const fetchModelSizes = async () => {
    try {
      const data = await api.getModelSizes()
      modelSizes.value = data.model_sizes
      return data.model_sizes
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const calculateConfig = async (config) => {
    loading.value = true
    try {
      result.value = await api.calculateConfig(config)
      appStore.showSuccess('配置计算完成')
      return result.value
    } catch (error) {
      appStore.showError(error.message)
      throw error
    } finally {
      loading.value = false
    }
  }

  const exportYaml = () => {
    if (!result.value?.yaml) return
    const blob = new Blob([result.value.yaml], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'verl_config.yaml'
    a.click()
    URL.revokeObjectURL(url)
    appStore.showSuccess('YAML 配置已导出')
  }

  return {
    gpuTypes,
    modelSizes,
    result,
    loading,
    fetchGpuTypes,
    fetchModelSizes,
    calculateConfig,
    exportYaml
  }
})
