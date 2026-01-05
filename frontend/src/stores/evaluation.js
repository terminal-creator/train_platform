import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as api from '@/api'
import { useAppStore } from './app'

export const useEvaluationStore = defineStore('evaluation', () => {
  const datasets = ref([])
  const evalTasks = ref([])
  const loading = ref(false)
  const datasetsLoading = ref(false)

  const appStore = useAppStore()

  // Grouped results by capability
  const resultsByCapability = computed(() => {
    const grouped = {}
    for (const task of evalTasks.value) {
      if (task.status !== 'completed' || !task.capability) continue
      if (!grouped[task.capability]) {
        grouped[task.capability] = []
      }
      grouped[task.capability].push({
        step: task.checkpoint_step,
        score: task.score,
        datasetName: task.dataset_name,
      })
    }
    // Sort by step
    for (const cap in grouped) {
      grouped[cap].sort((a, b) => a.step - b.step)
    }
    return grouped
  })

  // Fetch all datasets
  const fetchDatasets = async (params = {}) => {
    datasetsLoading.value = true
    try {
      const data = await api.getEvalDatasets(params)
      datasets.value = data.datasets || []
      return data
    } catch (error) {
      appStore.showError(error.message)
      throw error
    } finally {
      datasetsLoading.value = false
    }
  }

  // Upload new dataset
  const uploadDataset = async (file, metadata) => {
    try {
      const formData = new FormData()
      formData.append('file', file)
      const dataset = await api.uploadEvalDataset(formData, metadata)
      datasets.value.unshift(dataset)
      appStore.showSuccess(`数据集 "${dataset.name}" 上传成功`)
      return dataset
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  // Delete dataset
  const deleteDataset = async (uuid) => {
    try {
      await api.deleteEvalDataset(uuid)
      datasets.value = datasets.value.filter(d => d.uuid !== uuid)
      appStore.showSuccess('数据集已删除')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  // Preview dataset
  const previewDataset = async (uuid, limit = 10) => {
    try {
      return await api.previewEvalDataset(uuid, limit)
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  // Trigger evaluation
  const triggerEvaluation = async (params) => {
    try {
      const tasks = await api.triggerEvaluation(params)
      evalTasks.value.push(...tasks)
      appStore.showSuccess(`已启动 ${tasks.length} 个评估任务`)
      return tasks
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  // Fetch tasks for a job
  const fetchJobTasks = async (jobUuid) => {
    loading.value = true
    try {
      evalTasks.value = await api.getJobEvalTasks(jobUuid)
      return evalTasks.value
    } catch (error) {
      appStore.showError(error.message)
      throw error
    } finally {
      loading.value = false
    }
  }

  // Get task details
  const getTaskDetails = async (taskUuid) => {
    try {
      return await api.getEvalTaskDetails(taskUuid)
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  // Get results grouped by capability
  const fetchJobResults = async (jobUuid) => {
    try {
      return await api.getJobEvalResults(jobUuid)
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  // Refresh task status
  const refreshTask = async (taskUuid) => {
    try {
      const task = await api.getEvalTask(taskUuid)
      const index = evalTasks.value.findIndex(t => t.uuid === taskUuid)
      if (index >= 0) {
        evalTasks.value[index] = task
      }
      return task
    } catch (error) {
      console.error('Failed to refresh task:', error)
      throw error
    }
  }

  return {
    datasets,
    evalTasks,
    loading,
    datasetsLoading,
    resultsByCapability,
    fetchDatasets,
    uploadDataset,
    deleteDataset,
    previewDataset,
    triggerEvaluation,
    fetchJobTasks,
    getTaskDetails,
    fetchJobResults,
    refreshTask,
  }
})
