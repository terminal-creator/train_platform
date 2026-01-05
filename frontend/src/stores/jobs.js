import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as api from '@/api'
import { useAppStore } from './app'

export const useJobsStore = defineStore('jobs', () => {
  const jobs = ref([])
  const loading = ref(false)
  const currentJob = ref(null)
  const availableModels = ref([])
  const availableDatasets = ref([])

  const appStore = useAppStore()

  const runningJobs = computed(() =>
    jobs.value.filter(j => ['running', 'queued', 'paused'].includes(j.status))
  )

  const fetchJobs = async (params = {}) => {
    loading.value = true
    try {
      const data = await api.getJobs(params)
      jobs.value = data.jobs
      return data
    } catch (error) {
      appStore.showError(error.message)
      throw error
    } finally {
      loading.value = false
    }
  }

  const fetchJob = async (id) => {
    try {
      currentJob.value = await api.getJob(id)
      return currentJob.value
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const createJob = async (data) => {
    try {
      const job = await api.createJob(data)
      jobs.value.unshift(job)
      appStore.showSuccess(`Job "${job.name}" created successfully`)
      return job
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const updateJob = async (id, data) => {
    try {
      const job = await api.updateJob(id, data)
      const index = jobs.value.findIndex(j => j.id === id)
      if (index !== -1) jobs.value[index] = job
      return job
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const deleteJob = async (id) => {
    try {
      await api.deleteJob(id)
      jobs.value = jobs.value.filter(j => j.id !== id)
      appStore.showSuccess('Job deleted')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const startJob = async (id) => {
    try {
      await api.startJob(id)
      await fetchJobs()
      appStore.showSuccess('Job started')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const stopJob = async (id) => {
    try {
      await api.stopJob(id)
      await fetchJobs()
      appStore.showSuccess('Job stopped')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const pauseJob = async (id) => {
    try {
      await api.pauseJob(id)
      await fetchJobs()
      appStore.showSuccess('Job paused')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const resumeJob = async (id) => {
    try {
      await api.resumeJob(id)
      await fetchJobs()
      appStore.showSuccess('Job resumed')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const fetchAvailableModels = async () => {
    try {
      availableModels.value = await api.getAvailableModels()
      return availableModels.value
    } catch (error) {
      console.error('Failed to fetch available models:', error)
      return []
    }
  }

  const fetchAvailableDatasets = async () => {
    try {
      availableDatasets.value = await api.getAvailableDatasets()
      return availableDatasets.value
    } catch (error) {
      console.error('Failed to fetch available datasets:', error)
      return []
    }
  }

  return {
    jobs,
    loading,
    currentJob,
    runningJobs,
    availableModels,
    availableDatasets,
    fetchJobs,
    fetchJob,
    createJob,
    updateJob,
    deleteJob,
    startJob,
    stopJob,
    pauseJob,
    resumeJob,
    fetchAvailableModels,
    fetchAvailableDatasets
  }
})
