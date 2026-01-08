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
  const availableRewardScripts = ref([])

  // Run mode state
  const runMode = ref('local')  // 'local' or 'ssh'
  const runModeConfig = ref(null)
  const runModeLoading = ref(false)

  const appStore = useAppStore()

  const runningJobs = computed(() =>
    jobs.value.filter(j => ['running', 'queued', 'paused'].includes(j.status))
  )

  const isSSHMode = computed(() => runMode.value === 'ssh')

  // Fetch current run mode configuration
  const fetchRunModeConfig = async () => {
    runModeLoading.value = true
    try {
      runModeConfig.value = await api.getRunModeConfig()
      runMode.value = runModeConfig.value.mode || 'local'
      return runModeConfig.value
    } catch (error) {
      console.error('Failed to fetch run mode config:', error)
      runMode.value = 'local'
      return null
    } finally {
      runModeLoading.value = false
    }
  }

  // Switch run mode (quick switch, saves to backend)
  const switchRunMode = async (mode) => {
    if (mode === runMode.value) return

    runModeLoading.value = true
    try {
      if (mode === 'local') {
        await api.setRunModeConfig({ mode: 'local' })
        runMode.value = 'local'
      } else if (mode === 'ssh' && runModeConfig.value?.ssh_configured) {
        // Use existing SSH config
        await api.setRunModeConfig({
          mode: 'ssh',
          ssh_config: {
            host: runModeConfig.value.ssh_host,
            username: runModeConfig.value.ssh_username,
            working_dir: runModeConfig.value.ssh_working_dir,
            conda_env: runModeConfig.value.ssh_conda_env,
          }
        })
        runMode.value = 'ssh'
      } else {
        appStore.showError('请先在设置页面配置SSH连接')
        return false
      }

      // Reload models and datasets for new mode
      await Promise.all([
        fetchAvailableModels(),
        fetchAvailableDatasets()
      ])

      appStore.showSuccess(`已切换到${mode === 'local' ? '本地' : 'SSH远程'}模式`)
      return true
    } catch (error) {
      appStore.showError(error.message)
      return false
    } finally {
      runModeLoading.value = false
    }
  }

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
      appStore.showSuccess(`任务 "${job.name}" 创建成功`)
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
      appStore.showSuccess('任务已删除')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const startJob = async (id) => {
    try {
      await api.startJob(id)
      await fetchJobs()
      appStore.showSuccess('任务已启动')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const stopJob = async (id) => {
    try {
      await api.stopJob(id)
      await fetchJobs()
      appStore.showSuccess('任务已停止')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const pauseJob = async (id) => {
    try {
      await api.pauseJob(id)
      await fetchJobs()
      appStore.showSuccess('任务已暂停')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  const resumeJob = async (id) => {
    try {
      await api.resumeJob(id)
      await fetchJobs()
      appStore.showSuccess('任务已恢复')
    } catch (error) {
      appStore.showError(error.message)
      throw error
    }
  }

  // Fetch available models based on current run mode
  const fetchAvailableModels = async () => {
    try {
      if (runMode.value === 'ssh') {
        const result = await api.getRemoteModels()
        availableModels.value = result.models || []
      } else {
        availableModels.value = await api.getAvailableModels()
      }
      return availableModels.value
    } catch (error) {
      console.error('Failed to fetch available models:', error)
      availableModels.value = []
      return []
    }
  }

  // Fetch available datasets based on current run mode
  const fetchAvailableDatasets = async () => {
    try {
      if (runMode.value === 'ssh') {
        const result = await api.getRemoteDatasets()
        availableDatasets.value = result.datasets || []
      } else {
        availableDatasets.value = await api.getAvailableDatasets()
      }
      return availableDatasets.value
    } catch (error) {
      console.error('Failed to fetch available datasets:', error)
      availableDatasets.value = []
      return []
    }
  }

  // Fetch available reward scripts
  const fetchAvailableRewardScripts = async () => {
    try {
      availableRewardScripts.value = await api.getAvailableRewardScripts()
      return availableRewardScripts.value
    } catch (error) {
      console.error('Failed to fetch available reward scripts:', error)
      availableRewardScripts.value = []
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
    availableRewardScripts,
    // Run mode
    runMode,
    runModeConfig,
    runModeLoading,
    isSSHMode,
    fetchRunModeConfig,
    switchRunMode,
    // Job operations
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
    fetchAvailableDatasets,
    fetchAvailableRewardScripts
  }
})
