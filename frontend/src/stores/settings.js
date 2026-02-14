import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import * as api from '@/api'
import { useAppStore } from './app'

const STORAGE_KEY = 'settings_store'

// Helper to load from localStorage
function loadFromStorage() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      return JSON.parse(stored)
    }
  } catch (e) {
    console.error('Failed to load settings from localStorage:', e)
  }
  return null
}

// Helper to save to localStorage
function saveToStorage(data) {
  try {
    // Don't save password to localStorage for security
    const safeData = {
      ...data,
      sshConfig: {
        ...data.sshConfig,
        password: '' // Clear password for security
      }
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(safeData))
  } catch (e) {
    console.error('Failed to save settings to localStorage:', e)
  }
}

export const useSettingsStore = defineStore('settings', () => {
  // Helper function to get app store (lazy initialization)
  const getAppStore = () => useAppStore()

  // Load stored state
  const stored = loadFromStorage()

  // State - persisted across tab switches and page refreshes
  const runMode = ref(stored?.runMode || 'local')
  const authMethod = ref(stored?.authMethod || 'password')
  const sshConfig = ref(stored?.sshConfig || {
    host: '',
    port: 22,
    username: '',
    password: '',
    key_path: '',
    working_dir: '~/verl_jobs',
    conda_env: ''
  })

  const currentConfig = ref(stored?.currentConfig || {
    mode: 'local',
    ssh_configured: false,
    ssh_host: '',
    ssh_port: 22,
    ssh_username: '',
    ssh_working_dir: '',
    ssh_conda_env: ''
  })
  const connectionStatus = ref(stored?.connectionStatus || null)
  const gpuInfo = ref(stored?.gpuInfo || null)
  const testingConnection = ref(false)
  const saving = ref(false)
  const loaded = ref(false)

  // Watch for changes and persist to localStorage
  watch(
    [runMode, authMethod, sshConfig, currentConfig, connectionStatus, gpuInfo],
    () => {
      saveToStorage({
        runMode: runMode.value,
        authMethod: authMethod.value,
        sshConfig: sshConfig.value,
        currentConfig: currentConfig.value,
        connectionStatus: connectionStatus.value,
        gpuInfo: gpuInfo.value
      })
    },
    { deep: true }
  )

  // Load current configuration
  const loadConfig = async () => {
    if (loaded.value) return currentConfig.value

    try {
      const apiConfig = await api.getRunModeConfig()

      // Merge API response with current config
      currentConfig.value = {
        ...currentConfig.value,
        ...apiConfig
      }

      // Only update runMode and sshConfig from API if current sshConfig is empty
      // (meaning user hasn't entered any data yet)
      const hasCurrentConfig = sshConfig.value.host && sshConfig.value.host.trim() !== ''

      if (!hasCurrentConfig) {
        // No current config data, use API values
        runMode.value = currentConfig.value.mode || 'local'

        if (currentConfig.value.ssh_host) {
          sshConfig.value.host = currentConfig.value.ssh_host
          sshConfig.value.port = currentConfig.value.ssh_port || 22
          sshConfig.value.username = currentConfig.value.ssh_username || ''
          sshConfig.value.working_dir = currentConfig.value.ssh_working_dir || '~/verl_jobs'
          sshConfig.value.conda_env = currentConfig.value.ssh_conda_env || ''
        }
      }

      // Set connection status based on config
      if (runMode.value === 'ssh') {
        if (currentConfig.value.ssh_configured || sshConfig.value.host) {
          connectionStatus.value = { success: true, message: '已配置' }
        }
      } else if (runMode.value === 'local') {
        connectionStatus.value = { success: true, message: '本地模式' }
      }

      // Load GPU info
      await loadGpuInfo()

      loaded.value = true
      return currentConfig.value
    } catch (error) {
      console.error('Failed to load config:', error)
      // Still set loaded to true to avoid retrying on error
      loaded.value = true
      return null
    }
  }

  // Load GPU information
  const loadGpuInfo = async () => {
    try {
      if (runMode.value === 'ssh' && currentConfig.value?.ssh_configured) {
        // Get GPU info from remote server using saved config
        const config = {
          host: sshConfig.value.host,
          port: sshConfig.value.port || 22,
          username: sshConfig.value.username,
          password: sshConfig.value.password,
          working_dir: sshConfig.value.working_dir,
          conda_env: sshConfig.value.conda_env || null
        }
        try {
          gpuInfo.value = await api.getSSHGpuInfo(config)
        } catch (e) {
          // If SSH GPU info fails, try local
          gpuInfo.value = await api.getGpuInfo()
        }
      } else {
        gpuInfo.value = await api.getGpuInfo()
      }
    } catch (error) {
      console.error('Failed to load GPU info:', error)
      gpuInfo.value = { gpus: [], gpu_count: 0 }
    }
  }

  // Test SSH connection
  const testSSHConnection = async () => {
    testingConnection.value = true
    connectionStatus.value = null

    try {
      const config = {
        host: sshConfig.value.host,
        port: sshConfig.value.port || 22,
        username: sshConfig.value.username,
        working_dir: sshConfig.value.working_dir,
        conda_env: sshConfig.value.conda_env || null
      }

      if (authMethod.value === 'password') {
        config.password = sshConfig.value.password
      } else {
        config.key_path = sshConfig.value.key_path
      }

      connectionStatus.value = await api.testSSHConnection(config)

      if (connectionStatus.value.success) {
        // Also get GPU info from remote server
        gpuInfo.value = await api.getSSHGpuInfo(config)

        // Update currentConfig to reflect the test result
        currentConfig.value = {
          ...currentConfig.value,
          mode: 'ssh',
          ssh_configured: true,
          ssh_host: sshConfig.value.host,
          ssh_port: sshConfig.value.port,
          ssh_username: sshConfig.value.username,
          ssh_working_dir: sshConfig.value.working_dir,
          ssh_conda_env: sshConfig.value.conda_env
        }

        getAppStore().showSuccess('连接成功！')
      } else {
        getAppStore().showError(connectionStatus.value.error || '连接失败')
      }
    } catch (error) {
      connectionStatus.value = { success: false, error: error.message }
      getAppStore().showError(error.message)
    } finally {
      testingConnection.value = false
    }
  }

  // Save configuration
  const saveConfig = async () => {
    saving.value = true

    try {
      const data = {
        mode: runMode.value
      }

      if (runMode.value === 'ssh') {
        data.ssh_config = {
          host: sshConfig.value.host,
          port: sshConfig.value.port || 22,
          username: sshConfig.value.username,
          working_dir: sshConfig.value.working_dir,
          conda_env: sshConfig.value.conda_env || null
        }

        if (authMethod.value === 'password') {
          data.ssh_config.password = sshConfig.value.password
        } else {
          data.ssh_config.key_path = sshConfig.value.key_path
        }
      }

      currentConfig.value = await api.setRunModeConfig(data)

      // Update connection status
      if (runMode.value === 'ssh' && currentConfig.value.ssh_configured) {
        connectionStatus.value = { success: true, message: '已保存并配置' }
      } else if (runMode.value === 'local') {
        connectionStatus.value = { success: true, message: '本地模式' }
      }

      getAppStore().showSuccess('配置已保存！')

      // Refresh GPU info
      await loadGpuInfo()
    } catch (error) {
      getAppStore().showError(error.message)
    } finally {
      saving.value = false
    }
  }

  // Switch run mode
  const switchMode = async (newMode) => {
    runMode.value = newMode
    if (newMode === 'local') {
      connectionStatus.value = { success: true, message: '本地模式' }
      await loadGpuInfo()
    } else {
      if (!currentConfig.value?.ssh_configured) {
        connectionStatus.value = null
        gpuInfo.value = null
      }
    }
  }

  return {
    // State
    runMode,
    authMethod,
    sshConfig,
    currentConfig,
    connectionStatus,
    gpuInfo,
    testingConnection,
    saving,
    loaded,
    // Actions
    loadConfig,
    loadGpuInfo,
    testSSHConnection,
    saveConfig,
    switchMode
  }
})
