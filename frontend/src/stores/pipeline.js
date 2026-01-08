/**
 * Pipeline Store
 * Phase 4: Pipeline 状态管理
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as pipelineAPI from '@/api/pipeline'
import { useAppStore } from './app'

export const usePipelineStore = defineStore('pipeline', () => {
  const pipelines = ref([])
  const currentPipeline = ref(null)
  const currentPipelineStages = ref([])
  const loading = ref(false)
  const creating = ref(false)
  const total = ref(0)

  const appStore = useAppStore()

  // Computed
  const runningPipelines = computed(() =>
    pipelines.value.filter(p => p.status === 'running')
  )

  const pendingPipelines = computed(() =>
    pipelines.value.filter(p => p.status === 'pending')
  )

  const completedPipelines = computed(() =>
    pipelines.value.filter(p => p.status === 'completed')
  )

  const failedPipelines = computed(() =>
    pipelines.value.filter(p => p.status === 'failed')
  )

  /**
   * 获取 Pipeline 列表
   */
  const fetchPipelines = async (params = {}) => {
    loading.value = true
    try {
      const response = await pipelineAPI.listPipelines(params)
      pipelines.value = response.pipelines || []
      total.value = response.total || 0
      return response
    } catch (error) {
      appStore.showError(`获取 Pipeline 列表失败: ${error.message}`)
      throw error
    } finally {
      loading.value = false
    }
  }

  /**
   * 创建 Pipeline
   */
  const createPipeline = async (data) => {
    creating.value = true
    try {
      const pipeline = await pipelineAPI.createPipeline(data)
      pipelines.value.unshift(pipeline)
      appStore.showSuccess(`Pipeline "${pipeline.name}" 创建成功`)
      return pipeline
    } catch (error) {
      appStore.showError(`创建 Pipeline 失败: ${error.message}`)
      throw error
    } finally {
      creating.value = false
    }
  }

  /**
   * 获取 Pipeline 详情
   */
  const fetchPipeline = async (uuid) => {
    loading.value = true
    try {
      const pipeline = await pipelineAPI.getPipeline(uuid)
      currentPipeline.value = pipeline
      return pipeline
    } catch (error) {
      appStore.showError(`获取 Pipeline 详情失败: ${error.message}`)
      throw error
    } finally {
      loading.value = false
    }
  }

  /**
   * 获取 Pipeline 状态（包含所有 stages）
   */
  const fetchPipelineStatus = async (uuid) => {
    try {
      const response = await pipelineAPI.getPipelineStatus(uuid)
      currentPipeline.value = response.pipeline
      currentPipelineStages.value = response.stages || []
      return response
    } catch (error) {
      appStore.showError(`获取 Pipeline 状态失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 启动 Pipeline
   */
  const startPipeline = async (uuid) => {
    try {
      const response = await pipelineAPI.startPipeline(uuid)
      appStore.showSuccess('Pipeline 已启动')

      // 更新列表中的状态
      const index = pipelines.value.findIndex(p => p.uuid === uuid)
      if (index !== -1) {
        pipelines.value[index].status = 'running'
      }

      // 更新当前 Pipeline
      if (currentPipeline.value?.uuid === uuid) {
        currentPipeline.value.status = 'running'
      }

      return response
    } catch (error) {
      appStore.showError(`启动 Pipeline 失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 取消 Pipeline
   */
  const cancelPipeline = async (uuid) => {
    try {
      const response = await pipelineAPI.cancelPipeline(uuid)
      appStore.showSuccess('Pipeline 已取消')

      // 更新列表中的状态
      const index = pipelines.value.findIndex(p => p.uuid === uuid)
      if (index !== -1) {
        pipelines.value[index].status = 'cancelled'
      }

      // 更新当前 Pipeline
      if (currentPipeline.value?.uuid === uuid) {
        currentPipeline.value.status = 'cancelled'
      }

      return response
    } catch (error) {
      appStore.showError(`取消 Pipeline 失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 删除 Pipeline
   */
  const deletePipeline = async (uuid) => {
    try {
      await pipelineAPI.deletePipeline(uuid)
      appStore.showSuccess('Pipeline 已删除')

      // 从列表中移除
      const index = pipelines.value.findIndex(p => p.uuid === uuid)
      if (index !== -1) {
        pipelines.value.splice(index, 1)
        total.value--
      }

      // 清空当前 Pipeline
      if (currentPipeline.value?.uuid === uuid) {
        currentPipeline.value = null
        currentPipelineStages.value = []
      }
    } catch (error) {
      appStore.showError(`删除 Pipeline 失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 清空当前 Pipeline
   */
  const clearCurrentPipeline = () => {
    currentPipeline.value = null
    currentPipelineStages.value = []
  }

  /**
   * 获取 Pipeline 状态颜色
   */
  const getStatusColor = (status) => {
    const colorMap = {
      pending: 'info',
      running: 'primary',
      completed: 'success',
      failed: 'danger',
      cancelled: 'warning'
    }
    return colorMap[status] || 'info'
  }

  /**
   * 获取 Pipeline 状态图标
   */
  const getStatusIcon = (status) => {
    const iconMap = {
      pending: 'Clock',
      running: 'Loading',
      completed: 'CircleCheck',
      failed: 'CircleClose',
      cancelled: 'WarningFilled'
    }
    return iconMap[status] || 'QuestionFilled'
  }

  /**
   * 获取 Stage 状态颜色
   */
  const getStageStatusColor = (status) => {
    const colorMap = {
      pending: 'info',
      running: 'primary',
      completed: 'success',
      failed: 'danger',
      skipped: 'warning'
    }
    return colorMap[status] || 'info'
  }

  return {
    // State
    pipelines,
    currentPipeline,
    currentPipelineStages,
    loading,
    creating,
    total,

    // Computed
    runningPipelines,
    pendingPipelines,
    completedPipelines,
    failedPipelines,

    // Actions
    fetchPipelines,
    createPipeline,
    fetchPipeline,
    fetchPipelineStatus,
    startPipeline,
    cancelPipeline,
    deletePipeline,
    clearCurrentPipeline,

    // Helpers
    getStatusColor,
    getStatusIcon,
    getStageStatusColor
  }
})
