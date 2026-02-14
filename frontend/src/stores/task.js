/**
 * Task Store
 * Phase 4: Celery 任务状态管理
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as taskAPI from '@/api/task'
import { useAppStore } from './app'

export const useTaskStore = defineStore('task', () => {
  const tasks = ref([])
  const currentTask = ref(null)
  const loading = ref(false)
  const stats = ref({
    workers: [],
    worker_count: 0,
    active_tasks: 0,
    scheduled_tasks: 0,
    reserved_tasks: 0,
    registered_tasks: 0
  })

  const appStore = useAppStore()

  // Computed
  const activeTasks = computed(() =>
    tasks.value.filter(t => t.state === 'STARTED' || t.state === 'PENDING')
  )

  const successTasks = computed(() =>
    tasks.value.filter(t => t.state === 'SUCCESS')
  )

  const failedTasks = computed(() =>
    tasks.value.filter(t => t.state === 'FAILURE')
  )

  /**
   * 获取任务列表
   */
  const fetchTasks = async (params = {}) => {
    // Don't show loading spinner - just fetch in background
    loading.value = false
    try {
      const response = await taskAPI.listTasks(params)
      tasks.value = response.tasks || []
      return response
    } catch (error) {
      // Silently fail - just show empty list
      tasks.value = []
      return { tasks: [], total: 0 }
    }
  }

  /**
   * 获取任务状态
   */
  const fetchTaskStatus = async (taskId) => {
    loading.value = true
    try {
      const task = await taskAPI.getTaskStatus(taskId)
      currentTask.value = task
      return task
    } catch (error) {
      appStore.showError(`获取任务状态失败: ${error.message}`)
      throw error
    } finally {
      loading.value = false
    }
  }

  /**
   * 获取任务结果
   */
  const fetchTaskResult = async (taskId) => {
    try {
      const result = await taskAPI.getTaskResult(taskId)
      return result
    } catch (error) {
      appStore.showError(`获取任务结果失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 取消任务
   */
  const cancelTask = async (taskId) => {
    try {
      const response = await taskAPI.cancelTask(taskId)
      appStore.showSuccess('任务已取消')

      // 更新列表中的状态
      const index = tasks.value.findIndex(t => t.task_id === taskId)
      if (index !== -1) {
        tasks.value[index].state = 'REVOKED'
      }

      // 更新当前任务
      if (currentTask.value?.task_id === taskId) {
        currentTask.value.state = 'REVOKED'
      }

      return response
    } catch (error) {
      appStore.showError(`取消任务失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 重试失败的任务
   */
  const retryTask = async (taskId) => {
    try {
      const response = await taskAPI.retryTask(taskId)
      appStore.showSuccess('任务已重试')
      return response
    } catch (error) {
      appStore.showError(`重试任务失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 获取任务统计信息
   */
  const fetchTaskStats = async () => {
    try {
      const response = await taskAPI.getTaskStats()
      stats.value = response
      return response
    } catch (error) {
      // Silently fail - just keep default stats
      console.warn('Failed to fetch task stats:', error.message)
      return stats.value
    }
  }

  /**
   * 清除所有待处理的任务
   */
  const purgeTasks = async () => {
    try {
      const response = await taskAPI.purgeTasks()
      appStore.showSuccess(`已清除 ${response.tasks_purged} 个待处理任务`)
      await fetchTasks()
      return response
    } catch (error) {
      appStore.showError(`清除任务失败: ${error.message}`)
      throw error
    }
  }

  /**
   * 清空当前任务
   */
  const clearCurrentTask = () => {
    currentTask.value = null
  }

  /**
   * 获取任务状态颜色
   */
  const getTaskStateColor = (state) => {
    const colorMap = {
      PENDING: 'info',
      STARTED: 'primary',
      SUCCESS: 'success',
      FAILURE: 'danger',
      RETRY: 'warning',
      REVOKED: 'warning'
    }
    return colorMap[state] || 'info'
  }

  /**
   * 获取任务状态图标
   */
  const getTaskStateIcon = (state) => {
    const iconMap = {
      PENDING: 'Clock',
      STARTED: 'Loading',
      SUCCESS: 'CircleCheck',
      FAILURE: 'CircleClose',
      RETRY: 'RefreshRight',
      REVOKED: 'WarningFilled'
    }
    return iconMap[state] || 'QuestionFilled'
  }

  /**
   * 获取任务状态文本
   */
  const getTaskStateText = (state) => {
    const textMap = {
      PENDING: '等待中',
      STARTED: '运行中',
      SUCCESS: '成功',
      FAILURE: '失败',
      RETRY: '重试中',
      REVOKED: '已取消'
    }
    return textMap[state] || state
  }

  return {
    // State
    tasks,
    currentTask,
    loading,
    stats,

    // Computed
    activeTasks,
    successTasks,
    failedTasks,

    // Actions
    fetchTasks,
    fetchTaskStatus,
    fetchTaskResult,
    cancelTask,
    retryTask,
    fetchTaskStats,
    purgeTasks,
    clearCurrentTask,

    // Helpers
    getTaskStateColor,
    getTaskStateIcon,
    getTaskStateText
  }
})
