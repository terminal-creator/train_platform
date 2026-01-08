/**
 * Celery Task API
 * Phase 4: Celery 任务管理接口
 */

import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1/celery-tasks',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Response interceptor
api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || 'Network Error'
    return Promise.reject(new Error(message))
  }
)

/**
 * 列出所有 Celery 任务
 * @param {Object} params - 查询参数
 * @param {string} params.state - 状态筛选 (PENDING/STARTED/SUCCESS/FAILURE/RETRY/REVOKED)
 * @param {number} params.limit - 每页数量
 */
export const listTasks = (params) => api.get('', { params })

/**
 * 获取任务状态
 * @param {string} taskId - Celery 任务 ID
 */
export const getTaskStatus = (taskId) => api.get(`/${taskId}`)

/**
 * 获取任务结果
 * @param {string} taskId - Celery 任务 ID
 */
export const getTaskResult = (taskId) => api.get(`/${taskId}/result`)

/**
 * 取消任务
 * @param {string} taskId - Celery 任务 ID
 */
export const cancelTask = (taskId) => api.post(`/${taskId}/cancel`)

/**
 * 重试失败的任务
 * @param {string} taskId - Celery 任务 ID
 */
export const retryTask = (taskId) => api.post(`/${taskId}/retry`)

/**
 * 获取任务统计信息
 */
export const getTaskStats = () => api.get('/stats/overview')

/**
 * 清除所有待处理的任务
 */
export const purgeTasks = () => api.post('/purge')

export default {
  listTasks,
  getTaskStatus,
  getTaskResult,
  cancelTask,
  retryTask,
  getTaskStats,
  purgeTasks
}
