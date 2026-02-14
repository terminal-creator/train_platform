/**
 * Pipeline API
 * Phase 4: Pipeline 管理接口
 */

import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1/pipelines',
  timeout: 10000,  // 10 seconds
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
 * 列出所有 Pipelines
 * @param {Object} params - 查询参数
 * @param {string} params.status - 状态筛选 (pending/running/completed/failed/cancelled)
 * @param {number} params.offset - 偏移量
 * @param {number} params.limit - 每页数量
 */
export const listPipelines = (params) => api.get('', { params })

/**
 * 创建 Pipeline
 * @param {Object} data - Pipeline 配置
 * @param {string} data.name - Pipeline 名称
 * @param {string} data.description - 描述
 * @param {Array} data.stages - Stage 配置数组
 * @param {number} data.priority - 优先级 (1-10)
 * @param {number} data.max_retries - 最大重试次数
 */
export const createPipeline = (data) => api.post('', data)

/**
 * 获取 Pipeline 详情
 * @param {string} uuid - Pipeline UUID
 */
export const getPipeline = (uuid) => api.get(`/${uuid}`)

/**
 * 获取 Pipeline 状态（包含所有 stages）
 * @param {string} uuid - Pipeline UUID
 */
export const getPipelineStatus = (uuid) => api.get(`/${uuid}/status`)

/**
 * 启动 Pipeline
 * @param {string} uuid - Pipeline UUID
 */
export const startPipeline = (uuid) => api.post(`/${uuid}/start`)

/**
 * 取消 Pipeline
 * @param {string} uuid - Pipeline UUID
 */
export const cancelPipeline = (uuid) => api.post(`/${uuid}/cancel`)

/**
 * 删除 Pipeline
 * @param {string} uuid - Pipeline UUID
 */
export const deletePipeline = (uuid) => api.delete(`/${uuid}`)

export default {
  listPipelines,
  createPipeline,
  getPipeline,
  getPipelineStatus,
  startPipeline,
  cancelPipeline,
  deletePipeline
}
