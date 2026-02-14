/**
 * Demo模式入口
 */
export { useDemoStore } from './demoStore'
export * from './mockData'

// Demo模式API封装
export const demoApi = {
  // 获取Demo状态
  async getStatus() {
    const response = await fetch('/api/v1/demo/status')
    return response.json()
  },

  // 启用Demo模式
  async enable(config = {}) {
    const response = await fetch('/api/v1/demo/enable', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    return response.json()
  },

  // 禁用Demo模式
  async disable() {
    const response = await fetch('/api/v1/demo/disable', {
      method: 'POST',
    })
    return response.json()
  },

  // 获取Demo任务列表
  async getJobs(params = {}) {
    const query = new URLSearchParams(params).toString()
    const response = await fetch(`/api/v1/demo/jobs?${query}`)
    return response.json()
  },

  // 获取Demo任务详情
  async getJob(jobId) {
    const response = await fetch(`/api/v1/demo/jobs/${jobId}`)
    return response.json()
  },

  // 获取Demo任务指标
  async getJobMetrics(jobId, params = {}) {
    const query = new URLSearchParams(params).toString()
    const response = await fetch(`/api/v1/demo/jobs/${jobId}/metrics?${query}`)
    return response.json()
  },

  // 获取Demo梯度热力图
  async getGradientHeatmap(jobId) {
    const response = await fetch(`/api/v1/demo/jobs/${jobId}/gradient-heatmap`)
    return response.json()
  },

  // 获取Demo仪表板
  async getDashboard() {
    const response = await fetch('/api/v1/demo/dashboard')
    return response.json()
  },

  // 获取Demo数据集
  async getDatasets() {
    const response = await fetch('/api/v1/demo/datasets')
    return response.json()
  },

  // 获取Demo流水线
  async getPipelines() {
    const response = await fetch('/api/v1/demo/pipelines')
    return response.json()
  },

  // 获取评估对比
  async getEvaluationComparison() {
    const response = await fetch('/api/v1/demo/evaluation-comparison')
    return response.json()
  },
}
