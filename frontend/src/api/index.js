import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,  // 10 seconds
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
api.interceptors.request.use(
  config => config,
  error => Promise.reject(error)
)

// Response interceptor
api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || 'Network Error'
    return Promise.reject(new Error(message))
  }
)

// Dashboard
export const getDashboard = () => api.get('/monitoring/dashboard')

// Compute Calculator
export const calculateConfig = (data) => api.post('/compute/calculate', data)
export const estimateMemory = (data) => api.post('/compute/memory', data)
export const getGpuTypes = () => api.get('/compute/gpu-types')
export const getModelSizes = () => api.get('/compute/model-sizes')

// Jobs
export const getJobs = (params) => api.get('/jobs', { params })
export const getJob = (id) => api.get(`/jobs/${id}`)
export const createJob = (data) => api.post('/jobs', data)
export const getAvailableModels = () => api.get('/jobs/available-models')
export const getAvailableDatasets = () => api.get('/jobs/available-datasets')
export const getAvailableRewardScripts = () => api.get('/jobs/available-reward-scripts')
export const getDatasetPreview = (path) => api.post('/jobs/dataset-preview', { path })
export const updateJob = (id, data) => api.patch(`/jobs/${id}`, data)
export const deleteJob = (id) => api.delete(`/jobs/${id}`)
export const startJob = (id) => api.post(`/jobs/${id}/start`)
export const stopJob = (id) => api.post(`/jobs/${id}/stop`)
export const pauseJob = (id) => api.post(`/jobs/${id}/pause`)
export const resumeJob = (id) => api.post(`/jobs/${id}/resume`)
export const getJobLogs = (id, params) => api.get(`/jobs/${id}/logs`, { params })
export const getJobMetrics = (id, params) => api.get(`/jobs/${id}/metrics`, { params })
export const getJobConfig = (id) => api.get(`/jobs/${id}/config`)

// Model Surgery
export const mergeModels = (data) => api.post('/surgery/merge', data)
export const selectCheckpoint = (data) => api.post('/surgery/checkpoint/select', data)
export const selectBestCheckpoint = (data) => api.post('/surgery/checkpoint/select', data)
export const performSWA = (data) => api.post('/surgery/swa', data)
export const createSwaMerge = (data) => api.post('/surgery/swa', data)
export const getMergeMethods = () => api.get('/surgery/methods')
export const getCheckpoints = (jobId) => api.get(`/jobs/${jobId}/checkpoints`)

// Monitoring
export const getGradientHeatmap = (jobId, params) => api.get(`/monitoring/${jobId}/gradient-heatmap`, { params })
export const getGradientStats = (jobId, step) => api.get(`/monitoring/${jobId}/gradient-stats`, { params: step ? { step } : {} })
export const getEvaluations = (jobId, params) => api.get(`/monitoring/${jobId}/evaluations`, { params })
export const getResourceUsage = (jobId) => api.get(`/monitoring/${jobId}/resources`)
export const getGpuUsage = (jobId) => api.get(`/monitoring/${jobId}/gpu-usage`)
export const getAlerts = (jobId, params) => api.get(`/monitoring/${jobId}/alerts`, { params })
export const setAlertRules = (jobId, rules) => api.post(`/monitoring/${jobId}/alert-rules`, rules)
export const acknowledgeAlert = (jobId, alertId) => api.post(`/monitoring/${jobId}/alerts/${alertId}/acknowledge`)

// Phase 1 - Diagnostics and Anomaly Detection
export const getJobAnomalies = (jobId) => api.get(`/monitoring/${jobId}/anomalies/detected`)
export const getJobHealth = (jobId) => api.get(`/monitoring/${jobId}/health`)
export const diagnoseJob = (jobId) => api.post(`/monitoring/${jobId}/diagnose`)
export const syncJobMetrics = (jobId) => api.post(`/monitoring/${jobId}/metrics/sync`)

// Evaluation
export const getEvalDatasets = (params) => api.get('/evaluation/datasets', { params })
export const uploadEvalDataset = (formData, params) => {
  const queryParams = new URLSearchParams(params).toString()
  return api.post(`/evaluation/datasets/upload?${queryParams}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000  // 2 minutes for file upload
  })
}
export const getEvalDataset = (uuid) => api.get(`/evaluation/datasets/${uuid}`)
export const previewEvalDataset = (uuid, limit = 10) =>
  api.get(`/evaluation/datasets/${uuid}/preview`, { params: { limit } })
export const deleteEvalDataset = (uuid) => api.delete(`/evaluation/datasets/${uuid}`)
export const triggerEvaluation = (data) => api.post('/evaluation/trigger', data)
export const getEvalTask = (uuid) => api.get(`/evaluation/tasks/${uuid}`)
export const getEvalTaskDetails = (uuid) => api.get(`/evaluation/tasks/${uuid}/details`)
export const getJobEvalTasks = (jobUuid, params) =>
  api.get(`/evaluation/jobs/${jobUuid}/tasks`, { params })
export const getJobEvalResults = (jobUuid) => api.get(`/evaluation/jobs/${jobUuid}/results`)
export const getEvalTasks = (params) => api.get('/evaluation/tasks', { params })

// Evaluation Comparisons
export const getComparisons = (params) => api.get('/evaluation/comparisons', { params })
export const createComparison = (data) => api.post('/evaluation/comparisons', data)
export const getComparison = (uuid) => api.get(`/evaluation/comparisons/${uuid}`)
export const getComparisonDiffs = (uuid, params) =>
  api.get(`/evaluation/comparisons/${uuid}/diffs`, { params })
export const deleteComparison = (uuid) => api.delete(`/evaluation/comparisons/${uuid}`)

// Training Datasets
export const getTrainingDatasets = (params) => api.get('/training-datasets', { params })
export const uploadTrainingDataset = (formData, params) => {
  const queryParams = new URLSearchParams(params).toString()
  return api.post(`/training-datasets/upload?${queryParams}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000  // 2 minutes for file upload
  })
}
export const getTrainingDataset = (uuid) => api.get(`/training-datasets/${uuid}`)
export const deleteTrainingDataset = (uuid) => api.delete(`/training-datasets/${uuid}`)
export const getTrainingDatasetDistribution = (uuid, fields) =>
  api.get(`/training-datasets/${uuid}/distribution`, { params: { fields } })
export const getTrainingDatasetSample = (uuid, index) =>
  api.get(`/training-datasets/${uuid}/sample/${index}`)
export const getTrainingDatasetSamples = (uuid, params) =>
  api.get(`/training-datasets/${uuid}/samples`, { params })
export const configureTrainingDatasetLabels = (uuid, labelFields) =>
  api.post(`/training-datasets/${uuid}/configure-labels`, { label_fields: labelFields })
export const configureTrainingDatasetLoss = (uuid, promptField, responseField) =>
  api.post(`/training-datasets/${uuid}/configure-loss`, { prompt_field: promptField, response_field: responseField })
export const reanalyzeTrainingDataset = (uuid, autoDetect = false) =>
  api.post(`/training-datasets/${uuid}/reanalyze`, null, { params: { auto_detect_labels: autoDetect } })
export const syncTrainingDataset = (uuid, force = false) =>
  api.post(`/training-datasets/${uuid}/sync`, null, { params: { force } })
export const syncAllTrainingDatasets = (force = false) =>
  api.post('/training-datasets/sync-all', null, { params: { force } })
export const getTrainingDatasetStats = (uuid) => api.get(`/training-datasets/${uuid}/stats`)
export const getTrainingDatasetQuality = (uuid) => api.get(`/training-datasets/${uuid}/quality-check`)

// Run Mode Configuration
export const getRunModeConfig = () => api.get('/run-mode/config')
export const setRunModeConfig = (data) => api.post('/run-mode/config', data)
export const testConnection = () => api.post('/run-mode/test-connection')
export const testSSHConnection = (config) => api.post('/run-mode/test-ssh', config)
export const getGpuInfo = () => api.get('/run-mode/gpu-info')
export const getSSHGpuInfo = (config) => api.post('/run-mode/ssh/gpu-info', config)
export const getRunnerStatus = () => api.get('/run-mode/status')
export const getRemoteModels = () => api.get('/run-mode/remote/models')
export const getRemoteDatasets = () => api.get('/run-mode/remote/datasets')

// WebSocket for live metrics (using enhanced WebSocket utility)
import {
  createJobMetricsWS,
  createJobLogsWS,
  createDashboardWS
} from '../utils/websocket'

export const createMetricsWebSocket = (jobId, options = {}) => {
  return createJobMetricsWS(jobId, options)
}

export const createLogsWebSocket = (jobId, options = {}) => {
  return createJobLogsWS(jobId, options)
}

export const createDashboardWebSocket = (options = {}) => {
  return createDashboardWS(options)
}

// Legacy function for backwards compatibility
export const connectMetricsWs = (jobId, onMessage) => {
  const ws = createMetricsWebSocket(jobId, { debug: false })
  ws.on('message', onMessage)
  ws.connect()
  return ws
}

export default api
