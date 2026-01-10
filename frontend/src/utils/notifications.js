/**
 * Unified notification and error handling utility
 *
 * Provides consistent user feedback for:
 * - Success messages
 * - Error handling
 * - Warnings
 * - Info messages
 * - Loading states
 */

import { ElNotification, ElMessage, ElMessageBox, ElLoading } from 'element-plus'

/**
 * Display success notification
 */
export function notifySuccess(message, title = '成功', duration = 3000) {
  ElNotification({
    title,
    message,
    type: 'success',
    duration,
  })
}

/**
 * Display error notification
 */
export function notifyError(message, title = '错误', duration = 5000) {
  ElNotification({
    title,
    message,
    type: 'error',
    duration,
  })
}

/**
 * Display warning notification
 */
export function notifyWarning(message, title = '警告', duration = 4000) {
  ElNotification({
    title,
    message,
    type: 'warning',
    duration,
  })
}

/**
 * Display info notification
 */
export function notifyInfo(message, title = '提示', duration = 3000) {
  ElNotification({
    title,
    message,
    type: 'info',
    duration,
  })
}

/**
 * Display simple success message
 */
export function messageSuccess(message, duration = 2000) {
  ElMessage({
    message,
    type: 'success',
    duration,
  })
}

/**
 * Display simple error message
 */
export function messageError(message, duration = 3000) {
  ElMessage({
    message,
    type: 'error',
    duration,
  })
}

/**
 * Display simple warning message
 */
export function messageWarning(message, duration = 2500) {
  ElMessage({
    message,
    type: 'warning',
    duration,
  })
}

/**
 * Display simple info message
 */
export function messageInfo(message, duration = 2000) {
  ElMessage({
    message,
    type: 'info',
    duration,
  })
}

/**
 * Show confirmation dialog
 */
export async function confirm(message, title = '确认操作', options = {}) {
  try {
    await ElMessageBox.confirm(message, title, {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
      ...options,
    })
    return true
  } catch {
    return false
  }
}

/**
 * Show prompt dialog
 */
export async function prompt(message, title = '请输入', options = {}) {
  try {
    const { value } = await ElMessageBox.prompt(message, title, {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      ...options,
    })
    return value
  } catch {
    return null
  }
}

/**
 * Handle API errors with user-friendly messages
 */
export function handleApiError(error, defaultMessage = '操作失败，请稍后重试') {
  console.error('API Error:', error)

  let message = defaultMessage
  let details = null

  // Extract error message from different error formats
  if (error.response?.data) {
    const data = error.response.data

    // New error format with error object
    if (data.error) {
      message = data.error.message || defaultMessage
      details = data.error.details

      // Format validation errors
      if (details?.errors && Array.isArray(details.errors)) {
        const fieldErrors = details.errors
          .map(e => `${e.field}: ${e.message}`)
          .join('\n')
        message = `${message}\n\n${fieldErrors}`
      }
    }
    // Legacy format with detail field
    else if (data.detail) {
      message = typeof data.detail === 'string' ? data.detail : defaultMessage
    }
  } else if (error.message) {
    message = error.message
  }

  // Network errors
  if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
    message = '网络连接失败，请检查您的网络连接'
  }

  // Timeout errors
  if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
    message = '请求超时，请稍后重试'
  }

  notifyError(message)
  return message
}

/**
 * Execute async operation with loading and error handling
 */
export async function withLoading(
  asyncFn,
  options = {}
) {
  const {
    loadingText = '加载中...',
    successMessage = null,
    errorMessage = '操作失败',
    showSuccess = true,
  } = options

  let loadingInstance = null

  try {
    // Show loading
    if (loadingText) {
      loadingInstance = ElLoading.service({
        lock: true,
        text: loadingText,
        background: 'rgba(0, 0, 0, 0.7)',
      })
    }

    // Execute async function
    const result = await asyncFn()

    // Show success message
    if (showSuccess && successMessage) {
      messageSuccess(successMessage)
    }

    return { success: true, data: result, error: null }
  } catch (error) {
    // Handle error
    const errorMsg = handleApiError(error, errorMessage)
    return { success: false, data: null, error: errorMsg }
  } finally {
    // Hide loading
    if (loadingInstance) {
      loadingInstance.close()
    }
  }
}

/**
 * Format error for display
 */
export function formatError(error) {
  if (typeof error === 'string') {
    return error
  }

  if (error.response?.data) {
    const data = error.response.data
    if (data.error?.message) {
      return data.error.message
    }
    if (data.detail) {
      return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    }
  }

  if (error.message) {
    return error.message
  }

  return '未知错误'
}

/**
 * Create a notification manager for component lifecycle
 */
export function createNotificationManager() {
  const notifications = []

  return {
    success(message, title, duration) {
      const notification = notifySuccess(message, title, duration)
      notifications.push(notification)
      return notification
    },

    error(message, title, duration) {
      const notification = notifyError(message, title, duration)
      notifications.push(notification)
      return notification
    },

    warning(message, title, duration) {
      const notification = notifyWarning(message, title, duration)
      notifications.push(notification)
      return notification
    },

    info(message, title, duration) {
      const notification = notifyInfo(message, title, duration)
      notifications.push(notification)
      return notification
    },

    clear() {
      notifications.forEach(n => {
        if (n && typeof n.close === 'function') {
          n.close()
        }
      })
      notifications.length = 0
    },
  }
}

export default {
  success: notifySuccess,
  error: notifyError,
  warning: notifyWarning,
  info: notifyInfo,
  messageSuccess,
  messageError,
  messageWarning,
  messageInfo,
  confirm,
  prompt,
  handleApiError,
  withLoading,
  formatError,
  createNotificationManager,
}
