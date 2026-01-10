/**
 * Enhanced WebSocket utility with auto-reconnection and heartbeat
 */

export class ReconnectingWebSocket {
  constructor(getUrl, options = {}) {
    this.getUrl = getUrl
    this.options = {
      maxReconnectAttempts: options.maxReconnectAttempts ?? 10,
      reconnectInterval: options.reconnectInterval ?? 1000,
      maxReconnectInterval: options.maxReconnectInterval ?? 30000,
      heartbeatInterval: options.heartbeatInterval ?? 25000, // Send ping every 25s
      debug: options.debug ?? false,
      ...options
    }

    this.ws = null
    this.reconnectAttempts = 0
    this.reconnectTimer = null
    this.heartbeatTimer = null
    this.messageQueue = []
    this.isIntentionallyClosed = false
    this.listeners = {
      open: [],
      message: [],
      error: [],
      close: []
    }
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.log('Already connected')
      return
    }

    this.isIntentionallyClosed = false
    const url = typeof this.getUrl === 'function' ? this.getUrl() : this.getUrl

    try {
      this.log('Connecting to:', url)
      this.ws = new WebSocket(url)

      this.ws.onopen = (event) => {
        this.log('Connected')
        this.reconnectAttempts = 0
        this.startHeartbeat()
        this.flushMessageQueue()
        this.emit('open', event)
      }

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          // Handle heartbeat response
          if (data.type === 'pong' || data.type === 'heartbeat') {
            this.log('Received heartbeat response')
            return
          }

          this.emit('message', data)
        } catch (error) {
          this.log('Failed to parse message:', error)
          this.emit('error', error)
        }
      }

      this.ws.onerror = (error) => {
        this.log('WebSocket error:', error)
        this.emit('error', error)
      }

      this.ws.onclose = (event) => {
        this.log('Connection closed:', event.code, event.reason)
        this.stopHeartbeat()
        this.emit('close', event)

        // Auto-reconnect if not intentionally closed
        if (!this.isIntentionallyClosed) {
          this.scheduleReconnect()
        }
      }
    } catch (error) {
      this.log('Failed to create WebSocket:', error)
      this.emit('error', error)
      this.scheduleReconnect()
    }
  }

  send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = typeof data === 'string' ? data : JSON.stringify(data)
      this.ws.send(message)
    } else {
      // Queue message for sending when connected
      this.messageQueue.push(data)
    }
  }

  close() {
    this.isIntentionallyClosed = true
    this.stopHeartbeat()
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  // Event listeners
  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback)
    }
  }

  off(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback)
    }
  }

  emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          this.log('Error in event listener:', error)
        }
      })
    }
  }

  // Internal methods
  startHeartbeat() {
    this.stopHeartbeat()
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.log('Sending ping')
        this.send({ type: 'ping' })
      }
    }, this.options.heartbeatInterval)
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      this.log('Max reconnect attempts reached')
      this.emit('error', new Error('Max reconnect attempts reached'))
      return
    }

    const interval = Math.min(
      this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts),
      this.options.maxReconnectInterval
    )

    this.log(`Reconnecting in ${interval}ms (attempt ${this.reconnectAttempts + 1})`)

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++
      this.connect()
    }, interval)
  }

  flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()
      this.send(message)
    }
  }

  log(...args) {
    if (this.options.debug) {
      console.log('[WebSocket]', ...args)
    }
  }

  getState() {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }

  isConnected() {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

/**
 * Create a WebSocket connection for job metrics
 */
export function createJobMetricsWS(jobId, options = {}) {
  const getUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    return `${protocol}//${host}/api/v1/ws/jobs/${jobId}`
  }

  return new ReconnectingWebSocket(getUrl, {
    debug: true,
    ...options
  })
}

/**
 * Create a WebSocket connection for job logs
 */
export function createJobLogsWS(jobId, options = {}) {
  const getUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    return `${protocol}//${host}/api/v1/ws/jobs/${jobId}/logs`
  }

  return new ReconnectingWebSocket(getUrl, {
    debug: true,
    ...options
  })
}

/**
 * Create a WebSocket connection for dashboard
 */
export function createDashboardWS(options = {}) {
  const getUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    return `${protocol}//${host}/api/v1/ws/dashboard`
  }

  return new ReconnectingWebSocket(getUrl, {
    debug: true,
    ...options
  })
}
