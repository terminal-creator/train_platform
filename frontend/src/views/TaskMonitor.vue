<template>
  <div class="task-monitor">
    <!-- Header -->
    <div class="page-header">
      <h2>任务监控</h2>
      <el-button :icon="Refresh" @click="handleRefresh">刷新</el-button>
    </div>

    <!-- Statistics Cards -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <el-card class="stat-card stat-running">
          <div class="stat-content">
            <div class="stat-icon">
              <el-icon :size="40"><Loading /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ taskStore.stats.active_tasks }}</div>
              <div class="stat-label">运行中</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card stat-scheduled">
          <div class="stat-content">
            <div class="stat-icon">
              <el-icon :size="40"><Clock /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ taskStore.stats.scheduled_tasks }}</div>
              <div class="stat-label">待处理</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card stat-workers">
          <div class="stat-content">
            <div class="stat-icon">
              <el-icon :size="40"><Monitor /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ taskStore.stats.worker_count }}</div>
              <div class="stat-label">Worker 数量</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card stat-registered">
          <div class="stat-content">
            <div class="stat-icon">
              <el-icon :size="40"><Files /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ taskStore.stats.registered_tasks }}</div>
              <div class="stat-label">注册任务数</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Workers Status -->
    <el-card v-if="taskStore.stats.workers.length > 0" class="workers-card">
      <template #header>
        <span>Worker 状态</span>
      </template>
      <div class="workers-list">
        <div
          v-for="(worker, index) in taskStore.stats.workers"
          :key="index"
          class="worker-item"
        >
          <div class="worker-info">
            <el-icon color="#67C23A"><Select /></el-icon>
            <span class="worker-name">{{ worker }}</span>
          </div>
          <el-tag type="success" size="small">在线</el-tag>
        </div>
      </div>
    </el-card>

    <!-- Task List -->
    <el-card class="tasks-card">
      <template #header>
        <div class="card-header">
          <span>任务列表</span>
          <el-space>
            <el-select
              v-model="stateFilter"
              placeholder="状态筛选"
              clearable
              style="width: 150px"
              @change="handleFilterChange"
            >
              <el-option label="全部" value="" />
              <el-option label="等待中" value="PENDING" />
              <el-option label="运行中" value="STARTED" />
              <el-option label="成功" value="SUCCESS" />
              <el-option label="失败" value="FAILURE" />
              <el-option label="已取消" value="REVOKED" />
            </el-select>
          </el-space>
        </div>
      </template>

      <el-table
        :data="taskStore.tasks"
        stripe
        style="width: 100%"
        empty-text="暂无任务数据"
      >
        <el-table-column prop="task_id" label="Task ID" width="280">
          <template #default="{ row }">
            <el-tag size="small" type="info">
              {{ row.task_id.slice(0, 8) }}...
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column prop="name" label="任务名称" min-width="200">
          <template #default="{ row }">
            {{ row.name || '-' }}
          </template>
        </el-table-column>

        <el-table-column prop="state" label="状态" width="120">
          <template #default="{ row }">
            <el-tag
              :type="taskStore.getTaskStateColor(row.state)"
              :icon="getTaskIconComponent(row.state)"
            >
              {{ taskStore.getTaskStateText(row.state) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-space>
              <el-button
                v-if="row.state === 'STARTED'"
                type="warning"
                size="small"
                @click="handleCancel(row)"
              >
                取消
              </el-button>
              <el-button
                v-if="row.state === 'FAILURE'"
                type="primary"
                size="small"
                @click="handleRetry(row)"
              >
                重试
              </el-button>
              <el-button
                type="info"
                size="small"
                @click="handleViewResult(row)"
              >
                详情
              </el-button>
            </el-space>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Task Result Dialog -->
    <el-dialog
      v-model="resultDialogVisible"
      title="任务详情"
      width="800px"
    >
      <el-descriptions v-if="currentTaskResult" :column="1" border>
        <el-descriptions-item label="Task ID">
          {{ currentTaskResult.task_id }}
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="taskStore.getTaskStateColor(currentTaskResult.state)">
            {{ taskStore.getTaskStateText(currentTaskResult.state) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item v-if="currentTaskResult.result" label="结果">
          <el-input
            :model-value="JSON.stringify(currentTaskResult.result, null, 2)"
            type="textarea"
            :rows="10"
            readonly
          />
        </el-descriptions-item>
        <el-descriptions-item v-if="currentTaskResult.error" label="错误">
          <el-alert type="error" :closable="false">
            {{ currentTaskResult.error }}
          </el-alert>
        </el-descriptions-item>
        <el-descriptions-item v-if="currentTaskResult.traceback" label="Traceback">
          <el-input
            :model-value="currentTaskResult.traceback"
            type="textarea"
            :rows="10"
            readonly
          />
        </el-descriptions-item>
      </el-descriptions>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessageBox } from 'element-plus'
import {
  Refresh,
  Loading,
  Clock,
  Monitor,
  Files,
  Select,
  CircleCheck,
  CircleClose,
  WarningFilled,
  RefreshRight
} from '@element-plus/icons-vue'
import { useTaskStore } from '@/stores/task'

const taskStore = useTaskStore()

const stateFilter = ref('')
const resultDialogVisible = ref(false)
const currentTaskResult = ref(null)

// Auto refresh interval - disabled by default to avoid timeout issues
let refreshInterval = null
const autoRefreshEnabled = ref(false)

// Methods
const fetchTasks = async () => {
  await taskStore.fetchTasks({
    state: stateFilter.value || undefined,
    limit: 50
  })
}

const fetchStats = async () => {
  await taskStore.fetchTaskStats()
}

const handleRefresh = async () => {
  await Promise.all([fetchTasks(), fetchStats()])
}

const handleFilterChange = () => {
  fetchTasks()
}

const handleCancel = async (task) => {
  try {
    await ElMessageBox.confirm(
      `确定要取消任务 "${task.task_id}" 吗？`,
      '确认取消',
      { type: 'warning' }
    )
    await taskStore.cancelTask(task.task_id)
    await fetchTasks()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to cancel task:', error)
    }
  }
}

const handleRetry = async (task) => {
  try {
    await taskStore.retryTask(task.task_id)
    await fetchTasks()
  } catch (error) {
    console.error('Failed to retry task:', error)
  }
}

const handleViewResult = async (task) => {
  try {
    const result = await taskStore.fetchTaskResult(task.task_id)
    currentTaskResult.value = result
    resultDialogVisible.value = true
  } catch (error) {
    console.error('Failed to fetch task result:', error)
  }
}

const getTaskIconComponent = (state) => {
  const iconMap = {
    PENDING: Clock,
    STARTED: Loading,
    SUCCESS: CircleCheck,
    FAILURE: CircleClose,
    RETRY: RefreshRight,
    REVOKED: WarningFilled
  }
  return iconMap[state] || Clock
}

const startAutoRefresh = () => {
  if (!autoRefreshEnabled.value) return
  stopAutoRefresh()
  refreshInterval = setInterval(() => {
    handleRefresh()
  }, 30000) // Refresh every 30 seconds (less aggressive)
}

const stopAutoRefresh = () => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

onMounted(async () => {
  // Load once on mount, no auto-refresh to avoid timeout issues
  await handleRefresh()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>

<style scoped>
.task-monitor {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-header h2 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}

.stats-row {
  margin-bottom: 20px;
}

.stat-card {
  border: none;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.stat-card.stat-running {
  border-left: 4px solid #409EFF;
}

.stat-card.stat-scheduled {
  border-left: 4px solid #909399;
}

.stat-card.stat-workers {
  border-left: 4px solid #67C23A;
}

.stat-card.stat-registered {
  border-left: 4px solid #E6A23C;
}

.stat-content {
  display: flex;
  align-items: center;
  gap: 20px;
}

.stat-icon {
  flex-shrink: 0;
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 32px;
  font-weight: 600;
  color: #303133;
}

.stat-label {
  font-size: 14px;
  color: #606266;
  margin-top: 5px;
}

.workers-card,
.tasks-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.workers-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.worker-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background: #f5f7fa;
  border-radius: 4px;
}

.worker-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.worker-name {
  font-weight: 500;
}
</style>
