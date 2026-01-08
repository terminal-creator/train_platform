<template>
  <div class="pipeline-detail" v-loading="pipelineStore.loading">
    <div v-if="currentPipeline">
      <!-- Header -->
      <div class="page-header">
        <el-button :icon="ArrowLeft" @click="goBack">返回</el-button>
        <h2>{{ currentPipeline.name }}</h2>
        <div class="header-actions">
          <el-button
            v-if="currentPipeline.status === 'pending'"
            type="primary"
            :icon="VideoPlay"
            @click="handleStart"
          >
            启动
          </el-button>
          <el-button
            v-if="currentPipeline.status === 'running'"
            type="warning"
            :icon="VideoPause"
            @click="handleCancel"
          >
            取消
          </el-button>
          <el-button :icon="Refresh" @click="handleRefresh">刷新</el-button>
        </div>
      </div>

      <!-- Status Card -->
      <el-card class="status-card">
        <el-descriptions :column="4" border>
          <el-descriptions-item label="状态">
            <el-tag
              :type="pipelineStore.getStatusColor(currentPipeline.status)"
              :icon="getStatusIconComponent(currentPipeline.status)"
              size="large"
            >
              {{ getStatusText(currentPipeline.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="优先级">
            <el-rate
              :model-value="currentPipeline.priority"
              disabled
              :max="10"
              show-score
            />
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ formatDate(currentPipeline.created_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="开始时间">
            {{ currentPipeline.started_at ? formatDate(currentPipeline.started_at) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="完成时间" :span="2">
            {{ currentPipeline.completed_at ? formatDate(currentPipeline.completed_at) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="重试次数">
            {{ currentPipeline.retry_count }} / {{ currentPipeline.max_retries }}
          </el-descriptions-item>
          <el-descriptions-item label="描述" :span="4">
            {{ currentPipeline.description || '-' }}
          </el-descriptions-item>
          <el-descriptions-item v-if="currentPipeline.error_message" label="错误信息" :span="4">
            <el-alert type="error" :closable="false">
              {{ currentPipeline.error_message }}
            </el-alert>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>

      <!-- Progress -->
      <el-card v-if="currentPipelineStages.length > 0" class="progress-card">
        <template #header>
          <span>执行进度</span>
        </template>
        <el-steps :active="activeStepIndex" align-center>
          <el-step
            v-for="(stage, index) in currentPipelineStages"
            :key="index"
            :title="stage.stage_name"
            :icon="getStageIconComponent(stage.status)"
            :status="getStageStepStatus(stage.status)"
          >
            <template #description>
              <div>{{ getStageStatusText(stage.status) }}</div>
              <div v-if="stage.started_at" style="font-size: 12px; color: #909399">
                {{ formatDuration(stage.started_at, stage.completed_at) }}
              </div>
            </template>
          </el-step>
        </el-steps>
      </el-card>

      <!-- DAG Visualization -->
      <el-card class="dag-card">
        <template #header>
          <span>Pipeline 流程图</span>
        </template>
        <pipeline-dag :stages="currentPipelineStages" />
      </el-card>

      <!-- Stages List -->
      <el-card class="stages-card">
        <template #header>
          <span>阶段列表</span>
        </template>
        <el-timeline>
          <el-timeline-item
            v-for="(stage, index) in currentPipelineStages"
            :key="index"
            :icon="getStageIconComponent(stage.status)"
            :type="pipelineStore.getStageStatusColor(stage.status)"
            :hollow="stage.status === 'pending'"
            :timestamp="stage.started_at ? formatDate(stage.started_at) : '等待中'"
            placement="top"
          >
            <el-card :body-style="{ padding: '15px' }">
              <div class="stage-title">
                <h4>{{ stage.stage_name }}</h4>
                <el-tag
                  :type="pipelineStore.getStageStatusColor(stage.status)"
                  size="small"
                >
                  {{ getStageStatusText(stage.status) }}
                </el-tag>
              </div>

              <el-descriptions :column="2" size="small" border style="margin-top: 10px">
                <el-descriptions-item label="任务类型">
                  {{ stage.task_name }}
                </el-descriptions-item>
                <el-descriptions-item label="执行顺序">
                  第 {{ stage.stage_order + 1 }} 阶段
                </el-descriptions-item>
                <el-descriptions-item label="依赖阶段">
                  {{ stage.depends_on.length > 0 ? stage.depends_on.join(', ') : '无' }}
                </el-descriptions-item>
                <el-descriptions-item label="Celery Task ID">
                  <el-tag v-if="stage.celery_task_id" size="small" type="info">
                    {{ stage.celery_task_id.slice(0, 8) }}...
                  </el-tag>
                  <span v-else>-</span>
                </el-descriptions-item>
                <el-descriptions-item v-if="stage.completed_at" label="耗时" :span="2">
                  {{ formatDuration(stage.started_at, stage.completed_at) }}
                </el-descriptions-item>
                <el-descriptions-item v-if="stage.error_message" label="错误信息" :span="2">
                  <el-alert type="error" :closable="false" size="small">
                    {{ stage.error_message }}
                  </el-alert>
                </el-descriptions-item>
                <el-descriptions-item v-if="Object.keys(stage.result).length > 0" label="执行结果" :span="2">
                  <el-input
                    :model-value="JSON.stringify(stage.result, null, 2)"
                    type="textarea"
                    :rows="4"
                    readonly
                  />
                </el-descriptions-item>
              </el-descriptions>
            </el-card>
          </el-timeline-item>
        </el-timeline>
      </el-card>
    </div>

    <el-empty v-else description="Pipeline 不存在" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import {
  ArrowLeft,
  VideoPlay,
  VideoPause,
  Refresh,
  Clock,
  Loading,
  CircleCheck,
  CircleClose,
  WarningFilled
} from '@element-plus/icons-vue'
import { usePipelineStore } from '@/stores/pipeline'
import PipelineDag from '@/components/PipelineDag.vue'
import dayjs from 'dayjs'
import duration from 'dayjs/plugin/duration'

dayjs.extend(duration)

const route = useRoute()
const router = useRouter()
const pipelineStore = usePipelineStore()

const pipelineUuid = route.params.uuid
const currentPipeline = computed(() => pipelineStore.currentPipeline)
const currentPipelineStages = computed(() => pipelineStore.currentPipelineStages)

// Auto refresh interval
let refreshInterval = null

// Active step index for progress
const activeStepIndex = computed(() => {
  if (!currentPipelineStages.value) return 0

  const runningIndex = currentPipelineStages.value.findIndex(s => s.status === 'running')
  if (runningIndex !== -1) return runningIndex

  const completedCount = currentPipelineStages.value.filter(s => s.status === 'completed').length
  return completedCount
})

// Methods
const fetchPipelineStatus = async () => {
  try {
    await pipelineStore.fetchPipelineStatus(pipelineUuid)
  } catch (error) {
    console.error('Failed to fetch pipeline status:', error)
  }
}

const handleStart = async () => {
  try {
    await pipelineStore.startPipeline(pipelineUuid)
    await fetchPipelineStatus()
    startAutoRefresh()
  } catch (error) {
    console.error('Failed to start pipeline:', error)
  }
}

const handleCancel = async () => {
  try {
    await ElMessageBox.confirm(
      `确定要取消 Pipeline "${currentPipeline.value.name}" 吗？`,
      '确认取消',
      { type: 'warning' }
    )
    await pipelineStore.cancelPipeline(pipelineUuid)
    await fetchPipelineStatus()
    stopAutoRefresh()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to cancel pipeline:', error)
    }
  }
}

const handleRefresh = () => {
  fetchPipelineStatus()
}

const goBack = () => {
  router.push('/pipelines')
}

const startAutoRefresh = () => {
  if (refreshInterval) return

  refreshInterval = setInterval(() => {
    if (currentPipeline.value?.status === 'running') {
      fetchPipelineStatus()
    } else {
      stopAutoRefresh()
    }
  }, 5000) // Refresh every 5 seconds
}

const stopAutoRefresh = () => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

const getStatusText = (status) => {
  const textMap = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消'
  }
  return textMap[status] || status
}

const getStageStatusText = (status) => {
  const textMap = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    skipped: '已跳过'
  }
  return textMap[status] || status
}

const getStatusIconComponent = (status) => {
  const iconMap = {
    pending: Clock,
    running: Loading,
    completed: CircleCheck,
    failed: CircleClose,
    cancelled: WarningFilled
  }
  return iconMap[status] || Clock
}

const getStageIconComponent = (status) => {
  const iconMap = {
    pending: Clock,
    running: Loading,
    completed: CircleCheck,
    failed: CircleClose,
    skipped: WarningFilled
  }
  return iconMap[status] || Clock
}

const getStageStepStatus = (status) => {
  const statusMap = {
    pending: 'wait',
    running: 'process',
    completed: 'success',
    failed: 'error',
    skipped: 'wait'
  }
  return statusMap[status] || 'wait'
}

const formatDate = (dateString) => {
  return dayjs(dateString).format('YYYY-MM-DD HH:mm:ss')
}

const formatDuration = (startTime, endTime) => {
  if (!startTime) return '-'

  const start = dayjs(startTime)
  const end = endTime ? dayjs(endTime) : dayjs()
  const diff = end.diff(start)

  const d = dayjs.duration(diff)
  const hours = Math.floor(d.asHours())
  const minutes = d.minutes()
  const seconds = d.seconds()

  if (hours > 0) {
    return `${hours}小时 ${minutes}分钟`
  } else if (minutes > 0) {
    return `${minutes}分钟 ${seconds}秒`
  } else {
    return `${seconds}秒`
  }
}

onMounted(async () => {
  await fetchPipelineStatus()

  // Start auto refresh if running
  if (currentPipeline.value?.status === 'running') {
    startAutoRefresh()
  }
})

onUnmounted(() => {
  stopAutoRefresh()
  pipelineStore.clearCurrentPipeline()
})
</script>

<style scoped>
.pipeline-detail {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.page-header h2 {
  flex: 1;
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.status-card,
.progress-card,
.dag-card,
.stages-card {
  margin-bottom: 20px;
}

.stage-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stage-title h4 {
  margin: 0;
  font-size: 16px;
}

:deep(.el-step__title) {
  font-size: 14px;
}

:deep(.el-step__description) {
  padding-right: 10px;
  font-size: 12px;
}
</style>
