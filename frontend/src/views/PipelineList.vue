<template>
  <div class="pipeline-list">
    <!-- Header -->
    <div class="page-header">
      <h2>Pipeline 管理</h2>
      <el-button type="primary" @click="goToCreate">
        <el-icon><Plus /></el-icon>
        创建 Pipeline
      </el-button>
    </div>

    <!-- Filters -->
    <div class="filters">
      <el-space wrap>
        <el-radio-group v-model="statusFilter" @change="handleFilterChange">
          <el-radio-button label="">全部</el-radio-button>
          <el-radio-button label="pending">等待中</el-radio-button>
          <el-radio-button label="running">运行中</el-radio-button>
          <el-radio-button label="completed">已完成</el-radio-button>
          <el-radio-button label="failed">失败</el-radio-button>
          <el-radio-button label="cancelled">已取消</el-radio-button>
        </el-radio-group>

        <el-input
          v-model="searchKeyword"
          placeholder="搜索 Pipeline"
          clearable
          style="width: 300px"
          @input="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>

        <el-button :icon="Refresh" @click="handleRefresh">刷新</el-button>
      </el-space>
    </div>

    <!-- Pipeline Table -->
    <el-table
      v-loading="pipelineStore.loading"
      :data="filteredPipelines"
      stripe
      style="width: 100%"
    >
      <el-table-column prop="name" label="名称" min-width="200">
        <template #default="{ row }">
          <el-link type="primary" @click="goToDetail(row.uuid)">
            {{ row.name }}
          </el-link>
          <div v-if="row.description" class="description">
            {{ row.description }}
          </div>
        </template>
      </el-table-column>

      <el-table-column prop="status" label="状态" width="120">
        <template #default="{ row }">
          <el-tag
            :type="pipelineStore.getStatusColor(row.status)"
            :icon="getStatusIconComponent(row.status)"
          >
            {{ getStatusText(row.status) }}
          </el-tag>
        </template>
      </el-table-column>

      <el-table-column prop="priority" label="优先级" width="100" align="center">
        <template #default="{ row }">
          <el-rate
            v-model="row.priority"
            disabled
            :max="10"
            show-score
            text-color="#ff9900"
          />
        </template>
      </el-table-column>

      <el-table-column prop="created_at" label="创建时间" width="180">
        <template #default="{ row }">
          {{ formatDate(row.created_at) }}
        </template>
      </el-table-column>

      <el-table-column prop="started_at" label="开始时间" width="180">
        <template #default="{ row }">
          {{ row.started_at ? formatDate(row.started_at) : '-' }}
        </template>
      </el-table-column>

      <el-table-column prop="completed_at" label="完成时间" width="180">
        <template #default="{ row }">
          {{ row.completed_at ? formatDate(row.completed_at) : '-' }}
        </template>
      </el-table-column>

      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-space>
            <el-button
              v-if="row.status === 'pending'"
              type="primary"
              size="small"
              @click="handleStart(row)"
            >
              启动
            </el-button>

            <el-button
              v-if="row.status === 'running'"
              type="warning"
              size="small"
              @click="handleCancel(row)"
            >
              取消
            </el-button>

            <el-button
              type="info"
              size="small"
              @click="goToDetail(row.uuid)"
            >
              详情
            </el-button>

            <el-button
              v-if="['completed', 'failed', 'cancelled'].includes(row.status)"
              type="danger"
              size="small"
              @click="handleDelete(row)"
            >
              删除
            </el-button>
          </el-space>
        </template>
      </el-table-column>
    </el-table>

    <!-- Pagination -->
    <div class="pagination">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50, 100]"
        :total="pipelineStore.total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="handleSizeChange"
        @current-change="handlePageChange"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import {
  Plus,
  Search,
  Refresh,
  Clock,
  Loading,
  CircleCheck,
  CircleClose,
  WarningFilled
} from '@element-plus/icons-vue'
import { usePipelineStore } from '@/stores/pipeline'
import dayjs from 'dayjs'

const router = useRouter()
const pipelineStore = usePipelineStore()

// Filters
const statusFilter = ref('')
const searchKeyword = ref('')
const currentPage = ref(1)
const pageSize = ref(20)

// Computed
const filteredPipelines = computed(() => {
  let result = pipelineStore.pipelines

  // 状态筛选
  if (statusFilter.value) {
    result = result.filter(p => p.status === statusFilter.value)
  }

  // 搜索筛选
  if (searchKeyword.value) {
    const keyword = searchKeyword.value.toLowerCase()
    result = result.filter(p =>
      p.name.toLowerCase().includes(keyword) ||
      p.description?.toLowerCase().includes(keyword)
    )
  }

  return result
})

// Methods
const fetchPipelines = async () => {
  await pipelineStore.fetchPipelines({
    status: statusFilter.value || undefined,
    offset: (currentPage.value - 1) * pageSize.value,
    limit: pageSize.value
  })
}

const handleFilterChange = () => {
  currentPage.value = 1
  fetchPipelines()
}

const handleSearch = () => {
  // 搜索是在前端过滤的，不需要重新请求
}

const handleRefresh = () => {
  fetchPipelines()
}

const handlePageChange = (page) => {
  currentPage.value = page
  fetchPipelines()
}

const handleSizeChange = (size) => {
  pageSize.value = size
  currentPage.value = 1
  fetchPipelines()
}

const handleStart = async (pipeline) => {
  try {
    await pipelineStore.startPipeline(pipeline.uuid)
    await fetchPipelines()
  } catch (error) {
    console.error('Failed to start pipeline:', error)
  }
}

const handleCancel = async (pipeline) => {
  try {
    await ElMessageBox.confirm(
      `确定要取消 Pipeline "${pipeline.name}" 吗？`,
      '确认取消',
      {
        type: 'warning'
      }
    )
    await pipelineStore.cancelPipeline(pipeline.uuid)
    await fetchPipelines()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to cancel pipeline:', error)
    }
  }
}

const handleDelete = async (pipeline) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除 Pipeline "${pipeline.name}" 吗？此操作不可恢复。`,
      '确认删除',
      {
        type: 'error'
      }
    )
    await pipelineStore.deletePipeline(pipeline.uuid)
    await fetchPipelines()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete pipeline:', error)
    }
  }
}

const goToCreate = () => {
  router.push('/pipelines/create')
}

const goToDetail = (uuid) => {
  router.push(`/pipelines/${uuid}`)
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

const formatDate = (dateString) => {
  return dayjs(dateString).format('YYYY-MM-DD HH:mm:ss')
}

onMounted(() => {
  fetchPipelines()
})
</script>

<style scoped>
.pipeline-list {
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

.filters {
  margin-bottom: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 4px;
}

.description {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

:deep(.el-rate__text) {
  font-size: 12px;
}
</style>
