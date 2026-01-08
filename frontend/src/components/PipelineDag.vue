<template>
  <div class="pipeline-dag">
    <VueFlow
      v-model:nodes="nodes"
      v-model:edges="edges"
      :default-viewport="{ zoom: 1 }"
      :min-zoom="0.2"
      :max-zoom="4"
      fit-view-on-init
      class="vue-flow"
    >
      <Background pattern-color="#aaa" :gap="16" />
      <Controls />

      <template #node-custom="{ data }">
        <div :class="['custom-node', `status-${data.status}`]">
          <div class="node-header">
            <el-icon :size="20">
              <component :is="getStageIcon(data.status)" />
            </el-icon>
            <span class="node-title">{{ data.label }}</span>
          </div>
          <div class="node-body">
            <div class="node-info">
              <span class="node-task">{{ data.task }}</span>
            </div>
            <el-tag
              :type="getStatusType(data.status)"
              size="small"
              class="node-status"
            >
              {{ getStatusText(data.status) }}
            </el-tag>
          </div>
        </div>
      </template>
    </VueFlow>

    <div v-if="nodes.length === 0" class="empty-dag">
      <el-empty description="暂无流程图数据" />
    </div>
  </div>
</template>

<script setup>
import { ref, watch, computed } from 'vue'
import { VueFlow } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import {
  Clock,
  Loading,
  CircleCheck,
  CircleClose,
  WarningFilled
} from '@element-plus/icons-vue'

const props = defineProps({
  stages: {
    type: Array,
    default: () => []
  }
})

const nodes = ref([])
const edges = ref([])

// Build DAG from stages
const buildDag = () => {
  if (!props.stages || props.stages.length === 0) {
    nodes.value = []
    edges.value = []
    return
  }

  // Create nodes
  const newNodes = props.stages.map((stage, index) => ({
    id: stage.stage_name,
    type: 'custom',
    position: { x: 0, y: index * 150 },
    data: {
      label: stage.stage_name,
      task: stage.task_name,
      status: stage.status,
      order: stage.stage_order
    }
  }))

  // Create edges based on dependencies
  const newEdges = []
  props.stages.forEach(stage => {
    if (stage.depends_on && stage.depends_on.length > 0) {
      stage.depends_on.forEach(dep => {
        newEdges.push({
          id: `${dep}-${stage.stage_name}`,
          source: dep,
          target: stage.stage_name,
          type: 'smoothstep',
          animated: stage.status === 'running',
          style: getEdgeStyle(stage.status)
        })
      })
    }
  })

  // Layout nodes (simple vertical layout with dependencies)
  layoutNodes(newNodes, newEdges)

  nodes.value = newNodes
  edges.value = newEdges
}

// Simple layout algorithm
const layoutNodes = (nodes, edges) => {
  // Build dependency map
  const depMap = new Map()
  nodes.forEach(node => depMap.set(node.id, []))

  edges.forEach(edge => {
    const deps = depMap.get(edge.target) || []
    deps.push(edge.source)
    depMap.set(edge.target, deps)
  })

  // Assign levels based on dependencies
  const levels = new Map()
  const visited = new Set()

  const assignLevel = (nodeId, level = 0) => {
    if (visited.has(nodeId)) return

    visited.add(nodeId)
    const currentLevel = levels.get(nodeId) || 0
    levels.set(nodeId, Math.max(currentLevel, level))

    const deps = depMap.get(nodeId) || []
    deps.forEach(dep => {
      assignLevel(dep, level + 1)
    })
  }

  // Start from nodes with no dependencies
  nodes.forEach(node => {
    const deps = depMap.get(node.id) || []
    if (deps.length === 0) {
      assignLevel(node.id, 0)
    }
  })

  // Assign remaining nodes
  nodes.forEach(node => {
    if (!visited.has(node.id)) {
      assignLevel(node.id, 0)
    }
  })

  // Position nodes
  const levelGroups = new Map()
  nodes.forEach(node => {
    const level = levels.get(node.id) || 0
    const group = levelGroups.get(level) || []
    group.push(node)
    levelGroups.set(level, group)
  })

  const nodeWidth = 280
  const nodeHeight = 120
  const horizontalGap = 100
  const verticalGap = 80

  levelGroups.forEach((group, level) => {
    group.forEach((node, index) => {
      node.position = {
        x: level * (nodeWidth + horizontalGap),
        y: index * (nodeHeight + verticalGap)
      }
    })
  })
}

const getStageIcon = (status) => {
  const iconMap = {
    pending: Clock,
    running: Loading,
    completed: CircleCheck,
    failed: CircleClose,
    skipped: WarningFilled
  }
  return iconMap[status] || Clock
}

const getStatusType = (status) => {
  const typeMap = {
    pending: 'info',
    running: 'primary',
    completed: 'success',
    failed: 'danger',
    skipped: 'warning'
  }
  return typeMap[status] || 'info'
}

const getStatusText = (status) => {
  const textMap = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    skipped: '已跳过'
  }
  return textMap[status] || status
}

const getEdgeStyle = (targetStatus) => {
  const colorMap = {
    pending: { stroke: '#909399', strokeWidth: 2 },
    running: { stroke: '#409EFF', strokeWidth: 2 },
    completed: { stroke: '#67C23A', strokeWidth: 2 },
    failed: { stroke: '#F56C6C', strokeWidth: 2 },
    skipped: { stroke: '#E6A23C', strokeWidth: 2 }
  }
  return colorMap[targetStatus] || { stroke: '#909399', strokeWidth: 2 }
}

// Watch stages and rebuild DAG
watch(
  () => props.stages,
  () => {
    buildDag()
  },
  { immediate: true, deep: true }
)
</script>

<style scoped>
.pipeline-dag {
  width: 100%;
  height: 500px;
  position: relative;
}

.vue-flow {
  background: #f5f7fa;
}

.empty-dag {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.custom-node {
  padding: 12px;
  border-radius: 8px;
  background: white;
  border: 2px solid #dcdfe6;
  min-width: 250px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  transition: all 0.3s;
}

.custom-node:hover {
  box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.custom-node.status-pending {
  border-color: #909399;
}

.custom-node.status-running {
  border-color: #409EFF;
  animation: pulse 2s infinite;
}

.custom-node.status-completed {
  border-color: #67C23A;
  background: #f0f9ff;
}

.custom-node.status-failed {
  border-color: #F56C6C;
  background: #fef0f0;
}

.custom-node.status-skipped {
  border-color: #E6A23C;
  background: #fdf6ec;
}

.node-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid #ebeef5;
}

.node-title {
  font-weight: 600;
  font-size: 14px;
  color: #303133;
}

.node-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.node-info {
  font-size: 12px;
  color: #606266;
}

.node-task {
  font-family: monospace;
  background: #f5f7fa;
  padding: 2px 6px;
  border-radius: 3px;
}

.node-status {
  align-self: flex-start;
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 2px 12px 0 rgba(64, 158, 255, 0.3);
  }
  50% {
    box-shadow: 0 4px 20px 0 rgba(64, 158, 255, 0.6);
  }
}

/* Vue Flow default styles */
@import '@vue-flow/core/dist/style.css';
@import '@vue-flow/core/dist/theme-default.css';
@import '@vue-flow/controls/dist/style.css';
</style>
