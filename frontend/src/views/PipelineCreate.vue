<template>
  <div class="pipeline-create">
    <!-- Header -->
    <div class="page-header">
      <el-button :icon="ArrowLeft" @click="goBack">返回</el-button>
      <h2>创建 Pipeline</h2>
    </div>

    <!-- Form -->
    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-width="120px"
      class="pipeline-form"
    >
      <!-- Basic Info -->
      <el-card header="基本信息" class="form-section">
        <el-form-item label="Pipeline 名称" prop="name">
          <el-input
            v-model="form.name"
            placeholder="请输入 Pipeline 名称"
            maxlength="100"
            show-word-limit
          />
        </el-form-item>

        <el-form-item label="描述" prop="description">
          <el-input
            v-model="form.description"
            type="textarea"
            :rows="3"
            placeholder="请输入描述（可选）"
            maxlength="500"
            show-word-limit
          />
        </el-form-item>

        <el-form-item label="优先级" prop="priority">
          <el-slider
            v-model="form.priority"
            :min="1"
            :max="10"
            :marks="priorityMarks"
            show-stops
          />
          <div class="priority-hint">
            <span>优先级越高，Pipeline 越优先执行</span>
          </div>
        </el-form-item>

        <el-form-item label="最大重试次数" prop="max_retries">
          <el-input-number
            v-model="form.max_retries"
            :min="0"
            :max="10"
          />
        </el-form-item>
      </el-card>

      <!-- Stages -->
      <el-card header="阶段配置" class="form-section">
        <el-button
          type="primary"
          :icon="Plus"
          @click="addStage"
          style="margin-bottom: 15px"
        >
          添加阶段
        </el-button>

        <div v-if="form.stages.length === 0" class="empty-stages">
          <el-empty description="暂无阶段，请添加阶段" />
        </div>

        <draggable
          v-model="form.stages"
          item-key="id"
          handle=".drag-handle"
          class="stages-list"
        >
          <template #item="{ element, index }">
            <el-card class="stage-card">
              <template #header>
                <div class="stage-header">
                  <el-icon class="drag-handle" style="cursor: move">
                    <Rank />
                  </el-icon>
                  <span>阶段 {{ index + 1 }}: {{ element.name || '未命名' }}</span>
                  <el-button
                    type="danger"
                    size="small"
                    :icon="Delete"
                    circle
                    @click="removeStage(index)"
                  />
                </div>
              </template>

              <el-form-item
                :prop="`stages.${index}.name`"
                :rules="rules.stageName"
                label="阶段名称"
              >
                <el-input
                  v-model="element.name"
                  placeholder="例如: 数据预处理"
                />
              </el-form-item>

              <el-form-item
                :prop="`stages.${index}.task`"
                :rules="rules.stageTask"
                label="任务类型"
              >
                <el-select
                  v-model="element.task"
                  placeholder="请选择任务类型"
                  style="width: 100%"
                >
                  <el-option
                    v-for="task in availableTasks"
                    :key="task.value"
                    :label="task.label"
                    :value="task.value"
                  >
                    <span>{{ task.label }}</span>
                    <span style="color: #8492a6; font-size: 13px; margin-left: 10px">
                      {{ task.description }}
                    </span>
                  </el-option>
                </el-select>
              </el-form-item>

              <el-form-item label="任务参数">
                <el-input
                  v-model="element.paramsText"
                  type="textarea"
                  :rows="3"
                  placeholder='JSON 格式，例如: {"job_uuid": "test-job"}'
                  @blur="validateParams(element)"
                />
                <div v-if="element.paramsError" class="params-error">
                  {{ element.paramsError }}
                </div>
              </el-form-item>

              <el-form-item label="依赖阶段">
                <el-select
                  v-model="element.depends_on"
                  multiple
                  placeholder="选择此阶段依赖的其他阶段"
                  style="width: 100%"
                >
                  <el-option
                    v-for="(stage, i) in form.stages"
                    :key="i"
                    :label="`阶段 ${i + 1}: ${stage.name || '未命名'}`"
                    :value="stage.name"
                    :disabled="i === index || !stage.name"
                  />
                </el-select>
                <div class="depends-hint">
                  依赖的阶段必须先完成，此阶段才会开始执行
                </div>
              </el-form-item>

              <el-row :gutter="20">
                <el-col :span="12">
                  <el-form-item label="最大重试次数">
                    <el-input-number
                      v-model="element.max_retries"
                      :min="0"
                      :max="10"
                      style="width: 100%"
                    />
                  </el-form-item>
                </el-col>
                <el-col :span="12">
                  <el-form-item label="重试延迟(秒)">
                    <el-input-number
                      v-model="element.retry_delay"
                      :min="10"
                      :max="600"
                      :step="10"
                      style="width: 100%"
                    />
                  </el-form-item>
                </el-col>
              </el-row>
            </el-card>
          </template>
        </draggable>
      </el-card>

      <!-- Actions -->
      <div class="form-actions">
        <el-button @click="goBack">取消</el-button>
        <el-button type="primary" :loading="pipelineStore.creating" @click="handleSubmit">
          创建 Pipeline
        </el-button>
      </div>
    </el-form>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft, Plus, Delete, Rank } from '@element-plus/icons-vue'
import { usePipelineStore } from '@/stores/pipeline'
import draggable from 'vuedraggable'

const router = useRouter()
const pipelineStore = usePipelineStore()

// Form data
const formRef = ref(null)
const form = reactive({
  name: '',
  description: '',
  priority: 5,
  max_retries: 3,
  stages: []
})

// Priority marks
const priorityMarks = {
  1: '低',
  5: '中',
  10: '高'
}

// Available tasks
const availableTasks = [
  { value: 'preprocess_dataset', label: '数据预处理', description: '清洗和转换数据' },
  { value: 'train_model', label: '模型训练', description: '训练 LLM 模型' },
  { value: 'run_evaluation', label: '模型评测', description: '评估模型性能' },
  { value: 'cleanup_checkpoints', label: '清理检查点', description: '删除旧的检查点' },
]

// Form rules
const rules = {
  name: [
    { required: true, message: '请输入 Pipeline 名称', trigger: 'blur' },
    { min: 3, max: 100, message: '长度在 3 到 100 个字符', trigger: 'blur' }
  ],
  priority: [
    { required: true, message: '请选择优先级', trigger: 'change' }
  ],
  max_retries: [
    { required: true, message: '请输入最大重试次数', trigger: 'blur' }
  ],
  stageName: [
    { required: true, message: '请输入阶段名称', trigger: 'blur' }
  ],
  stageTask: [
    { required: true, message: '请选择任务类型', trigger: 'change' }
  ]
}

// Stage ID counter
let stageIdCounter = 0

// Methods
const addStage = () => {
  form.stages.push({
    id: stageIdCounter++,
    name: '',
    task: '',
    params: {},
    paramsText: '{}',
    paramsError: '',
    depends_on: [],
    max_retries: 3,
    retry_delay: 60
  })
}

const removeStage = (index) => {
  const stageName = form.stages[index].name
  form.stages.splice(index, 1)

  // 移除其他阶段对此阶段的依赖
  if (stageName) {
    form.stages.forEach(stage => {
      stage.depends_on = stage.depends_on.filter(dep => dep !== stageName)
    })
  }
}

const validateParams = (stage) => {
  stage.paramsError = ''
  if (!stage.paramsText || stage.paramsText.trim() === '') {
    stage.params = {}
    return
  }

  try {
    stage.params = JSON.parse(stage.paramsText)
    if (typeof stage.params !== 'object' || Array.isArray(stage.params)) {
      stage.paramsError = '参数必须是 JSON 对象'
      stage.params = {}
    }
  } catch (e) {
    stage.paramsError = 'JSON 格式错误: ' + e.message
    stage.params = {}
  }
}

const handleSubmit = async () => {
  try {
    // Validate form
    await formRef.value.validate()

    // Validate stages
    if (form.stages.length === 0) {
      ElMessage.error('请至少添加一个阶段')
      return
    }

    // Validate all stage params
    let hasParamsError = false
    form.stages.forEach(stage => {
      validateParams(stage)
      if (stage.paramsError) {
        hasParamsError = true
      }
    })

    if (hasParamsError) {
      ElMessage.error('请修正阶段参数错误')
      return
    }

    // Prepare data
    const data = {
      name: form.name,
      description: form.description || undefined,
      stages: form.stages.map(stage => ({
        name: stage.name,
        task: stage.task,
        params: stage.params,
        depends_on: stage.depends_on,
        max_retries: stage.max_retries,
        retry_delay: stage.retry_delay
      })),
      priority: form.priority,
      max_retries: form.max_retries
    }

    // Create pipeline
    const pipeline = await pipelineStore.createPipeline(data)

    // Redirect to detail page
    router.push(`/pipelines/${pipeline.uuid}`)
  } catch (error) {
    console.error('Failed to create pipeline:', error)
  }
}

const goBack = () => {
  router.back()
}

// Add initial stage
addStage()
</script>

<style scoped>
.pipeline-create {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.page-header h2 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}

.pipeline-form {
  margin-bottom: 20px;
}

.form-section {
  margin-bottom: 20px;
}

.priority-hint {
  color: #909399;
  font-size: 12px;
  margin-top: 5px;
}

.empty-stages {
  padding: 40px 0;
}

.stages-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.stage-card {
  border: 1px solid #dcdfe6;
}

.stage-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.stage-header span {
  flex: 1;
  font-weight: 600;
}

.params-error {
  color: #f56c6c;
  font-size: 12px;
  margin-top: 5px;
}

.depends-hint {
  color: #909399;
  font-size: 12px;
  margin-top: 5px;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 20px;
  background: #f5f7fa;
  border-radius: 4px;
}

.drag-handle {
  cursor: move;
}

:deep(.el-card__header) {
  padding: 12px 20px;
  background: #f5f7fa;
}
</style>
