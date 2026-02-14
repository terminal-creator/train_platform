<script setup>
import { ref, onMounted } from 'vue'
import { FileText, Upload, Sparkles, Download, Copy, Check, BookOpen, Briefcase, X } from 'lucide-vue-next'
import api from '@/api'

const resumeFile = ref(null)
const resumeFileName = ref('')
const resumeText = ref('')
const fileInputRef = ref(null)
const modifiedResume = ref('')
const loading = ref(false)
const copied = ref(false)
const selectedProjects = ref([])

// Course projects from TRAIN.AI
const courseProjects = [
  {
    id: 'data-engineering',
    title: '数据工程实战',
    description: '高质量训练数据构建、数据清洗、格式转换、质量评估',
    skills: ['数据处理', 'Python', 'JSON/JSONL', '数据质量分析'],
    highlight: '构建了包含10万+样本的高质量SFT训练数据集'
  },
  {
    id: 'sft',
    title: 'SFT 监督微调',
    description: '在自建数据集上完成LLM监督微调，掌握训练参数调优',
    skills: ['PyTorch', 'Transformers', 'DeepSpeed', 'LoRA'],
    highlight: '使用LoRA技术在7B模型上实现高效微调，显存占用降低60%'
  },
  {
    id: 'reward-model',
    title: '奖励模型训练',
    description: '构建偏好数据集，训练Reward Model用于RLHF',
    skills: ['偏好学习', 'Bradley-Terry模型', '对比学习'],
    highlight: '训练的RM在测试集上达到85%的偏好预测准确率'
  },
  {
    id: 'rlhf-grpo',
    title: 'RLHF/GRPO 对齐训练',
    description: '使用PPO/GRPO算法进行人类偏好对齐训练',
    skills: ['强化学习', 'PPO', 'GRPO', 'KL散度控制'],
    highlight: '通过GRPO训练使模型在GSM8K上提升24%准确率'
  },
  {
    id: 'evaluation',
    title: '模型评估与迭代',
    description: '设计评估方案，使用GSM8K/MATH等基准测试',
    skills: ['Benchmark评测', 'A/B测试', '统计分析'],
    highlight: '建立完整评估pipeline，支持自动化回归测试'
  },
  {
    id: 'model-surgery',
    title: '模型手术与融合',
    description: '使用SLERP/TIES等方法进行模型融合优化',
    skills: ['模型融合', 'SLERP', 'SWA', 'Checkpoint选择'],
    highlight: '通过模型融合技术将多个checkpoint整合，性能提升5%'
  }
]

const toggleProject = (projectId) => {
  const index = selectedProjects.value.indexOf(projectId)
  if (index === -1) {
    selectedProjects.value.push(projectId)
  } else {
    selectedProjects.value.splice(index, 1)
  }
}

const triggerFileInput = () => {
  fileInputRef.value?.click()
}

const handleFileUpload = async (event) => {
  const file = event.target.files?.[0]
  if (!file) return

  resumeFile.value = file
  resumeFileName.value = file.name

  // Read file content
  const reader = new FileReader()
  reader.onload = (e) => {
    resumeText.value = e.target.result
  }
  reader.readAsText(file)
}

const removeFile = () => {
  resumeFile.value = null
  resumeFileName.value = ''
  resumeText.value = ''
  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

const processResume = async () => {
  if (!resumeText.value.trim() || selectedProjects.value.length === 0) return

  loading.value = true

  // Simulate processing
  await new Promise(resolve => setTimeout(resolve, 2000))

  const selected = courseProjects.filter(p => selectedProjects.value.includes(p.id))

  let result = `# 简历优化建议\n\n`
  result += `基于你选择的 ${selected.length} 个TRAIN.AI课程项目，以下是如何将这些经历融入简历的建议：\n\n`

  result += `---\n\n`
  result += `## 项目经历部分建议\n\n`

  selected.forEach((project, index) => {
    result += `### ${index + 1}. ${project.title}\n\n`
    result += `**推荐写法：**\n`
    result += `> ${project.highlight}\n\n`
    result += `**技能关键词：** ${project.skills.join('、')}\n\n`
    result += `**STAR法则示例：**\n`
    result += `- **Situation:** 在LLM Post-Training项目中负责${project.description}\n`
    result += `- **Task:** 需要${project.description.split('，')[0]}\n`
    result += `- **Action:** 使用${project.skills.slice(0, 2).join('和')}等技术完成开发\n`
    result += `- **Result:** ${project.highlight}\n\n`
  })

  result += `---\n\n`
  result += `## 技能清单建议\n\n`
  result += `根据你选择的项目，建议在技能部分添加：\n\n`

  const allSkills = [...new Set(selected.flatMap(p => p.skills))]
  result += `- **LLM训练：** ${allSkills.filter(s => ['PyTorch', 'Transformers', 'DeepSpeed', 'LoRA'].includes(s)).join('、') || 'PyTorch、Transformers'}\n`
  result += `- **算法：** ${allSkills.filter(s => ['强化学习', 'PPO', 'GRPO', '偏好学习'].includes(s)).join('、') || 'RLHF、SFT'}\n`
  result += `- **工程能力：** ${allSkills.filter(s => ['数据处理', 'Python', 'Benchmark评测'].includes(s)).join('、') || '数据工程、评估系统'}\n\n`

  result += `---\n\n`
  result += `## 亮点数据参考\n\n`
  result += `简历中可以使用的量化数据：\n\n`
  selected.forEach(project => {
    result += `- ${project.highlight}\n`
  })

  modifiedResume.value = result
  loading.value = false
}

const copyToClipboard = async () => {
  if (!modifiedResume.value) return
  await navigator.clipboard.writeText(modifiedResume.value)
  copied.value = true
  setTimeout(() => { copied.value = false }, 2000)
}

const downloadResume = () => {
  if (!modifiedResume.value) return
  const blob = new Blob([modifiedResume.value], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'resume_suggestions.md'
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div class="resume-container">
    <div class="page-header">
      <div class="header-icon">
        <FileText class="w-6 h-6" />
      </div>
      <div>
        <h1>简历优化助手</h1>
        <p>将 TRAIN.AI 课程项目经历融入你的简历，展示 LLM 训练实战能力</p>
      </div>
    </div>

    <div class="resume-grid">
      <!-- Left: Project Selection -->
      <div class="resume-card">
        <div class="card-header">
          <h3>
            <BookOpen class="w-4 h-4" />
            选择课程项目
          </h3>
          <span class="selected-count">已选 {{ selectedProjects.length }} 个</span>
        </div>

        <div class="card-body">
          <p class="hint-text">选择你在 TRAIN.AI 课程中完成的项目，我们将帮你生成简历优化建议：</p>

          <div class="project-list">
            <div
              v-for="project in courseProjects"
              :key="project.id"
              :class="['project-card', { selected: selectedProjects.includes(project.id) }]"
              @click="toggleProject(project.id)"
            >
              <div class="project-checkbox">
                <Check v-if="selectedProjects.includes(project.id)" class="w-4 h-4" />
              </div>
              <div class="project-info">
                <h4>{{ project.title }}</h4>
                <p>{{ project.description }}</p>
                <div class="project-skills">
                  <span v-for="skill in project.skills.slice(0, 3)" :key="skill">{{ skill }}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="input-group">
            <label>
              <Briefcase class="w-4 h-4" />
              上传你的简历 (可选)
            </label>
            <input
              ref="fileInputRef"
              type="file"
              accept=".txt,.md,.pdf,.doc,.docx"
              style="display: none"
              @change="handleFileUpload"
            />
            <div v-if="!resumeFile" class="upload-area" @click="triggerFileInput">
              <Upload class="w-8 h-8 text-gray-400" />
              <p>点击上传简历文件</p>
              <span>支持 .txt, .md, .pdf, .doc, .docx</span>
            </div>
            <div v-else class="file-preview">
              <div class="file-info">
                <FileText class="w-5 h-5 text-accent-500" />
                <span>{{ resumeFileName }}</span>
              </div>
              <button class="remove-btn" @click="removeFile">
                <X class="w-4 h-4" />
              </button>
            </div>
          </div>

          <button
            class="process-btn"
            :disabled="selectedProjects.length === 0 || loading"
            @click="processResume"
          >
            <Sparkles class="w-4 h-4" />
            {{ loading ? '生成中...' : '生成优化建议' }}
          </button>
        </div>
      </div>

      <!-- Right: Output -->
      <div class="resume-card">
        <div class="card-header">
          <h3>
            <Sparkles class="w-4 h-4" />
            优化建议
          </h3>
          <div class="header-actions" v-if="modifiedResume">
            <button class="action-btn" @click="copyToClipboard">
              <Check v-if="copied" class="w-4 h-4" />
              <Copy v-else class="w-4 h-4" />
              {{ copied ? '已复制' : '复制' }}
            </button>
            <button class="action-btn" @click="downloadResume">
              <Download class="w-4 h-4" />
              下载
            </button>
          </div>
        </div>

        <div class="card-body">
          <div v-if="loading" class="loading-state">
            <div class="loading-spinner"></div>
            <p>正在根据课程项目生成简历优化建议...</p>
          </div>

          <div v-else-if="modifiedResume" class="result-content">
            <pre>{{ modifiedResume }}</pre>
          </div>

          <div v-else class="empty-state">
            <FileText class="w-12 h-12" />
            <p>选择课程项目后生成建议</p>
            <span>我们会告诉你如何将项目经历写入简历</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Info Section -->
    <div class="info-section">
      <h3>为什么要写 LLM 训练经历？</h3>
      <div class="info-grid">
        <div class="info-card">
          <div class="info-icon">🔥</div>
          <div class="info-content">
            <h4>市场需求旺盛</h4>
            <p>LLM/AIGC 领域人才缺口大，具备实战经验的候选人更受青睐</p>
          </div>
        </div>
        <div class="info-card">
          <div class="info-icon">💡</div>
          <div class="info-content">
            <h4>技能稀缺性</h4>
            <p>Post-Training 是 LLM 开发核心环节，掌握 RLHF/DPO 等技术是加分项</p>
          </div>
        </div>
        <div class="info-card">
          <div class="info-icon">📊</div>
          <div class="info-content">
            <h4>量化成果</h4>
            <p>课程项目提供真实的性能指标，让简历更有说服力</p>
          </div>
        </div>
        <div class="info-card">
          <div class="info-icon">🎯</div>
          <div class="info-content">
            <h4>完整项目经历</h4>
            <p>从数据到评估的全流程经验，展示系统性思维</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.resume-container {
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 32px;
}

.header-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.page-header h1 {
  font-size: 24px;
  font-weight: 700;
  color: #111827;
  margin-bottom: 4px;
}

.page-header p {
  font-size: 14px;
  color: #6b7280;
}

.resume-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 32px;
}

.resume-card {
  background: white;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  overflow: hidden;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.card-header h3 {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 600;
  color: #374151;
}

.selected-count {
  font-size: 12px;
  color: #10b981;
  font-weight: 500;
}

.header-actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
  background: white;
  font-size: 12px;
  color: #4b5563;
  cursor: pointer;
  transition: all 0.2s;
}

.action-btn:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}

.card-body {
  padding: 20px;
}

.hint-text {
  font-size: 13px;
  color: #6b7280;
  margin-bottom: 16px;
}

.project-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 20px;
}

.project-card {
  display: flex;
  gap: 12px;
  padding: 14px;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.project-card:hover {
  border-color: #d1d5db;
  background: #f9fafb;
}

.project-card.selected {
  border-color: #10b981;
  background: #ecfdf5;
}

.project-checkbox {
  width: 20px;
  height: 20px;
  border-radius: 6px;
  border: 2px solid #d1d5db;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
  transition: all 0.2s;
}

.project-card.selected .project-checkbox {
  background: #10b981;
  border-color: #10b981;
  color: white;
}

.project-info h4 {
  font-size: 14px;
  font-weight: 600;
  color: #111827;
  margin-bottom: 4px;
}

.project-info p {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 8px;
  line-height: 1.5;
}

.project-skills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.project-skills span {
  padding: 2px 8px;
  border-radius: 4px;
  background: #f3f4f6;
  color: #4b5563;
  font-size: 11px;
}

.project-card.selected .project-skills span {
  background: #d1fae5;
  color: #065f46;
}

.input-group {
  margin-bottom: 16px;
}

.input-group label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 500;
  color: #374151;
  margin-bottom: 8px;
}

.upload-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px;
  border: 2px dashed #e5e7eb;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
  background: #fafafa;
}

.upload-area:hover {
  border-color: #10b981;
  background: #ecfdf5;
}

.upload-area p {
  margin-top: 8px;
  font-size: 14px;
  color: #374151;
  font-weight: 500;
}

.upload-area span {
  margin-top: 4px;
  font-size: 12px;
  color: #9ca3af;
}

.file-preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border: 1px solid #d1fae5;
  border-radius: 8px;
  background: #ecfdf5;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  color: #065f46;
  font-weight: 500;
}

.remove-btn {
  padding: 4px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s;
}

.remove-btn:hover {
  background: #fee2e2;
  color: #dc2626;
}

.process-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  padding: 12px;
  border-radius: 8px;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  border: none;
  color: white;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.process-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.process-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: #6b7280;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e5e7eb;
  border-top-color: #10b981;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: #9ca3af;
  text-align: center;
}

.empty-state p {
  font-size: 14px;
  margin-top: 12px;
  color: #6b7280;
}

.empty-state span {
  font-size: 13px;
  margin-top: 4px;
}

.result-content {
  background: #f9fafb;
  border-radius: 8px;
  padding: 16px;
  max-height: 600px;
  overflow-y: auto;
}

.result-content pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: inherit;
  font-size: 13px;
  line-height: 1.7;
  color: #374151;
  margin: 0;
}

.info-section {
  background: white;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  padding: 24px;
}

.info-section h3 {
  font-size: 16px;
  font-weight: 600;
  color: #111827;
  margin-bottom: 20px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.info-card {
  display: flex;
  gap: 12px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 10px;
}

.info-icon {
  font-size: 24px;
  flex-shrink: 0;
}

.info-content h4 {
  font-size: 14px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 4px;
}

.info-content p {
  font-size: 12px;
  color: #6b7280;
  line-height: 1.5;
}

@media (max-width: 1024px) {
  .resume-grid {
    grid-template-columns: 1fr;
  }

  .info-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 640px) {
  .info-grid {
    grid-template-columns: 1fr;
  }
}
</style>
