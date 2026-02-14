import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'landing',
    component: () => import('@/views/LandingView.vue'),
    meta: { layout: 'blank' }
  },
  {
    path: '/dashboard',
    name: 'dashboard',
    component: () => import('@/views/DashboardView.vue')
  },
  {
    path: '/compute',
    name: 'compute',
    component: () => import('@/views/ComputeView.vue')
  },
  {
    path: '/datasets',
    name: 'datasets',
    component: () => import('@/views/DatasetView.vue')
  },
  {
    path: '/jobs',
    name: 'jobs',
    component: () => import('@/views/JobsView.vue')
  },
  {
    path: '/surgery',
    name: 'surgery',
    component: () => import('@/views/SurgeryView.vue')
  },
  {
    path: '/monitoring',
    name: 'monitoring',
    component: () => import('@/views/MonitoringView.vue')
  },
  {
    path: '/monitoring/:jobId',
    name: 'job-monitoring',
    component: () => import('@/views/MonitoringView.vue')
  },
  {
    path: '/evaluation',
    name: 'evaluation',
    component: () => import('@/views/EvaluationView.vue')
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue')
  },
  // Phase 4: Pipeline 管理
  {
    path: '/pipelines',
    name: 'pipelines',
    component: () => import('@/views/PipelineList.vue')
  },
  {
    path: '/pipelines/create',
    name: 'pipeline-create',
    component: () => import('@/views/PipelineCreate.vue')
  },
  {
    path: '/pipelines/:uuid',
    name: 'pipeline-detail',
    component: () => import('@/views/PipelineDetail.vue')
  },
  // Phase 4: 任务监控
  {
    path: '/tasks',
    name: 'tasks',
    component: () => import('@/views/TaskMonitor.vue')
  },
  // Data Factory
  {
    path: '/data-factory',
    name: 'data-factory',
    component: () => import('@/views/DataFactoryView.vue')
  },
  // Blog and Resume
  {
    path: '/blog',
    name: 'blog',
    component: () => import('@/views/BlogView.vue')
  },
  {
    path: '/blog/:postId',
    name: 'blog-post',
    component: () => import('@/views/BlogView.vue')
  },
  {
    path: '/resume',
    name: 'resume',
    component: () => import('@/views/ResumeView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
