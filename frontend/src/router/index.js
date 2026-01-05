import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
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
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
