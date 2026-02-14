<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { marked } from 'marked'
import { BookOpen, ChevronRight, ArrowLeft, List } from 'lucide-vue-next'
import api from '@/api'

const route = useRoute()
const router = useRouter()

const posts = ref([])
const currentPost = ref(null)
const loading = ref(false)
const sidebarOpen = ref(true)

// Configure marked for better rendering
marked.setOptions({
  gfm: true,
  breaks: true
})

// Get current post ID from route
const currentPostId = computed(() => route.params.postId || null)

// Render markdown content
const renderedContent = computed(() => {
  if (!currentPost.value?.content) return ''
  return marked(currentPost.value.content)
})

// Fetch all posts
const fetchPosts = async () => {
  try {
    const data = await api.get('/blog/posts')
    posts.value = data

    // If no post selected and we have posts, select the first one
    if (!currentPostId.value && posts.value.length > 0) {
      router.replace(`/blog/${posts.value[0].id}`)
    }
  } catch (error) {
    console.error('Failed to fetch posts:', error)
  }
}

// Fetch single post content
const fetchPost = async (postId) => {
  if (!postId) {
    currentPost.value = null
    return
  }

  loading.value = true
  try {
    const data = await api.get(`/blog/posts/${postId}`)
    currentPost.value = data
  } catch (error) {
    console.error('Failed to fetch post:', error)
    currentPost.value = null
  } finally {
    loading.value = false
  }
}

// Watch for route changes
watch(currentPostId, (newId) => {
  if (newId) {
    fetchPost(newId)
  }
}, { immediate: true })

// Select a post
const selectPost = (post) => {
  router.push(`/blog/${post.id}`)
}

onMounted(() => {
  fetchPosts()
})
</script>

<template>
  <div class="blog-container">
    <!-- Sidebar -->
    <aside :class="['blog-sidebar', { collapsed: !sidebarOpen }]">
      <div class="sidebar-header">
        <div class="sidebar-title">
          <BookOpen class="w-5 h-5 text-accent-500" />
          <span v-if="sidebarOpen">Post-Train 技术文档</span>
        </div>
        <button class="toggle-btn" @click="sidebarOpen = !sidebarOpen">
          <List class="w-4 h-4" />
        </button>
      </div>

      <nav class="post-list" v-if="sidebarOpen">
        <a
          v-for="post in posts"
          :key="post.id"
          :class="['post-item', { active: currentPostId === post.id }]"
          @click="selectPost(post)"
        >
          <span class="post-order">{{ String(post.order).padStart(2, '0') }}</span>
          <span class="post-title">{{ post.title }}</span>
          <ChevronRight class="w-4 h-4 post-arrow" />
        </a>
      </nav>
    </aside>

    <!-- Content -->
    <main class="blog-content">
      <div v-if="loading" class="loading">
        <div class="loading-spinner"></div>
        <span>加载中...</span>
      </div>

      <article v-else-if="currentPost" class="markdown-body">
        <header class="article-header">
          <h1>{{ currentPost.title }}</h1>
        </header>
        <div class="article-content" v-html="renderedContent"></div>
      </article>

      <div v-else class="empty-state">
        <BookOpen class="w-16 h-16 text-gray-300" />
        <p>请从左侧选择一篇文章</p>
      </div>
    </main>
  </div>
</template>

<style scoped>
.blog-container {
  display: flex;
  height: calc(100vh - 120px);
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.blog-sidebar {
  width: 280px;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  transition: width 0.2s ease;
  flex-shrink: 0;
}

.blog-sidebar.collapsed {
  width: 48px;
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid #e5e7eb;
  background: #f9fafb;
}

.sidebar-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  font-size: 14px;
  color: #374151;
}

.toggle-btn {
  padding: 6px;
  border-radius: 6px;
  background: transparent;
  border: none;
  cursor: pointer;
  color: #6b7280;
  transition: all 0.2s;
}

.toggle-btn:hover {
  background: #e5e7eb;
  color: #374151;
}

.post-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.post-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  text-decoration: none;
  color: #4b5563;
  font-size: 13px;
}

.post-item:hover {
  background: #f3f4f6;
}

.post-item.active {
  background: #ecfdf5;
  color: #059669;
}

.post-order {
  font-family: monospace;
  font-size: 11px;
  padding: 2px 6px;
  background: #e5e7eb;
  border-radius: 4px;
  color: #6b7280;
  flex-shrink: 0;
}

.post-item.active .post-order {
  background: #d1fae5;
  color: #059669;
}

.post-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.post-arrow {
  opacity: 0;
  transition: opacity 0.2s;
  flex-shrink: 0;
}

.post-item:hover .post-arrow,
.post-item.active .post-arrow {
  opacity: 1;
}

.blog-content {
  flex: 1;
  overflow-y: auto;
  padding: 32px 48px;
  background: #fff;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  gap: 16px;
  color: #6b7280;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e5e7eb;
  border-top-color: #10b981;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  gap: 16px;
  color: #9ca3af;
}

.article-header {
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid #e5e7eb;
}

.article-header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #111827;
  line-height: 1.3;
}

/* Markdown Styles */
.article-content {
  color: #374151;
  line-height: 1.8;
  font-size: 15px;
}

.article-content :deep(h1) {
  font-size: 24px;
  font-weight: 700;
  color: #111827;
  margin: 40px 0 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e5e7eb;
}

.article-content :deep(h2) {
  font-size: 20px;
  font-weight: 600;
  color: #1f2937;
  margin: 32px 0 16px;
}

.article-content :deep(h3) {
  font-size: 17px;
  font-weight: 600;
  color: #374151;
  margin: 24px 0 12px;
}

.article-content :deep(h4) {
  font-size: 15px;
  font-weight: 600;
  color: #4b5563;
  margin: 20px 0 10px;
}

.article-content :deep(p) {
  margin: 16px 0;
}

.article-content :deep(ul),
.article-content :deep(ol) {
  margin: 16px 0;
  padding-left: 24px;
}

.article-content :deep(li) {
  margin: 8px 0;
}

.article-content :deep(code) {
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Fira Code', 'SF Mono', Monaco, monospace;
  font-size: 13px;
  color: #dc2626;
}

.article-content :deep(pre) {
  background: #1f2937;
  padding: 20px;
  border-radius: 10px;
  overflow-x: auto;
  margin: 20px 0;
}

.article-content :deep(pre code) {
  background: transparent;
  padding: 0;
  color: #e5e7eb;
  font-size: 13px;
  line-height: 1.6;
}

.article-content :deep(blockquote) {
  border-left: 4px solid #10b981;
  background: #ecfdf5;
  padding: 16px 20px;
  margin: 20px 0;
  border-radius: 0 8px 8px 0;
  color: #065f46;
}

.article-content :deep(blockquote p) {
  margin: 0;
}

.article-content :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
  font-size: 14px;
}

.article-content :deep(th),
.article-content :deep(td) {
  border: 1px solid #e5e7eb;
  padding: 12px 16px;
  text-align: left;
}

.article-content :deep(th) {
  background: #f9fafb;
  font-weight: 600;
  color: #374151;
}

.article-content :deep(tr:hover td) {
  background: #f9fafb;
}

.article-content :deep(a) {
  color: #10b981;
  text-decoration: none;
}

.article-content :deep(a:hover) {
  text-decoration: underline;
}

.article-content :deep(img) {
  max-width: 100%;
  border-radius: 8px;
  margin: 20px 0;
}

.article-content :deep(hr) {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 32px 0;
}

.article-content :deep(strong) {
  font-weight: 600;
  color: #111827;
}
</style>
