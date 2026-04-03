<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="header-content">
        <h1 class="title">炸金花 AI 博弈系统</h1>
        <nav class="nav-menu">
          <el-menu mode="horizontal" :default-active="currentRoute" router>
            <el-menu-item index="/">大厅</el-menu-item>
            <el-menu-item index="/training">训练监控</el-menu-item>
            <el-menu-item index="/analysis">策略分析</el-menu-item>
          </el-menu>
        </nav>
      </div>
    </el-header>
    <el-main class="app-main">
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const currentRoute = computed(() => route.path)
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #e0e0e0; }
.app-container { min-height: 100vh; }
.app-header {
  background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
  border-bottom: 2px solid #e94560;
}
.header-content { display: flex; align-items: center; justify-content: space-between; max-width: 1400px; margin: 0 auto; }
.title { color: #e94560; font-size: 20px; white-space: nowrap; }
.nav-menu .el-menu { background: transparent; border: none; }
.nav-menu .el-menu-item { color: #e0e0e0; border-bottom-color: transparent !important; }
.nav-menu .el-menu-item:hover { color: #e94560; background: rgba(233, 69, 96, 0.1); }
.nav-menu .el-menu-item.is-active { color: #e94560; border-bottom-color: #e94560 !important; }
.app-main { max-width: 1400px; margin: 0 auto; padding: 20px; width: 100%; }
</style>
