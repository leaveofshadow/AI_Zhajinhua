<template>
  <div class="action-log">
    <h4>操作日志</h4>
    <div class="log-list" ref="logList">
      <div v-for="(log, i) in logs" :key="i" class="log-item">
        <span class="log-player">P{{ log.player }}</span>
        <span class="log-action">{{ log.action }}</span>
      </div>
      <div v-if="!logs.length" class="log-empty">暂无操作</div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  logs: { player: number; action: string; amount?: number }[]
}>()
</script>

<style scoped>
.action-log {
  background: rgba(0,0,0,0.3);
  border-radius: 10px;
  padding: 12px;
  max-height: 400px;
}
h4 { color: #4ecdc4; margin-bottom: 8px; font-size: 14px; }
.log-list {
  max-height: 340px;
  overflow-y: auto;
  font-size: 13px;
}
.log-item {
  padding: 4px 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.log-player { color: #e94560; margin-right: 8px; font-weight: bold; }
.log-action { color: #e0e0e0; }
.log-empty { color: #666; font-size: 12px; text-align: center; padding: 20px; }
</style>
