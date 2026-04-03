<template>
  <div class="strategy-analysis">
    <h2>策略分析</h2>

    <!-- 回放列表 -->
    <el-card class="replay-panel">
      <h3>对局回放</h3>
      <el-table :data="replays" stripe @row-click="selectReplay" style="cursor: pointer;">
        <el-table-column prop="id" label="回放ID" width="120" />
        <el-table-column prop="timestamp" label="时间" width="180" />
        <el-table-column prop="num_players" label="人数" width="80" />
      </el-table>
    </el-card>

    <!-- 回放详情 -->
    <el-card v-if="selectedReplay" class="replay-detail">
      <h3>回放详情 - {{ selectedReplay.id }}</h3>

      <!-- 动作时间线 -->
      <el-timeline>
        <el-timeline-item
          v-for="(action, i) in selectedReplay.actions"
          :key="i"
          :timestamp="`第${action.round}轮`"
          placement="top"
        >
          <el-card>
            <p>P{{ action.player }} - {{ action.action }}
              <span v-if="action.multiplier"> ({{ action.multiplier }}x)</span>
              <span v-if="action.target"> → P{{ action.target }}</span>
            </p>
          </el-card>
        </el-timeline-item>
      </el-timeline>

      <!-- 结果 -->
      <div v-if="selectedReplay.result" class="result-section">
        <h4>对局结果</h4>
        <p>获胜者: P{{ selectedReplay.result.winner }}</p>
        <div v-for="(change, i) in selectedReplay.result.chip_changes" :key="i">
          P{{ i }}: <span :style="{ color: change >= 0 ? '#4ecdc4' : '#e94560' }">
            {{ change >= 0 ? '+' : '' }}{{ change }}
          </span>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import * as api from '../utils/api'

const replays = ref<any[]>([])
const selectedReplay = ref<any>(null)

onMounted(async () => {
  try {
    const res = await api.listReplays(0, 50)
    replays.value = res.data
  } catch {
    // server not available
  }
})

async function selectReplay(row: any) {
  try {
    const res = await api.getReplay(row.id)
    selectedReplay.value = res.data
  } catch {
    // ignore
  }
}
</script>

<style scoped>
.strategy-analysis h2 { color: #e94560; margin-bottom: 20px; }
.replay-panel, .replay-detail { background: #16213e; border-color: #2a3a5e; margin-bottom: 20px; }
.replay-panel h3, .replay-detail h3 { color: #4ecdc4; margin-bottom: 16px; }
.result-section { margin-top: 20px; }
.result-section h4 { color: #ffd700; margin-bottom: 8px; }
</style>
