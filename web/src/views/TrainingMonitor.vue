<template>
  <div class="training-monitor">
    <h2>训练监控</h2>

    <!-- 训练控制 -->
    <el-card class="control-panel">
      <h3>训练控制</h3>
      <el-form inline>
        <el-form-item label="训练局数">
          <el-input-number v-model="numEpisodes" :min="1000" :step="1000" />
        </el-form-item>
        <el-form-item label="玩家人数">
          <el-input-number v-model="numPlayers" :min="3" :max="6" />
        </el-form-item>
        <el-form-item>
          <el-button
            type="success"
            @click="startTraining"
            :disabled="trainingStore.isRunning"
          >开始训练</el-button>
          <el-button
            type="danger"
            @click="trainingStore.stop()"
            :disabled="!trainingStore.isRunning"
          >停止训练</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 训练状态 -->
    <el-card class="status-panel">
      <h3>训练状态</h3>
      <el-row :gutter="20">
        <el-col :span="6">
          <el-statistic title="状态" :value="trainingStore.isRunning ? '运行中' : '已停止'" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="当前局数" :value="trainingStore.currentEpisode" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="总局数" :value="trainingStore.totalEpisodes" />
        </el-col>
        <el-col :span="6">
          <el-statistic
            title="进度"
            :value="trainingStore.totalEpisodes > 0
              ? ((trainingStore.currentEpisode / trainingStore.totalEpisodes) * 100).toFixed(1) + '%'
              : '0%'"
          />
        </el-col>
      </el-row>
    </el-card>

    <!-- 训练曲线 -->
    <el-row :gutter="20">
      <el-col :span="12">
        <TrainingChart title="策略损失" :data="policyLossData" color="#e94560" />
      </el-col>
      <el-col :span="12">
        <TrainingChart title="价值损失" :data="valueLossData" color="#4ecdc4" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useTrainingStore } from '../stores/training'
import TrainingChart from '../components/TrainingChart.vue'

const trainingStore = useTrainingStore()
const numEpisodes = ref(50000)
const numPlayers = ref(4)

const policyLossData = ref<{ x: number; y: number }[]>([])
const valueLossData = ref<{ x: number; y: number }[]>([])

let timer: ReturnType<typeof setInterval> | null = null

onMounted(async () => {
  await trainingStore.fetchStatus()
  timer = setInterval(() => trainingStore.fetchStatus(), 5000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})

async function startTraining() {
  await trainingStore.start(numEpisodes.value, numPlayers.value)
}
</script>

<style scoped>
.training-monitor h2 { color: #e94560; margin-bottom: 20px; }
.control-panel, .status-panel { background: #16213e; border-color: #2a3a5e; margin-bottom: 20px; }
.control-panel h3, .status-panel h3 { color: #4ecdc4; margin-bottom: 16px; }
</style>
