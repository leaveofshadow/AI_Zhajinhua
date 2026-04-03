<template>
  <div class="lobby">
    <h2>对局大厅</h2>

    <!-- 快速开始 -->
    <el-row :gutter="20" class="quick-start">
      <el-col :span="8" v-for="mode in modes" :key="mode.key">
        <el-card class="mode-card" shadow="hover">
          <h3>{{ mode.title }}</h3>
          <p>{{ mode.desc }}</p>
          <div v-if="mode.hasAI" class="ai-level-select">
            <span class="ai-label">AI 难度:</span>
            <el-radio-group v-model="aiLevel" size="small">
              <el-radio-button value="easy">初级AI</el-radio-button>
              <el-radio-button value="medium">中级AI</el-radio-button>
            </el-radio-group>
          </div>
          <el-button type="primary" round @click="quickStart(mode.key)" style="margin-top: 8px">快速开始</el-button>
        </el-card>
      </el-col>
    </el-row>

    <!-- 自定义房间 -->
    <el-card class="custom-room">
      <h3>自定义房间</h3>
      <el-form :model="roomConfig" label-width="80px" inline>
        <el-form-item label="人数">
          <el-input-number v-model="roomConfig.num_players" :min="3" :max="6" />
        </el-form-item>
        <el-form-item label="初始筹码">
          <el-input-number v-model="roomConfig.initial_chips" :min="100" :step="100" />
        </el-form-item>
        <el-form-item label="底注">
          <el-input-number v-model="roomConfig.min_bet" :min="1" :step="5" />
        </el-form-item>
        <el-form-item label="速度">
          <el-select v-model="roomConfig.speed">
            <el-option label="正常" value="normal" />
            <el-option label="快速" value="fast" />
            <el-option label="极速" value="turbo" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="success" @click="createCustomRoom">创建房间</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 房间列表 -->
    <el-card class="room-list">
      <h3>活跃房间</h3>
      <el-table :data="roomStore.rooms" stripe style="width: 100%">
        <el-table-column prop="id" label="房间ID" width="120" />
        <el-table-column prop="num_players" label="人数" width="80" />
        <el-table-column prop="phase" label="状态" width="100" />
        <el-table-column prop="speed" label="速度" width="80" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button size="small" type="primary" @click="joinRoom(row.id)">加入</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useRoomStore } from '../stores/room'

const router = useRouter()
const roomStore = useRoomStore()
const aiLevel = ref('medium')

const modes = [
  { key: 'ai_vs_ai', title: 'AI vs AI', desc: '全部AI对局，可加速观摩', hasAI: true },
  { key: 'ai_vs_human', title: 'AI vs Human', desc: '人机对战，挑战AI策略', hasAI: true },
  { key: 'human_vs_human', title: '混合对局', desc: '多人混合模式', hasAI: false },
]

const roomConfig = reactive({
  num_players: 4,
  initial_chips: 1000,
  min_bet: 10,
  speed: 'normal',
})

onMounted(() => {
  roomStore.fetchRooms()
})

async function quickStart(mode: string) {
  const numPlayersMap: Record<string, number> = {
    ai_vs_ai: 6,
    ai_vs_human: 4,
    human_vs_human: 6,
  }
  const speedMap: Record<string, string> = {
    ai_vs_ai: 'fast',
    ai_vs_human: 'normal',
    human_vs_human: 'normal',
  }
  const n = numPlayersMap[mode]

  // 根据模式生成座位配置
  const level = aiLevel.value
  const seats: { player_type: string; display_name: string; ai_level?: string }[] = []
  if (mode === 'ai_vs_ai') {
    for (let i = 0; i < n; i++) seats.push({ player_type: 'ai', display_name: `AI-${i + 1}`, ai_level: level })
  } else if (mode === 'ai_vs_human') {
    seats.push({ player_type: 'human', display_name: 'You' })
    for (let i = 1; i < n; i++) seats.push({ player_type: 'ai', display_name: `AI-${i}`, ai_level: level })
  } else {
    seats.push({ player_type: 'human', display_name: 'You' })
    seats.push({ player_type: 'human', display_name: 'Player 2' })
    for (let i = 2; i < n; i++) seats.push({ player_type: 'ai', display_name: `AI-${i - 1}`, ai_level: level })
  }

  const room = await roomStore.createRoom({
    num_players: n,
    initial_chips: roomConfig.initial_chips,
    min_bet: roomConfig.min_bet,
    speed: speedMap[mode],
    seats,
  })
  router.push(`/game/${room.id}`)
}

async function createCustomRoom() {
  const n = roomConfig.num_players
  const seats = []
  seats.push({ player_type: 'human', display_name: 'You' })
  for (let i = 1; i < n; i++) seats.push({ player_type: 'ai', display_name: `AI-${i}`, ai_level: aiLevel.value })
  const room = await roomStore.createRoom({ ...roomConfig, seats })
  router.push(`/game/${room.id}`)
}

function joinRoom(id: string) {
  router.push(`/game/${id}`)
}
</script>

<style scoped>
.lobby h2 { color: #e94560; margin-bottom: 20px; }
.quick-start { margin-bottom: 24px; }
.mode-card {
  text-align: center;
  background: #16213e;
  border-color: #2a3a5e;
  cursor: pointer;
  transition: transform 0.2s;
}
.mode-card:hover { transform: translateY(-4px); }
.mode-card h3 { color: #4ecdc4; margin-bottom: 8px; }
.mode-card p { color: #888; font-size: 13px; margin-bottom: 12px; }
.custom-room, .room-list { background: #16213e; border-color: #2a3a5e; margin-bottom: 20px; }
.custom-room h3, .room-list h3 { color: #4ecdc4; margin-bottom: 16px; }
.ai-level-select { margin: 8px 0; }
.ai-label { color: #888; font-size: 13px; margin-right: 8px; }
</style>
