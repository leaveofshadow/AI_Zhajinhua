<template>
  <div class="game-table">
    <div class="game-header">
      <span>房间: {{ $route.params.roomId }}</span>
      <span>轮数: {{ gameStore.roundCount }}</span>
      <span>存活: {{ gameStore.activePlayers }}</span>
    </div>

    <div class="game-content">
      <div class="game-main">
        <SeatRing
          :seats="gameStore.playerStates"
          :current-player="gameStore.currentPlayer"
          :my-position="gameStore.myPosition"
          :pot="gameStore.pot"
          :num-players="gameStore.playerStates.length || 4"
        />

        <div class="my-hand">
          <CardHand
            :cards="gameStore.myCards"
            :face-up="gameStore.hasLooked"
          />
        </div>

        <ActionBar
          :is-my-turn="gameStore.currentPlayer === gameStore.myPosition && !gameStore.gameOver"
          :has-looked="gameStore.hasLooked"
          :player-states="gameStore.playerStates"
          :my-position="gameStore.myPosition"
          @action="handleAction"
        />
      </div>

      <div class="game-sidebar">
        <ActionLog :logs="gameStore.actionLog" />
      </div>
    </div>

    <!-- 游戏结束弹窗 -->
    <el-dialog v-model="gameStore.gameOver" title="对局结束" width="500px" :show-close="false">
      <div class="end-header">
        <span class="winner-name">{{ winnerName }} 获胜!</span>
      </div>

      <!-- 所有玩家的手牌 -->
      <div class="all-hands">
        <div v-for="(hand, i) in gameStore.allHands" :key="i" class="hand-row" :class="{ winner: hand.is_active }">
          <div class="hand-name">{{ hand.name }}</div>
          <div class="hand-cards">
            <span v-for="(card, ci) in hand.cards" :key="ci" class="card-tag" :class="cardColor(card)">{{ card }}</span>
          </div>
          <div class="hand-type">{{ hand.hand_type }}</div>
          <div class="hand-chips">
            筹码: {{ hand.chips }}
            <span v-if="gameStore.chipChanges[i] !== undefined" :class="gameStore.chipChanges[i] >= 0 ? 'gain' : 'loss'">
              ({{ gameStore.chipChanges[i] >= 0 ? '+' : '' }}{{ gameStore.chipChanges[i] }})
            </span>
          </div>
        </div>
      </div>

      <div class="end-actions">
        <el-button type="success" size="large" @click="startNewGame">再来一局</el-button>
        <el-button size="large" @click="backToLobby">返回大厅</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useGameStore } from '../stores/game'
import SeatRing from '../components/SeatRing.vue'
import CardHand from '../components/CardHand.vue'
import ActionBar from '../components/ActionBar.vue'
import ActionLog from '../components/ActionLog.vue'

const route = useRoute()
const router = useRouter()
const gameStore = useGameStore()

const winnerName = computed(() => {
  const hands = gameStore.allHands
  const w = gameStore.winner
  if (w >= 0 && w < hands.length) return hands[w].name
  return '??'
})

function cardColor(card: string) {
  const s = card.slice(-1)
  return s === '♥' || s === '♦' ? 'red-suit' : 'black-suit'
}

onMounted(() => {
  const roomId = route.params.roomId as string
  gameStore.connectGame(roomId)
})

onUnmounted(() => {
  gameStore.reset()
})

function handleAction(action: string, multiplier?: number, target?: number) {
  gameStore.sendAction(action, multiplier, target)
}

function startNewGame() {
  gameStore.startNewGame()
}

function backToLobby() {
  gameStore.reset()
  router.push('/')
}
</script>

<style scoped>
.game-table { padding: 10px; }
.game-header {
  display: flex;
  gap: 20px;
  color: #888;
  font-size: 14px;
  margin-bottom: 16px;
}
.game-content { display: flex; gap: 20px; }
.game-main { flex: 1; }
.game-sidebar { width: 250px; }
.my-hand { margin: 20px 0; text-align: center; }

/* 结束弹窗 */
.end-header { text-align: center; margin-bottom: 20px; }
.winner-name { font-size: 22px; color: #ffd700; font-weight: bold; }
.all-hands { display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px; }
.hand-row {
  display: flex; align-items: center; gap: 16px;
  padding: 12px;
  border-radius: 8px;
  background: rgba(255,255,255,0.05);
  border: 1px solid #3a5a4a;
}
.hand-row.winner { border-color: #ffd700; background: rgba(255,215,0,0.1); }
.hand-name { width: 80px; font-weight: bold; color: #e0e0e0; }
.hand-cards { display: flex; gap: 6px; }
.card-tag {
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: bold;
  font-size: 14px;
}
.red-suit { background: #e94560; color: #fff; }
.black-suit { background: #333; color: #fff; }
.hand-type { width: 100px; color: #4ecdc4; font-size: 13px; }
.hand-chips { color: #ffd700; font-size: 13px; }
.hand-chips .gain { color: #4ecdc4; }
.hand-chips .loss { color: #e94560; }
.end-actions { display: flex; gap: 12px; justify-content: center; }
</style>
