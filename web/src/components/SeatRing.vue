<template>
  <div class="seat-ring">
    <div class="pot-display">
      <span class="pot-label">底池</span>
      <span class="pot-amount">{{ pot }}</span>
    </div>
    <div
      v-for="(seat, i) in seats"
      :key="i"
      class="seat"
      :class="{
        active: i === currentPlayer && seat.is_active,
        folded: !seat.is_active,
        'my-seat': i === myPosition,
      }"
      :style="seatStyle(i)"
    >
      <div class="seat-avatar">{{ seat.name || `P${i}` }}</div>
      <div class="seat-chips">{{ seat.chip_count }}</div>
      <div class="seat-status">
        <span v-if="!seat.is_active" class="status-fold">弃牌</span>
        <span v-else-if="seat.is_looked" class="status-looked">明牌</span>
        <span v-else class="status-dark">暗牌</span>
      </div>
      <div v-if="seat.total_bet" class="seat-bet">注: {{ seat.total_bet }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  seats: any[]
  currentPlayer: number
  myPosition: number
  pot: number
  numPlayers: number
}>()

function seatStyle(index: number) {
  const angle = (2 * Math.PI * index) / props.numPlayers - Math.PI / 2
  const radius = 38
  const x = 50 + radius * Math.cos(angle)
  const y = 50 + radius * Math.sin(angle)
  return {
    left: `${x}%`,
    top: `${y}%`,
    transform: 'translate(-50%, -50%)',
  }
}
</script>

<style scoped>
.seat-ring {
  position: relative;
  width: 100%;
  max-width: 500px;
  aspect-ratio: 1;
  margin: 0 auto;
  background: radial-gradient(circle, #1a3a2a 0%, #0d1f17 70%);
  border-radius: 50%;
  border: 3px solid #2d5a3d;
}
.pot-display {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}
.pot-label { display: block; color: #8ab; font-size: 12px; }
.pot-amount { display: block; color: #ffd700; font-size: 24px; font-weight: bold; }
.seat {
  position: absolute;
  width: 80px;
  text-align: center;
  padding: 6px;
  border-radius: 10px;
  background: rgba(255,255,255,0.05);
  border: 2px solid #3a5a4a;
  transition: all 0.3s;
}
.seat.active { border-color: #e94560; box-shadow: 0 0 12px rgba(233,69,96,0.5); }
.seat.folded { opacity: 0.4; }
.seat.my-seat { border-color: #4ecdc4; }
.seat-avatar { font-size: 13px; color: #e0e0e0; font-weight: bold; }
.seat-chips { font-size: 11px; color: #ffd700; }
.seat-status { font-size: 10px; margin-top: 2px; }
.status-fold { color: #e94560; }
.status-looked { color: #4ecdc4; }
.status-dark { color: #888; }
.seat-bet { font-size: 10px; color: #aaa; }
</style>
