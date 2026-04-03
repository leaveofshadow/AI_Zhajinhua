<template>
  <div class="card-hand">
    <div
      v-for="(card, i) in displayCards"
      :key="i"
      class="card"
      :class="{ faceUp: faceUp && card, faceDown: !faceUp || !card }"
    >
      <template v-if="faceUp && card">
        <span class="card-rank" :class="suitColor(card)">{{ cardRank(card) }}</span>
        <span class="card-suit" :class="suitColor(card)">{{ cardSuit(card) }}</span>
      </template>
      <template v-else>
        <div class="card-back">?</div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  cards: string[]
  faceUp: boolean
}>()

const displayCards = computed(() => {
  if (props.cards.length > 0) return props.cards
  return ['', '', ''] // 暗牌时显示3张牌背
})

function cardRank(card: string) {
  return card ? card.slice(0, -1) : ''
}
function cardSuit(card: string) {
  return card ? card.slice(-1) : ''
}
function suitColor(card: string) {
  const s = cardSuit(card)
  return s === '♥' || s === '♦' ? 'red' : 'black'
}
</script>

<style scoped>
.card-hand { display: flex; gap: 8px; justify-content: center; }
.card {
  width: 52px;
  height: 72px;
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  transition: transform 0.3s;
}
.card:hover { transform: translateY(-4px); }
.faceUp {
  background: #fff;
  border: 1px solid #ccc;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.faceDown {
  background: linear-gradient(135deg, #1a3a5c, #2a5a8c);
  border: 1px solid #3a6a9c;
}
.card-rank { font-size: 16px; }
.card-suit { font-size: 20px; }
.red { color: #e94560; }
.black { color: #1a1a2e; }
.card-back { font-size: 24px; color: rgba(255,255,255,0.3); }
</style>
