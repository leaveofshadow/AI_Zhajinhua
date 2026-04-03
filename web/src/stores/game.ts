import { defineStore } from 'pinia'
import { ref, reactive } from 'vue'
import { gameWs } from '../utils/ws'

export interface PlayerState {
  is_active: boolean
  is_looked: boolean
  total_bet: number
  last_action: string
  chip_count: number
}

export const useGameStore = defineStore('game', () => {
  const connected = ref(false)
  const roomId = ref('')
  const myPosition = ref(-1)
  const myCards = ref<string[]>([])
  const hasLooked = ref(false)
  const myChips = ref(0)
  const pot = ref(0)
  const currentBet = ref(0)
  const roundCount = ref(0)
  const activePlayers = ref(0)
  const currentPlayer = ref(-1)
  const playerStates = ref<PlayerState[]>([])
  const actionLog = ref<{ player: number; action: string; amount?: number }[]>([])
  const gameOver = ref(false)
  const winner = ref(-1)
  const chipChanges = ref<number[]>([])
  const allHands = ref<any[]>([])

  function connectGame(room: string) {
    roomId.value = room
    gameWs.connect(room)
    connected.value = true

    gameWs.on('game_start', (data: any) => {
      myPosition.value = data.position
      currentPlayer.value = data.current_player ?? -1
      updateState(data.state)
      if (data.state?.my_cards) {
        myCards.value = data.state.my_cards
      }
    })

    gameWs.on('your_turn', (data: any) => {
      currentPlayer.value = myPosition.value
      if (data.state) {
        updateState(data.state)
        if (data.state.my_cards) {
          myCards.value = data.state.my_cards
        }
      }
    })

    gameWs.on('player_action', (data: any) => {
      actionLog.value.push({
        player: data.player,
        action: data.action,
        amount: data.pot,
      })
      pot.value = data.pot
      if (data.current_player !== undefined) {
        currentPlayer.value = data.current_player
      }
    })

    gameWs.on('round_end', (data: any) => {
      gameOver.value = true
      winner.value = data.winner
      chipChanges.value = data.chip_changes
      allHands.value = data.all_hands ?? []
    })

    gameWs.on('error', (data: any) => {
      console.error('Game error:', data.message)
    })
  }

  function updateState(state: any) {
    myChips.value = state.my_chips ?? 0
    hasLooked.value = state.has_looked ?? false
    pot.value = state.pot ?? 0
    currentBet.value = state.current_bet ?? 0
    roundCount.value = state.round_count ?? 0
    activePlayers.value = state.active_players ?? 0
    playerStates.value = state.player_states ?? []
  }

  function sendAction(action: string, multiplier?: number, target?: number) {
    gameWs.send({ action, multiplier, target })
  }

  function startNewGame() {
    gameOver.value = false
    winner.value = -1
    allHands.value = []
    actionLog.value = []
    myCards.value = []
    hasLooked.value = false
    roundCount.value = 0
    gameWs.send({ action: 'new_game' })
  }

  function reset() {
    connected.value = false
    gameOver.value = false
    winner.value = -1
    allHands.value = []
    actionLog.value = []
    gameWs.disconnect()
  }

  return {
    connected, roomId, myPosition, myCards, hasLooked, myChips,
    pot, currentBet, roundCount, activePlayers, currentPlayer,
    playerStates, actionLog, gameOver, winner, chipChanges, allHands,
    connectGame, sendAction, reset, startNewGame,
  }
})
