import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as api from '../utils/api'

export const useTrainingStore = defineStore('training', () => {
  const isRunning = ref(false)
  const currentEpisode = ref(0)
  const totalEpisodes = ref(0)
  const metrics = ref<Record<string, number>>({})

  async function fetchStatus() {
    try {
      const res = await api.getTrainingStatus()
      isRunning.value = res.data.is_running
      currentEpisode.value = res.data.current_episode
      totalEpisodes.value = res.data.total_episodes
      metrics.value = res.data.metrics || {}
    } catch {
      // ignore
    }
  }

  async function start(numEpisodes = 50000, numPlayers = 4) {
    await api.startTraining({ num_episodes: numEpisodes, num_players: numPlayers })
    isRunning.value = true
    totalEpisodes.value = numEpisodes
  }

  async function stop() {
    await api.stopTraining()
    isRunning.value = false
  }

  return { isRunning, currentEpisode, totalEpisodes, metrics, fetchStatus, start, stop }
})
