import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as api from '../utils/api'

export interface RoomInfo {
  id: string
  num_players: number
  initial_chips: number
  min_bet: number
  speed: string
  seats: any[]
  phase: string
  pot: number
  current_player: number
}

export const useRoomStore = defineStore('room', () => {
  const rooms = ref<RoomInfo[]>([])
  const currentRoom = ref<RoomInfo | null>(null)

  async function fetchRooms() {
    const res = await api.listRooms()
    rooms.value = res.data
  }

  async function createRoom(config: {
    num_players?: number
    initial_chips?: number
    min_bet?: number
    speed?: string
    seats?: { player_type: string; display_name: string }[]
  }) {
    const res = await api.createRoom(config)
    currentRoom.value = res.data
    rooms.value.push(res.data)
    return res.data
  }

  async function fetchRoom(id: string) {
    const res = await api.getRoom(id)
    currentRoom.value = res.data
    return res.data
  }

  return { rooms, currentRoom, fetchRooms, createRoom, fetchRoom }
})
