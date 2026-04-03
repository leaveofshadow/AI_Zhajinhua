import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
})

// 房间
export const createRoom = (data: {
  num_players?: number
  initial_chips?: number
  min_bet?: number
  speed?: string
  seats?: { player_type: string; display_name: string }[]
}) => api.post('/rooms', data)

export const getRoom = (id: string) => api.get(`/rooms/${id}`)
export const listRooms = () => api.get('/rooms')
export const updateSeats = (id: string, seats: any[]) => api.patch(`/rooms/${id}/seats`, { seats })
export const closeRoom = (id: string) => api.delete(`/rooms/${id}`)

// 回放
export const listReplays = (skip = 0, limit = 20) => api.get('/replays', { params: { skip, limit } })
export const getReplay = (id: string) => api.get(`/replays/${id}`)

// 模型
export const listModels = () => api.get('/models')

// 训练
export const startTraining = (data: { num_episodes?: number; num_players?: number }) =>
  api.post('/training/start', data)
export const stopTraining = () => api.post('/training/stop')
export const getTrainingStatus = () => api.get('/training/status')

export default api
