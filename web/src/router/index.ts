import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'lobby', component: () => import('../views/Lobby.vue') },
    { path: '/game/:roomId', name: 'game', component: () => import('../views/GameTable.vue') },
    { path: '/training', name: 'training', component: () => import('../views/TrainingMonitor.vue') },
    { path: '/analysis', name: 'analysis', component: () => import('../views/StrategyAnalysis.vue') },
  ],
})

export default router
