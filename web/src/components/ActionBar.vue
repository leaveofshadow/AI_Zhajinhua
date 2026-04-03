<template>
  <div class="action-bar">
    <el-button type="danger" @click="doAction('fold')" :disabled="!isMyTurn">弃牌</el-button>
    <el-button type="primary" @click="doAction('look')" :disabled="!isMyTurn || hasLooked">看牌</el-button>
    <el-button type="success" @click="doAction('call')" :disabled="!isMyTurn">跟注</el-button>
    <el-dropdown v-if="isMyTurn" @command="doRaise" trigger="click">
      <el-button type="warning">加注 <el-icon><arrow-down /></el-icon></el-button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item v-for="m in [2,3,4,5,6]" :key="m" :command="m">加注 {{ m }}x</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
    <el-dropdown v-if="isMyTurn" @command="doCompare" trigger="click">
      <el-button type="info">比牌 <el-icon><arrow-down /></el-icon></el-button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item
            v-for="p in activeOpponents"
            :key="p.index"
            :command="p.index"
          >与 P{{ p.index }} 比牌</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { ArrowDown } from '@element-plus/icons-vue'

const props = defineProps<{
  isMyTurn: boolean
  hasLooked: boolean
  playerStates: any[]
  myPosition: number
}>()

const emit = defineEmits<{
  action: [action: string, multiplier?: number, target?: number]
}>()

const activeOpponents = computed(() =>
  props.playerStates
    .map((p, i) => ({ ...p, index: i }))
    .filter((p) => p.index !== props.myPosition && p.is_active)
)

function doAction(action: string) {
  emit('action', action)
}

function doRaise(mult: number) {
  emit('action', 'raise', mult)
}

function doCompare(target: number) {
  emit('action', 'compare', undefined, target)
}
</script>

<style scoped>
.action-bar {
  display: flex;
  gap: 10px;
  justify-content: center;
  padding: 16px;
  background: rgba(0,0,0,0.3);
  border-radius: 10px;
}
</style>
