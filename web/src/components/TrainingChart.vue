<template>
  <div class="training-chart" ref="chartRef" style="width: 100%; height: 300px;"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps<{
  title: string
  data: { x: number; y: number }[]
  color?: string
}>()

const chartRef = ref<HTMLElement>()
let chart: echarts.ECharts | null = null

onMounted(() => {
  if (chartRef.value) {
    chart = echarts.init(chartRef.value, 'dark')
    updateChart()
  }
})

watch(() => props.data, updateChart, { deep: true })

function updateChart() {
  if (!chart) return
  chart.setOption({
    title: { text: props.title, textStyle: { color: '#e0e0e0', fontSize: 14 } },
    backgroundColor: 'transparent',
    grid: { top: 40, right: 20, bottom: 30, left: 50 },
    xAxis: { type: 'category', data: props.data.map((d) => d.x) },
    yAxis: { type: 'value' },
    series: [{
      type: 'line',
      data: props.data.map((d) => d.y),
      smooth: true,
      lineStyle: { color: props.color || '#e94560' },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: (props.color || '#e94560') + '40' },
        { offset: 1, color: 'transparent' },
      ]) },
    }],
  })
}
</script>
