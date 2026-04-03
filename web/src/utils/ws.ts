export interface WSMessage {
  event: string
  [key: string]: any
}

export class GameWebSocket {
  private ws: WebSocket | null = null
  private listeners: Map<string, ((data: any) => void)[]> = new Map()

  connect(roomId: string) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${location.host}/ws/room/${roomId}/play`
    this.ws = new WebSocket(url)

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const handlers = this.listeners.get(data.event) || []
        handlers.forEach((h) => h(data))
        const allHandlers = this.listeners.get('*') || []
        allHandlers.forEach((h) => h(data))
      } catch (e) {
        console.error('WS parse error:', e)
      }
    }

    this.ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    this.ws.onerror = (err) => {
      console.error('WebSocket error:', err)
    }
  }

  send(action: { action: string; multiplier?: number; target?: number }) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(action))
    }
  }

  on(event: string, handler: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event)!.push(handler)
  }

  off(event: string, handler: (data: any) => void) {
    const handlers = this.listeners.get(event)
    if (handlers) {
      const idx = handlers.indexOf(handler)
      if (idx >= 0) handlers.splice(idx, 1)
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.listeners.clear()
  }
}

export const gameWs = new GameWebSocket()
