#!/bin/bash
# 炸金花游戏服务器启停脚本
# 用法: ./scripts.sh {start|stop|restart|status|logs}

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$PROJECT_DIR/.pids"
LOG_DIR="$PROJECT_DIR/.logs"

BACKEND_PID="$PID_DIR/backend.pid"
FRONTEND_PID="$PID_DIR/frontend.pid"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

BACKEND_PORT=8001
FRONTEND_PORT=5173

mkdir -p "$PID_DIR" "$LOG_DIR"

get_backend_pid() {
    # 先检查 PID 文件
    if [ -f "$BACKEND_PID" ]; then
        local pid
        pid=$(cat "$BACKEND_PID")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
        rm -f "$BACKEND_PID"
    fi
    # 再检查端口
    local pid
    pid=$(lsof -ti:$BACKEND_PORT 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    return 1
}

get_frontend_pid() {
    if [ -f "$FRONTEND_PID" ]; then
        local pid
        pid=$(cat "$FRONTEND_PID")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
        rm -f "$FRONTEND_PID"
    fi
    local pid
    pid=$(lsof -ti:$FRONTEND_PORT 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    return 1
}

start_backend() {
    if pid=$(get_backend_pid); then
        echo "[Backend] Already running (PID $pid)"
        return 0
    fi
    echo "[Backend] Starting on port $BACKEND_PORT..."
    cd "$PROJECT_DIR"
    nohup python3 -m uvicorn server.main:app \
        --host 0.0.0.0 \
        --port $BACKEND_PORT \
        --reload \
        >> "$BACKEND_LOG" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$BACKEND_PID"
    sleep 3
    if curl -s -o /dev/null http://localhost:$BACKEND_PORT/docs 2>/dev/null; then
        echo "[Backend] Started (PID $new_pid, port $BACKEND_PORT)"
    else
        echo "[Backend] Failed to start. Check $BACKEND_LOG"
        cat "$BACKEND_LOG" | tail -5
        return 1
    fi
}

start_frontend() {
    if pid=$(get_frontend_pid); then
        echo "[Frontend] Already running (PID $pid)"
        return 0
    fi
    echo "[Frontend] Starting on port $FRONTEND_PORT..."
    cd "$PROJECT_DIR/web"
    rm -rf node_modules/.vite
    nohup npx vite --host \
        >> "$FRONTEND_LOG" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$FRONTEND_PID"
    sleep 5
    if curl -s -o /dev/null http://localhost:$FRONTEND_PORT 2>/dev/null; then
        echo "[Frontend] Started (PID $new_pid, port $FRONTEND_PORT)"
    else
        echo "[Frontend] Failed to start. Check $FRONTEND_LOG"
        cat "$FRONTEND_LOG" | tail -5
        return 1
    fi
}

stop_backend() {
    if pid=$(get_backend_pid); then
        echo "[Backend] Stopping (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        # Also kill any uvicorn child processes
        pkill -f "uvicorn.*server.main" 2>/dev/null || true
        rm -f "$BACKEND_PID"
        echo "[Backend] Stopped"
    else
        echo "[Backend] Not running"
    fi
}

stop_frontend() {
    if pid=$(get_frontend_pid); then
        echo "[Frontend] Stopping (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        pkill -f "vite" 2>/dev/null || true
        rm -f "$FRONTEND_PID"
        echo "[Frontend] Stopped"
    else
        echo "[Frontend] Not running"
    fi
}

show_status() {
    echo "=== 炸金花游戏服务器状态 ==="
    if pid=$(get_backend_pid); then
        echo "[Backend]  Running (PID $pid, port $BACKEND_PORT)"
    else
        echo "[Backend]  Stopped"
    fi
    if pid=$(get_frontend_pid); then
        echo "[Frontend] Running (PID $pid, port $FRONTEND_PORT)"
    else
        echo "[Frontend] Stopped"
    fi
}

show_logs() {
    local target="${1:-all}"
    case "$target" in
        backend|b)
            echo "=== Backend logs (tail -30) ==="
            tail -30 "$BACKEND_LOG" 2>/dev/null || echo "No logs yet"
            ;;
        frontend|f)
            echo "=== Frontend logs (tail -30) ==="
            tail -30 "$FRONTEND_LOG" 2>/dev/null || echo "No logs yet"
            ;;
        all|*)
            show_logs backend
            echo ""
            show_logs frontend
            ;;
    esac
}

case "${1:-}" in
    start)
        start_backend
        start_frontend
        echo ""
        show_status
        ;;
    stop)
        stop_frontend
        stop_backend
        echo ""
        show_status
        ;;
    restart)
        stop_frontend
        stop_backend
        sleep 1
        start_backend
        start_frontend
        echo ""
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-all}"
        ;;
    backend)
        case "${2:-}" in
            start) start_backend ;;
            stop) stop_backend ;;
            restart) stop_backend; sleep 1; start_backend ;;
            logs) show_logs backend ;;
            *) echo "Usage: $0 backend {start|stop|restart|logs}" ;;
        esac
        ;;
    frontend)
        case "${2:-}" in
            start) start_frontend ;;
            stop) stop_frontend ;;
            restart) stop_frontend; sleep 1; start_frontend ;;
            logs) show_logs frontend ;;
            *) echo "Usage: $0 frontend {start|stop|restart|logs}" ;;
        esac
        ;;
    *)
        echo "炸金花游戏服务器管理脚本"
        echo ""
        echo "用法: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "命令:"
        echo "  start              启动前后端"
        echo "  stop               停止前后端"
        echo "  restart            重启前后端"
        echo "  status             查看运行状态"
        echo "  logs [backend|f]   查看日志"
        echo ""
        echo "  backend {start|stop|restart|logs}   单独管理后端"
        echo "  frontend {start|stop|restart|logs}  单独管理前端"
        ;;
esac
