#!/bin/bash
set -e

echo "[ENTRYPOINT] Starting ElixpoSearch services..."

APP_MODE=${APP_MODE:-worker}
WORKER_ID=${WORKER_ID:-1}
WORKER_PORT=${WORKER_PORT:-8001}

echo "[ENTRYPOINT] Configuration:"
echo "  APP_MODE: $APP_MODE"
echo "  WORKER_ID: $WORKER_ID"
echo "  WORKER_PORT: $WORKER_PORT"
echo "  CHROMA_API_IMPL: ${CHROMA_API_IMPL:-http}"
echo "  CHROMA_SERVER_HOST: ${CHROMA_SERVER_HOST:-chroma-server}"
echo "  CHROMA_SERVER_PORT: ${CHROMA_SERVER_PORT:-8000}"

if [ "$APP_MODE" = "load_balancer" ]; then
    echo "[ENTRYPOINT] [LB] Starting shared IPC Service on port 5010..."
    
    python3 -m lixsearch.ipcService.main &
    IPC_PID=$!
    echo "[ENTRYPOINT] [LB] IPC Service started with PID $IPC_PID"
    
    echo "[ENTRYPOINT] [LB] Waiting for IPC Service to be ready..."
    max_attempts=30
    attempt=0
    while ! nc -z localhost 5010 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "[ENTRYPOINT] [LB] FAILED: IPC Service did not start within timeout"
            kill $IPC_PID 2>/dev/null || true
            exit 1
        fi
        echo "[ENTRYPOINT] [LB] Waiting for IPC Service... (attempt $attempt/$max_attempts)"
        sleep 1
    done
    echo "[ENTRYPOINT] [LB] ✓ IPC Service is ready on port 5010"
    
    export APP_MODE=load_balancer
    
    echo "[ENTRYPOINT] [LB] Starting Load Balancer App..."
    exec python3 lixsearch/app.py

elif [ "$APP_MODE" = "worker" ]; then
    echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Waiting for shared IPC Service on port 5010..."
    
    max_attempts=120
    attempt=0
    while ! nc -z localhost 5010 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "[ENTRYPOINT] [WORKER-$WORKER_ID] FAILED: IPC Service unavailable after timeout"
            echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Make sure Load Balancer is running first"
            exit 1
        fi
        if [ $((attempt % 10)) -eq 0 ]; then
            echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Still waiting for IPC Service... (attempt $attempt/$max_attempts)"
        fi
        sleep 1
    done
    echo "[ENTRYPOINT] [WORKER-$WORKER_ID] ✓ IPC Service is ready on port 5010"
    
    export APP_MODE=worker
    
    echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Starting Worker App on port $WORKER_PORT..."
    exec python3 lixsearch/app.py

else
    echo "[ENTRYPOINT] ERROR: Unknown APP_MODE='$APP_MODE'"
    echo "[ENTRYPOINT] Supported modes: 'load_balancer', 'worker'"
    exit 1
fi
