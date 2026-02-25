#!/bin/bash
set -e

echo "[ENTRYPOINT] Starting ElixpoSearch services..."

# Determine if this is load balancer or worker mode
APP_MODE=${APP_MODE:-worker}
WORKER_ID=${WORKER_ID:-1}

echo "[ENTRYPOINT] Running in APP_MODE=$APP_MODE"

# Start IPC Service in background (shared by all workers)
# The IPC service should be started ONLY ONCE, ideally on the load balancer
# For now, we'll start it in each container but use a lock mechanism
if [ "$APP_MODE" = "load_balancer" ]; then
    echo "[ENTRYPOINT] [LB] Starting shared IPC Service..."
    python3 -m lixsearch.ipcService.main &
    IPC_PID=$!
    echo "[ENTRYPOINT] [LB] IPC Service started with PID $IPC_PID"

    # Wait for IPC Service to be ready (port 5010)
    echo "[ENTRYPOINT] [LB] Waiting for IPC Service to be ready on port 5010..."
    max_attempts=30
    attempt=0
    while ! nc -z localhost 5010 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "[ENTRYPOINT] [LB] IPC Service failed to start within timeout"
            kill $IPC_PID 2>/dev/null || true
            exit 1
        fi
        echo "[ENTRYPOINT] [LB] Waiting for IPC Service... (attempt $attempt/$max_attempts)"
        sleep 1
    done
    echo "[ENTRYPOINT] [LB] IPC Service is ready!"

    # Start Load Balancer in foreground
    echo "[ENTRYPOINT] [LB] Starting Load Balancer on port 8000..."
    exec python3 lixsearch/load_balancer_app.py

elif [ "$APP_MODE" = "worker" ]; then
    # Workers wait for IPC Service on shared network
    echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Waiting for shared IPC Service on port 5010..."
    max_attempts=60
    attempt=0
    while ! nc -z localhost 5010 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Shared IPC Service unavailable within timeout"
            exit 1
        fi
        echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Waiting for IPC Service... (attempt $attempt/$max_attempts)"
        sleep 1
    done
    echo "[ENTRYPOINT] [WORKER-$WORKER_ID] IPC Service is ready!"

    # Start Worker App in foreground
    echo "[ENTRYPOINT] [WORKER-$WORKER_ID] Starting Worker App on port ${WORKER_PORT:-8001}..."
    exec python3 lixsearch/app.py
else
    echo "[ENTRYPOINT] ERROR: Unknown APP_MODE=$APP_MODE"
    exit 1
fi
