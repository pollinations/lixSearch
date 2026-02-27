#!/bin/bash
set -e

# LixSearch Entrypoint Script
# Supports load_balancer, ipc, and worker modes

APP_MODE=${APP_MODE:-worker}
WORKER_PORT=${WORKER_PORT:-9002}
WORKER_ID=${WORKER_ID:-1}
LOG_LEVEL=${LOG_LEVEL:-INFO}
IPC_PORT=${IPC_PORT:-9510}

echo "Starting LixSearch in $APP_MODE mode..."
echo "Log level: $LOG_LEVEL"

cd /app

if [ "$APP_MODE" = "load_balancer" ]; then
    echo "Starting Load Balancer on port 9000..."
    python lixsearch/load_balancer.py
elif [ "$APP_MODE" = "ipc" ]; then
    echo "Starting IPC Service on port $IPC_PORT..."
    python lixsearch/ipcService/main.py
elif [ "$APP_MODE" = "worker" ]; then
    echo "Starting Worker $WORKER_ID on port $WORKER_PORT..."
    export WORKER_PORT=$WORKER_PORT
    python lixsearch/app/main.py
else
    echo "Unknown APP_MODE: $APP_MODE"
    exit 1
fi
