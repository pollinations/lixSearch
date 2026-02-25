import os
from app.utils import setup_logger
from app.main import create_app
from load_balancer import create_load_balancer

logger = setup_logger("lixsearch")


def run_worker():
    port = int(os.getenv("WORKER_PORT", "8001"))
    worker_id = int(os.getenv("WORKER_ID", "1"))
    
    logger.info(f"[WORKER-{worker_id}] Initializing ElixpoSearch Worker on port {port}...")
    
    elixpo_app = create_app()
    
    logger.info(f"[WORKER-{worker_id}] Starting worker server on 0.0.0.0:{port}")
    elixpo_app.run(host="0.0.0.0", port=port, workers=1)


def run_load_balancer():
    from pipeline.config import (
        LOAD_BALANCER_PORT,
        LOAD_BALANCER_HOST,
        WORKER_START_PORT,
        WORKER_COUNT
    )
    
    logger.info("[LOAD_BALANCER] Initializing ElixpoSearch Load Balancer...")
    logger.info(f"[LOAD_BALANCER] Configuration: {WORKER_COUNT} workers on ports {WORKER_START_PORT}-{WORKER_START_PORT + WORKER_COUNT - 1}")
    
    lb = create_load_balancer(num_workers=WORKER_COUNT, start_port=WORKER_START_PORT)
    
    logger.info(f"[LOAD_BALANCER] Starting on {LOAD_BALANCER_HOST}:{LOAD_BALANCER_PORT}")
    lb.app.run(host=LOAD_BALANCER_HOST, port=LOAD_BALANCER_PORT, workers=1)


if __name__ == "__main__":
    app_mode = os.getenv("APP_MODE", "worker").lower()
    
    logger.info(f"[MAIN] Starting lixSearch in {app_mode.upper()} mode")
    
    if app_mode == "load_balancer":
        run_load_balancer()
    elif app_mode == "worker":
        run_worker()
    else:
        logger.error(f"[MAIN] Unknown APP_MODE: {app_mode}")
        logger.info("[MAIN] Supported modes: 'load_balancer', 'worker'")
        exit(1)
