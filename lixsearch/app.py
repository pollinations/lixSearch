import os
from app.main import create_app
from app.utils import setup_logger

logger = setup_logger("lixsearch-api")


if __name__ == "__main__":
    # Get port from environment variable or default to 8001
    # This allows running multiple worker instances on different ports
    port = int(os.getenv("WORKER_PORT", "8001"))
    worker_id = int(os.getenv("WORKER_ID", "1"))
    
    logger.info(f"[MAIN] Initializing ElixpoSearch API Worker {worker_id} on port {port}...")
    elixpo_app = create_app()
    elixpo_app.run(host="0.0.0.0", port=port, workers=1)
