"""
Load Balancer entry point for distributing requests across worker instances
Port 8000 - Load Balancer
Ports 8001-8010 - Worker instances
"""
import logging
from app.utils import setup_logger
from load_balancer import create_load_balancer

logger = setup_logger("lixsearch-lb")


if __name__ == "__main__":
    logger.info("[LOAD_BALANCER] Initializing ElixpoSearch Load Balancer...")
    
    # Create load balancer for 10 workers on ports 8001-8010
    lb = create_load_balancer(num_workers=10, start_port=8001)
    
    logger.info("[LOAD_BALANCER] Starting load balancer on port 8000...")
    lb.app.run(host="0.0.0.0", port=8000, workers=1)
