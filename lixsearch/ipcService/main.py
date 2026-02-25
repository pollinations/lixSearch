import warnings
import os
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing.managers import BaseManager
from loguru import logger
from ipcService.coreEmbeddingService import CoreEmbeddingService
from ipcService.searchPortManager import accessSearchAgents, _ensure_background_loop, run_async_on_bg_loop, agent_pool, shutdown_graceful

warnings.filterwarnings('ignore', message='Can\'t initialize NVML')
os.environ['CHROMA_TELEMETRY_DISABLED'] = '1'

# Configure Chroma client
# Support both embedded and server modes via environment variables
chroma_api_impl = os.getenv("CHROMA_API_IMPL", "embedded").lower()
os.environ.setdefault("CHROMA_API_IMPL", chroma_api_impl)

if chroma_api_impl == "http":
    # Server mode configuration
    chroma_host = os.getenv("CHROMA_SERVER_HOST", "localhost")
    chroma_port = os.getenv("CHROMA_SERVER_PORT", "8000")
    os.environ["CHROMA_SERVER_HOST"] = chroma_host
    os.environ["CHROMA_SERVER_PORT"] = chroma_port
    logger.info(f"[IPC] Chroma configured for server mode: {chroma_host}:{chroma_port}")
else:
    # Embedded mode
    logger.info(f"[IPC] Chroma configured for embedded mode")

logging.getLogger('chromadb').setLevel(logging.ERROR)

if __name__ == "__main__":
    class ModelManager(BaseManager):
        pass

    ModelManager.register("CoreEmbeddingService", CoreEmbeddingService)
    ModelManager.register("accessSearchAgents", accessSearchAgents)
    core_service = CoreEmbeddingService()
    search_agents = accessSearchAgents()
    manager = ModelManager(address=("localhost", 5010), authkey=b"ipcService")
    server = manager.get_server()
    logger.info("[MAIN] Core service started on port 5010...")
    logger.info(f"[MAIN] Vector store stats: {core_service.get_vector_store_stats()}")

    try:
        _ensure_background_loop()
        run_async_on_bg_loop(agent_pool.initialize_pool())
    except Exception as e:
        logger.error(f"[MAIN] Failed to initialize agent pool: {e}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[MAIN] Shutdown signal received...")
    except Exception as e:
        logger.error(f"[MAIN] Server error: {e}")
    finally:
        shutdown_graceful()
        logger.info("[MAIN] Shutdown complete")
