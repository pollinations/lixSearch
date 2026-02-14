
from loguru import logger
import time
from multiprocessing.managers import BaseManager

search_service = None
_ipc_ready = False
_ipc_initialized = False

class ModelManager(BaseManager):
    pass

ModelManager.register('CoreEmbeddingService')
ModelManager.register('accessSearchAgents')

def _init_ipc_manager(max_retries: int = 3, retry_delay: float = 1.0):
    """Lazily initialize IPC connection to the model server."""
    global search_service, _ipc_ready, _ipc_initialized
    
    # Avoid re-attempting if already tried
    if _ipc_initialized:
        return _ipc_ready
    
    _ipc_initialized = True
    
    for attempt in range(max_retries):
        try:
            manager = ModelManager(address=("localhost", 5010), authkey=b"ipcService")
            manager.connect()
            search_service = manager.accessSearchAgents()
            _ipc_ready = True
            logger.info("[Utility] IPC connection established with model_server")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"[Utility] IPC connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logger.debug(f"[Utility] IPC server not available - running in standalone mode")
                _ipc_ready = False
                return False
    
    _ipc_ready = False
    return False

