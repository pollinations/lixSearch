import threading
from typing import Optional
from loguru import logger
from multiprocessing.managers import BaseManager
from pipeline.config import IPC_HOST, IPC_PORT, IPC_AUTHKEY, IPC_TIMEOUT


class ModelServerClient(BaseManager):
    pass


ModelServerClient.register('CoreEmbeddingService')


class CoreServiceManager:
    _instance = None
    _lock = threading.Lock()
    _core_service = None
    _manager = None
    _connection_ready = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CoreServiceManager()
        return cls._instance

    def __init__(self):
        self._initialized = False
        self._attempt_connection()

    def _attempt_connection(self):
        if self._initialized:
            return

        try:
            logger.info(
                f"[CoreServiceManager] Connecting to IPC CoreEmbeddingService "
                f"at {IPC_HOST}:{IPC_PORT} (timeout: {IPC_TIMEOUT}s)"
            )

            self._manager = ModelServerClient(
                address=(IPC_HOST, IPC_PORT),
                authkey=IPC_AUTHKEY
            )
            self._manager.connect()
            self._core_service = self._manager.CoreEmbeddingService()
            self._connection_ready = True
            self._initialized = True

            logger.info(
                f"[CoreServiceManager] ✅ Connected to CoreEmbeddingService. "
                f"Vector store: {self._core_service.get_vector_store_stats()}"
            )
        except Exception as e:
            self._connection_ready = False
            logger.error(
                f"[CoreServiceManager] Failed to connect to IPC service: {e}. "
                f"Make sure the IPC service is running: python -m ipcService.main"
            )
            raise

    def get_core_service(self):
        if not self._connection_ready or self._core_service is None:
            self._attempt_connection()
        return self._core_service

    def is_ready(self) -> bool:
        return self._connection_ready

    def get_vector_store_stats(self):
        try:
            return self._core_service.get_vector_store_stats()
        except Exception as e:
            logger.error(f"[CoreServiceManager] Failed to get stats: {e}")
            return {}


def get_core_embedding_service():
    manager = CoreServiceManager.get_instance()
    return manager.get_core_service()


def is_ipc_ready() -> bool:
    try:
        manager = CoreServiceManager.get_instance()
        return manager.is_ready()
    except Exception:
        return False
