"""Main Quart server and route setup."""
import logging
import sys
import os
import subprocess
import asyncio
import time
from datetime import datetime
import signal

from quart import Quart, request, jsonify
from quart_cors import cors

from pipeline.searchPipeline import run_elixposearch_pipeline
from sessions.main import get_session_manager
from ragService.main import get_retrieval_system
from chatEngine.main import initialize_chat_engine, get_chat_engine
from commons.requestID import RequestIDMiddleware

# Import gateway functions
from app.gateways import health, search, session, chat, stats, websocket

logger = logging.getLogger("elixpo-api")


class lixSearch:
    """Main application container."""
    
    def __init__(self):
        self.app = Quart(__name__)
        self.pipeline_initialized = False
        self.initialization_lock = asyncio.Lock()
        self.model_server_process = None
        
        self._setup_cors()
        self._setup_middleware()
        self._register_routes()
        self._register_error_handlers()
        self._register_lifecycle_hooks()
    
    def _setup_cors(self):
        """Setup CORS middleware."""
        cors(self.app)
    
    def _setup_middleware(self):
        """Setup ASGI middleware."""
        middleware = RequestIDMiddleware(self.app.asgi_app)
        self.app.asgi_app = middleware
    
    def _register_routes(self):
        """Register all application routes."""
        # Create wrapper functions to pass pipeline_initialized state
        async def health_check_wrapper():
            return await health.health_check(self.pipeline_initialized)
        
        async def search_wrapper():
            return await search.search(self.pipeline_initialized)
        
        async def chat_wrapper():
            return await chat.chat(self.pipeline_initialized)
        
        async def session_chat_wrapper(session_id):
            return await chat.session_chat(session_id, self.pipeline_initialized)
        
        async def chat_completions_wrapper(session_id):
            return await chat.chat_completions(session_id, self.pipeline_initialized)
        
        # Health check
        self.app.route('/api/health', methods=['GET'])(health_check_wrapper)
        
        # Search
        self.app.route('/api/search', methods=['POST'])(search_wrapper)
        
        # Session management
        self.app.route('/api/session/create', methods=['POST'])(session.create_session)
        self.app.route('/api/session/<session_id>', methods=['GET'])(session.get_session_info)
        self.app.route('/api/session/<session_id>/kg', methods=['GET'])(session.get_session_kg)
        self.app.route('/api/session/<session_id>/query', methods=['POST'])(session.query_session_kg)
        self.app.route('/api/session/<session_id>/entity/<entity>', methods=['GET'])(
            session.get_entity_evidence
        )
        self.app.route('/api/session/<session_id>/summary', methods=['GET'])(session.get_session_summary)
        self.app.route('/api/session/<session_id>', methods=['DELETE'])(session.delete_session)
        
        # Chat
        self.app.route('/api/chat', methods=['POST'])(chat_wrapper)
        self.app.route('/api/session/<session_id>/chat', methods=['POST'])(session_chat_wrapper)
        self.app.route('/api/session/<session_id>/chat/completions', methods=['POST'])(chat_completions_wrapper)
        self.app.route('/api/session/<session_id>/history', methods=['GET'])(chat.get_chat_history)
        
        # Stats
        self.app.route('/api/stats', methods=['GET'])(stats.get_stats)
        
        # WebSocket
        self.app.websocket('/ws/search')(websocket.websocket_search)
    
    def _register_error_handlers(self):
        """Register error handlers."""
        @self.app.errorhandler(404)
        async def not_found(error):
            return jsonify({"error": "Not found"}), 404
        
        @self.app.errorhandler(500)
        async def internal_error(error):
            request_id = request.headers.get("X-Request-ID", "")
            logger.error(f"[{request_id}] Internal error: {error}", exc_info=True)
            return jsonify({
                "error": "Internal server error",
                "request_id": request_id
            }), 500
    
    def _register_lifecycle_hooks(self):
        """Register startup and shutdown hooks."""
        @self.app.before_serving
        async def startup():
            async with self.initialization_lock:
                if self.pipeline_initialized:
                    return
                
                logger.info("[APP] Starting ElixpoSearch and IPC Service...")
                try:
                    self._start_ipc_service()
                    await asyncio.sleep(2)
                    
                    session_manager = get_session_manager()
                    retrieval_system = get_retrieval_system()
                    initialize_chat_engine(session_manager, retrieval_system)
                    
                    self.pipeline_initialized = True
                    logger.info("[APP] ElixpoSearch ready with IPC Service")
                except Exception as e:
                    logger.error(f"[APP] Initialization failed: {e}", exc_info=True)
                    raise
        
        @self.app.after_serving
        async def shutdown():
            logger.info("[APP] Shutting down...")
            self._stop_ipc_service()
    
    def _start_ipc_service(self):
        """Start IPC service subprocess."""
        if self.model_server_process is not None:
            logger.info(f"[APP] IPC service already running with PID {self.model_server_process.pid}")
            return
        
        try:
            logger.info("[APP] Starting IPC service...")
            python_executable = sys.executable
            ipc_service_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "ipcService", 
                "main.py"
            )
            
            self.model_server_process = subprocess.Popen(
                [python_executable, ipc_service_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )
            logger.info(f"[APP] IPC service started with PID {self.model_server_process.pid}")
        except Exception as e:
            logger.error(f"[APP] Failed to start IPC service: {e}", exc_info=True)
            raise
    
    def _stop_ipc_service(self):
        """Stop IPC service subprocess."""
        if not self.model_server_process:
            return
        
        try:
            logger.info(f"[APP] Terminating IPC service (PID {self.model_server_process.pid})...")
            if sys.platform != 'win32':
                os.killpg(os.getpgid(self.model_server_process.pid), signal.SIGTERM)
            else:
                self.model_server_process.terminate()
            
            self.model_server_process.wait(timeout=5)
            logger.info("[APP] IPC service terminated")
        except subprocess.TimeoutExpired:
            logger.warning("[APP] IPC service did not terminate gracefully, killing...")
            if sys.platform != 'win32':
                os.killpg(os.getpgid(self.model_server_process.pid), signal.SIGKILL)
            else:
                self.model_server_process.kill()
        except Exception as e:
            logger.warning(f"[APP] Error terminating IPC service: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run the Quart server."""
        import hypercorn.asyncio
        from hypercorn.config import Config
        
        config = Config()
        config.bind = [f"{host}:{port}"]
        config.workers = workers
        
        logger.info("[APP] Starting ElixpoSearch...")
        logger.info(f"[APP] Listening on http://{host}:{port}")
        
        asyncio.run(hypercorn.asyncio.serve(self.app, config))


def create_app() -> lixSearch:
    """Factory function to create and configure the lixSearch."""
    return lixSearch()
