import os
import time
from multiprocessing.managers import BaseManager
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_core_service = None


def get_core_service():
    """Connect to model_server's CoreEmbeddingService via IPC."""
    global _core_service
    if _core_service is None:
        try:
            class ModelManager(BaseManager):
                pass
            
            ModelManager.register("CoreEmbeddingService")
            manager = ModelManager(address=("localhost", 5010), authkey=b"ipcService")
            manager.connect()
            _core_service = manager.CoreEmbeddingService()
            logger.info("[TRANSCRIBE] Connected to model_server")
        except Exception as e:
            logger.error(f"[TRANSCRIBE] Failed to connect to model_server: {e}")
            raise
    
    return _core_service


# --- TRANSCRIPTION ------------------

def transcribe_long(
    audio_path: str,
    chunk_sec: int = 120,
    beam_size: int = 5,
) -> str:
    try:
        core_service = get_core_service()
        
        # Verify file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"[TRANSCRIBE] Starting transcription of {audio_path}")
        t0 = time.perf_counter()
        
        # Use the model server's transcribe_audio method
        transcript = core_service.transcribe_audio(audio_path)
        
        t1 = time.perf_counter()
        logger.info(f"[TRANSCRIBE] Transcription completed in {t1 - t0:.2f}s")
        
        return transcript
    
    except Exception as e:
        logger.error(f"[TRANSCRIBE] Error during transcription: {e}")
        raise


# --- MAIN ENTRY ------------------

if __name__ == "__main__":
    AUDIO_FILE = "audio_cache/2gUAxUWXelg.wav"

    try:
        t_start = time.perf_counter()
        transcript = transcribe_long(AUDIO_FILE)
        t_end = time.perf_counter()

        print("\n=== DONE ===")
        print(f"Total time: {t_end - t_start:.2f}s")
        print(transcript)
    except Exception as e:
        logger.error(f"[TRANSCRIBE] Fatal error: {e}")
        raise
