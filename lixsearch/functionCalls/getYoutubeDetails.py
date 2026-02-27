import os
from pipeline.config import BASE_CACHE_DIR, AUDIO_TRANSCRIBE_SIZE, ERROR_MESSAGE_TRUNCATE
from pytubefix import AsyncYouTube
from pydub import AudioSegment
from loguru import logger
import torch
import asyncio
import re
import time
from urllib.parse import urlparse, parse_qs
from typing import Optional
from faster_whisper import WhisperModel

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"[YoutubeDetails] Loading faster-whisper model {AUDIO_TRANSCRIBE_SIZE} on {device}")
whisper_model = WhisperModel(AUDIO_TRANSCRIBE_SIZE, device=device, compute_type="auto")

def _init_ipc_manager(max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """Initialize IPC connection using centralized manager."""
    try:
        from ipcService.coreServiceManager import CoreServiceManager
        manager = CoreServiceManager.get_instance()
        is_ready = manager.is_ready()
        if is_ready:
            logger.info("[YoutubeDetails] IPC connection established with CoreEmbeddingService")
        else:
            logger.warning("[YoutubeDetails] IPC service not yet ready")
        return is_ready
    except Exception as e:
        logger.warning(f"[YoutubeDetails] Could not initialize IPC connection: {e}")
        return False

async def youtubeMetadata(url: str):
    """Get YouTube metadata via IPC service."""
    if not _init_ipc_manager():
        logger.warning("[YoutubeDetails] IPC service not available for YouTube metadata")
        return None
    try:
        from ipcService.coreServiceManager import get_core_embedding_service
        core_service = get_core_embedding_service()
        metadata = await asyncio.to_thread(core_service.get_youtube_metadata, url)
        return metadata
    except Exception as e:
        logger.error(f"[YoutubeDetails] Error fetching YouTube metadata: {e}")
        return None

def ensure_cache_dir(video_id):
    path = os.path.join(BASE_CACHE_DIR, video_id)
    os.makedirs(path, exist_ok=True)
    return path

def get_youtube_video_id(url):
    print("[INFO] Getting Youtube video ID")
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        video_id = parse_qs(parsed_url.query).get('v')
        if video_id:
            return video_id[0]
        if parsed_url.path:
            match = re.search(r'/(?:embed|v)/([^/?#&]+)', parsed_url.path)
            if match:
                return match.group(1)
    elif "youtu.be" in parsed_url.netloc:
        path = parsed_url.path.lstrip('/')
        if path:
            video_id = path.split('/')[0].split('?')[0].split('#')[0]
            video_id = video_id.split('&')[0]
            return video_id
    return None

async def download_audio(url):
    video_id = get_youtube_video_id(url)
    cache_folder = ensure_cache_dir(video_id)
    wav_path = os.path.join(cache_folder, f"{video_id}.wav")
    if os.path.exists(wav_path):
        logger.info(f"[Download] Using cached audio: {wav_path}")
        return wav_path
    
    yt = AsyncYouTube(url, use_oauth=True, allow_oauth_cache=True)
    streams = await yt.streams()
    audio_streams = streams.filter(only_audio=True)
    preferred_codecs = ["opus", "aac", "mp4a.40.2", "vorbis"]
    audio_streams = [s for s in audio_streams if s.audio_codec in preferred_codecs]
    audio_stream = max(audio_streams, key=lambda s: int(s.abr.replace("kbps", "")))
    extension = audio_stream.mime_type.split("/")[1]
    tmp_path = os.path.join(cache_folder, f"{video_id}.{extension}")
    audio_stream.download(output_path=os.path.dirname(tmp_path), filename=os.path.basename(tmp_path))
    audio = AudioSegment.from_file(tmp_path, format=extension)
    audio.export(wav_path, format="wav")
    os.remove(tmp_path)
    logger.info(f"[Download] Audio saved to {wav_path}")
    return wav_path

async def transcribe_audio(
    url: str,
    full_transcript: bool = False,
    query: Optional[str] = None,
    timeout: float = 300.0
) -> str:
    start_time = time.perf_counter()
    video_id = get_youtube_video_id(url)
    if not video_id:
        logger.error(f"[Transcribe] Invalid YouTube URL: {url}")
        return "[ERROR] Unable to extract video ID from URL"
    
    try:
        logger.info(f"[Transcribe] Starting transcription for video {video_id}")
        
        if _init_ipc_manager() and core_service is not None:
            try:
                logger.info(f"[Transcribe] Using IPC CoreEmbeddingService for transcription")
                audio_path = await download_audio(url)
                transcription = await asyncio.to_thread(
                    core_service.transcribe_audio, 
                    audio_path
                )
                metadata = await youtubeMetadata(url)
                elapsed = time.perf_counter() - start_time
                logger.info(f"[Transcribe] IPC transcription completed in {elapsed:.2f}s")
                if metadata:
                    transcription = f"{transcription}\n\n[Source: {metadata}]"
                return transcription
            except Exception as e:
                logger.warning(f"[Transcribe] IPC transcription failed: {e}. Falling back to local faster-whisper")
        
        logger.info(f"[Transcribe] Using local faster-whisper model on {device}")
        audio_path = await download_audio(url)
        
        segments, info = await asyncio.to_thread(
            whisper_model.transcribe,
            audio_path,
            language="en",
            beam_size=5
        )
        
        transcription = " ".join([segment.text for segment in segments])
        
        metadata = await youtubeMetadata(url)
        elapsed = time.perf_counter() - start_time
        logger.info(f"[Transcribe] Local transcription completed in {elapsed:.2f}s")
        
        if metadata:
            transcription = f"{transcription}\n\n[Source: {metadata}]"
        
        return transcription
    
    except asyncio.TimeoutError:
        logger.error(f"[Transcribe] Transcription timed out after {timeout}s")
        return "[TIMEOUT] Video transcription took too long"
    except Exception as e:
        logger.error(f"[Transcribe] Transcription failed: {e}")
        return f"[ERROR] Failed to transcribe: {str(e)[:ERROR_MESSAGE_TRUNCATE]}"

async def transcribe_long(
    url: str,
    chunk_sec: int = 120,
    beam_size: int = 5,
    query: Optional[str] = None
) -> str:
    start_time = time.perf_counter()
    video_id = get_youtube_video_id(url)
    
    if not video_id:
        logger.error(f"[TranscribeLong] Invalid YouTube URL: {url}")
        return "[ERROR] Unable to extract video ID from URL"
    
    try:
        logger.info(f"[TranscribeLong] Starting long-form transcription for video {video_id}")
        audio_path = await download_audio(url)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"[TranscribeLong] Using faster-whisper on {device} with beam_size={beam_size}")
        segments, info = await asyncio.to_thread(
            whisper_model.transcribe,
            audio_path,
            language="en",
            beam_size=beam_size,
            vad_filter=True
        )
        
        transcription = " ".join([segment.text for segment in segments])
        
        metadata = await youtubeMetadata(url)
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"[TranscribeLong] Transcription completed in {elapsed:.2f}s")
        
        response = f"{transcription}\n\n---\n**Transcription Stats:**\n"
        response += f"- Duration: {info.duration:.1f}s\n"
        response += f"- Language: {info.language}\n"
        response += f"- Processing time: {elapsed:.2f}s\n"
        
        if metadata:
            response += f"- Source: {metadata}\n"
        
        return response
    
    except Exception as e:
        logger.error(f"[TranscribeLong] Transcription failed: {e}")
        return f"[ERROR] Failed to transcribe: {str(e)[:ERROR_MESSAGE_TRUNCATE]}"

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=FLal-KvTNAQ"
    full_transcript = True
    query = None
    transcript = asyncio.run(transcribe_audio(url, full_transcript, query))
