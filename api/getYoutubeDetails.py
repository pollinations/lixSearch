import os
from config import BASE_CACHE_DIR
from pytubefix import AsyncYouTube
from pydub import AudioSegment
from multiprocessing.managers import BaseManager
import torch
import whisper
import asyncio
import re
import time
from urllib.parse import urlparse, parse_qs
from typing import Optional, Iterable



# Remove model server dependency for STT, only keep search_service if needed
class modelManager(BaseManager): pass
modelManager.register("accessSearchAgents")
manager = modelManager(address=("localhost", 5010), authkey=b"ipcService")
manager.connect()
search_service = manager.accessSearchAgents()

# Load Whisper model directly here
AUDIO_TRANSCRIBE_SIZE = "small"  # or set to your preferred model size
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(AUDIO_TRANSCRIBE_SIZE).to(device)

async def youtubeMetadata(url: str):
    """
    Async wrapper for YouTube metadata retrieval via search_service.
    Returns metadata like title, description, views, etc. from YouTube.
    """
    metadata = search_service.get_youtube_metadata(url)
    return metadata

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
    
    return wav_path


async def transcribe_audio_deprecated(url, full_transcript: Optional[str] = None, query: Optional[str] = None):
    start_time = time.time()
    transcription = ""
    video_id = get_youtube_video_id(url)
    print(f"[INFO] Starting transcription for video ID: {video_id}")
    meta_data = youtubeMetadata(url)
    print(f"[INFO] Video title: {meta_data}")
    audio = await download_audio(url)
    # Use Whisper model directly
    print(f"[INFO] Transcribing audio with Whisper model on device: {device}")
    result = whisper_model.transcribe(audio)
    transcription = result["text"]
    print(f"[INFO] Completed transcription for video ID: {video_id}")
    # If full_transcript is provided, use it
    if full_transcript:
        print("[INFO] Using provided full transcript.")
        transcription = full_transcript
    # Optionally, you can add query-based extraction here if needed, but modelService.extract_relevant is not available
    transcription += f" Video Titled as {meta_data}"
    end_time = time.time()
    print(f"[INFO] Transcription took {end_time - start_time:.2f} seconds.")
    return transcription

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=FLal-KvTNAQ"
    full_transcript = True
    query = None

    transcript = asyncio.run(transcribe_audio_deprecated(url, full_transcript, query))
    
