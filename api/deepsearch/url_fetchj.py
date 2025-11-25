import asyncio
from pytubefix import AsyncYouTube
import whisper
import os
from pydub import AudioSegment
import torch
import time

URL = "https://www.youtube.com/watch?v=v8eUhkhC8lw"
BASE_CACHE_DIR = "./cached_audio"  

def ensure_cache_dir(video_id):
    path = os.path.join(BASE_CACHE_DIR, video_id)
    os.makedirs(path, exist_ok=True)
    return path

async def download_audio(video_id, url):
    cache_folder = ensure_cache_dir(video_id)
    wav_path = os.path.join(cache_folder, f"{video_id}.wav")
    
    # Reuse cached audio if exists
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
    
    # Convert to WAV
    audio = AudioSegment.from_file(tmp_path, format=extension)
    audio.export(wav_path, format="wav")
    os.remove(tmp_path)
    
    return wav_path

async def main(url):
    video_id = url.split("v=")[-1].split("&")[0]
    wav_path = await download_audio(video_id, url)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small", device=device)
    
    start_time = time.time()
    # Full audio transcription
    result = await asyncio.to_thread(model.transcribe, wav_path, language="en")
    final_text = result["text"]
    end_time = time.time()
    
    print("="*50)
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    print("="*50)
    print(final_text)

if __name__ == "__main__":
    asyncio.run(main(URL))
