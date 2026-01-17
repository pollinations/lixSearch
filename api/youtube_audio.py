import os
import yt_dlp

def download_youtube_audio(
    url: str,
    out_dir: str = "audio_cache",
    audio_format: str = "wav",
    sample_rate: int = 16000
):
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_dir}/%(id)s.%(ext)s",
        "noplaylist": True,
        "quiet": True,

        # SINGLE valid postprocessor
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": "0",
        }],

        # Correct way to force mono + 16kHz
        "postprocessor_args": {
            "FFmpegExtractAudio": [
                "-ac", "1",
                "-ar", str(sample_rate),
            ]
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]

    return os.path.join(out_dir, f"{video_id}.{audio_format}")




download_youtube_audio("https://www.youtube.com/watch?v=sI5Ftm1-jik")