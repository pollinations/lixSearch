import os
import time
import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment
import torch

# ------------------ MODEL INIT (ONCE) ------------------

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="int8_float32",
    download_root="model_cache",
    local_files_only=True
)

# ------------------ AUDIO UTIL ------------------

def load_audio_mono_float32(path: str, target_sr: int = 16000) -> np.ndarray:
    audio = (
        AudioSegment
        .from_file(path)
        .set_channels(1)
        .set_frame_rate(target_sr)  # <-- THIS WAS MISSING
    )

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max
    return samples


def chunk_audio(samples: np.ndarray, sr: int, chunk_sec: int):
    chunk_size = chunk_sec * sr
    for i in range(0, len(samples), chunk_size):
        yield samples[i:i + chunk_size]

# ------------------ TRANSCRIPTION ------------------

def transcribe_long(
    audio_path: str,
    chunk_sec: int = 120,
    beam_size: int = 5,
) -> str:

    samples = load_audio_mono_float32(audio_path)
    sample_rate = 16000  # Whisper expectation

    if sample_rate != 16000:
        raise RuntimeError("Resampling required (Whisper expects 16kHz).")

    total_chunks = (len(samples) + chunk_sec * sample_rate - 1) // (chunk_sec * sample_rate)
    results = []

    print(f"[INFO] audio length = {len(samples)/sample_rate/60:.2f} min")
    print(f"[INFO] chunks = {total_chunks}")

    for idx, chunk in enumerate(chunk_audio(samples, sample_rate, chunk_sec)):
        print(f"[CALL] chunk {idx+1}/{total_chunks}")

        t0 = time.perf_counter()

        segments, _ = model.transcribe(
            chunk,
            beam_size=beam_size,
            vad_filter=True,
            language="en",
            task="transcribe"
        )

        text = "".join(seg.text for seg in segments)
        results.append(text)

        t1 = time.perf_counter()
        print(f"[CHUNK] {idx+1} time = {t1 - t0:.3f}s")

        # IMPORTANT: do NOT call torch.cuda.empty_cache()

    return " ".join(results)

# ------------------ ENTRY ------------------

if __name__ == "__main__":
    AUDIO_FILE = "audio_cache/2gUAxUWXelg.wav"

    t_start = time.perf_counter()
    transcript = transcribe_long(AUDIO_FILE)
    t_end = time.perf_counter()

    print("\n=== DONE ===")
    print(f"Total time: {t_end - t_start:.2f}s")
    print(transcript)
