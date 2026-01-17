import os
from faster_whisper import WhisperModel
import time
from pydub import AudioSegment

t0 = time.perf_counter()
model = WhisperModel(
    "small",
    device="cuda",
    compute_type="int8_float32",
    download_root="model_cache",
    local_files_only=True
)

def _transcribe_chunk(chunk_path: str, model: WhisperModel):
    segments, _ = model.transcribe(chunk_path, beam_size=5)
    transcription = "".join([segment.text.strip() for segment in segments])
    return transcription

def chunk_audio_to_disk(audio: AudioSegment, chunk_length_ms: int, reqID: str, base_tmp: str = "tmp_cache"):
    import shutil
    chunk_dir = os.path.join(base_tmp, reqID)
    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        end = min(start + chunk_length_ms, len(audio))
        chunk_audio = audio[start:end]
        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
        chunk_audio.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    return chunk_paths

def transcribe(AUDIO_FILE: str, reqID: str) -> str:
    t1 = time.perf_counter()
    print(f"Model loaded in {t1 - t0:.2f} seconds")
    t2 = time.perf_counter()
    audio = AudioSegment.from_file(AUDIO_FILE)
    duration_sec = len(audio) / 1000
    chunk_length_ms = 2 * 60 * 1000
    if duration_sec <= 5 * 60:
        segments, _ = model.transcribe(AUDIO_FILE, beam_size=5)
        transcription = "".join([segment.text.strip() for segment in segments])
        t3 = time.perf_counter()
        print(f"Transcription completed in {t3 - t2:.2f} seconds")
        return transcription
    print(f"Audio is {duration_sec/60:.2f} minutes, chunking to 2 min and saving to disk...")
    chunking_start = time.perf_counter()
    chunk_paths = chunk_audio_to_disk(audio, chunk_length_ms, reqID)
    chunking_end = time.perf_counter()
    print(f"Chunking completed in {chunking_end - chunking_start:.3f} seconds. {len(chunk_paths)} chunks created.")
    transcriptions = []
    for idx, chunk_path in enumerate(chunk_paths):
        text = _transcribe_chunk(chunk_path, model)
        print(f"[INFO] Transcribed chunk {idx+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")
        transcriptions.append(text)
    full_transcription = " ".join(transcriptions)
    t3 = time.perf_counter()
    print(f"Transcription completed in {t3 - t2:.2f} seconds")
    print("[DONE]")
    return full_transcription

if __name__ == "__main__":
    AUDIO_FILE  = "tmp_cache/test123/chunk_0.wav"
    reqID = "test123"
    trans  = transcribe(AUDIO_FILE, reqID)
    print(trans)
