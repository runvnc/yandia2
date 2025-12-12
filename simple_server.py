"""Simple Dia2 TTS Server - Minimal Changes Approach

This server uses the standard Dia2 API without any modifications.
For each request, it calls generate() with prefix files.

The key insight: continuous conversation doesn't require maintaining
state between calls. Each generate() call is independent, with the
prefix audio providing voice conditioning and context.
"""
import os
import tempfile
import time
import hashlib
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
import numpy as np

from dia2 import Dia2, GenerationConfig, SamplingConfig

app = FastAPI(title="Simple Dia2 TTS Server")

# Configuration
MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

# Global model instance
dia: Optional[Dia2] = None

# Directory for storing voice samples
VOICE_DIR = Path(tempfile.gettempdir()) / "yandia2_voices"
VOICE_DIR.mkdir(exist_ok=True)

# Simple caching for prefix processing
_transcription_cache = {}
_audio_token_cache = {}


def _hash_file(path: str) -> str:
    """Get hash of a file for caching."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup_caching():
    """Monkey-patch dia2's voice_clone to use caching."""
    from dia2.runtime import voice_clone
    from dia2.runtime.voice_clone import WhisperWord
    from dia2.runtime.audio_io import load_mono_audio, encode_audio_tokens
    
    _whisper_model = None
    
    def get_whisper_model(device):
        nonlocal _whisper_model
        if _whisper_model is None:
            import whisper_timestamped as wts
            print("[Dia2] Loading Whisper model...")
            _whisper_model = wts.load_model("openai/whisper-large-v3", device=str(device))
        return _whisper_model
    
    def cached_transcribe_words(audio_path, device, language=None):
        file_hash = _hash_file(audio_path)
        if file_hash in _transcription_cache:
            print(f"[Cache] Transcription cache hit for {audio_path}")
            return _transcription_cache[file_hash]
        
        print(f"[Cache] Transcription cache miss for {audio_path}")
        import whisper_timestamped as wts
        model = get_whisper_model(device)
        result = wts.transcribe(model, audio_path, language=language)
        
        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                text = (word.get("text") or word.get("word") or "").strip()
                if not text:
                    continue
                words.append(WhisperWord(
                    text=text,
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", 0.0))
                ))
        
        _transcription_cache[file_hash] = words
        return words
    
    # Replace the transcribe function
    voice_clone.transcribe_words = cached_transcribe_words
    print("[Dia2] Caching enabled for transcription")


@app.on_event("startup")
async def startup():
    global dia
    print(f"[Dia2] Loading model from {MODEL_REPO}...")
    print(f"[Dia2] Device: {DEVICE}, Dtype: {DTYPE}")
    start = time.time()
    
    # Setup caching before loading model
    _setup_caching()
    
    dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)
    # Warm up the runtime
    _ = dia._ensure_runtime()
    print(f"[Dia2] Model loaded in {time.time() - start:.1f}s")
    print(f"[Dia2] Sample rate: {dia.sample_rate}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_REPO, "device": DEVICE}


@app.post("/upload_voice")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """Upload a voice sample for later use as a prefix."""
    voice_path = VOICE_DIR / f"{name}.wav"
    content = await file.read()
    voice_path.write_bytes(content)
    return {"status": "ok", "name": name, "path": str(voice_path)}


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    speaker_1: Optional[str] = Form(None),
    speaker_2: Optional[str] = Form(None),
    cfg_scale: float = Form(2.0),
    temperature: float = Form(0.8),
    top_k: int = Form(50),
    use_cuda_graph: bool = Form(True),
):
    """
    Generate speech from text.
    
    Args:
        text: The text to speak (use [S1] and [S2] tags for speakers)
        speaker_1: Path to speaker 1 voice sample, or name of uploaded voice
        speaker_2: Path to speaker 2 voice sample, or name of uploaded voice  
        cfg_scale: Classifier-free guidance scale (1.0 = disabled, 2.0+ for better quality)
        temperature: Sampling temperature for audio
        top_k: Top-k sampling parameter
        use_cuda_graph: Whether to use CUDA graphs for faster inference
    
    Returns:
        WAV audio file
    """
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Resolve voice paths
    s1_path = _resolve_voice_path(speaker_1) if speaker_1 else None
    s2_path = _resolve_voice_path(speaker_2) if speaker_2 else None
    
    # Create config
    config = GenerationConfig(
        cfg_scale=cfg_scale,
        audio=SamplingConfig(temperature=temperature, top_k=top_k),
        text=SamplingConfig(temperature=0.6, top_k=50),
        use_cuda_graph=use_cuda_graph,
    )
    
    print(f"[Dia2] Generating: {text[:80]}..." if len(text) > 80 else f"[Dia2] Generating: {text}")
    print(f"[Dia2] Speaker 1: {s1_path}, Speaker 2: {s2_path}")
    start = time.time()
    
    try:
        result = dia.generate(
            text,
            config=config,
            prefix_speaker_1=s1_path,
            prefix_speaker_2=s2_path,
            verbose=True,
        )
        
        elapsed = time.time() - start
        duration = result.waveform.shape[-1] / result.sample_rate
        print(f"[Dia2] Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/duration:.2f})")
        
        # Convert to WAV bytes
        wav_bytes = _tensor_to_wav(result.waveform, result.sample_rate)
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Audio-Duration": str(duration)}
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Dia2] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts_file")
async def text_to_speech_file(
    text: str = Form(...),
    speaker_1: Optional[str] = Form(None),
    speaker_2: Optional[str] = Form(None),
    cfg_scale: float = Form(2.0),
    temperature: float = Form(0.8),
    output_name: str = Form("output"),
):
    """
    Generate speech and save to a file. Returns the file info.
    Useful for testing and debugging.
    """
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    s1_path = _resolve_voice_path(speaker_1) if speaker_1 else None
    s2_path = _resolve_voice_path(speaker_2) if speaker_2 else None
    
    config = GenerationConfig(
        cfg_scale=cfg_scale,
        audio=SamplingConfig(temperature=temperature, top_k=50),
        use_cuda_graph=True,
    )
    
    output_path = VOICE_DIR / f"{output_name}.wav"
    
    print(f"[Dia2] Generating to file: {output_path}")
    start = time.time()
    
    result = dia.generate(
        text,
        config=config,
        prefix_speaker_1=s1_path,
        prefix_speaker_2=s2_path,
        output_wav=str(output_path),
        verbose=True,
    )
    
    elapsed = time.time() - start
    duration = result.waveform.shape[-1] / result.sample_rate
    
    return {
        "status": "ok",
        "path": str(output_path),
        "duration": duration,
        "generation_time": elapsed,
        "rtf": elapsed / duration,
    }


def _resolve_voice_path(voice: str) -> str:
    """Resolve a voice name or path to an actual file path."""
    if not voice:
        return None
        
    # If it's already an absolute path and exists, use it
    if os.path.isabs(voice) and os.path.exists(voice):
        return voice
    
    # Check if it's a name in our voice directory
    voice_file = VOICE_DIR / f"{voice}.wav"
    if voice_file.exists():
        return str(voice_file)
    
    # Also check without .wav extension
    voice_file_direct = VOICE_DIR / voice
    if voice_file_direct.exists():
        return str(voice_file_direct)
    
    # Check if it's a relative path that exists
    if os.path.exists(voice):
        return os.path.abspath(voice)
    
    raise HTTPException(status_code=404, detail=f"Voice not found: {voice}")


def _tensor_to_wav(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Convert a waveform tensor to WAV bytes."""
    import wave
    import io
    
    # Ensure 1D
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    
    # Convert to int16
    audio_np = waveform.detach().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    # Write to WAV
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    
    return buf.getvalue()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
