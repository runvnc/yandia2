"""Dia2 Conversation Server - Stateful Single Conversation

This server manages a single conversation, keeping track of:
- Last AI audio (used as speaker_1 prefix for voice consistency)
- Last user audio (used as speaker_2 prefix for context)

Flow:
1. POST /set_voice - Send AI warmup audio (e.g., "Hi, I'm your friendly AI...")
2. POST /user_spoke - Send user audio (e.g., "Hello?")
3. POST /generate - Send text to generate (e.g., "Hi, this is Mary AI from Services.")
   -> Returns generated audio, which becomes the new "last AI audio"
4. Repeat steps 2-3 for the conversation
"""
import os
import tempfile
import time
import hashlib
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
import numpy as np

from dia2 import Dia2, GenerationConfig, SamplingConfig

app = FastAPI(title="Dia2 Conversation Server")

# Configuration
MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

# Global model instance
dia: Optional[Dia2] = None

# Conversation state directory
CONV_DIR = Path(tempfile.gettempdir()) / "yandia2_conversation"
CONV_DIR.mkdir(exist_ok=True)

# Transcription cache
_transcription_cache = {}


@dataclass
class ConversationState:
    """Tracks the current conversation state."""
    last_ai_audio: Optional[str] = None      # Path to last AI audio (warmup or generated)
    last_user_audio: Optional[str] = None    # Path to last user audio
    turn_count: int = 0
    is_initialized: bool = False


# Global conversation state (single conversation)
conversation = ConversationState()


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
            print(f"[Cache] Transcription HIT: {Path(audio_path).name}")
            return _transcription_cache[file_hash]
        
        print(f"[Cache] Transcription MISS: {Path(audio_path).name} - transcribing...")
        import whisper_timestamped as wts
        model = get_whisper_model(device)
        
        start = time.time()
        result = wts.transcribe(model, audio_path, language=language)
        elapsed = time.time() - start
        
        words = []
        transcript = []
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
                transcript.append(text)
        
        print(f"[Cache] Transcribed in {elapsed:.1f}s: {' '.join(transcript)[:80]}...")
        _transcription_cache[file_hash] = words
        return words
    
    voice_clone.transcribe_words = cached_transcribe_words
    print("[Dia2] Caching enabled for transcription")


@app.on_event("startup")
async def startup():
    global dia
    print(f"[Dia2] Loading model from {MODEL_REPO}...")
    print(f"[Dia2] Device: {DEVICE}, Dtype: {DTYPE}")
    start = time.time()
    
    _setup_caching()
    
    dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)
    _ = dia._ensure_runtime()
    print(f"[Dia2] Model loaded in {time.time() - start:.1f}s")
    print(f"[Dia2] Sample rate: {dia.sample_rate}")
    
    # Clean up old conversation files
    for f in CONV_DIR.glob("*.wav"):
        f.unlink()
    print(f"[Dia2] Conversation directory: {CONV_DIR}")


@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model": MODEL_REPO, 
        "device": DEVICE,
        "conversation": {
            "initialized": conversation.is_initialized,
            "turn_count": conversation.turn_count,
            "has_ai_audio": conversation.last_ai_audio is not None,
            "has_user_audio": conversation.last_user_audio is not None,
        }
    }


@app.post("/reset")
async def reset_conversation():
    """Reset the conversation state."""
    global conversation
    
    # Clean up files
    for f in CONV_DIR.glob("*.wav"):
        f.unlink()
    
    conversation = ConversationState()
    print("[Dia2] Conversation reset")
    
    return {"status": "ok", "message": "Conversation reset"}


@app.post("/set_voice")
async def set_voice(file: UploadFile = File(...)):
    """
    Initialize the conversation with an AI voice warmup.
    
    Send an audio clip of the AI speaking to establish the voice.
    This will be transcribed and used as the voice reference.
    
    Example: AI saying "Hi, I'm your friendly AI customer service warming up my voice."
    """
    global conversation
    
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save the audio file
    voice_path = CONV_DIR / "ai_voice_warmup.wav"
    content = await file.read()
    voice_path.write_bytes(content)
    
    print(f"[Dia2] AI voice set: {voice_path} ({len(content)} bytes)")
    
    # Update conversation state
    conversation.last_ai_audio = str(voice_path)
    conversation.is_initialized = True
    conversation.turn_count = 0
    conversation.last_user_audio = None
    
    # Pre-transcribe to warm up cache
    try:
        from dia2.runtime.voice_clone import transcribe_words
        runtime = dia._ensure_runtime()
        transcribe_words(str(voice_path), runtime.device)
    except Exception as e:
        print(f"[Dia2] Warning: Pre-transcription failed: {e}")
    
    return {
        "status": "ok",
        "message": "AI voice initialized",
        "path": str(voice_path)
    }


@app.post("/user_spoke")
async def user_spoke(file: UploadFile = File(...)):
    """
    Add user audio to the conversation.
    
    Send an audio clip of the user speaking. This will be transcribed
    and used as context for the next AI response.
    
    Example: User saying "Hello?"
    """
    global conversation
    
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not conversation.is_initialized:
        raise HTTPException(status_code=400, detail="Conversation not initialized. Call /set_voice first.")
    
    # Save the audio file
    user_path = CONV_DIR / f"user_turn_{conversation.turn_count}.wav"
    content = await file.read()
    user_path.write_bytes(content)
    
    print(f"[Dia2] User audio received: {user_path} ({len(content)} bytes)")
    
    # Update conversation state
    conversation.last_user_audio = str(user_path)
    
    # Pre-transcribe to warm up cache
    try:
        from dia2.runtime.voice_clone import transcribe_words
        runtime = dia._ensure_runtime()
        transcribe_words(str(user_path), runtime.device)
    except Exception as e:
        print(f"[Dia2] Warning: Pre-transcription failed: {e}")
    
    return {
        "status": "ok",
        "message": "User audio added",
        "path": str(user_path)
    }


@app.post("/generate")
async def generate(
    text: str = Form(...),
    cfg_scale: float = Form(1.0),
    temperature: float = Form(0.8),
    top_k: int = Form(50),
):
    """
    Generate AI speech for the given text.
    
    Uses the current conversation context:
    - speaker_1 (AI voice): Last AI audio or warmup
    - speaker_2 (user): Last user audio (if any)
    
    The generated audio becomes the new "last AI audio" for the next turn.
    
    Args:
        text: The text for the AI to speak (without [S1] tag - it's added automatically)
    
    Returns:
        WAV audio of the AI speaking
    """
    global conversation
    
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not conversation.is_initialized:
        raise HTTPException(status_code=400, detail="Conversation not initialized. Call /set_voice first.")
    
    # Prepare the text with S1 tag
    if not text.strip().startswith("[S1]"):
        text = f"[S1] {text}"
    
    config = GenerationConfig(
        cfg_scale=cfg_scale,
        audio=SamplingConfig(temperature=temperature, top_k=top_k),
        text=SamplingConfig(temperature=0.6, top_k=50),
        use_cuda_graph=True,
    )
    
    print(f"\n[Dia2] === Turn {conversation.turn_count} ===")
    print(f"[Dia2] Generating: {text}")
    print(f"[Dia2] AI prefix: {conversation.last_ai_audio}")
    print(f"[Dia2] User prefix: {conversation.last_user_audio}")
    
    start = time.time()
    
    try:
        # IMPORTANT: Dia2 needs BOTH speakers for voice cloning!
        # If no user audio, fall back to AI audio for speaker_2
        speaker_2_audio = conversation.last_user_audio or conversation.last_ai_audio
        
        result = dia.generate(
            text,
            config=config,
            prefix_speaker_1=conversation.last_ai_audio,
            prefix_speaker_2=speaker_2_audio,
            verbose=True,
        )
        
        elapsed = time.time() - start
        duration = result.waveform.shape[-1] / result.sample_rate
        print(f"[Dia2] Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/duration:.2f})")
        
        # Save the generated audio as the new AI audio for next turn
        ai_output_path = CONV_DIR / f"ai_turn_{conversation.turn_count}.wav"
        wav_bytes = _tensor_to_wav(result.waveform, result.sample_rate)
        ai_output_path.write_bytes(wav_bytes)
        
        # Update conversation state
        conversation.last_ai_audio = str(ai_output_path)
        conversation.turn_count += 1
        
        print(f"[Dia2] Saved AI audio: {ai_output_path}")
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(duration),
                "X-Turn-Count": str(conversation.turn_count),
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Dia2] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get the current conversation state (for debugging)."""
    return {
        "initialized": conversation.is_initialized,
        "turn_count": conversation.turn_count,
        "last_ai_audio": conversation.last_ai_audio,
        "last_user_audio": conversation.last_user_audio,
    }


def _tensor_to_wav(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Convert a waveform tensor to WAV bytes."""
    import wave
    import io
    
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    
    audio_np = waveform.detach().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    
    return buf.getvalue()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
