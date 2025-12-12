"""Dia2 Streaming Conversation Server

Extends conversation_server.py with WebSocket streaming for low-latency TTS.

Flow:
1. POST /set_voice - Send AI warmup audio (same as before)
2. POST /user_spoke - Send user audio (same as before)  
3. WebSocket /ws/generate - Stream audio chunks as they're generated

Expected latency:
- Warmup (unavoidable): ~1-2s
- First audio chunk: ~600-800ms after generation starts
- Subsequent chunks: ~150-250ms intervals
- Total to first audio: ~2-3s
"""
import os
import tempfile
import time
import hashlib
import struct
import json
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.generation import PrefixConfig
from dia2.runtime.voice_clone import build_prefix_plan, PrefixPlan
from dia2.runtime.script_parser import parse_script
from dia2.runtime.generator import (
    build_initial_state,
    warmup_with_prefix,
    run_generation_loop,
    decode_audio,
    GenerationState,
    CachedGraphState,
    create_graph_cache,
    reset_graph_cache,
    _allocate_network_buffers,
    _ensure_graph_cublas_ready,
    _fill_audio_channels,
    _execute_transformer_step,
    _execute_transformer_graph,
    _execute_depformer_stage,
    _execute_depformer_graph,
)
from dia2.audio.grid import undelay_frames
from dia2.runtime.guidance import apply_classifier_guidance, sample_audio_logits
from dia2.runtime.sampler import sample_token
from dia2.audio.grid import mask_audio_logits, delay_frames, undelay_frames

app = FastAPI(title="Dia2 Streaming Conversation Server")

# Configuration
MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"


# Global model instance
dia: Optional[Dia2] = None

# Conversation state directory
CONV_DIR = Path(tempfile.gettempdir()) / "yandia2_streaming"
CONV_DIR.mkdir(exist_ok=True)

# Transcription cache
_transcription_cache = {}


@dataclass
class ConversationState:
    """Tracks the current conversation state."""
    last_ai_audio: Optional[str] = None
    last_user_audio: Optional[str] = None
    turn_count: int = 0
    is_initialized: bool = False


# Global conversation state
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
        "streaming": True,
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
    
    for f in CONV_DIR.glob("*.wav"):
        f.unlink()
    
    conversation = ConversationState()
    
    # Clear graph cache for fresh start
    if dia is not None:
        dia.clear_graph_cache()
    
    print("[Dia2] Conversation reset")
    return {"status": "ok", "message": "Conversation reset"}


@app.post("/set_voice")
async def set_voice(file: UploadFile = File(...)):
    """Initialize the conversation with an AI voice warmup."""
    global conversation
    
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    voice_path = CONV_DIR / "ai_voice_warmup.wav"
    content = await file.read()
    voice_path.write_bytes(content)
    
    print(f"[Dia2] AI voice set: {voice_path} ({len(content)} bytes)")
    
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
    """Add user audio to the conversation."""
    global conversation
    
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not conversation.is_initialized:
        raise HTTPException(status_code=400, detail="Conversation not initialized. Call /set_voice first.")
    
    user_path = CONV_DIR / f"user_turn_{conversation.turn_count}.wav"
    content = await file.read()
    user_path.write_bytes(content)
    
    print(f"[Dia2] User audio received: {user_path} ({len(content)} bytes)")
    
    conversation.last_user_audio = str(user_path)
    
    # Pre-transcribe
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


async def run_generation_with_dia_generate(
    dia_instance,
    text: str,
    last_ai_audio: Optional[str],
    last_user_audio: Optional[str],
    config: GenerationConfig,
) -> AsyncGenerator[Tuple[bytes, bool], None]:
    """
    DIAGNOSTIC VERSION 2: Uses dia.generate() directly (the known working method),
    then streams the result through websocket.
    
    If this works -> the issue is in my prefix_plan setup
    If this fails -> something else is wrong
    """
    print(f"[DiagGen] Using dia.generate() directly")
    print(f"[DiagGen] prefix_speaker_1={last_ai_audio}")
    print(f"[DiagGen] prefix_speaker_2={last_user_audio}")
    
    gen_start = time.time()
    
    # Use the EXACT same call as the working /generate endpoint
    result = dia_instance.generate(
        text,
        config=config,
        prefix_speaker_1=last_ai_audio,
        prefix_speaker_2=last_user_audio,
        verbose=True,
    )
    
    gen_time = time.time() - gen_start
    duration = result.waveform.shape[-1] / result.sample_rate
    print(f"[DiagGen] Generation done in {gen_time:.2f}s, audio={duration:.2f}s")
    
    # Convert to PCM and yield
    waveform = result.waveform
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    waveform = torch.clamp(waveform, -1.0, 1.0)
    pcm16 = (waveform.detach().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
    
    yield (pcm16, True)


async def run_generation_with_original_loop(
    runtime,
    text: str,
    prefix_plan: Optional[PrefixPlan],
    config: GenerationConfig,
    graph_cache: Optional[CachedGraphState],
) -> AsyncGenerator[Tuple[bytes, bool], None]:
    """
    DIAGNOSTIC VERSION: Uses the original run_generation_loop from generator.py,
    then decodes and streams the audio AFTER generation completes.
    
    This tests whether the issue is in my custom generation loop or elsewhere.
    """
    from dia2.runtime.logger import RuntimeLogger
    
    token_ids = runtime.constants
    
    # Build entries from prefix + new text (same as engine.py)
    entries = []
    if prefix_plan is not None:
        entries.extend(prefix_plan.entries)
    entries.extend(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
    
    runtime.machine.initial_padding = config.initial_padding
    
    # Create FRESH state machine
    state = runtime.machine.new_state(entries)
    
    # Setup generation state (same as engine.py)
    if graph_cache is not None:
        reset_graph_cache(graph_cache, runtime, prefix_plan)
        gen_state = graph_cache.generation
    else:
        gen_state = build_initial_state(runtime, prefix=prefix_plan)
    
    start_step = 0
    if prefix_plan is not None:
        print(f"[DiagLoop] Warming up with prefix ({prefix_plan.aligned_frames} frames)...")
        start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
        print(f"[DiagLoop] Warmup done, start_step={start_step}")
    
    logger = RuntimeLogger(True)  # verbose=True as positional arg
    
    # Use the ORIGINAL run_generation_loop from generator.py
    print(f"[DiagLoop] Running ORIGINAL generation loop...")
    gen_start = time.time()
    
    first_word_frame, audio_buf = run_generation_loop(
        runtime,
        state=state,
        generation=gen_state,
        config=config,
        start_step=start_step,
        logger=logger,
        graph_cache=graph_cache,
    )
    
    gen_time = time.time() - gen_start
    print(f"[DiagLoop] Generation done in {gen_time:.2f}s, first_word_frame={first_word_frame}")
    
    # Decode using the SAME logic as engine.py
    include_prefix_audio = False  # Don't include prefix
    aligned = undelay_frames(audio_buf[0], runtime.audio_delays, runtime.constants.audio_pad).unsqueeze(0)
    crop = 0 if include_prefix_audio else max(first_word_frame, 0)
    if crop > 0 and crop < aligned.shape[-1]:
        aligned = aligned[:, :, crop:]
    
    print(f"[DiagLoop] Decoding {aligned.shape[-1]} frames (cropped {crop} prefix frames)...")
    waveform = decode_audio(runtime, aligned)
    
    duration = waveform.shape[-1] / runtime.mimi.sample_rate
    print(f"[DiagLoop] Decoded {duration:.2f}s audio")
    
    # Convert to PCM and yield as single chunk
    waveform = torch.clamp(waveform, -1.0, 1.0)
    pcm16 = (waveform.detach().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
    
    yield (pcm16, True)


async def run_streaming_generation(
    runtime,
    text: str,
    prefix_plan: Optional[PrefixPlan],
    config: GenerationConfig,
    graph_cache: Optional[CachedGraphState],
) -> AsyncGenerator[Tuple[bytes, bool], None]:
    """
    Generator that yields (audio_bytes, is_final) tuples as audio is generated.
    
    This is the core streaming logic - it runs the generation loop and yields
    audio chunks incrementally instead of waiting for full generation.
    """
    # Streaming config
    CHUNK_FRAMES = 6  # Decode every N frames after undelaying
    
    token_ids = runtime.constants
    
    # Build entries from prefix + new text
    entries = []
    if prefix_plan is not None:
        entries.extend(prefix_plan.entries)
    entries.extend(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
    
    runtime.machine.initial_padding = config.initial_padding
    
    # Create FRESH state machine - critical!
    state = runtime.machine.new_state(entries)
    
    # Setup generation state
    if graph_cache is not None:
        reset_graph_cache(graph_cache, runtime, prefix_plan)
        gen_state = graph_cache.generation
    else:
        gen_state = build_initial_state(runtime, prefix=prefix_plan)
    
    # Warmup with prefix (builds KV cache) - this part can't be streamed
    start_step = 0
    if prefix_plan is not None:
        print(f"[Stream] Warming up with prefix ({prefix_plan.aligned_frames} frames)...")
        warmup_start = time.time()
        start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
        print(f"[Stream] Warmup done in {time.time() - warmup_start:.2f}s")
    
    # Note: We use regular decode() for each chunk instead of streaming decode
    # The HuggingFace MimiModel doesn't have decode_streaming, but chunk-by-chunk
    # decoding works fine since Mimi's decoder is relatively frame-independent
    
    # Get delay info for proper undelaying
    delays = runtime.audio_delays  # List of delays per codebook
    max_delay = int(runtime.audio_delay_tensor.max().item()) if runtime.audio_delay_tensor.numel() else 0
    
    # Setup tensors for generation loop
    step_tokens = gen_state.step_tokens
    audio_buf = gen_state.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    
    if graph_cache is not None:
        positions = graph_cache.positions
        main_tokens = graph_cache.main_tokens
        aux_tokens = graph_cache.aux_tokens
        buffers = graph_cache.buffers
        transformer_capture = graph_cache.transformer_capture
        dep_captures = graph_cache.dep_captures
    else:
        positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
        main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
        aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
        buffers = _allocate_network_buffers(runtime, branches)
        transformer_capture = None
        dep_captures = None
    
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    
    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)
    
    delay_tensor = runtime.audio_delay_tensor
    positions_view = positions.expand(branches, -1)
    
    # Streaming state
    # Start decoding from start_step - that's where new content begins after prefix warmup
    # (Don't wait for new_word token - it may not appear for all text entries)
    content_start = start_step + 1  # First frame of new content is start_step + 1
    decode_pos = 0  # Next position to decode (in undelayed space)
    eos_cutoff = None
    total_samples = 0
    
    print(f"[Stream] Starting generation loop from step {start_step}...")
    print(f"[Stream] max_delay={max_delay}, num_codebooks={len(delays)}")
    print(f"[Stream] content_start={content_start} (will decode from here)")
    gen_start = time.time()
    first_chunk_sent = False
    last_step = start_step - 1
    
    with torch.inference_mode():
        for offset in range(max_context):
            t = start_step + offset
            
            # Check for end of generation
            if eos_cutoff is not None and t >= eos_cutoff:
                break
            if t + 1 >= audio_buf.shape[-1]:
                break
            
            gen_state.reset_dep_cache()
            positions.fill_(t)
            _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)
            
            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad
            
            # Transformer step
            if use_graph:
                transformer_capture, dep_captures = _execute_transformer_graph(
                    runtime=runtime,
                    step_tokens=step_tokens,
                    positions_view=positions_view,
                    branches=branches,
                    generation=gen_state,
                    transformer_step=runtime.transformer_step,
                    buffers=buffers,
                    transformer_capture=transformer_capture,
                    dep_captures=dep_captures,
                )
                hidden_t = transformer_capture[1]
            else:
                hidden_t = _execute_transformer_step(
                    step_tokens, positions_view, gen_state, 
                    runtime.transformer_step, buffers
                )
            
            # Text sampling
            guided_text = apply_classifier_guidance(
                buffers.text, config.cfg_scale != 1.0, config.cfg_scale, config.cfg_filter_k
            )
            if guided_text.shape[0] > 1:
                guided_text = guided_text[:1]
            
            text_token = sample_token(
                guided_text, temp=config.text.temperature, top_k=config.text.top_k
            ).item()
            
            # State machine
            main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
            second_token = aux_token if aux_token != -1 else token_ids.pad
            
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = second_token
            
            # Audio sampling (CB0)
            guided_cb0 = apply_classifier_guidance(
                buffers.cb0, config.cfg_scale != 1.0, config.cfg_scale, config.cfg_filter_k
            )
            if guided_cb0.shape[0] > 1:
                guided_cb0 = guided_cb0[:1]
            masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
            codebook_token = sample_audio_logits(masked_cb0, config.audio.temperature, config.audio.top_k)
            audio_buf[:, 0, t + 1] = codebook_token
            
            # Depformer stages
            prev_audio = codebook_token.expand(branches)
            main_tokens.fill_(main_token)
            aux_tokens.fill_(second_token)
            
            for stage in range(runtime.model.depformer.num_depth):
                if use_graph and dep_captures is not None:
                    dep_captures[stage] = _execute_depformer_graph(
                        stage=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=gen_state,
                        depformer_step=runtime.depformer_step,
                        main_tokens=main_tokens,
                        aux_tokens=aux_tokens,
                        buffers=buffers,
                        capture=dep_captures[stage],
                    )
                else:
                    _execute_depformer_stage(
                        stage_index=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=gen_state,
                        depformer_step=runtime.depformer_step,
                        main_tokens=main_tokens,
                        second_tokens=aux_tokens,
                        buffers=buffers,
                    )
                
                dep_logits = apply_classifier_guidance(
                    buffers.dep[stage], config.cfg_scale != 1.0, config.cfg_scale, config.cfg_filter_k
                )
                if dep_logits.shape[0] > 1:
                    dep_logits = dep_logits[:1]
                stage_token = sample_audio_logits(
                    dep_logits, config.audio.temperature, config.audio.top_k
                )
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            
            # Check for EOS
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            
            last_step = t
            generation_pos = t + 1  # We just wrote to position t+1
            
            # Decode and yield audio chunks
            # We can safely undelay position p if we have data up to p + max_delay
            # So we can decode up to: generation_pos - max_delay
            # Start decoding from content_start (after prefix)
            if decode_pos < content_start:
                decode_pos = content_start
            
            decodable_end = generation_pos - max_delay
            frames_to_decode = decodable_end - decode_pos
                
            if frames_to_decode >= CHUNK_FRAMES:
                # Extract and undelay the frames
                num_codebooks = len(delays)
                chunk_tokens = torch.zeros(1, num_codebooks, frames_to_decode, dtype=torch.long, device=runtime.device)
                    
                for cb in range(num_codebooks):
                    d = int(delays[cb]) if hasattr(delays, '__getitem__') else int(runtime.audio_delay_tensor[cb].item())
                    chunk_tokens[0, cb, :] = audio_buf[0, cb, decode_pos + d : decodable_end + d]
                
                chunk_tokens = torch.clamp(chunk_tokens, 0, 2047)
                
                # Decode chunk
                pcm = runtime.mimi.decode(chunk_tokens)
                waveform = pcm[0, 0] if pcm.dim() > 2 else pcm.squeeze()
                
                if waveform.shape[0] > 0:
                    pcm16 = (waveform.detach().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
                    total_samples += waveform.shape[0]
                    
                    if not first_chunk_sent:
                        first_chunk_sent = True
                        print(f"[Stream] First chunk sent after {time.time() - gen_start:.2f}s")
                    
                    yield (pcm16, False)
                
                decode_pos = decodable_end
                
            # Yield to event loop periodically
            if offset % 3 == 0:
                await asyncio.sleep(0)
    
    # Flush remaining frames
    generation_pos = last_step + 2
    decodable_end = min(generation_pos - max_delay, audio_buf.shape[-1] - max_delay)
    
    if decode_pos < content_start:
        decode_pos = content_start
        
    if decodable_end > decode_pos:
        num_codebooks = len(delays)
        remaining_frames = decodable_end - decode_pos
        chunk_tokens = torch.zeros(1, num_codebooks, remaining_frames, dtype=torch.long, device=runtime.device)
        
        for cb in range(num_codebooks):  
            d = int(delays[cb]) if hasattr(delays, '__getitem__') else int(runtime.audio_delay_tensor[cb].item())
            chunk_tokens[0, cb, :] = audio_buf[0, cb, decode_pos + d : decodable_end + d]
        
        chunk_tokens = torch.clamp(chunk_tokens, 0, 2047)
        
        pcm = runtime.mimi.decode(chunk_tokens)
        waveform = pcm[0, 0] if pcm.dim() > 2 else pcm.squeeze()
        
        if waveform.shape[0] > 0:
            pcm16 = (waveform.detach().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
            total_samples += waveform.shape[0]
            yield (pcm16, True)
            
    
    # Update graph cache
    if graph_cache is not None:
        graph_cache.transformer_capture = transformer_capture
        graph_cache.dep_captures = dep_captures
    
    total_time = time.time() - gen_start
    duration = total_samples / runtime.mimi.sample_rate if total_samples > 0 else 0
    print(f"[Stream] Generation complete: {duration:.2f}s audio in {total_time:.2f}s (RTF: {total_time/max(duration, 0.01):.2f})")


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS generation.
    
    Protocol:
    1. Client sends: {"text": "Hello world", "cfg_scale": 1.0, "temperature": 0.8, "top_k": 50}
    2. Server streams binary audio chunks (16-bit PCM, 24kHz)
    3. Server sends: {"event": "done", "duration": 1.5, "sample_rate": 24000}
    
    Audio format: Each binary message is:
    - 1 byte: is_final (0 or 1)
    - Remaining bytes: 16-bit PCM samples
    """
    await websocket.accept()
    
    if dia is None:
        await websocket.send_text(json.dumps({"error": "Model not loaded"}))
        await websocket.close()
        return
    
    if not conversation.is_initialized:
        await websocket.send_text(json.dumps({"error": "Conversation not initialized. Call /set_voice first."}))
        await websocket.close()
        return
    
    runtime = dia._ensure_runtime()
    
    # Send ready message
    await websocket.send_text(json.dumps({
        "event": "ready",
        "sample_rate": dia.sample_rate,
        "channels": 1,
        "sample_width": 2,  # 16-bit
    }))
    
    # Config flag for diagnostic mode
    use_original_loop = False
    
    try:
        while True:
            # Wait for text input
            msg = await websocket.receive_text()
            data = json.loads(msg)
            
            if data.get("type") == "close":
                break
            
            if data.get("type") == "config":
                use_original_loop = data.get("use_original_loop", False)
                use_dia_generate = data.get("use_dia_generate", False)
                print(f"[WS] Config: use_original_loop={use_original_loop}, use_dia_generate={use_dia_generate}")
                if use_dia_generate:
                    use_original_loop = "dia_generate"  # Special flag
                continue
            
            text = data.get("text", "")
            if not text:
                await websocket.send_text(json.dumps({"error": "No text provided"}))
                continue
            
            # Prepare text with S1 tag
            if not text.strip().startswith("[S1]"):
                text = f"[S1] {text}"
            
            # Get config from request
            cfg_scale = data.get("cfg_scale", 1.0)
            temperature = data.get("temperature", 0.8)
            top_k = data.get("top_k", 50)
            
            config = GenerationConfig(
                cfg_scale=cfg_scale,
                audio=SamplingConfig(temperature=temperature, top_k=top_k),
                text=SamplingConfig(temperature=0.6, top_k=50),
                use_cuda_graph=True,
            )
            
            # Build prefix plan
            prefix_config = PrefixConfig(
                speaker_1=conversation.last_ai_audio,
                speaker_2=conversation.last_user_audio,
            )
            prefix_plan = build_prefix_plan(runtime, prefix_config)
            
            print(f"\n[WS] === Turn {conversation.turn_count} ===")
            print(f"[WS] Generating: {text}")
            print(f"[WS] AI prefix: {conversation.last_ai_audio}")
            print(f"[WS] User prefix: {conversation.last_user_audio}")
            
            # Send generating event
            await websocket.send_text(json.dumps({"event": "generating"}))
            
            total_start = time.time()
            
            # Get or create graph cache
            if dia._graph_cache is None:
                dia._graph_cache = create_graph_cache(runtime)
            
            # Stream audio chunks
            if use_original_loop == "dia_generate":
                print(f"[WS] Using dia.generate() directly (diagnostic mode 2)")
                generator = run_generation_with_dia_generate(
                    dia, text, 
                    conversation.last_ai_audio,
                    conversation.last_user_audio,
                    config
                )
            elif use_original_loop:
                print(f"[WS] Using ORIGINAL generation loop (diagnostic mode)")
                generator = run_generation_with_original_loop(
                    runtime, text, prefix_plan, config, dia._graph_cache
                )
            else:
                generator = run_streaming_generation(
                    runtime, text, prefix_plan, config, dia._graph_cache
                )
            
            audio_chunks = []
            async for chunk, is_final in generator:
                # Send binary chunk with header
                header = struct.pack("!?", is_final)
                await websocket.send_bytes(header + chunk)
                audio_chunks.append(chunk)
            
            total_time = time.time() - total_start
            
            # Combine chunks for saving
            if audio_chunks:
                all_audio = b"".join(audio_chunks)
                samples = len(all_audio) // 2  # 16-bit = 2 bytes per sample
                duration = samples / dia.sample_rate
                
                # Save as conversation state
                ai_output_path = CONV_DIR / f"ai_turn_{conversation.turn_count}.wav"
                _save_pcm_wav(all_audio, dia.sample_rate, str(ai_output_path))
                
                conversation.last_ai_audio = str(ai_output_path)
                conversation.turn_count += 1
                
                print(f"[WS] Saved AI audio: {ai_output_path}")
            else:
                duration = 0
            
            # Send completion event
            await websocket.send_text(json.dumps({
                "event": "done",
                "duration": duration,
                "total_time": total_time,
                "sample_rate": dia.sample_rate,
                "turn_count": conversation.turn_count,
            }))
            
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass


def _save_pcm_wav(pcm_bytes: bytes, sample_rate: int, path: str):
    """Save raw PCM bytes as a WAV file."""
    import wave
    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)


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


# Also keep the non-streaming generate endpoint for compatibility
@app.post("/generate")
async def generate(
    text: str = Form(...),
    cfg_scale: float = Form(1.0),
    temperature: float = Form(0.8),
    top_k: int = Form(50),
):
    """Non-streaming generate endpoint (same as conversation_server.py)."""
    global conversation
    
    if dia is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not conversation.is_initialized:
        raise HTTPException(status_code=400, detail="Conversation not initialized. Call /set_voice first.")
    
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
    
    start = time.time()
    
    try:
        result = dia.generate(
            text,
            config=config,
            prefix_speaker_1=conversation.last_ai_audio,
            prefix_speaker_2=conversation.last_user_audio,
            verbose=True,
        )
        
        elapsed = time.time() - start
        duration = result.waveform.shape[-1] / result.sample_rate
        print(f"[Dia2] Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/duration:.2f})")
        
        ai_output_path = CONV_DIR / f"ai_turn_{conversation.turn_count}.wav"
        wav_bytes = _tensor_to_wav(result.waveform, result.sample_rate)
        ai_output_path.write_bytes(wav_bytes)
        
        conversation.last_ai_audio = str(ai_output_path)
        conversation.turn_count += 1
        
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get the current conversation state."""
    return {
        "initialized": conversation.is_initialized,
        "turn_count": conversation.turn_count,
        "last_ai_audio": conversation.last_ai_audio,
        "last_user_audio": conversation.last_user_audio,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
