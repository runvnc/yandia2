"""Streaming Conversation WebSocket Handler

This module provides a WebSocket endpoint for streaming conversational TTS with:
1. Streaming audio input (process as user speaks)
2. External timestamps from Deepgram (bypass Whisper)
3. Incremental KV cache warmup
4. Low-latency response generation

The key optimization is processing user audio AS IT ARRIVES rather than
waiting for the user to finish speaking.
"""
import asyncio
import io
import json
import struct
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch

from dia2.runtime.voice_clone import (
    build_prefix_plan_with_timestamps,
    WhisperWord,
    words_to_entries,
    PrefixPlan,
)
from dia2.runtime.audio_io import encode_audio_tokens, load_mono_audio
from dia2.runtime.generator import (
    build_initial_state,
    warmup_with_prefix,
    GenerationState,
    CachedGraphState,
    create_graph_cache,
    reset_graph_cache,
)
from dia2.runtime.script_parser import parse_script
from dia2.audio.grid import delay_frames


@dataclass
class StreamingConversationState:
    """State for a streaming conversation session."""
    # AI voice (static across conversation)
    ai_voice_path: Optional[str] = None
    ai_voice_timestamps: Optional[List[dict]] = None
    ai_voice_tokens: Optional[torch.Tensor] = None  # Pre-encoded
    
    # Current user turn (streaming)
    user_audio_buffer: bytes = b""
    user_timestamps: List[dict] = field(default_factory=list)
    user_audio_tokens: Optional[torch.Tensor] = None
    
    # Incremental warmup state
    warmup_frames_processed: int = 0
    kv_cache_warmed: bool = False
    
    # Last AI response (for next turn's context)
    last_ai_audio_path: Optional[str] = None
    last_ai_timestamps: Optional[List[dict]] = None
    
    # Session info
    turn_count: int = 0
    is_initialized: bool = False


class StreamingConversationHandler:
    """Handles streaming conversation with incremental warmup."""
    
    def __init__(self, runtime, graph_cache: Optional[CachedGraphState] = None):
        self.runtime = runtime
        self.graph_cache = graph_cache or create_graph_cache(runtime)
        self.state = StreamingConversationState()
        self.temp_dir = Path(tempfile.gettempdir()) / "yandia2_stream"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Audio encoding state
        self._audio_frame_size = runtime.mimi.samples_per_frame  # Samples per Mimi frame
        self._sample_rate = runtime.mimi.sample_rate
        self._pending_samples = np.array([], dtype=np.float32)
    
    async def handle_init(self, data: dict) -> dict:
        """Initialize with AI voice."""
        ai_voice_path = data.get("ai_voice_path")
        ai_timestamps = data.get("ai_voice_timestamps", [])
        
        if not ai_voice_path:
            return {"event": "error", "message": "ai_voice_path required"}
        
        # Pre-encode AI voice audio
        try:
            audio = load_mono_audio(ai_voice_path, self._sample_rate)
            tokens = encode_audio_tokens(self.runtime.mimi, audio)
            
            self.state.ai_voice_path = ai_voice_path
            self.state.ai_voice_timestamps = ai_timestamps
            self.state.ai_voice_tokens = tokens.to(self.runtime.device)
            self.state.is_initialized = True
            
            print(f"[StreamConv] Initialized with AI voice: {ai_voice_path}")
            print(f"[StreamConv] AI voice: {tokens.shape[-1]} frames, {len(ai_timestamps)} words")
            
            return {
                "event": "initialized",
                "ai_frames": tokens.shape[-1],
                "ai_words": len(ai_timestamps),
            }
        except Exception as e:
            return {"event": "error", "message": str(e)}
    
    def handle_audio_chunk(self, audio_bytes: bytes) -> dict:
        """Process incoming audio chunk.
        
        Audio should be raw PCM 16-bit mono at 24kHz.
        """
        # Accumulate audio
        self.state.user_audio_buffer += audio_bytes
        
        # Convert to float32
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Add to pending samples
        self._pending_samples = np.concatenate([self._pending_samples, audio_float])
        
        # Encode complete frames
        frames_encoded = 0
        while len(self._pending_samples) >= self._audio_frame_size:
            frame_samples = self._pending_samples[:self._audio_frame_size]
            self._pending_samples = self._pending_samples[self._audio_frame_size:]
            
            # Encode this frame
            frame_tensor = torch.from_numpy(frame_samples).unsqueeze(0).unsqueeze(0)
            frame_tokens = encode_audio_tokens(self.runtime.mimi, frame_samples)
            
            # Append to user tokens
            if self.state.user_audio_tokens is None:
                self.state.user_audio_tokens = frame_tokens.to(self.runtime.device)
            else:
                self.state.user_audio_tokens = torch.cat(
                    [self.state.user_audio_tokens, frame_tokens.to(self.runtime.device)],
                    dim=1
                )
            frames_encoded += 1
        
        return {
            "event": "audio_received",
            "bytes": len(audio_bytes),
            "frames_encoded": frames_encoded,
            "total_frames": self.state.user_audio_tokens.shape[-1] if self.state.user_audio_tokens is not None else 0,
        }
    
    def handle_word_timestamp(self, data: dict) -> dict:
        """Process a word timestamp from Deepgram."""
        word = data.get("word", "")
        start = data.get("start", 0.0)
        end = data.get("end", 0.0)
        
        self.state.user_timestamps.append({
            "word": word,
            "start": start,
            "end": end,
        })
        
        return {
            "event": "word_received",
            "word": word,
            "total_words": len(self.state.user_timestamps),
        }
    
    async def handle_generate(self, data: dict, send_chunk) -> dict:
        """Generate AI response.
        
        Args:
            data: Contains "ai_text" - the text for AI to speak
            send_chunk: Async function to send audio chunks
        """
        ai_text = data.get("ai_text", "")
        if not ai_text:
            return {"event": "error", "message": "ai_text required"}
        
        if not self.state.is_initialized:
            return {"event": "error", "message": "Not initialized. Send init first."}
        
        gen_start = time.time()
        
        # Save user audio to temp file
        user_audio_path = None
        if self.state.user_audio_buffer:
            user_audio_path = self.temp_dir / f"user_turn_{self.state.turn_count}.wav"
            self._save_audio_buffer(self.state.user_audio_buffer, str(user_audio_path))
        
        # Build prefix plan with external timestamps
        # Use last AI audio if available, otherwise use initial AI voice
        ai_audio = self.state.last_ai_audio_path or self.state.ai_voice_path
        ai_timestamps = self.state.last_ai_timestamps or self.state.ai_voice_timestamps
        
        print(f"[StreamConv] Building prefix plan...")
        print(f"[StreamConv] AI audio: {ai_audio}, {len(ai_timestamps or [])} words")
        print(f"[StreamConv] User audio: {user_audio_path}, {len(self.state.user_timestamps)} words")
        
        prefix_plan = build_prefix_plan_with_timestamps(
            self.runtime,
            speaker_1_audio=ai_audio,
            speaker_1_timestamps=ai_timestamps or [],
            speaker_2_audio=str(user_audio_path) if user_audio_path else None,
            speaker_2_timestamps=self.state.user_timestamps if user_audio_path else None,
        )
        
        build_time = time.time() - gen_start
        print(f"[StreamConv] Prefix plan built in {build_time*1000:.1f}ms")
        
        # Import the streaming generation function
        from streaming_server import run_streaming_generation, ENABLE_PREFIX_CACHE
        from dia2 import GenerationConfig, SamplingConfig
        from dia2.runtime.prefix_cache import (
            PrefixCacheStore, compute_prefix_key, save_prefix_cache, restore_prefix_cache
        )
        
        config = GenerationConfig(
            cfg_scale=1.0,
            audio=SamplingConfig(temperature=0.8, top_k=50),
            text=SamplingConfig(temperature=0.6, top_k=50),
            use_cuda_graph=True,
        )
        
        # Prepare text
        if not ai_text.strip().startswith("[S1]"):
            ai_text = f"[S1] {ai_text}"
        
        # Generate and stream
        audio_chunks = []
        first_chunk_time = None
        
        async for chunk, is_final in run_streaming_generation(
            self.runtime,
            ai_text,
            ai_audio,
            str(user_audio_path) if user_audio_path else ai_audio,
            prefix_plan,
            config,
            self.graph_cache,
        ):
            if first_chunk_time is None:
                first_chunk_time = time.time()
                latency = first_chunk_time - gen_start
                print(f"[StreamConv] First chunk latency: {latency:.3f}s")
            
            # Send chunk
            header = struct.pack("!?", is_final)
            await send_chunk(header + chunk)
            audio_chunks.append(chunk)
        
        total_time = time.time() - gen_start
        
        # Save AI audio for next turn
        if audio_chunks:
            all_audio = b"".join(audio_chunks)
            ai_output_path = self.temp_dir / f"ai_turn_{self.state.turn_count}.wav"
            self._save_audio_buffer(all_audio, str(ai_output_path))
            self.state.last_ai_audio_path = str(ai_output_path)
            # TODO: Extract timestamps from generation result
            self.state.last_ai_timestamps = []  # Would need to get from generation
            
            duration = len(all_audio) / 2 / self._sample_rate  # 16-bit = 2 bytes
        else:
            duration = 0
        
        # Reset user state for next turn
        self.state.user_audio_buffer = b""
        self.state.user_timestamps = []
        self.state.user_audio_tokens = None
        self._pending_samples = np.array([], dtype=np.float32)
        self.state.turn_count += 1
        
        return {
            "event": "done",
            "duration": duration,
            "total_time": total_time,
            "first_chunk_latency": first_chunk_time - gen_start if first_chunk_time else None,
            "turn_count": self.state.turn_count,
        }
    
    def handle_reset(self) -> dict:
        """Reset conversation state."""
        # Keep AI voice, reset everything else
        self.state.user_audio_buffer = b""
        self.state.user_timestamps = []
        self.state.user_audio_tokens = None
        self.state.last_ai_audio_path = None
        self.state.last_ai_timestamps = None
        self.state.turn_count = 0
        self._pending_samples = np.array([], dtype=np.float32)
        
        return {"event": "reset", "message": "Conversation reset"}
    
    def _save_audio_buffer(self, audio_bytes: bytes, path: str):
        """Save raw PCM bytes as WAV file."""
        with wave.open(path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(audio_bytes)


async def create_conversation_websocket_handler(app, dia_instance):
    """Create and register the streaming conversation WebSocket endpoint."""
    from fastapi import WebSocket, WebSocketDisconnect
    
    @app.websocket("/ws/conversation_stream")
    async def websocket_conversation_stream(websocket: WebSocket):
        await websocket.accept()
        
        runtime = dia_instance._ensure_runtime()
        handler = StreamingConversationHandler(runtime, dia_instance._graph_cache)
        
        # Send ready message
        await websocket.send_text(json.dumps({
            "event": "ready",
            "sample_rate": runtime.mimi.sample_rate,
            "channels": 1,
            "sample_width": 2,
        }))
        
        try:
            while True:
                msg = await websocket.receive()
                
                if msg["type"] == "websocket.disconnect":
                    break
                
                if "bytes" in msg:
                    # Binary audio data
                    result = handler.handle_audio_chunk(msg["bytes"])
                    # Don't send ack for every chunk to reduce overhead
                    
                elif "text" in msg:
                    data = json.loads(msg["text"])
                    msg_type = data.get("type", "")
                    
                    if msg_type == "init":
                        result = await handler.handle_init(data)
                        await websocket.send_text(json.dumps(result))
                        
                    elif msg_type == "word":
                        result = handler.handle_word_timestamp(data)
                        # Don't send ack for every word
                        
                    elif msg_type == "generate":
                        async def send_chunk(chunk_bytes):
                            await websocket.send_bytes(chunk_bytes)
                        
                        result = await handler.handle_generate(data, send_chunk)
                        await websocket.send_text(json.dumps(result))
                        
                    elif msg_type == "reset":
                        result = handler.handle_reset()
                        await websocket.send_text(json.dumps(result))
                        
                    elif msg_type == "close":
                        break
                        
        except WebSocketDisconnect:
            print("[StreamConv] Client disconnected")
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                await websocket.send_text(json.dumps({"event": "error", "message": str(e)}))
            except:
                pass
