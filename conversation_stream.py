"""Streaming Conversation WebSocket Handler with Incremental Warmup

This module provides a WebSocket endpoint for streaming conversational TTS with:
1. Streaming audio input (process as user speaks)
2. External timestamps from Deepgram (bypass Whisper)
3. Incremental KV cache warmup
4. Low-latency response generation

The key optimization is processing user audio AS IT ARRIVES rather than
waiting for the user to finish speaking.

INCREMENTAL WARMUP: As audio chunks arrive, we encode them and run transformer
warmup steps immediately. By the time generate is called, KV cache is pre-warmed.
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
from typing import Optional, List, Dict, Any

import numpy as np
import torch

from dia2.runtime.voice_clone import (
    WhisperWord,
    words_to_entries,
    build_prefix_plan_with_timestamps,
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
from dia2.audio.grid import delay_frames, undelay_frames
from dia2.core.cache import KVCacheSnapshot


class IncrementalWarmup:
    """Manages incremental KV cache warmup as audio streams in.
    
    Instead of waiting for all audio before running warmup, this processes
    audio frames through the transformer as they arrive. By the time
    generation is triggered, the KV cache is already warmed.
    """
    
    def __init__(self, runtime, ai_voice_tokens: torch.Tensor, ai_voice_word_steps: List[int]):
        """Initialize incremental warmup manager.
        
        Args:
            runtime: The Dia2 runtime context
            ai_voice_tokens: Pre-encoded AI voice audio tokens [num_codebooks, frames]
            ai_voice_word_steps: Frame indices where words start in AI voice
        """
        self.runtime = runtime
        self.device = runtime.device
        self.token_ids = runtime.constants
        self.branches = 2  # CFG branches
        
        # AI voice (static, pre-encoded)
        self.ai_tokens = ai_voice_tokens.to(self.device)
        self.ai_frames = ai_voice_tokens.shape[-1]
        self.ai_word_steps = ai_voice_word_steps
        
        # User audio (streaming)
        self.user_tokens: Optional[torch.Tensor] = None
        self.user_frames = 0
        self.user_word_steps: List[int] = []
        
        # Warmup state - separate from graph cache
        self.warmup_state: Optional[GenerationState] = None
        self.frames_warmed = 0
        self.ai_warmup_done = False
        
        # Step tokens for warmup (separate from generation)
        self.step_tokens: Optional[torch.Tensor] = None
        
        # Initialize warmup state
        self._init_warmup_state()
    
    def _init_warmup_state(self):
        """Initialize the warmup generation state."""
        dep_q = self.runtime.model.depformer.num_audio_channels
        channels = 2 + dep_q  # main_token, second_token, + audio channels
        
        # Allocate step_tokens for warmup
        self.step_tokens = torch.full(
            (self.branches, channels, 1),
            self.token_ids.pad,
            dtype=torch.long,
            device=self.device,
        )
        self.step_tokens[0, 0, 0] = self.token_ids.bos
        self.step_tokens[0, 1, 0] = self.token_ids.pad
        self.step_tokens[1, 0, 0] = self.token_ids.zero
        self.step_tokens[1, 1, 0] = self.token_ids.pad
        
        # Build initial generation state (allocates KV cache, audio buffer)
        # We pass prefix=None because we'll fill the audio buffer incrementally
        self.warmup_state = build_initial_state(self.runtime, prefix=None) 
        self.frames_warmed = 0
        self.ai_warmup_done = False
    
    def warmup_ai_voice(self):
        """Warm up with AI voice (call once at init)."""
        if self.ai_warmup_done:
            return
        
        print(f"[IncrWarmup] Warming AI voice: {self.ai_frames} frames")
        start = time.time()
        
        # Copy AI tokens into audio buffer
        delayed = delay_frames(
            self.ai_tokens,
            self.runtime.audio_delays,
            self.runtime.constants.audio_pad
        ).to(self.device)
        self.warmup_state.audio_buf[0, :, :delayed.shape[1]] = delayed
        
        # Run warmup for AI frames
        self._warmup_frames(0, self.ai_frames, self.ai_word_steps)
        
        self.ai_warmup_done = True
        self.frames_warmed = self.ai_frames
        
        print(f"[IncrWarmup] AI voice warmed in {(time.time()-start)*1000:.1f}ms")
        return self.frames_warmed
    
    def add_user_tokens(self, tokens: torch.Tensor, word_steps: List[int] = None):
        """Add encoded user audio tokens and optionally warm them.
        
        Args:
            tokens: Audio tokens [num_codebooks, frames]
            word_steps: Frame indices where words start (relative to user audio start)
        """
        tokens = tokens.to(self.device)
        
        if self.user_tokens is None:
            self.user_tokens = tokens
        else:
            self.user_tokens = torch.cat([self.user_tokens, tokens], dim=1)
        
        old_frames = self.user_frames
        self.user_frames = self.user_tokens.shape[-1]
        
        # Add word steps (offset by AI frames)
        if word_steps:
            for step in word_steps:
                self.user_word_steps.append(step + self.ai_frames)
        
        # Copy new tokens into audio buffer
        delayed = delay_frames(
            tokens,
            self.runtime.audio_delays,
            self.runtime.constants.audio_pad
        ).to(self.device)
        
        buf_start = self.ai_frames + old_frames
        self.warmup_state.audio_buf[0, :, buf_start:buf_start + delayed.shape[1]] = delayed
    
    def warmup_new_frames(self):
        """Warm up any new frames that haven't been processed yet."""
        total_frames = self.ai_frames + self.user_frames
        
        if self.frames_warmed >= total_frames:
            return 0  # Nothing new to warm
        
        # Get word steps for the frames we're about to process
        word_steps_in_range = [
            s for s in self.user_word_steps 
            if self.frames_warmed <= s < total_frames
        ]
        
        frames_to_warm = total_frames - self.frames_warmed
        start = time.time()
        
        self._warmup_frames(self.frames_warmed, total_frames, word_steps_in_range)
        
        self.frames_warmed = total_frames
        elapsed = (time.time() - start) * 1000
        
        if frames_to_warm > 0:
            print(f"[IncrWarmup] Warmed {frames_to_warm} frames in {elapsed:.1f}ms")
        
        return frames_to_warm

    def _warmup_frames(self, start_frame: int, end_frame: int, word_steps: List[int]):
        """Run transformer warmup for a range of frames.
        
        This processes frames through the transformer to build up the KV cache,
        similar to warmup_with_prefix but operating on a frame range.
        
        Args:
            start_frame: First frame to process (inclusive)
            end_frame: Last frame to process (exclusive)
            word_steps: Frame indices where new words start (for state machine)
        """
        if self.warmup_state is None or self.step_tokens is None:
            raise RuntimeError("Warmup state not initialized")
        
        model_state = self.warmup_state.decode
        audio_buf = self.warmup_state.audio_buf
        positions = torch.empty(1, 1, dtype=torch.long, device=self.device)
        word_steps_set = set(word_steps)
        
        num_codebooks = audio_buf.shape[1]
        delays = self.runtime.audio_delays
        
        with torch.inference_mode():
            for t in range(start_frame, end_frame):
                positions.fill_(t)
                
                # Fill audio channels from buffer (with delay compensation)
                for cb in range(num_codebooks):
                    delay = delays[cb] if cb < len(delays) else 0
                    idx = t - delay
                    if idx >= 0 and idx < audio_buf.shape[-1]:
                        value = audio_buf[0, cb, idx]
                    else:
                        value = self.token_ids.audio_bos
                    self.step_tokens[:, 2 + cb, 0] = value
                
                # Run transformer step (builds KV cache)
                hidden, text_logits, cb0_logits, present = self.runtime.model.transformer.forward_step(
                    self.step_tokens,
                    positions.expand(self.branches, -1),
                    model_state.transformer,
                )
                model_state.transformer = present
                
                # Update step tokens for next iteration
                # During warmup, we use forced tokens (new_word at word boundaries, pad otherwise)
                forced = self.token_ids.new_word if t in word_steps_set else self.token_ids.pad
                self.step_tokens[0, 0, 0] = forced
                self.step_tokens[0, 1, 0] = self.token_ids.pad
                if self.branches > 1:
                    self.step_tokens[1:, 0, 0] = self.token_ids.zero
                    self.step_tokens[1:, 1, 0] = self.token_ids.pad
    
    def get_kv_snapshot(self) -> KVCacheSnapshot:
        """Get a snapshot of the current KV cache state."""
        if self.warmup_state is None:
            raise RuntimeError("Warmup state not initialized")
        return self.warmup_state.transformer_cache.snapshot()
    
    def transfer_to_generation(self, gen_state: GenerationState) -> int:
        """Transfer warmed KV cache to a generation state.
        
        Args:
            gen_state: The generation state to transfer to
            
        Returns:
            The start_step for generation (frames_warmed - 1)
        """
        snapshot = self.get_kv_snapshot()
        gen_state.transformer_cache.restore(snapshot)
        return max(self.frames_warmed - 1, 0)


async def run_generation_with_prewarmed_cache(  
    runtime,
    text: str,
    incremental_warmup: IncrementalWarmup,
    config,
    graph_cache: Optional[CachedGraphState],
):
    """Run generation using a pre-warmed KV cache from incremental warmup.
    
    
    This is the key optimization - instead of running warmup at generation time,
    we use the KV cache that was built incrementally as audio arrived.
    
    Args:
        runtime: Dia2 runtime context
        text: Text to generate (should include [S1] tag)
        incremental_warmup: The warmup manager with pre-warmed KV cache
        config: Generation config
        graph_cache: Optional CUDA graph cache for generation loop
    
    Yields:
        (audio_bytes, is_final) tuples
    """
    import asyncio
    from dia2.runtime.generator import (
        _allocate_network_buffers,
        _ensure_graph_cublas_ready,
        _fill_audio_channels,
        _execute_transformer_step,
        _execute_transformer_graph,
        _execute_depformer_stage,
        _execute_depformer_graph,
    )
    from dia2.runtime.guidance import apply_classifier_guidance, sample_audio_logits
    from dia2.runtime.sampler import sample_token
    from dia2.audio.grid import mask_audio_logits
    
    gen_start = time.time()
    
    # Parse text entries
    text_entries = parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate)
    runtime.machine.initial_padding = config.initial_padding
    
    # Create fresh state machine with text entries only
    # (prefix was already processed during incremental warmup)
    from dia2.runtime.state_machine import State
    state = runtime.machine.new_state(text_entries)
    
    # Setup generation state
    if graph_cache is not None:
        # Reset graph cache but DON'T run warmup - we'll restore from incremental warmup
        reset_graph_cache(graph_cache, runtime, prefix=None)
        gen_state = graph_cache.generation
    else:
        gen_state = build_initial_state(runtime, prefix=None)
    
    # Transfer pre-warmed KV cache from incremental warmup
    transfer_start = time.time()
    start_step = incremental_warmup.transfer_to_generation(gen_state)
    
    # Also copy the audio buffer from warmup state
    warmup_audio = incremental_warmup.warmup_state.audio_buf
    frames_to_copy = incremental_warmup.frames_warmed + 1
    gen_state.audio_buf[:, :, :frames_to_copy].copy_(
        warmup_audio[:, :, :frames_to_copy]
    )
    
    # CRITICAL: Replace any remaining 'ungenerated' tokens (-2) with audio_pad
    # This prevents CUDA errors from invalid token IDs in embedding lookups
    ungenerated_mask = gen_state.audio_buf == token_ids.ungenerated
    gen_state.audio_buf[ungenerated_mask] = token_ids.audio_pad
    
    # Also ensure the warmup portion doesn't have ungenerated tokens
    # (it shouldn't, but just in case)
    
    transfer_time = (time.time() - transfer_start) * 1000
    print(f"[IncrGen] KV cache transferred in {transfer_time:.1f}ms, start_step={start_step}")
    print(f"[IncrGen] Skipped warmup of {incremental_warmup.frames_warmed} frames!")
    
    # Now run the generation loop (same as run_streaming_generation but starting from warmed state)
    # Import the streaming generation internals
    from streaming_server import run_streaming_generation
    
    # For now, we'll create a minimal prefix plan that just has the audio tokens
    # but no entries (since state machine already processed them)
    # This lets us reuse run_streaming_generation's loop
    
    # Actually, let's just inline the generation loop here for full control
    token_ids = runtime.constants
    step_tokens = gen_state.step_tokens
    audio_buf = gen_state.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    
    # Get delay info
    delays = runtime.audio_delays
    max_delay = int(runtime.audio_delay_tensor.max().item()) if runtime.audio_delay_tensor.numel() else 0
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    
    # Streaming state
    content_start = start_step + 1
    decode_pos = 0
    eos_cutoff = None
    total_samples = 0
    first_chunk_sent = False
    last_step = start_step - 1
    
    CHUNK_FRAMES = 1
    
    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)
    
    # Setup tensors
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
    
    delay_tensor = runtime.audio_delay_tensor
    positions_view = positions.expand(branches, -1)
    
    print(f"[IncrGen] Time to generation start: {time.time() - gen_start:.4f}s")
    
    loop_start = time.perf_counter()
    
    # Enable Mimi streaming mode
    with torch.inference_mode(), runtime.mimi.streaming(batch_size=1):
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
            cfg_active = config.cfg_scale != 1.0
            guided_text = apply_classifier_guidance(
                buffers.text, cfg_active, config.cfg_scale, config.cfg_filter_k
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
                buffers.cb0, cfg_active, config.cfg_scale, config.cfg_filter_k
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
                    buffers.dep[stage], cfg_active, config.cfg_scale, config.cfg_filter_k
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
            generation_pos = t + 1
            
            # Decode and yield audio chunks
            if decode_pos < content_start:
                decode_pos = content_start
            
            decodable_end = generation_pos - max_delay
            frames_to_decode = decodable_end - decode_pos
            
            if frames_to_decode >= CHUNK_FRAMES:
                num_codebooks = len(delays)
                chunk_tokens = torch.zeros(1, num_codebooks, frames_to_decode, dtype=torch.long, device=runtime.device)
                
                for cb in range(num_codebooks):
                    d = int(delays[cb]) if hasattr(delays, '__getitem__') else int(runtime.audio_delay_tensor[cb].item())
                    chunk_tokens[0, cb, :] = audio_buf[0, cb, decode_pos + d : decodable_end + d]
                
                chunk_tokens = torch.clamp(chunk_tokens, 0, 2047)
                
                pcm = runtime.mimi.decode(chunk_tokens)
                waveform = pcm[0, 0] if pcm.dim() > 2 else pcm.squeeze()
                
                if waveform.shape[0] > 0:
                    pcm16 = (waveform.detach().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
                    total_samples += waveform.shape[0]
                    
                    if not first_chunk_sent:
                        first_chunk_sent = True
                        first_chunk_time = time.perf_counter() - loop_start
                        total_time_to_first = time.time() - gen_start
                        print(f"[IncrGen] First chunk: {first_chunk_time:.3f}s (total: {total_time_to_first:.3f}s)")
                    
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
    
    total_time = time.perf_counter() - loop_start
    duration = total_samples / runtime.mimi.sample_rate if total_samples > 0 else 0
    print(f"[IncrGen] Generation complete: {duration:.2f}s audio in {total_time:.2f}s (RTF: {total_time/max(duration, 0.01):.2f})")


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
    
    # Generated audio from last turn (for extracting timestamps)
    last_generated_audio: Optional[bytes] = None
    last_generated_timestamps: Optional[List[dict]] = None
    
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
        
        # Incremental warmup manager (created on init)
        self._incremental_warmup: Optional[IncrementalWarmup] = None
        self._warmup_enabled = True  # Can be disabled for comparison
        
        # Track if we need to reinitialize warmup for next turn
        self._needs_warmup_reinit = False
        self._pending_ai_audio_for_warmup: Optional[str] = None
    
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
            
            # Convert timestamps to word steps for incremental warmup
            ai_word_steps = []
            if ai_timestamps:
                frame_rate = self.runtime.frame_rate
                for ts in ai_timestamps:
                    step = int(round(ts["start"] * frame_rate))
                    ai_word_steps.append(step)
            
            # Initialize incremental warmup with AI voice
            if self._warmup_enabled:
                self._incremental_warmup = IncrementalWarmup(
                    self.runtime,
                    tokens,
                    ai_word_steps,
                )
                # Pre-warm AI voice (this happens once)
                warmup_start = time.time()
                self._incremental_warmup.warmup_ai_voice()
                warmup_time = (time.time() - warmup_start) * 1000
                print(f"[StreamConv] AI voice pre-warmed in {warmup_time:.1f}ms")
            
            print(f"[StreamConv] Initialized with AI voice: {ai_voice_path}")
            print(f"[StreamConv] AI voice: {tokens.shape[-1]} frames, {len(ai_timestamps)} words")
            
            return {
                "event": "initialized",
                "ai_frames": tokens.shape[-1],
                "ai_words": len(ai_timestamps),
                "warmup_enabled": self._warmup_enabled,
            }
        except Exception as e:
            return {"event": "error", "message": str(e)}
    
    def _reinit_warmup_for_new_turn(self):
        """Reinitialize incremental warmup with the last AI audio for a new turn.
        
        This is called at the start of a new turn (when user starts speaking)
        to set up warmup with the previous AI response as the prefix.
        """
        if not self._needs_warmup_reinit:
            return
        
        ai_audio_path = self._pending_ai_audio_for_warmup or self.state.last_ai_audio_path
        ai_timestamps = self.state.last_ai_timestamps or []
        
        if not ai_audio_path:
            print(f"[StreamConv] No AI audio for warmup reinit, using initial voice")
            ai_audio_path = self.state.ai_voice_path
            ai_timestamps = self.state.ai_voice_timestamps or []
        
        if not ai_audio_path:
            print(f"[StreamConv] WARNING: No audio available for warmup reinit")
            self._needs_warmup_reinit = False
            return
        
        print(f"[StreamConv] Reinitializing warmup for turn {self.state.turn_count + 1}")
        print(f"[StreamConv] Using AI audio: {ai_audio_path}")
        
        try:
            # Load and encode the AI audio
            audio = load_mono_audio(ai_audio_path, self._sample_rate)
            tokens = encode_audio_tokens(self.runtime.mimi, audio)
            
            # Convert timestamps to word steps
            ai_word_steps = []
            if ai_timestamps:
                frame_rate = self.runtime.frame_rate
                for ts in ai_timestamps:
                    step = int(round(ts["start"] * frame_rate))
                    ai_word_steps.append(step)
            
            # Create new incremental warmup
            self._incremental_warmup = IncrementalWarmup(
                self.runtime,
                tokens,
                ai_word_steps,
            )
            
            # Pre-warm the AI audio
            warmup_start = time.time()
            self._incremental_warmup.warmup_ai_voice()
            warmup_time = (time.time() - warmup_start) * 1000
            print(f"[StreamConv] AI audio pre-warmed in {warmup_time:.1f}ms ({tokens.shape[-1]} frames)")
            
            self._needs_warmup_reinit = False
            self._pending_ai_audio_for_warmup = None
            
        except Exception as e:
            print(f"[StreamConv] ERROR reinitializing warmup: {e}")
            import traceback
            traceback.print_exc()
            self._needs_warmup_reinit = False
    
    def handle_audio_chunk(self, audio_bytes: bytes) -> dict:
        """Process incoming audio chunk.
        
        Audio should be raw PCM 16-bit mono at 24kHz.
        """
        # Accumulate audio
        self.state.user_audio_buffer += audio_bytes
        
        # Check if we need to reinitialize warmup for this turn
        if self._needs_warmup_reinit:
            self._reinit_warmup_for_new_turn()
        
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
        
        # If incremental warmup is enabled, add tokens and warm immediately
        if self._incremental_warmup is not None and frames_encoded > 0:
            # Get word steps for these frames (from timestamps received so far)
            # Note: In real usage, word timestamps arrive via handle_word_timestamp
            # and are converted to frame steps there
            word_steps = self._get_pending_word_steps()
            
            # Add tokens to warmup manager
            self._incremental_warmup.add_user_tokens(self.state.user_audio_tokens, word_steps)
            # Warm the new frames immediately
            self._incremental_warmup.warmup_new_frames()
        
        return {
            "event": "audio_received",
            "bytes": len(audio_bytes),
            "frames_encoded": frames_encoded,
            "total_frames": self.state.user_audio_tokens.shape[-1] if self.state.user_audio_tokens is not None else 0,
        }
    
    def _get_pending_word_steps(self) -> List[int]:
        """Convert pending word timestamps to frame steps."""
        if not self.state.user_timestamps:
            return []
        
        frame_rate = self.runtime.frame_rate
        word_steps = []
        for ts in self.state.user_timestamps:
            step = int(round(ts["start"] * frame_rate))
            word_steps.append(step)
        
        # Clear processed timestamps
        # (In a more sophisticated implementation, we'd track which are processed)
        return word_steps
    
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
        has_user_audio = bool(self.state.user_audio_buffer)
        
        if self.state.user_audio_buffer:
            user_audio_path = self.temp_dir / f"user_turn_{self.state.turn_count}.wav"
            self._save_audio_buffer(self.state.user_audio_buffer, str(user_audio_path))
        
        # Build prefix plan with external timestamps
        # Use last AI audio if available, otherwise use initial AI voice
        ai_audio = self.state.last_ai_audio_path or self.state.ai_voice_path
        ai_timestamps = self.state.last_ai_timestamps or self.state.ai_voice_timestamps
        
        # Try to use incremental warmup path
        # Works when we have a pre-warmed cache (either from init or from previous turn)
        use_incremental = (
            self._incremental_warmup is not None 
            and self._incremental_warmup.frames_warmed > 0
            and not self._needs_warmup_reinit  # Make sure warmup is current
        )
        
        if use_incremental:
            print(f"[StreamConv] Using INCREMENTAL warmup path!")
            print(f"[StreamConv] Pre-warmed frames: {self._incremental_warmup.frames_warmed}")
            
            from dia2 import GenerationConfig, SamplingConfig
            config = GenerationConfig(
                cfg_scale=1.0,
                audio=SamplingConfig(temperature=0.8, top_k=50),
                text=SamplingConfig(temperature=0.6, top_k=50),
                use_cuda_graph=True,
            )
            
            if not ai_text.strip().startswith("[S1]"):
                ai_text = f"[S1] {ai_text}"
            
            # Use the incremental warmup generation path!
            audio_chunks = []
            first_chunk_time = None
            
            async for chunk, is_final in run_generation_with_prewarmed_cache(
                self.runtime,
                ai_text,
                self._incremental_warmup,
                config,
                self.graph_cache,
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - gen_start
                    print(f"[StreamConv] First chunk latency (incremental): {latency:.3f}s")
                
                header = struct.pack("!?", is_final)
                await send_chunk(header + chunk)
                audio_chunks.append(chunk)
            
            total_time = time.time() - gen_start
            
            if audio_chunks:
                all_audio = b"".join(audio_chunks)
                duration = len(all_audio) / 2 / self._sample_rate
            else:
                duration = 0
            
            # Save AI audio for next turn and schedule warmup reinit
            if audio_chunks:
                all_audio = b"".join(audio_chunks)
                ai_output_path = self.temp_dir / f"ai_turn_{self.state.turn_count}.wav"
                self._save_audio_buffer(all_audio, str(ai_output_path))
                self.state.last_ai_audio_path = str(ai_output_path)
                # TODO: Extract actual timestamps from generation
                self.state.last_ai_timestamps = []  
                
                # Schedule warmup reinit for next turn
                self._pending_ai_audio_for_warmup = str(ai_output_path)
                self._needs_warmup_reinit = True
            
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
                "used_incremental_warmup": True,
                "has_user_audio": has_user_audio,
            }
        
        # Standard path with full warmup
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
            
            # Schedule warmup reinit for next turn
            self._pending_ai_audio_for_warmup = str(ai_output_path)
            self._needs_warmup_reinit = True
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
            "used_incremental_warmup": False,
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
        self._needs_warmup_reinit = False
        self._pending_ai_audio_for_warmup = None
        
        # Reinitialize warmup with original AI voice
        if self.state.ai_voice_tokens is not None and self._warmup_enabled:
            # Will be reinitialized on next audio chunk or can force it
            self._pending_ai_audio_for_warmup = self.state.ai_voice_path
            self._needs_warmup_reinit = True
        
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
