from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .assets import resolve_assets
from .runtime.context import RuntimeContext, build_runtime
from .runtime.generator import (
    build_initial_state,
    decode_audio,
    run_generation_loop,
    warmup_with_prefix,
    # NEW: Graph caching support
    CachedGraphState,
    create_graph_cache,
    reset_graph_cache,
)
from .runtime.script_parser import parse_script
from .audio.grid import undelay_frames, write_wav
from .runtime.voice_clone import build_prefix_plan
from .generation import (
    GenerationConfig,
    GenerationResult,
    merge_generation_config,
    normalize_script,
)
from .runtime.logger import RuntimeLogger


class Dia2:
    def __init__(
        self,
        *,
        repo: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        weights_path: Optional[str | Path] = None,
        tokenizer_id: Optional[str | Path] = None,
        mimi_id: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "auto",
        default_config: Optional[GenerationConfig] = None,
    ) -> None:
        bundle = resolve_assets(
            repo=repo,
            config_path=config_path,
            weights_path=weights_path,
        )
        self._config_path = bundle.config_path
        self._weights_path = bundle.weights_path
        self._tokenizer_id = (str(tokenizer_id) if tokenizer_id else None) or bundle.tokenizer_id
        self._repo_id = bundle.repo_id
        self._mimi_id = mimi_id or bundle.mimi_id
        self.device = device
        self._dtype_pref = dtype or "auto"
        self.default_config = default_config or GenerationConfig()
        self._runtime: Optional[RuntimeContext] = None
        # NEW: Graph cache for faster generation after first call
        self._graph_cache: Optional[CachedGraphState] = None

    @classmethod
    def from_repo(
        cls,
        repo: str,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        tokenizer_id: Optional[str] = None,
        mimi_id: Optional[str] = None,
    ) -> "Dia2":
        return cls(repo=repo, device=device, dtype=dtype, tokenizer_id=tokenizer_id, mimi_id=mimi_id)

    @classmethod
    def from_local(
        cls,
        config_path: str | Path,
        weights_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        tokenizer_id: Optional[str | Path] = None,
        mimi_id: Optional[str] = None,
    ) -> "Dia2":
        return cls(
            config_path=config_path,
            weights_path=weights_path,
            tokenizer_id=tokenizer_id,
            device=device,
            dtype=dtype,
            mimi_id=mimi_id,
        )

    def set_device(self, device: str, *, dtype: Optional[str] = None) -> None:
        desired_dtype = dtype or self._dtype_pref
        if self.device == device and desired_dtype == self._dtype_pref:
            return
        self.device = device
        self._dtype_pref = desired_dtype
        self._runtime = None
        # Clear graph cache when device changes
        self._graph_cache = None

    def close(self) -> None:
        self._runtime = None
        self._graph_cache = None

    def clear_graph_cache(self) -> None:
        """Clear the CUDA graph cache, forcing recompilation on next generate()."""
        self._graph_cache = None

    def _ensure_runtime(self) -> RuntimeContext:
        if self._runtime is None:
            self._runtime = self._build_runtime()
        return self._runtime

    def generate(
        self,
        script: str | Sequence[str],
        *,
        config: Optional[GenerationConfig] = None,
        output_wav: Optional[str | Path] = None,
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
        include_prefix: Optional[bool] = None,
        verbose: bool = False,
        use_graph_cache: bool = True,  # NEW: Enable/disable graph caching
        **overrides,
    ):
        """Generate speech from text.
        
        Args:
            script: Text to speak (use [S1]/[S2] tags for speaker changes)
            config: Generation configuration
            output_wav: Optional path to save output WAV file
            prefix_speaker_1: Path to speaker 1 voice sample
            prefix_speaker_2: Path to speaker 2 voice sample
            include_prefix: Whether to include prefix audio in output
            verbose: Enable verbose logging
            use_graph_cache: If True, reuse CUDA graphs across calls (faster)
            **overrides: Additional config overrides
        """
        runtime = self._ensure_runtime()
        logger = RuntimeLogger(verbose)
        merged_overrides = dict(overrides)
        if prefix_speaker_1 is not None:
            merged_overrides["prefix_speaker_1"] = prefix_speaker_1
        if prefix_speaker_2 is not None:
            merged_overrides["prefix_speaker_2"] = prefix_speaker_2
        if include_prefix is not None:
            merged_overrides["include_prefix"] = include_prefix
        merged = merge_generation_config(base=config or self.default_config, overrides=merged_overrides)
        max_context = runtime.config.runtime.max_context_steps
        text = normalize_script(script)
        
        # Build prefix plan (always fresh)
        prefix_plan = build_prefix_plan(runtime, merged.prefix)
        
        # Build entries list: prefix entries + new text entries
        # This is the SAME logic as the working version
        entries = []
        if prefix_plan is not None:
            entries.extend(prefix_plan.entries)
        entries.extend(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
        
        runtime.machine.initial_padding = merged.initial_padding
        logger.event(
            f"starting generation: max_context={max_context} cfg_scale={merged.cfg_scale:.2f} "
            f"device={self.device} dtype={self._dtype_pref}"
        )
        
        # Create FRESH state machine - CRITICAL for avoiding "re-read" bug
        # The state tracks what text to generate and MUST be fresh each call
        state = runtime.machine.new_state(entries)
        
        cfg_active = merged.cfg_scale != 1.0
        if cfg_active:
            logger.event(f"classifier-free guidance enabled (scale={merged.cfg_scale:.2f})")
        else:
            logger.event("classifier-free guidance disabled (scale=1.0)")
        
        # Determine if we should use graph caching
        should_use_cache = (
            use_graph_cache 
            and merged.use_cuda_graph 
            and runtime.device.type == "cuda"
        )
        
        if should_use_cache:
            # Get or create graph cache
            if self._graph_cache is None:
                logger.event("creating graph cache (first generation will compile graphs)")
                self._graph_cache = create_graph_cache(runtime)
            else:
                logger.event("using cached graphs (fast path)")
            
            # Reset cache for new generation - resets tensor VALUES, keeps tensor OBJECTS
            # This copies prefix audio into the buffer
            reset_graph_cache(self._graph_cache, runtime, prefix_plan)
            
            # Use the cached GenerationState - same tensor objects as captured graphs
            gen_state = self._graph_cache.generation
            graph_cache = self._graph_cache
        else:
            # Original behavior - build fresh state each time
            gen_state = build_initial_state(runtime, prefix=prefix_plan)
            graph_cache = None
        
        include_prefix_audio = bool(prefix_plan and merged.prefix and merged.prefix.include_audio)
        start_step = 0
        
        # Warmup with prefix - builds KV cache
        # This runs in EAGER mode (no graphs) because prefix length varies
        if prefix_plan is not None:
            logger.event(f"warming up with prefix ({prefix_plan.aligned_frames} frames)")
            start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
            if include_prefix_audio:
                logger.event("prefix audio will be kept in output")
            else:
                logger.event("prefix audio trimmed from output")
        
        # Run generation loop - uses cached graphs if available
        first_word_frame, audio_buf = run_generation_loop(
            runtime,
            state=state,  # Fresh state machine
            generation=gen_state,  # Cached or fresh GenerationState
            config=merged,
            start_step=start_step,
            logger=logger,
            graph_cache=graph_cache,  # Pass cache for graph reuse
        )
        
        aligned = undelay_frames(audio_buf[0], runtime.audio_delays, runtime.constants.audio_pad).unsqueeze(0)
        crop = 0 if include_prefix_audio else max(first_word_frame, 0)
        if crop > 0 and crop < aligned.shape[-1]:
            aligned = aligned[:, :, crop:]
        elif crop >= aligned.shape[-1]:
            crop = 0
        logger.event(f"decoding {aligned.shape[-1]} Mimi frames")
        waveform = decode_audio(runtime, aligned)
        if output_wav is not None:
            write_wav(str(output_wav), waveform.detach().cpu().numpy(), runtime.mimi.sample_rate)
            duration = waveform.shape[-1] / max(runtime.mimi.sample_rate, 1)
            logger.event(f"saved {output_wav} ({duration:.2f}s)")
        frame_rate = max(runtime.frame_rate, 1.0)
        prefix_entry_count = len(prefix_plan.entries) if prefix_plan is not None else 0
        transcript_entries = state.transcript
        if prefix_plan is not None and not include_prefix_audio:
            if len(transcript_entries) > prefix_entry_count:
                transcript_entries = transcript_entries[prefix_entry_count:]
            else:
                transcript_entries = []
        timestamps = []
        for word, step in transcript_entries:
            adj = step - crop
            if adj < 0:
                continue
            timestamps.append((word, adj / frame_rate))
        logger.event(f"generation finished in {logger.elapsed():.2f}s")
        return GenerationResult(aligned, waveform, runtime.mimi.sample_rate, timestamps)

    def save_wav(self, script: str | Sequence[str], path: str | Path, **kwargs):
        return self.generate(script, output_wav=path, **kwargs)

    @property
    def sample_rate(self) -> int:
        return self._ensure_runtime().mimi.sample_rate

    @property
    def tokenizer_id(self) -> Optional[str]:
        if self._tokenizer_id:
            return self._tokenizer_id
        if self._runtime is not None:
            return getattr(self._runtime.tokenizer, "name_or_path", None)
        return self._repo_id

    @property
    def dtype(self) -> str:
        return self._dtype_pref

    @property
    def max_context_steps(self) -> int:
        return self._ensure_runtime().config.runtime.max_context_steps

    @property
    def repo(self) -> Optional[str]:
        return self._repo_id

    def _build_runtime(self) -> RuntimeContext:
        runtime, tokenizer_ref, mimi_ref = build_runtime(
            config_path=self._config_path,
            weights_path=self._weights_path,
            tokenizer_id=self._tokenizer_id,
            repo_id=self._repo_id,
            mimi_id=self._mimi_id,
            device=self.device,
            dtype_pref=self._dtype_pref,
        )
        self._tokenizer_id = tokenizer_ref
        self._mimi_id = mimi_ref
        return runtime
