# codec.py - Kyutai Mimi with native streaming support
# This uses the original Kyutai Mimi from the moshi package which has
# built-in StreamingModule support for efficient incremental decoding.
# Added: compile() method for torch.compile optimization
# Added: warmup_decode() method for pre-warming CUDA graphs
#
# To revert to HuggingFace MimiModel (no native streaming):
#   cp dia2/audio/codec_hf.py dia2/audio/codec.py

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional, Any

import torch
from torch import nn

# Import Kyutai Mimi from moshi package
from moshi.models import get_mimi
from moshi.models.compression import MimiModel as KyutaiMimiModel


# Default model repo and weights file for Kyutai Mimi
# Note: The moshi library expects weights from moshiko repo, not kyutai/mimi
DEFAULT_MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
DEFAULT_MIMI_WEIGHTS = "tokenizer-e351c8d8-checkpoint125.safetensors"


@dataclass(frozen=True)
class MimiConfig:
    model_repo: str = DEFAULT_MIMI_REPO
    dtype: Optional[torch.dtype] = None


class MimiCodec(nn.Module):
    """Wrapper around Kyutai's native Mimi model with streaming support.
    
    This uses the original Kyutai Mimi from the moshi package, which has
    built-in streaming support via the StreamingModule pattern.
    
    Usage for streaming:
        codec = MimiCodec.from_pretrained(device=device)
        codec.start_streaming(batch_size=1)
        try:
            # Each decode() call maintains internal state
            audio1 = codec.decode(codes1)
            audio2 = codec.decode(codes2)
            # ...
        finally:
            codec.stop_streaming()
    
    Or use the context manager:
        with codec.streaming(batch_size=1):
            audio1 = codec.decode(codes1)
            audio2 = codec.decode(codes2)
    """

    def __init__(self, model: KyutaiMimiModel, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.sample_rate = model.sample_rate
        self.frame_rate = model.frame_rate
        self.samples_per_frame = model.frame_size
        self._streaming_context: Optional[ExitStack] = None
        self._compiled = False
        self._compile_mode: Optional[str] = None

    @classmethod
    def from_pretrained(
        cls,
        model_repo: str = DEFAULT_MIMI_REPO,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        num_codebooks: int = 32,
    ) -> "MimiCodec":
        """Load the Kyutai Mimi model.
        
        Args:
            model_repo: HuggingFace repo containing Mimi weights (ignored - always uses moshiko repo)
            device: Device to load model on
            dtype: Data type (note: Kyutai Mimi manages its own dtype)
            num_codebooks: Number of codebooks to use (default 32 - Mimi's max)
        """
        from huggingface_hub import hf_hub_download
        
        # Always use the moshiko repo - the moshi get_mimi() function requires weights
        # in a specific format that only exists in the moshiko repo
        # (The 'kyutai/mimi' repo has weights in HuggingFace format which is incompatible)
        if model_repo != DEFAULT_MIMI_REPO:
            print(f"[MimiCodec] Note: Ignoring repo '{model_repo}', using '{DEFAULT_MIMI_REPO}' for Kyutai Mimi")
            model_repo = DEFAULT_MIMI_REPO
        
        weights_path = hf_hub_download(model_repo, DEFAULT_MIMI_WEIGHTS)
        print(f"[MimiCodec] Loading weights from {weights_path}")
        
        # Load using Kyutai's get_mimi function
        model = get_mimi(
            filename=weights_path,
            device=device,
            num_codebooks=num_codebooks,
        )
        
        return cls(model, device)

    @property
    def is_streaming(self) -> bool:
        """Check if currently in streaming mode."""
        return self._streaming_context is not None

    @property
    def is_compiled(self) -> bool:
        """Check if model has been compiled with torch.compile."""
        return self._compiled

    def compile(self, mode: str = "reduce-overhead") -> None:
        """Compile model components with torch.compile for faster inference.
        
        Args:
            mode: Compilation mode. Options:
                - "reduce-overhead": Good balance of compile time and speedup (recommended)
                - "max-autotune": Maximum optimization, longer compile time
                - "default": Fastest compile, moderate speedup
        
        Note: This compiles the encoder, decoder, and transformer components.
        The first inference after compilation will be slower due to JIT compilation,
        so call warmup_decode() after this to pre-warm the compiled functions.
        """
        if self._compiled:
            print(f"[MimiCodec] Already compiled with mode '{self._compile_mode}'")
            return
        
        print(f"[MimiCodec] Compiling model components with mode='{mode}'...")
        
        # Compile the main components
        self.model.encoder = torch.compile(self.model.encoder, mode=mode)
        self.model.decoder = torch.compile(self.model.decoder, mode=mode)
        
        if self.model.encoder_transformer is not None:
            self.model.encoder_transformer = torch.compile(
                self.model.encoder_transformer, mode=mode
            )
        if self.model.decoder_transformer is not None:
            self.model.decoder_transformer = torch.compile(
                self.model.decoder_transformer, mode=mode
            )
        
        self._compiled = True
        self._compile_mode = mode
        print(f"[MimiCodec] Compilation complete")

    def warmup_decode(self, num_frames: int = 3, batch_size: int = 1, num_warmup_calls: int = 3) -> None:
        """Pre-warm the decode path to trigger JIT compilation and CUDA graph capture.
        
        Args:
            num_frames: Number of frames per decode call (should match typical usage)
            batch_size: Batch size for warmup
            num_warmup_calls: Number of decode calls to make for thorough warmup
        """
        print(f"[MimiCodec] Warming up decode path ({num_warmup_calls} calls, {num_frames} frames each)...")
        dummy_codes = torch.zeros(batch_size, 32, num_frames, dtype=torch.long, device=self.device)
        
        with self.streaming(batch_size):
            for i in range(num_warmup_calls):
                _ = self.decode(dummy_codes)
        print(f"[MimiCodec] Warmup complete")

    def start_streaming(self, batch_size: int = 1) -> None:
        """Enter streaming mode.
        
        In streaming mode, each call to decode() maintains internal state,
        allowing for incremental decoding of audio tokens.
        """
        if self._streaming_context is not None:
            raise RuntimeError("Already in streaming mode. Call stop_streaming() first.")
        self._streaming_context = self.model.streaming(batch_size)
        self._streaming_context.__enter__()

    def stop_streaming(self) -> None:
        """Exit streaming mode and reset internal state."""
        if self._streaming_context is not None:
            self._streaming_context.__exit__(None, None, None)
            self._streaming_context = None

    def streaming(self, batch_size: int = 1):
        """Context manager for streaming mode.
        
        Usage:
            with codec.streaming(batch_size=1):
                audio = codec.decode(codes)
        """
        return _StreamingContext(self, batch_size)

    def reset_streaming(self) -> None:
        """Reset the streaming state without exiting streaming mode."""
        if self._streaming_context is not None:
            self.model.reset_streaming()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform.
        
        Args:
            codes: Audio codes of shape [B, K, T] where K is num_codebooks
            
        Returns:
            Audio waveform of shape [B, C, samples]
        
        Note: In streaming mode, this maintains internal state for incremental
        decoding. Outside streaming mode, this is a stateless decode.
        """
        codes = codes.to(self.device)
        with torch.inference_mode():
            audio = self.model.decode(codes)
            return torch.clamp(audio, -1.0, 1.0)

    def encode(self, audio: torch.Tensor, *, return_dict: bool = False):
        """Encode audio waveform to codes.
        
        Args:
            audio: Audio waveform of shape [B, C, samples]
            return_dict: Ignored, kept for backwards compatibility with HuggingFace API
            
        Returns:
            Tuple of (codes, scale) to match HuggingFace API.
            codes: Audio codes of shape [B, K, T]
            scale: Always None (kept for compatibility)
        """
        audio = audio.to(self.device)
        with torch.inference_mode():
            codes = self.model.encode(audio)
            return (codes, None)  # Return tuple to match HuggingFace API


class _StreamingContext:
    """Context manager helper for streaming mode."""
    
    def __init__(self, codec: MimiCodec, batch_size: int):
        self.codec = codec
        self.batch_size = batch_size
    
    def __enter__(self):
        self.codec.start_streaming(self.batch_size)
        return self.codec
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.codec.stop_streaming()
        return False
