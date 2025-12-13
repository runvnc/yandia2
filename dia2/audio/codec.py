# codec.py - Kyutai Mimi with native streaming support
# This uses the original Kyutai Mimi from the moshi package which has
# built-in StreamingModule support for efficient incremental decoding.
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

    @classmethod
    def from_pretrained(
        cls,
        model_repo: str = DEFAULT_MIMI_REPO,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        num_codebooks: int = 8,
    ) -> "MimiCodec":
        """Load the Kyutai Mimi model.
        
        Args:
            model_repo: HuggingFace repo containing Mimi weights
            device: Device to load model on
            dtype: Data type (note: Kyutai Mimi manages its own dtype)
            num_codebooks: Number of codebooks to use (default 8 for Dia2)
        """
        from huggingface_hub import hf_hub_download
        
        # Download the weights from HuggingFace (moshiko repo has the correct format)
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

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to codes.
        
        Args:
            audio: Audio waveform of shape [B, C, samples]
            
        Returns:
            Audio codes of shape [B, K, T]
        """
        audio = audio.to(self.device)
        with torch.inference_mode():
            return self.model.encode(audio)


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
