"""Prefix caching for fast warmup restoration.

This module provides caching of KV cache state after processing prefix audio,
enabling near-instant restoration of warmed-up state for subsequent generations
with the same prefix configuration.

Typical usage:
    cache_store = PrefixCacheStore(max_entries=10)
    
    # First generation with a prefix - will do full warmup and cache
    start_step = warmup_with_prefix_cached(
        runtime, prefix_plan, state, generation, cache_store
    )
    
    # Subsequent generations with same prefix - instant restore
    start_step = warmup_with_prefix_cached(
        runtime, prefix_plan, state, generation, cache_store
    )
"""
from __future__ import annotations

import copy
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

from ..core.cache import KVCacheSnapshot

if TYPE_CHECKING:
    from .context import RuntimeContext
    from .generator import GenerationState
    from .state_machine import State
    from .voice_clone import PrefixPlan


@dataclass
class PrefixCacheEntry:
    """Cached state after processing a prefix.
    
    This stores everything needed to restore the generation state to the
    point immediately after prefix warmup, without re-running the transformer.
    
    Attributes:
        prefix_key: Unique identifier for this prefix configuration
        transformer_snapshot: KV cache state after processing prefix
        step_tokens_snapshot: Token state after warmup
        audio_buf_snapshot: Audio buffer with prefix tokens
        start_step: Step number after warmup (where generation begins)
        created_at: Timestamp when this entry was created
    """
    prefix_key: str
    transformer_snapshot: KVCacheSnapshot
    step_tokens_snapshot: torch.Tensor
    audio_buf_snapshot: torch.Tensor
    start_step: int
    created_at: float = field(default_factory=time.time)
    
    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage of this cache entry."""
        total = 0
        for k in self.transformer_snapshot.keys:
            total += k.numel() * k.element_size()
        for v in self.transformer_snapshot.values:
            total += v.numel() * v.element_size()
        total += self.step_tokens_snapshot.numel() * self.step_tokens_snapshot.element_size()
        total += self.audio_buf_snapshot.numel() * self.audio_buf_snapshot.element_size()
        return total


class PrefixCacheStore:
    """LRU cache store for prefix cache entries.
    
    Manages multiple cached prefix states with automatic eviction when
    the cache reaches its maximum size.
    
    Args:
        max_entries: Maximum number of prefix entries to cache
    """
    
    def __init__(self, max_entries: int = 10):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[PrefixCacheEntry]:
        """Get a cache entry, moving it to the end (most recently used)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, entry: PrefixCacheEntry) -> None:
        """Add or update a cache entry."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = entry
        else:
            if len(self._cache) >= self.max_entries:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = entry
    
    def remove(self, key: str) -> bool:
        """Remove a cache entry. Returns True if entry existed."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache
    
    @property
    def total_memory_bytes(self) -> int:
        """Total memory usage of all cached entries."""
        return sum(entry.memory_bytes for entry in self._cache.values())
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "memory_mb": self.total_memory_bytes / (1024 * 1024),
            "keys": list(self._cache.keys()),
        }


def compute_prefix_key(
    speaker_1_path: Optional[str],
    speaker_2_path: Optional[str],
    aligned_frames: int,
) -> str:
    """Compute a unique key for a prefix configuration.
    
    The key is based on the audio file paths and the number of aligned frames,
    which together uniquely identify the prefix processing result.
    """
    key_data = f"{speaker_1_path or 'none'}:{speaker_2_path or 'none'}:{aligned_frames}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def compute_prefix_key_from_plan(plan: "PrefixPlan") -> str:
    """Compute prefix key from a PrefixPlan object."""
    # Get paths from the plan's config if available
    s1_path = getattr(plan, 'speaker_1_path', None) or getattr(getattr(plan, 'config', None), 'speaker_1', None)
    s2_path = getattr(plan, 'speaker_2_path', None) or getattr(getattr(plan, 'config', None), 'speaker_2', None)
    return compute_prefix_key(s1_path, s2_path, plan.aligned_frames)


def save_prefix_cache(
    cache_store: PrefixCacheStore,
    prefix_key: str,
    generation: "GenerationState",
    start_step: int,
) -> PrefixCacheEntry:
    """Save the current generation state to the prefix cache.
    
    This should be called immediately after warmup_with_prefix() completes.
    
    Args:
        cache_store: The cache store to save to
        prefix_key: Unique key for this prefix
        generation: Current generation state with warmed-up KV cache
        start_step: Step number after warmup
    
    Returns:
        The created cache entry
    """
    entry = PrefixCacheEntry(
        prefix_key=prefix_key,
        transformer_snapshot=generation.transformer_cache.snapshot(),
        step_tokens_snapshot=generation.step_tokens.clone(),
        audio_buf_snapshot=generation.audio_buf[:, :, :start_step + 1].clone(),
        start_step=start_step,
    )
    cache_store.put(prefix_key, entry)
    return entry


def restore_prefix_cache(
    entry: PrefixCacheEntry,
    generation: "GenerationState",
) -> int:
    """Restore generation state from a cached prefix entry.
    
    Args:
        entry: The cache entry to restore from
        generation: Generation state to restore into
    
    Returns:
        The start_step for generation
    """
    # Restore transformer KV cache
    generation.transformer_cache.restore(entry.transformer_snapshot)
    
    # Restore step tokens
    generation.step_tokens.copy_(entry.step_tokens_snapshot)
    
    # Restore audio buffer prefix portion
    prefix_len = entry.audio_buf_snapshot.shape[-1]
    generation.audio_buf[:, :, :prefix_len].copy_(entry.audio_buf_snapshot)
    
    return entry.start_step


__all__ = [
    "PrefixCacheEntry",
    "PrefixCacheStore", 
    "compute_prefix_key",
    "compute_prefix_key_from_plan",
    "save_prefix_cache",
    "restore_prefix_cache",
]
