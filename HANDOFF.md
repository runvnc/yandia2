# Yandia2 Project Handoff Document

**Date**: December 12, 2024  
**Project**: `/files/yandia2`  
**GitHub**: `https://github.com/runvnc/yandia2`

---

## Executive Summary

Yandia2 is a **working Dia2 TTS conversation server** with CUDA graph caching for faster generation. It manages stateful conversations where:
1. An AI voice warmup is set
2. User audio is provided  
3. Text is generated as speech using the conversation context

**Current Performance**: ~5-6s total, RTF ~1.6 (1.6x slower than realtime)  
**Target**: Streaming output for ~2-3s to first audio chunk

---

## Project Structure

```
/files/yandia2/
├── dia2/                      # Dia2 library (modified for graph caching)
│   ├── engine.py              # Main Dia2 class with graph cache
│   ├── runtime/
│   │   ├── generator.py       # Generation loop + CachedGraphState
│   │   ├── voice_clone.py     # Prefix/voice handling
│   │   └── ...
│   └── ...
├── conversation_server.py     # FastAPI server (stateful conversation)
├── simple_server.py           # FastAPI server (stateless)
├── example_prefix1.wav        # Example AI voice (Speaker 1)
├── example_prefix2.wav        # Example user voice (Speaker 2)
├── test_conversation.sh       # Test script
├── requirements.txt           # For pip install
├── pyproject.toml             # For uv
└── HANDOFF.md                 # This document
```

---

## Git Tags (Safe Restore Points)

| Tag | Description |
|-----|-------------|
| `working_conv_very_slow` | Before graph caching (~10s, RTF ~2.5-3.0) |
| `working_rtf_1.6` | With graph caching (~5-6s, RTF ~1.6) |

```bash
# To restore if something breaks:
git checkout working_rtf_1.6
```

---

## API Endpoints (conversation_server.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + conversation state |
| `/reset` | POST | Reset conversation |
| `/set_voice` | POST | Upload AI warmup audio (file) |
| `/user_spoke` | POST | Upload user audio (file) |
| `/generate` | POST | Generate AI speech from text |
| `/state` | GET | Get current conversation state |

### Conversation Flow

```
1. POST /set_voice    [AI warmup audio]
2. POST /user_spoke   [User audio]  
3. POST /generate     {text: "Hello..."}  → Returns WAV
4. POST /user_spoke   [Next user audio]
5. POST /generate     {text: "..."}  → Returns WAV
... repeat 4-5 ...
```

---

## Key Implementation Details

### CUDA Graph Caching

Graphs are cached at the `Dia2` instance level to avoid recompilation:

```python
# dia2/runtime/generator.py
@dataclass
class CachedGraphState:
    generation: GenerationState  # Reused tensor objects
    positions: torch.Tensor
    main_tokens: torch.Tensor
    aux_tokens: torch.Tensor
    buffers: NetworkBuffers
    transformer_capture: Optional[...]  # CUDA graphs
    dep_captures: Optional[...]
```

**Critical**: Tensor OBJECTS must be reused (same memory addresses). Only reset VALUES with `.fill_()` or `.copy_()`.

### State Machine (MUST be fresh each call)

```python
# dia2/engine.py generate()
state = runtime.machine.new_state(entries)  # ALWAYS fresh!
```

The State tracks text entries to generate. If reused, causes "re-read prefix" bug.

### Prefix Handling

```python
entries = []
if prefix_plan:
    entries.extend(prefix_plan.entries)  # Prefix entries first
entries.extend(parse_script([text], ...))  # Then new text
state = runtime.machine.new_state(entries)  # Single state with ALL
```

---

## Critical Bugs Fixed

### 1. trim_audio Replacing Tensor (MAJOR)

**Bug**: `GenerationState.trim_audio()` replaced `self.audio_buf` with smaller tensor, breaking CUDA graph references.

**Fix**: Return trimmed tensor without modifying `self.audio_buf`:
```python
def trim_audio(self, limit, pad_token, ungenerated):
    trimmed = self.audio_buf[:, :, :limit]
    pad = torch.full_like(trimmed, pad_token)
    trimmed = torch.where(trimmed == ungenerated, pad, trimmed)
    # DON'T DO: self.audio_buf = trimmed
    return trimmed
```

### 2. PyTorch Compatibility (torch.backends.cudnn.conv)

**Bug**: Older PyTorch doesn't have `torch.backends.cudnn.conv`

**Fix**: Check with `hasattr()` before accessing.

---

## Lessons from Failed Attempts (/files/stream-dia2)

The stream-dia2 project has ~30 commits of failed fixes. Key lessons:

1. **Use SINGLE state machine with ALL entries** - Separating prefix/generation state machines breaks voice consistency
2. **Always run warmup_with_prefix()** - Builds KV cache, can't skip
3. **Don't cache State object** - Only cache tensors
4. **Don't replace tensor objects** - Breaks CUDA graphs

---

## Performance Breakdown

### Current (~5-6s total, RTF ~1.6)

| Phase | Time | Notes |
|-------|------|-------|
| Transcription | ~1s | Cached after first call per file |
| Prefix warmup | ~1-2s | Eager mode, varies with prefix length |
| Graph compile | ~0.5s | Only first call, then ~0 |
| Generation | ~2-3s | ~21 toks/s with graph replay |
| Mimi decode | ~0.5s | Could be streamed |

### Why Previous Experiments Showed 0.6 RTF

The 0.6 RTF (faster than realtime) was measuring ONLY the generation loop after graphs were warm, not including warmup/decode/transcription.

---

## Next Steps: Streaming Output

### Goal

Reduce time-to-first-audio from ~5-6s to ~2-3s by streaming audio chunks during generation.

### Architecture

```
Current:  Generate ALL tokens → Decode ALL → Return
Streaming: Generate 10 frames → Decode → Send chunk → Repeat
```

### Key Components Needed

1. **Mimi's `decode_streaming()`** - Already exists, maintains KV state
2. **Yield chunks in generation loop** - Every N frames
3. **WebSocket endpoint** - `/generate_stream` for pushing chunks
4. **Handle delay pattern** - First ~8 frames don't produce valid audio

### Expected Latency (Streaming)

| Phase | Time |
|-------|------|
| Warmup (unavoidable) | ~1-2s |
| First audio chunk | ~600-800ms after generation starts |
| Subsequent chunks | ~150-250ms intervals |
| **Total to first audio** | **~2-3s** |

### Reference Code

The `/files/stream-dia2` project has streaming infrastructure:
- `streaming_generator.py` - `run_streaming_generation()` yields `StreamingChunk`
- `realtime_dia2_server.py` - WebSocket handling, `ContinuousSession`

Can be ported to yandia2 now that core generation is working.

---

## Configuration

### Defaults (conversation_server.py)

```python
cfg_scale: float = Form(1.0)      # 1.0 = no CFG (faster), 2.0 = CFG (better quality)
temperature: float = Form(0.8)
top_k: int = Form(50)
use_cuda_graph: bool = True       # In engine.py generate()
```

### CFG Impact

| cfg_scale | Speed | Quality |
|-----------|-------|--------|
| 1.0 | ~21 toks/s | Good |
| 2.0 | ~10 toks/s | Better (2x slower, runs model twice) |

---

## Running the Server

```bash
cd /files/yandia2

# With uv:
uv sync
uv run python conversation_server.py

# With pip:
pip install -r requirements.txt
python conversation_server.py

# Or with uvicorn:
uvicorn conversation_server:app --host 0.0.0.0 --port 8000
```

### Testing

```bash
SERVER=http://localhost:8000 ./test_conversation.sh
```

---

## User Context

- **Use case**: Employment verification phone calls via SIP
- **Current TTS**: ElevenLabs (~150-200ms latency)
- **Goal**: Replace with self-hosted Dia2
- **Constraint**: Need low latency for natural conversation
- **Related projects**:
  - `mr_eleven_stream` - ElevenLabs streaming with µ-law output
  - `mr_sip` - SIP call handling
  - `stream-dia2` - Previous streaming attempts (many failed)

---

## µ-law Output (For SIP Integration)

Adding µ-law output won't speed up generation but is useful for SIP integration:

```python
import audioop

def pcm_to_ulaw(pcm_24khz: bytes) -> bytes:
    # Resample 24kHz → 8kHz
    pcm_8khz, _ = audioop.ratecv(pcm_24khz, 2, 1, 24000, 8000, None)
    # Convert to µ-law
    ulaw = audioop.lin2ulaw(pcm_8khz, 2)
    return ulaw
```

---

## Files Modified from Original Dia2

| File | Changes |
|------|------|
| `dia2/engine.py` | Added `_graph_cache`, `use_graph_cache` param, `clear_graph_cache()` |
| `dia2/runtime/generator.py` | Added `CachedGraphState`, `create_graph_cache()`, `reset_graph_cache()`, fixed `trim_audio()` |
| `dia2/runtime/context.py` | Fixed PyTorch compatibility for `cudnn.conv` |

---

## Contact / Resources

- **Dia2 Original**: https://github.com/nari-labs/dia2
- **Dia2 HuggingFace**: `nari-labs/Dia2-2B`
- **This Project**: https://github.com/runvnc/yandia2
