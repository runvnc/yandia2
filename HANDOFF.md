# Yandia2 Project Handoff Document

**Last Updated**: December 13, 2025  
**Project**: `/files/yandia2`  
**GitHub**: `https://github.com/runvnc/yandia2`  
**Current Tag**: `1s_new_streaming_ok`

---

## Executive Summary

Yandia2 is a **Dia2 TTS conversation server** with **native Kyutai Mimi streaming** for low-latency audio output. The server maintains stateful conversations with voice cloning support.

**Current Performance:**
- First audio chunk: ~1.14s after generation starts
- Total warmup: ~2.76s (prefix processing)
- RTF (Real-Time Factor): ~0.96 (nearly real-time)
- Audio generation: 2.16s audio in 2.07s

**Key Achievement:** Integrated the original Kyutai Mimi codec (from moshi package) with native `StreamingModule` support, replacing the HuggingFace MimiModel.

---

## ⚠️ CRITICAL: Dual Speaker Prefix Requirement

**Dia2 requires BOTH speaker prefixes for voice cloning to work!**

When only `prefix_speaker_1` is provided (without `prefix_speaker_2`), Dia2 produces **random voices** instead of cloning the target voice.

**Fix:** When no user audio exists, fall back to the AI audio for speaker_2:
```python
speaker_2_audio = conversation.last_user_audio or conversation.last_ai_audio
```

---

## Architecture Overview

### Kyutai Mimi Integration

We replaced the HuggingFace `transformers.MimiModel` with the original **Kyutai Mimi** from the `moshi` package. This provides:

1. **Native StreamingModule support** - Built-in state management for incremental decoding
2. **CUDA graph optimization** - Automatic graph caching within the streaming context
3. **Simpler API** - Just enter streaming context and call decode()

```python
# Old HuggingFace approach (decode_streaming with past_key_values)
audio, kv = mimi.decode(codes, decoder_past_key_values=kv, return_dict=True)

# New Kyutai Mimi approach (native streaming)
with mimi.streaming(batch_size=1):
    audio = mimi.decode(codes)  # State managed automatically
```

### Key Files

| File | Description |
|------|-------------|
| `dia2/audio/codec.py` | Kyutai Mimi wrapper with streaming support |
| `dia2/audio/codec_hf.py` | Backup of HuggingFace implementation |
| `streaming_server.py` | Main server with WebSocket streaming |
| `conversation_server.py` | REST-only server (no streaming decode) |

### Codec Architecture

```python
# dia2/audio/codec.py
class MimiCodec(nn.Module):
    def start_streaming(self, batch_size=1):
        self._streaming_context = self.model.streaming(batch_size)
        self._streaming_context.__enter__()
    
    def stop_streaming(self):
        self._streaming_context.__exit__(None, None, None)
        self._streaming_context = None
    
    def decode(self, codes):
        # In streaming mode, maintains internal KV state
        return self.model.decode(codes)
```

---

## Streaming Flow

### WebSocket `/ws/generate` Endpoint

```
1. Client connects to WebSocket
2. Server enters Mimi streaming mode: mimi.start_streaming(batch_size=1)
3. Warm up Mimi with prefix tokens (builds voice context in decoder KV cache)
4. Generation loop:
   a. Generate audio tokens (transformer + depformer)
   b. Every CHUNK_FRAMES (3), decode tokens via Mimi streaming
   c. Send PCM chunk to client via WebSocket
5. Flush remaining frames
6. Exit streaming mode: mimi.stop_streaming()
```

### Latency Breakdown

| Phase | Time | Notes |
|-------|------|-------|
| Prefix warmup (transformer KV) | ~2.76s | Unavoidable, processes prefix audio |
| First audio chunk | ~1.14s | After generation starts |
| Subsequent chunks | ~80-150ms | Every 3 frames |
| Total first audio | ~3.9s | warmup + first chunk |

---

## Optimization Opportunities

### 1. Pre-warm During User Speech (High Impact)

The ~2.76s warmup happens AFTER the user finishes speaking. Could pre-warm:
- Start prefix warmup as soon as user audio arrives
- Run transcription in parallel with warmup
- Have prefix ready when generation text arrives

### 2. Cache Transcription Results (Already Done)

Transcription results are cached by file hash. First transcription takes ~0.7-0.9s, subsequent calls hit cache.

### 3. Reduce CHUNK_FRAMES (Trade-off)

Currently decoding every 3 frames (~240ms of audio). Could reduce to 1-2 frames for lower latency but more decode overhead.

### 4. Mimi Warmup Optimization

Currently warming Mimi by decoding prefix tokens (discarding output). Could potentially:
- Use smaller prefix (less warmup time)
- Investigate if Mimi truly needs full prefix warmup for voice consistency

### 5. Persistent Session with KV Snapshots

The `stream-dia2` project explored KV cache snapshots to avoid per-request warmup. This is complex but could eliminate the 2.76s warmup for subsequent requests.

---

## Git Tags (Restore Points)

| Tag | Description |
|-----|-------------|
| `1s_new_streaming_ok` | **CURRENT** - Kyutai Mimi native streaming, working |
| `stream_clone_artifact` | Voice cloning works, minor artifact at start |
| `working_rtf_1.6` | With CUDA graph caching (~5-6s, RTF ~1.6) |
| `working_conv_very_slow` | Before graph caching (~10s, RTF ~2.5-3.0) |
| `realtime_prebad_rnd_voice` | Before random voice fix |

```bash
# To restore if something breaks:
git checkout 1s_new_streaming_ok
```

---

## Installation

### Using install.sh (Recommended)

```bash
cd /files/yandia2
./install.sh
```

This installs dependencies and handles the moshi package separately (due to version conflicts).

### Manual Installation

```bash
# Install main dependencies
pip install -r requirements.txt

# Install moshi from git with --no-deps (has version conflicts)
pip install --no-deps git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi
```

### Dependencies Note

The `moshi` package has strict version constraints that conflict with yandia2's requirements:
- moshi requires `torch<2.8` but we need `torch>=2.8.0`
- moshi requires `sphn<0.2.0` but we need `sphn>=0.2.0`

We use `--no-deps` and manage dependencies manually, or use uv's `override-dependencies` feature.

---

## Running the Server

```bash
cd /files/yandia2
python streaming_server.py  # Runs on port 3030
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + conversation state |
| `/reset` | POST | Reset conversation |
| `/set_voice` | POST | Upload AI warmup audio (file) |
| `/user_spoke` | POST | Upload user audio (file) |
| `/generate` | POST | Generate AI speech (REST, complete audio) |
| `/ws/generate` | WebSocket | Stream audio chunks (native Mimi streaming) |
| `/state` | GET | Get current conversation state |

---

## Testing

### Test REST Endpoints

```bash
./test_conversation.sh
```

Tests: /health, /reset, /set_voice, /user_spoke, /generate (REST), /state

### Test WebSocket Streaming

```bash
./test_streaming.sh
# or
python test_streaming.py "Hello, this is a streaming test."
```

Tests the WebSocket `/ws/generate` endpoint with native Mimi streaming.

### Expected Log Output (Streaming)

```
[Stream] Warming up with prefix (112 frames)...
[Stream] Warmup done in 2.76s
[Stream] Starting generation loop from step 111...
[Stream] max_delay=18, num_codebooks=32
[Stream] content_start=112 (will decode from here)
[Stream] First chunk sent after 1.14s
[Stream] Generation complete: 2.16s audio in 2.07s (RTF: 0.96)
```

---

## Reverting to HuggingFace Mimi

If the Kyutai Mimi integration causes issues, you can revert to the HuggingFace implementation:

```bash
cp dia2/audio/codec_hf.py dia2/audio/codec.py
```

Note: The HuggingFace version doesn't have native streaming support, so the streaming server will use chunk-by-chunk batch decoding instead.

---

## Project Structure

```
/files/yandia2/
├── dia2/                      # Dia2 library (modified for yandia2)
│   ├── audio/
│   │   ├── codec.py           # Kyutai Mimi wrapper (native streaming)
│   │   ├── codec_hf.py        # HuggingFace backup
│   │   └── grid.py            # Audio frame utilities
│   ├── engine.py              # Main Dia2 class
│   └── runtime/
│       ├── generator.py       # Generation loop + CUDA graphs
│       ├── voice_clone.py     # Prefix/voice handling
│       └── ...
├── streaming_server.py        # Main server (REST + WebSocket streaming)
├── conversation_server.py     # REST-only server
├── install.sh                 # Installation script
├── requirements.txt           # Dependencies
├── test_conversation.sh       # REST endpoint tests
├── test_streaming.sh          # WebSocket streaming test
├── test_streaming.py          # Python streaming client
├── example_prefix1.wav        # Example AI voice
├── example_prefix2.wav        # Example user voice
└── HANDOFF.md                 # This document
```

---

## Key Learnings

### 1. Moshi vs HuggingFace Mimi

- **HuggingFace `transformers.MimiModel`**: Port of Mimi, uses `decoder_past_key_values` for streaming
- **Kyutai Mimi (moshi package)**: Original implementation with `StreamingModule` pattern
- The weights format is different! Kyutai expects weights from `kyutai/moshiko-pytorch-bf16`, not `kyutai/mimi`

### 2. Codebook Count

- Mimi has 32 codebooks total
- Dia2 uses a delay pattern that requires access to all of them
- Must set `num_codebooks=32` (we initially tried 8, then 16, both failed)

### 3. API Compatibility

The new codec needed to maintain backwards compatibility:
- `encode()` returns tuple `(codes, None)` to match HuggingFace API
- `decode()` returns same tensor format
- Added `return_dict` parameter (ignored) for compatibility

---

## User Context

- **Use case**: Employment verification phone calls via SIP
- **Current TTS**: ElevenLabs (~150-200ms latency)
- **Goal**: Replace with self-hosted Dia2 for cost savings
- **Constraint**: Need low latency for natural conversation

---

## Related Projects

| Project | Path | Description |
|---------|------|-------------|
| stream-dia2 | `/files/stream-dia2` | Previous streaming experiments (150+ commits) |
| moshi | `/files/moshi` | Kyutai Moshi/Mimi library |
| mr_eleven_stream | `/xfiles/upd5/mr_eleven_stream` | ElevenLabs streaming plugin |
| mr_sip | `/xfiles/update_plugins/mr_sip` | SIP call handling |

---

## Contact

User (runvnc) has extensive context on this project and can clarify requirements.
