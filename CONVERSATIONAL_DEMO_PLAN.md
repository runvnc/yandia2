# Yandia2 Conversational Demo Plan

## Overview

Create a WebSocket-based conversational TTS server that:
1. Accepts streaming user audio (for voice context)
2. Receives completed user transcript + AI reply text (with word timestamps from Deepgram)
3. Generates AI speech that sounds natural in conversation
4. Streams audio back with ~1.5s latency to first audio

**Key Optimization: Bypass Whisper entirely by using Deepgram timestamps!**

## Architecture

```
┌───────────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      Mindroot           │     │    Yandia2      │     │   Dia2 Model    │
│                         │     │    Server       │     │                 │
│  - Deepgram STT         │────▶│  - User audio   │────▶│  - Voice clone  │
│    (with timestamps!)   │     │  - Timestamps   │     │  - Conversation │
│  - LLM                  │◀────│  - Stream audio │◀────│    context      │
│  - VAD                  │     │                 │     │                 │
└───────────────────────┘     └─────────────────┘     └─────────────────┘
```

## Why We Can Skip Whisper

Dia2 uses Whisper to get **word-level timestamps** for aligning audio with text tokens.

But we have better options:

1. **User Audio**: Deepgram already provides word-level timestamps!
2. **AI Audio**: Dia2's `GenerationResult.timestamps` gives us the timestamps!

### Latency Comparison:

```
With Whisper:     2-3 seconds per audio file
With Deepgram:    Already have timestamps (0ms additional)
With AI cache:    Reuse from previous generation (0ms additional)
```

## Current State (yandia2/streaming_server.py)

### What Works:
- `/set_voice` - Initialize AI voice from audio file
- `/user_spoke` - Add user audio to conversation
- `/ws/generate` - WebSocket streaming TTS generation
- Moshi Mimi codec with native streaming support
- CUDA graph caching for faster subsequent generations
- Warmup caching for repeated prefix configurations

### What Needs to Change:
- **Bypass Whisper** - Accept pre-computed timestamps
- Add streaming user audio input
- Save AI timestamps from each generation
- Modify `build_prefix_plan()` to use external timestamps

## Proposed API

### WebSocket Endpoint: `/ws/conversation`

A single WebSocket connection that handles the full conversation flow.

#### Client → Server Messages:

```json
// 1. Initialize with AI voice (with timestamps from initial recording)
{
  "type": "init",
  "ai_voice_path": "path/to/ai_voice.wav",
  "ai_voice_timestamps": [
    {"word": "Hello", "start": 0.0, "end": 0.4},
    {"word": "I'm", "start": 0.45, "end": 0.6},
    {"word": "your", "start": 0.65, "end": 0.8},
    {"word": "assistant", "start": 0.85, "end": 1.3}
  ]
}

// 2. Stream user audio chunks (during user speaking)
// Binary WebSocket message: raw PCM 16-bit 24kHz audio bytes

// 3. Finalize user turn and trigger AI response
{
  "type": "generate",
  "user_transcript": "Hello how are you today",
  "user_timestamps": [
    {"word": "Hello", "start": 0.0, "end": 0.35},
    {"word": "how", "start": 0.4, "end": 0.55},
    {"word": "are", "start": 0.6, "end": 0.75},
    {"word": "you", "start": 0.8, "end": 0.95},
    {"word": "today", "start": 1.0, "end": 1.4}
  ],
  "ai_text": "[S1] I'm doing great, thanks for asking! How can I help you?"
}

// 4. Cancel current generation (if user interrupts)
{
  "type": "cancel"
}

// 5. Reset conversation
{
  "type": "reset"
}
```

#### Server → Client Messages:

```json
// Ready acknowledgment
{
  "event": "ready",
  "sample_rate": 24000,
  "channels": 1
}

// Generation starting
{
  "event": "generating"
}

// Audio chunk (binary message)
// Format: 1 byte is_final flag + PCM 16-bit audio bytes

// Generation complete (includes timestamps for next turn!)
{
  "event": "done",
  "duration": 2.5,
  "total_time": 1.6,
  "ai_timestamps": [
    {"word": "I'm", "start": 0.0, "end": 0.15},
    {"word": "doing", "start": 0.2, "end": 0.4},
    ...
  ]
}

// Error
{
  "event": "error",
  "message": "Error description"
}
```

## Implementation Steps

### Phase 1: Whisper Bypass

1. **Modify `voice_clone.py` to accept external timestamps**
   
   Current code:
   ```python
   def transcribe_words(audio_path, device, language=None):
       # Uses Whisper to get word timestamps
       result = wts.transcribe(model, audio_path, language=language)
       ...
   ```
   
   New code:
   ```python
   def build_prefix_plan_with_timestamps(
       runtime,
       audio_path: str,
       timestamps: list[dict],  # [{"word": str, "start": float, "end": float}]
   ) -> PrefixPlan:
       # Skip Whisper, use provided timestamps directly
       words = [
           WhisperWord(text=t["word"], start=t["start"], end=t["end"])
           for t in timestamps
       ]
       # ... rest of prefix plan building
   ```

2. **Store AI timestamps from generation**
   
   ```python
   # After generation completes
   ai_timestamps = [
       {"word": word, "start": step / frame_rate}
       for word, step in result.timestamps
   ]
   conversation.last_ai_timestamps = ai_timestamps
   ```

### Phase 2: Streaming User Audio

1. **Add user audio buffer to conversation state**
   ```python
   @dataclass
   class ConversationState:
       initial_ai_audio: Optional[str] = None
       initial_ai_timestamps: Optional[list] = None
       last_ai_audio: Optional[str] = None
       last_ai_timestamps: Optional[list] = None  # From Dia2 generation
       last_user_audio: Optional[str] = None
       last_user_timestamps: Optional[list] = None  # From Deepgram
       turn_count: int = 0
       is_initialized: bool = False
       user_audio_buffer: bytes = b""
   ```

2. **Handle binary audio messages**
   ```python
   if msg["type"] == "bytes":
       conversation.user_audio_buffer += msg["bytes"]
   ```

### Phase 3: New WebSocket Endpoint

```python
@app.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    await websocket.accept()
    
    # Send ready message
    await websocket.send_text(json.dumps({
        "event": "ready",
        "sample_rate": dia.sample_rate,
        "channels": 1,
        "sample_width": 2
    }))
    
    try:
        while True:
            msg = await websocket.receive()
            
            if msg["type"] == "bytes":
                # Accumulate user audio
                conversation.user_audio_buffer += msg["bytes"]
                
            elif msg["type"] == "text":
                data = json.loads(msg["text"])
                
                if data["type"] == "init":
                    await handle_init(data, websocket)
                    
                elif data["type"] == "generate":
                    await handle_generate(data, websocket)
                    
                elif data["type"] == "cancel":
                    # TODO: Implement cancellation
                    pass
                    
                elif data["type"] == "reset":
                    await handle_reset(websocket)
                    
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
```

### Phase 4: Generation with External Timestamps

```python
async def handle_generate(data: dict, websocket: WebSocket):
    # Save user audio to temp file
    user_audio_path = save_user_audio_buffer()
    
    # Get timestamps from request (from Deepgram)
    user_timestamps = data.get("user_timestamps", [])
    
    # Update conversation state
    conversation.last_user_audio = user_audio_path
    conversation.last_user_timestamps = user_timestamps
    
    # Build prefix plan WITHOUT Whisper
    prefix_plan = build_prefix_plan_with_timestamps(
        runtime,
        ai_audio=conversation.last_ai_audio,
        ai_timestamps=conversation.last_ai_timestamps,
        user_audio=conversation.last_user_audio,
        user_timestamps=conversation.last_user_timestamps,
    )
    
    # Generate and stream
    ai_text = data["ai_text"]
    async for chunk, is_final in run_streaming_generation(...):
        header = struct.pack("!?", is_final)
        await websocket.send_bytes(header + chunk)
    
    # Save AI audio and timestamps for next turn
    conversation.last_ai_audio = save_ai_audio(all_chunks)
    conversation.last_ai_timestamps = generation_result.timestamps
    
    # Send completion with timestamps
    await websocket.send_text(json.dumps({
        "event": "done",
        "duration": duration,
        "total_time": total_time,
        "ai_timestamps": conversation.last_ai_timestamps
    }))
```

## Latency Breakdown (Optimized)

### With Whisper Bypass:
```
User finishes speaking
  │
  ├── Deepgram provides transcript + timestamps (already done)
  ├── LLM generates response (parallel, ~500-1500ms)
  │
  └── Yandia2 receives generate request
        │
        ├── Build prefix plan from timestamps (~10ms)
        ├── Warmup with prefix (~100ms, cached after first)
        ├── Generate 18 frames (~1.44s) ← Fundamental limit
        └── First audio chunk sent!

Total TTS latency: ~1.5s (down from 4-6s with Whisper!)
```

### First Turn vs Subsequent:
```
First turn:       ~1.5s (no Whisper overhead!)
Subsequent turns: ~1.5s (same, with warmup caching)
```

## Files to Modify

### 1. `/files/yandia2/dia2/runtime/voice_clone.py`
- Add `build_prefix_plan_with_timestamps()` function
- Accept external timestamps instead of calling Whisper

### 2. `/files/yandia2/streaming_server.py`
- Add `/ws/conversation` endpoint
- Add user audio streaming support
- Store and reuse AI timestamps
- Integrate with new prefix plan builder

### 3. `/files/yandia2/test_conversation.py` (new)
- Demo client script
- Simulates Mindroot integration
- Latency measurement tools

## Deepgram Timestamp Format

Deepgram returns timestamps like:
```json
{
  "words": [
    {"word": "hello", "start": 0.0, "end": 0.32, "confidence": 0.99},
    {"word": "how", "start": 0.4, "end": 0.56, "confidence": 0.98},
    ...
  ]
}
```

We just need to extract `word`, `start`, `end` and pass to Yandia2.

## Dia2 Timestamp Format

Dia2's `GenerationResult.timestamps` is:
```python
[("word1", step1), ("word2", step2), ...]
```

Convert to our format:
```python
ai_timestamps = [
    {"word": word, "start": step / frame_rate, "end": (step + estimated_duration) / frame_rate}
    for word, step in result.timestamps
]
```

## Demo Script

```python
import asyncio
import websockets
import json
import wave
import io

async def demo():
    async with websockets.connect("ws://localhost:3030/ws/conversation") as ws:
        # Wait for ready
        response = json.loads(await ws.recv())
        print(f"Server ready: {response}")
        
        # Initialize with AI voice
        await ws.send(json.dumps({
            "type": "init",
            "ai_voice_path": "example_prefix1.wav",
            "ai_voice_timestamps": [
                {"word": "Hello", "start": 0.0, "end": 0.4},
                {"word": "there", "start": 0.45, "end": 0.8}
            ]
        }))
        
        # Wait for init confirmation
        response = json.loads(await ws.recv())
        print(f"Init response: {response}")
        
        # Send user audio (binary)
        with open("user_audio.wav", "rb") as f:
            # Skip WAV header, send raw PCM
            f.read(44)  # Skip header
            audio_data = f.read()
            
        # Send in chunks (simulating streaming)
        chunk_size = 4800  # 100ms at 24kHz 16-bit
        for i in range(0, len(audio_data), chunk_size):
            await ws.send(audio_data[i:i+chunk_size])
            await asyncio.sleep(0.05)  # Simulate real-time
        
        # Trigger AI response with Deepgram timestamps
        import time
        start_time = time.time()
        
        await ws.send(json.dumps({
            "type": "generate",
            "user_transcript": "Hello how are you today",
            "user_timestamps": [
                {"word": "Hello", "start": 0.0, "end": 0.35},
                {"word": "how", "start": 0.4, "end": 0.55},
                {"word": "are", "start": 0.6, "end": 0.75},
                {"word": "you", "start": 0.8, "end": 0.95},
                {"word": "today", "start": 1.0, "end": 1.4}
            ],
            "ai_text": "[S1] I'm doing great, thanks for asking! How can I help you today?"
        }))
        
        # Receive audio stream
        first_chunk_time = None
        audio_chunks = []
        
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    print(f"Time to first audio: {first_chunk_time - start_time:.3f}s")
                
                is_final = msg[0]
                audio = msg[1:]
                audio_chunks.append(audio)
                print(f"Received {len(audio)} bytes, final={is_final}")
                
                if is_final:
                    break
            else:
                data = json.loads(msg)
                if data.get("event") == "done":
                    print(f"Generation complete: {data}")
                    # Save AI timestamps for next turn
                    ai_timestamps = data.get("ai_timestamps", [])
                    break
        
        # Save output audio
        all_audio = b"".join(audio_chunks)
        with wave.open("output.wav", "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(all_audio)
        
        print(f"Saved output.wav ({len(all_audio)} bytes)")

asyncio.run(demo())
```

## Success Criteria

1. **Functional**: Complete conversation turn works end-to-end
2. **Latency**: ~1.5s to first audio (no Whisper overhead)
3. **Quality**: Natural conversational prosody maintained
4. **Streaming**: Audio streams as it's generated
5. **Timestamps**: AI timestamps returned for next turn
6. **Stable**: No crashes or memory leaks over multiple turns

## Future Improvements

### If Faster Model Becomes Available:
1. Swap Dia2 for faster model (KyutaiTTS, future Dia3, etc.)
2. Keep same API - just change the backend
3. Conversation context handling may need adjustment

### Silence Padding for Lower Perceived Latency:
1. Start decoding before 18 frames are ready
2. Pad missing frames with silence tokens
3. First ~1s has artifacts but audio starts in ~500ms
4. Implement as optional flag: `"early_decode": true`

### Mindroot Integration:
1. Create Mindroot plugin that connects to Yandia2
2. Pipe Deepgram timestamps directly
3. Handle VAD → STT → LLM → TTS flow
