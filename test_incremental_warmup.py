#!/usr/bin/env python3
"""Test script for incremental warmup functionality.

This tests the IncrementalWarmup class and the streaming conversation handler
with pre-warmed KV cache across multiple turns.

Usage:
    python test_incremental_warmup.py

Environment variables:
    SERVER_URL - HTTP server URL (default: http://localhost:3030)
    WS_URL - WebSocket URL for conversation_stream (default: derived from SERVER_URL)
"""
import asyncio
import json
import time
import wave
import struct
import os
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    exit(1)


# Configurable via environment variables (same pattern as test_streaming.py)
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:3030")

# Derive WS_URL from SERVER_URL if not explicitly set
if "WS_URL" in os.environ:
    WS_URL = os.environ["WS_URL"]
else:
    # Convert http(s):// to ws(s)://
    if SERVER_URL.startswith("https://"):
        WS_URL = "wss://" + SERVER_URL[8:] + "/ws/conversation_stream"
    else:
        WS_URL = "ws://" + SERVER_URL[7:] + "/ws/conversation_stream"

AI_VOICE_PATH = "example_prefix1.wav"  # Adjust path as needed
USER_VOICE_PATH = "example_prefix2.wav"  # For simulating user audio

# Fake timestamps for testing (would come from Deepgram in production)
AI_VOICE_TIMESTAMPS = [
    {"word": "Hello", "start": 0.0, "end": 0.3},
    {"word": "there", "start": 0.35, "end": 0.6},
]

USER_TIMESTAMPS = [
    {"word": "Hi", "start": 0.0, "end": 0.2},
    {"word": "how", "start": 0.25, "end": 0.4},
    {"word": "are", "start": 0.45, "end": 0.6},
    {"word": "you", "start": 0.65, "end": 0.85},
]


def load_audio_as_pcm(path: str) -> bytes:
    """Load a WAV file and return raw PCM bytes."""
    with wave.open(path, 'rb') as wav:
        return wav.readframes(wav.getnframes())


async def receive_generation(ws, gen_start: float) -> tuple:
    """Receive audio chunks until generation is complete.
    
    Returns: (audio_chunks, result_data, first_chunk_latency)
    """
    audio_chunks = []
    first_chunk_time = None
    result_data = None
    
    while True:
        msg = await ws.recv()
        
        if isinstance(msg, bytes):
            is_final = struct.unpack("!?", msg[:1])[0]
            audio = msg[1:]
            audio_chunks.append(audio)
            
            if first_chunk_time is None:
                first_chunk_time = time.time()
                latency = first_chunk_time - gen_start
                print(f"    First chunk! Latency: {latency:.3f}s")
            
            if is_final:
                print(f"    Final chunk received")
        else:
            data = json.loads(msg)
            if data.get("event") == "done":
                result_data = data
                break
            elif data.get("event") == "error":
                print(f"    ERROR: {data['message']}")
                return [], data, None
    
    first_latency = first_chunk_time - gen_start if first_chunk_time else None
    return audio_chunks, result_data, first_latency


async def test_multi_turn_conversation():
    """Test multi-turn conversation with incremental warmup."""
    print(f"\n{'='*60}")
    print("Testing Multi-Turn Incremental Warmup")
    print(f"Server URL: {SERVER_URL}")
    print(f"WebSocket URL: {WS_URL}")
    print(f"{'='*60}\n")
    
    # Check if user audio exists for multi-turn test
    has_user_audio = Path(USER_VOICE_PATH).exists()
    if not has_user_audio:
        print(f"NOTE: {USER_VOICE_PATH} not found, will test without user audio")
    
    try:
        async with websockets.connect(WS_URL) as ws:
            # 1. Wait for ready message
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"[1] Server ready: {data}")
            assert data["event"] == "ready"
            
            # 2. Initialize with AI voice
            print(f"\n[2] Initializing with AI voice: {AI_VOICE_PATH}")
            init_start = time.time()
            
            await ws.send(json.dumps({
                "type": "init",
                "ai_voice_path": AI_VOICE_PATH,
                "ai_voice_timestamps": AI_VOICE_TIMESTAMPS,
            }))
            
            msg = await ws.recv()
            data = json.loads(msg)
            init_time = time.time() - init_start
            print(f"    Init response: {data}")
            print(f"    Init time: {init_time:.3f}s")
            
            if data["event"] == "error":
                print(f"    ERROR: {data['message']}")
                return False
            
            assert data["event"] == "initialized"
            
            # ============ TURN 1: AI speaks (no user audio) ============
            print(f"\n{'='*50}")
            print("TURN 1: AI speaks (pre-warmed AI voice)")
            print(f"{'='*50}")
            
            gen_start = time.time()
            await ws.send(json.dumps({
                "type": "generate",
                "ai_text": "Hello! Welcome to the incremental warmup test. How can I help you today?",
            }))
            
            chunks, result, latency = await receive_generation(ws, gen_start)
            
            print(f"\n    Turn 1 Results:")
            print(f"    - Duration: {result.get('duration', 0):.2f}s")
            print(f"    - Total time: {result.get('total_time', 0):.3f}s")
            print(f"    - First chunk latency: {latency:.3f}s" if latency else "    - No audio generated")
            print(f"    - Used incremental warmup: {result.get('used_incremental_warmup', False)}")
            print(f"    - Turn count: {result.get('turn_count', 'N/A')}")
            
            # Save turn 1 audio
            if chunks:
                all_audio = b"".join(chunks)
                with wave.open("test_turn1_output.wav", 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(all_audio)
                print(f"    - Saved: test_turn1_output.wav")
            
            # ============ TURN 2: User speaks, then AI responds ============
            print(f"\n{'='*50}")
            print("TURN 2: User speaks, then AI responds")
            print(f"{'='*50}")
            
            if has_user_audio:
                # Stream user audio
                print(f"\n    Streaming user audio from {USER_VOICE_PATH}...")
                user_pcm = load_audio_as_pcm(USER_VOICE_PATH)
                
                # Send in chunks (simulate real-time streaming)
                chunk_size = 4800  # 100ms at 24kHz 16-bit mono
                chunks_sent = 0
                for i in range(0, len(user_pcm), chunk_size):
                    chunk = user_pcm[i:i+chunk_size]
                    await ws.send(chunk)
                    chunks_sent += 1
                    await asyncio.sleep(0.05)  # Simulate real-time
                
                print(f"    Sent {chunks_sent} audio chunks ({len(user_pcm)} bytes total)")
                
                # Send word timestamps
                for ts in USER_TIMESTAMPS:
                    await ws.send(json.dumps({
                        "type": "word",
                        "word": ts["word"],
                        "start": ts["start"],
                        "end": ts["end"],
                    }))
                print(f"    Sent {len(USER_TIMESTAMPS)} word timestamps")
            else:
                print(f"    (Skipping user audio - file not found)")
            
            # Generate AI response
            print(f"\n    Generating AI response...")
            gen_start = time.time()
            await ws.send(json.dumps({
                "type": "generate",
                "ai_text": "I'm doing great, thank you for asking! The incremental warmup should make this response faster.",
            }))
            
            chunks, result, latency = await receive_generation(ws, gen_start)
            
            print(f"\n    Turn 2 Results:")
            print(f"    - Duration: {result.get('duration', 0):.2f}s")
            print(f"    - Total time: {result.get('total_time', 0):.3f}s")
            print(f"    - First chunk latency: {latency:.3f}s" if latency else "    - No audio generated")
            print(f"    - Used incremental warmup: {result.get('used_incremental_warmup', False)}")
            print(f"    - Has user audio: {result.get('has_user_audio', False)}")
            print(f"    - Turn count: {result.get('turn_count', 'N/A')}")
            
            # Save turn 2 audio
            if chunks:
                all_audio = b"".join(chunks)
                with wave.open("test_turn2_output.wav", 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(all_audio)
                print(f"    - Saved: test_turn2_output.wav")
            
            # ============ TURN 3: Another user turn ============
            print(f"\n{'='*50}")
            print("TURN 3: Another exchange")
            print(f"{'='*50}")
            
            if has_user_audio:
                print(f"\n    Streaming user audio again...")
                for i in range(0, min(len(user_pcm), chunk_size * 5), chunk_size):
                    chunk = user_pcm[i:i+chunk_size]
                    await ws.send(chunk)
                    await asyncio.sleep(0.05)
                print(f"    Sent user audio")
            
            print(f"\n    Generating AI response...")
            gen_start = time.time()
            await ws.send(json.dumps({
                "type": "generate",
                "ai_text": "This is turn three. Each turn should use the previous AI audio for warmup.",
            }))
            
            chunks, result, latency = await receive_generation(ws, gen_start)
            
            print(f"\n    Turn 3 Results:")
            print(f"    - Duration: {result.get('duration', 0):.2f}s")
            print(f"    - Total time: {result.get('total_time', 0):.3f}s")
            print(f"    - First chunk latency: {latency:.3f}s" if latency else "    - No audio generated")
            print(f"    - Used incremental warmup: {result.get('used_incremental_warmup', False)}")
            print(f"    - Turn count: {result.get('turn_count', 'N/A')}")
            
            # Close connection
            await ws.send(json.dumps({"type": "close"}))
            
            print(f"\n{'='*60}")
            print("TEST COMPLETE!")
            print(f"{'='*60}\n")
            return True
            
    except (ConnectionRefusedError, OSError) as e:
        print(f"ERROR: Could not connect to {WS_URL}")
        print("Make sure the server is running: python streaming_server.py")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_multi_turn_conversation())
    exit(0 if success else 1)
