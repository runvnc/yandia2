#!/usr/bin/env python3
import os
"""Test client for the streaming Dia2 server.

Usage:
    python test_streaming.py [text]

Examples:
    python test_streaming.py "Hello, this is a test of streaming TTS."
    python test_streaming.py  # Uses default text

Environment variables:
    SERVER_URL - HTTP server URL (default: http://localhost:8000)
    WS_URL - WebSocket URL (default: ws://localhost:8000/ws/generate)
    VOICE_FILE - Path to voice warmup file (default: example_prefix1.wav)
    USE_ORIGINAL_LOOP - Set to "1" to use diagnostic mode with original generation loop
"""
import asyncio
import json
import sys
import time
import wave
import struct
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)


# Configurable via environment variables
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
VOICE_FILE = os.environ.get("VOICE_FILE", "example_prefix1.wav")
USE_ORIGINAL_LOOP = os.environ.get("USE_ORIGINAL_LOOP", "") == "1"

# Derive WS_URL from SERVER_URL if not explicitly set
if "WS_URL" in os.environ:
    WS_URL = os.environ["WS_URL"]
else:
    # Convert http(s):// to ws(s)://
    if SERVER_URL.startswith("https://"):
        WS_URL = "wss://" + SERVER_URL[8:] + "/ws/generate"
    else:
        WS_URL = "ws://" + SERVER_URL[7:] + "/ws/generate"



async def setup_voice(server_url: str, voice_file: str):
    """Upload voice warmup file."""
    print(f"Setting voice from: {voice_file}")
    # Use longer timeout for Whisper transcription (can take 30-60s on first run)
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(voice_file, "rb") as f:
            response = await client.post(
                f"{server_url}/set_voice",
                files={"file": ("voice.wav", f, "audio/wav")}
            )
        response.raise_for_status()
        print(f"Voice set: {response.json()}")


async def stream_tts(text: str, output_file: str = "streaming_output.wav"):
    """Stream TTS and save to file."""
    print(f"\nConnecting to {WS_URL}...")
    
    async with websockets.connect(WS_URL) as ws:
        # Wait for ready
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"Server: {data}")
        
        if data.get("error"):
            print(f"Error: {data['error']}")
            return
        
        # Send config if using original loop
        if USE_ORIGINAL_LOOP:
            print("\n*** DIAGNOSTIC MODE: Using original generation loop ***\n")
            await ws.send(json.dumps({"type": "config", "use_original_loop": True}))
        else:
            print("\n*** STREAMING MODE: Using custom streaming loop ***\n")
        
        sample_rate = data.get("sample_rate", 24000)
        
        # Send text
        print(f"\nSending: {text}")
        await ws.send(json.dumps({"text": text}))
        
        # Receive generating event
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"Server: {data}")
        
        # Collect audio chunks
        audio_chunks = []
        start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                # Binary audio chunk
                is_final = struct.unpack("!?", msg[:1])[0]
                audio_data = msg[1:]
                audio_chunks.append(audio_data)
                chunk_count += 1
                
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    print(f"First chunk received after {first_chunk_time:.2f}s ({len(audio_data)} bytes)")
                else:
                    print(f"Chunk {chunk_count}: {len(audio_data)} bytes, final={is_final}")
                
            else:
                # JSON message
                data = json.loads(msg)
                print(f"Server: {data}")
                
                if data.get("event") == "done":
                    break
                elif data.get("error"):
                    print(f"Error: {data['error']}")
                    break
        
        total_time = time.time() - start_time
        
        # Save audio
        if audio_chunks:
            all_audio = b"".join(audio_chunks)
            samples = len(all_audio) // 2
            duration = samples / sample_rate
            
            with wave.open(output_file, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(all_audio)
            
            print(f"\n=== Results ===")
            print(f"Output saved to: {output_file}")
            print(f"Audio duration: {duration:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            print(f"Time to first chunk: {first_chunk_time:.2f}s")
            print(f"RTF: {total_time/duration:.2f}")
            print(f"Chunks received: {chunk_count}")
        else:
            print("No audio received!")


async def main():
    # Check if server is running
    try:
        # Use longer timeout for initial health check too
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{SERVER_URL}/health")
            health = response.json()
            print(f"Server health: {health}")
    except Exception as e:
        print(f"Could not connect to server at {SERVER_URL}: {e}")
        print("Make sure the server is running: python streaming_server.py")
        return
    
    # Setup voice if not initialized
    if not health.get("conversation", {}).get("initialized"):
        voice_file = Path(__file__).parent / VOICE_FILE
        if not voice_file.exists():
            print(f"Voice file not found: {voice_file}")
            print("Please provide a voice warmup file.")
            return
        await setup_voice(SERVER_URL, str(voice_file))
    
    # Get text from command line or use default
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello! This is a test of the streaming text to speech system. I hope you can hear me clearly."
    
    # Stream TTS
    await stream_tts(text)


if __name__ == "__main__":
    asyncio.run(main())
