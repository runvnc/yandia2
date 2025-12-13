#!/usr/bin/env python3
import os
"""Test client for the streaming Dia2 server.

Usage:
    python test_streaming.py

Features:
    - Interactive chat loop
    - Real-time audio playback (requires 'aplay' or 'sox')
    - Persistent WebSocket connection

Environment variables:
    SERVER_URL - HTTP server URL (default: http://localhost:3030)
    WS_URL - WebSocket URL (default: ws://localhost:3030/ws/generate)
    VOICE_FILE - Path to voice warmup file (default: example_prefix1.wav)
    PLAYER - Audio player command (default: auto-detect aplay/play)
    NO_DEPFORMER_GRAPHS - Set to "1" to disable depformer CUDA graphs
    USE_TORCH_COMPILE - Set to "1" to enable torch.compile optimization
"""
import asyncio
import json
import sys
import time
import struct
import shutil
import subprocess
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
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:3030")
VOICE_FILE = os.environ.get("VOICE_FILE", "example_prefix1.wav")
NO_DEPFORMER_GRAPHS = os.environ.get("NO_DEPFORMER_GRAPHS", "") == "1"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "") == "1"

# Derive WS_URL from SERVER_URL if not explicitly set
if "WS_URL" in os.environ:
    WS_URL = os.environ["WS_URL"]
else:
    # Convert http(s):// to ws(s)://
    if SERVER_URL.startswith("https://"):
        WS_URL = "wss://" + SERVER_URL[8:] + "/ws/generate"
    else:
        WS_URL = "ws://" + SERVER_URL[7:] + "/ws/generate"


def get_audio_player_cmd(sample_rate=24000):
    """Get command to play raw PCM audio from stdin."""
    if shutil.which("aplay"):
        return ["aplay", "-r", str(sample_rate), "-f", "S16_LE", "-c", "1", "-t", "raw"]
    elif shutil.which("play"):
        # sox
        return ["play", "-r", str(sample_rate), "-b", "16", "-c", "1", "-e", "signed-integer", "-t", "raw", "-"]
    return None


async def setup_voice(server_url: str, voice_file: str):
    """Upload voice warmup file."""
    print(f"Setting voice from: {voice_file}")
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(voice_file, "rb") as f:
            response = await client.post(
                f"{server_url}/set_voice",
                files={"file": ("voice.wav", f, "audio/wav")}
            )
        response.raise_for_status()
        print(f"Voice set: {response.json()}")


async def chat_loop():
    """Interactive chat loop with streaming audio playback."""
    print(f"\nConnecting to {WS_URL}...")
    
    if NO_DEPFORMER_GRAPHS:
        print("*** DEPFORMER CUDA GRAPHS DISABLED ***")
    if USE_TORCH_COMPILE:
        print("*** TORCH.COMPILE ENABLED (first request will be slow due to compilation) ***")
    
    async with websockets.connect(WS_URL) as ws:
        # Wait for ready
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"Server: {data}")
        
        if data.get("error"):
            print(f"Error: {data['error']}")
            return
            
        sample_rate = data.get("sample_rate", 24000)
        player_cmd = get_audio_player_cmd(sample_rate)
        
        if player_cmd:
            print(f"Audio playback enabled using: {' '.join(player_cmd)}")
        else:
            print("Warning: No audio player found (aplay/sox). Audio will not be played.")

        print("\n=== Ready! Type your text and press Enter (Ctrl+C to exit) ===")
        
        while True:
            try:
                text = await asyncio.get_event_loop().run_in_executor(None, input, "\nYou: ")
                if not text.strip():
                    continue
                    
                # Send text with optional flags
                request_data = {
                    "text": text,
                    "conversational": False,  # Default to non-conversational for speed
                    "use_depformer_graphs": not NO_DEPFORMER_GRAPHS,
                    "use_torch_compile": USE_TORCH_COMPILE
                }
                await ws.send(json.dumps(request_data))
                
                # Start player process
                player = None
                if player_cmd:
                    player = subprocess.Popen(
                        player_cmd, 
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL
                    )
                
                # Receive loop
                start_time = time.time()
                first_chunk = True
                
                while True:
                    msg = await ws.recv()
                    
                    if isinstance(msg, bytes):
                        # Binary audio chunk
                        # is_final = struct.unpack("!?", msg[:1])[0]
                        audio_data = msg[1:]
                        
                        if first_chunk:
                            latency = time.time() - start_time
                            print(f"[Latency: {latency:.3f}s] Playing...", end="", flush=True)
                            first_chunk = False
                        else:
                            print(".", end="", flush=True)
                            
                        if player:
                            player.stdin.write(audio_data)
                            player.stdin.flush()
                        
                    else:
                        # JSON message
                        data = json.loads(msg)
                        if data.get("event") == "done":
                            print(f" Done ({data.get('duration', 0):.2f}s)")
                            break
                        elif data.get("event") == "generating":
                            print("Generating...", end=" ", flush=True)
                        elif data.get("error"):
                            print(f"\nError: {data['error']}")
                            break
                
                if player:
                    player.stdin.close()
                    player.wait()
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
                break


async def main():
    # Check server health
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SERVER_URL}/health")
            health = response.json()
    except Exception as e:
        print(f"Could not connect to server at {SERVER_URL}: {e}")
        return
    
    # Setup voice if needed
    if not health.get("conversation", {}).get("initialized"):
        voice_file = Path(__file__).parent / VOICE_FILE
        if not voice_file.exists():
            print(f"Voice file not found: {voice_file}")
            return
        await setup_voice(SERVER_URL, str(voice_file))
    
    await chat_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
