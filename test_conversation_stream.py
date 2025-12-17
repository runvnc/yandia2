#!/usr/bin/env python3
"""Test script for streaming conversation WebSocket endpoint.

This demonstrates the streaming conversation flow:
1. Initialize with AI voice + timestamps
2. Stream user audio chunks (simulating real-time input)
3. Send word timestamps (simulating Deepgram)
4. Trigger generation and receive streaming audio

Usage:
    python test_conversation_stream.py [--server URL] [--ai-voice PATH]
"""
import asyncio
import argparse
import json
import struct
import time
import wave
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


async def test_streaming_conversation(
    server_url: str = "ws://localhost:3030/ws/conversation_stream",
    ai_voice_path: str = "example_prefix1.wav",
    user_audio_path: str = "example_prefix2.wav",
):
    """Test the streaming conversation endpoint."""
    print(f"Connecting to {server_url}...")
    
    async with websockets.connect(server_url) as ws:
        # Wait for ready
        response = json.loads(await ws.recv())
        print(f"Server ready: {response}")
        
        sample_rate = response.get("sample_rate", 24000)
        
        # 1. Initialize with AI voice
        print(f"\n=== Initializing with AI voice: {ai_voice_path} ===")
        
        # For demo, create fake timestamps based on audio duration
        ai_timestamps = [
            {"word": "Hello", "start": 0.0, "end": 0.3},
            {"word": "there", "start": 0.35, "end": 0.6},
        ]
        
        await ws.send(json.dumps({
            "type": "init",
            "ai_voice_path": ai_voice_path,
            "ai_voice_timestamps": ai_timestamps,
        }))
        
        response = json.loads(await ws.recv())
        print(f"Init response: {response}")
        
        if response.get("event") == "error":
            print(f"Error: {response.get('message')}")
            return
        
        # 2. Stream user audio (simulating real-time input)
        print(f"\n=== Streaming user audio: {user_audio_path} ===")
        
        if Path(user_audio_path).exists():
            with wave.open(user_audio_path, 'rb') as wav:
                # Read audio data
                audio_data = wav.readframes(wav.getnframes())
                wav_sample_rate = wav.getframerate()
                
            print(f"User audio: {len(audio_data)} bytes, {wav_sample_rate}Hz")
            
            # Stream in chunks (simulating real-time)
            chunk_size = sample_rate * 2 // 10  # 100ms chunks (16-bit = 2 bytes/sample)
            chunks_sent = 0
            
            stream_start = time.time()
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                await ws.send(chunk)
                chunks_sent += 1
                await asyncio.sleep(0.05)  # Simulate real-time streaming
            
            stream_time = time.time() - stream_start
            print(f"Streamed {chunks_sent} chunks in {stream_time:.2f}s")
            
            # 3. Send word timestamps (simulating Deepgram)
            print(f"\n=== Sending word timestamps ===")
            
            user_timestamps = [
                {"word": "How", "start": 0.0, "end": 0.2},
                {"word": "are", "start": 0.25, "end": 0.4},
                {"word": "you", "start": 0.45, "end": 0.6},
                {"word": "today", "start": 0.65, "end": 1.0},
            ]
            
            for ts in user_timestamps:
                await ws.send(json.dumps({
                    "type": "word",
                    "word": ts["word"],
                    "start": ts["start"],
                    "end": ts["end"],
                }))
            
            print(f"Sent {len(user_timestamps)} word timestamps")
        else:
            print(f"User audio file not found: {user_audio_path}")
            print("Continuing without user audio...")
        
        # 4. Trigger generation
        print(f"\n=== Triggering generation ===")
        
        ai_text = "I'm doing great, thanks for asking! How can I help you today?"
        
        gen_start = time.time()
        await ws.send(json.dumps({
            "type": "generate",
            "ai_text": ai_text,
        }))
        
        # Receive audio stream
        first_chunk_time = None
        audio_chunks = []
        total_bytes = 0
        
        print("Receiving audio stream...")
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - gen_start
                    print(f"\n*** First audio chunk received! Latency: {latency:.3f}s ***\n")
                
                is_final = struct.unpack("!?", msg[:1])[0]
                audio = msg[1:]
                audio_chunks.append(audio)
                total_bytes += len(audio)
                
                if is_final:
                    print(f"Final chunk received. Total: {total_bytes} bytes")
                    break
            else:
                data = json.loads(msg)
                event = data.get("event")
                
                if event == "done":
                    print(f"\nGeneration complete: {data}")
                    break
                elif event == "error":
                    print(f"Error: {data.get('message')}")
                    break
                else:
                    print(f"Event: {data}")
        
        total_time = time.time() - gen_start
        
        # Save output
        if audio_chunks:
            all_audio = b"".join(audio_chunks)
            output_path = "test_stream_output.wav"
            
            with wave.open(output_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(all_audio)
            
            duration = len(all_audio) / 2 / sample_rate
            print(f"\nSaved {output_path}: {duration:.2f}s audio")
        
        print(f"\n=== Summary ===")
        print(f"Total generation time: {total_time:.3f}s")
        if first_chunk_time:
            print(f"Time to first audio: {first_chunk_time - gen_start:.3f}s")
        
        # 5. Reset for next turn
        await ws.send(json.dumps({"type": "reset"}))
        response = json.loads(await ws.recv())
        print(f"Reset: {response}")


def main():
    parser = argparse.ArgumentParser(description="Test streaming conversation endpoint")
    parser.add_argument("--server", default="ws://localhost:3030/ws/conversation_stream",
                        help="WebSocket server URL")
    parser.add_argument("--ai-voice", default="example_prefix1.wav",
                        help="Path to AI voice audio file")
    parser.add_argument("--user-audio", default="example_prefix2.wav",
                        help="Path to user audio file (for simulation)")
    args = parser.parse_args()
    
    asyncio.run(test_streaming_conversation(
        server_url=args.server,
        ai_voice_path=args.ai_voice,
        user_audio_path=args.user_audio,
    ))


if __name__ == "__main__":
    main()
