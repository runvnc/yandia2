#!/bin/bash
# Test script for the simple Dia2 server

SERVER="http://localhost:8000"

echo "=== Testing Dia2 Simple Server ==="
echo ""

# Check health
echo "1. Checking health..."
curl -s "$SERVER/health" | python3 -m json.tool
echo ""

# Test TTS with the example prefix files
echo "2. Testing TTS with example prefixes..."
echo "   This will take a while on first run (Whisper transcription)..."

curl -X POST "$SERVER/tts" \
  -F "text=[S1] Hello! This is a test of the Dia 2 text to speech system. How does it sound?" \
  -F "speaker_1=/files/yandia2/example_prefix1.wav" \
  -F "speaker_2=/files/yandia2/example_prefix2.wav" \
  -F "cfg_scale=2.0" \
  -F "temperature=0.8" \
  --output /tmp/test_output.wav

echo ""
echo "Output saved to /tmp/test_output.wav"
echo "Duration: $(soxi -D /tmp/test_output.wav 2>/dev/null || echo 'install sox to see duration')s"
echo ""

# Test second request (should be faster due to caching)
echo "3. Testing second TTS (should be faster - cached transcription)..."

time curl -X POST "$SERVER/tts" \
  -F "text=[S1] This is a second test. The transcription should be cached now." \
  -F "speaker_1=/files/yandia2/example_prefix1.wav" \
  -F "speaker_2=/files/yandia2/example_prefix2.wav" \
  --output /tmp/test_output2.wav

echo ""
echo "Output saved to /tmp/test_output2.wav"
