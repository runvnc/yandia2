#!/bin/bash
# Test the streaming Dia2 server

SERVER=${SERVER:-http://localhost:8000}
VOICE_FILE=${VOICE_FILE:-example_prefix1.wav}

echo "Testing Dia2 Streaming Server at $SERVER"
echo "========================================"

# Check health
echo -e "\n1. Checking server health..."
curl -s "$SERVER/health" | python3 -m json.tool

# Set voice
echo -e "\n2. Setting AI voice..."
curl -s -X POST "$SERVER/set_voice" \
    -F "file=@$VOICE_FILE" | python3 -m json.tool

# Test streaming with Python client
echo -e "\n3. Testing streaming TTS..."
python3 test_streaming.py "Hello! This is a test of streaming text to speech."

echo -e "\n4. Playing output..."
if command -v aplay &> /dev/null; then
    aplay streaming_output.wav
elif command -v play &> /dev/null; then
    play streaming_output.wav
elif command -v afplay &> /dev/null; then
    afplay streaming_output.wav
else
    echo "No audio player found. Output saved to streaming_output.wav"
fi

echo -e "\nDone!"
