#!/bin/bash
# Test script for the Dia2 Conversation Server
# Run this from the yandia2 directory

# Default to localhost, override with: SERVER=https://your-url ./test_conversation.sh
SERVER="${SERVER:-http://localhost:8000}"
OUTDIR="/tmp/dia2_conversation"

echo "=== Testing Dia2 Conversation Server ==="
echo "Server: $SERVER"
echo "Output dir: $OUTDIR"
echo ""

# Create output directory
mkdir -p "$OUTDIR"

# Check health
echo "1. Checking server health..."
curl -s "$SERVER/health" | python3 -m json.tool
echo ""

# Reset conversation
echo "2. Resetting conversation..."
curl -s -X POST "$SERVER/reset" | python3 -m json.tool
echo ""

# Set AI voice (using example_prefix1.wav as the AI warmup)
echo "3. Setting AI voice (this will transcribe - takes ~10-20s first time)..."
cp example_prefix1.wav "$OUTDIR/00_ai_warmup.wav"
echo "   Copied example_prefix1.wav -> $OUTDIR/00_ai_warmup.wav"
time curl -s -X POST "$SERVER/set_voice" \
  -F "file=@example_prefix1.wav" | python3 -m json.tool
echo ""

# Simulate user speaking (using example_prefix2.wav as user audio)
echo "4. User spoke (this will transcribe - takes ~10-20s first time)..."
cp example_prefix2.wav "$OUTDIR/01_user_spoke.wav"
echo "   Copied example_prefix2.wav -> $OUTDIR/01_user_spoke.wav"
time curl -s -X POST "$SERVER/user_spoke" \
  -F "file=@example_prefix2.wav" | python3 -m json.tool
echo ""

# Generate AI response
echo "5. Generating AI response..."
time curl -s -X POST "$SERVER/generate" \
  -F "text=Hello! Thank you for calling. How can I help you today?" \
  -F "cfg_scale=1.0" \
  -F "temperature=0.8" \
  --output "$OUTDIR/02_ai_response.wav"
echo ""
echo "   Output: $OUTDIR/02_ai_response.wav"

# Check state
echo ""
echo "6. Current conversation state:"
curl -s "$SERVER/state" | python3 -m json.tool
echo ""

# Generate another response (user didn't speak, just AI continues)
echo "7. Generating second AI response (no new user audio)..."
time curl -s -X POST "$SERVER/generate" \
  -F "text=I'm Mary from customer service. Are you calling about your account?" \
  --output "$OUTDIR/03_ai_response.wav"
echo ""
echo "   Output: $OUTDIR/03_ai_response.wav"

echo ""
echo "=== Test Complete ==="
echo ""
echo "Conversation files in $OUTDIR:"
ls -la "$OUTDIR/"*.wav 2>/dev/null
echo ""
echo "To play the full conversation:"
echo "  for f in $OUTDIR/*.wav; do echo Playing \$f; aplay \$f; done"
echo ""
echo "Or concatenate into one file:"
echo "  sox $OUTDIR/*.wav $OUTDIR/full_conversation.wav"
