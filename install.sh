#!/bin/bash
# Install script for yandia2 with Kyutai Mimi streaming support

set -e

echo "Installing yandia2 dependencies..."

# Install main dependencies
pip install -r requirements.txt

# Install moshi from git with --no-deps to bypass version conflicts
# (moshi requires torch<2.8 and sphn<0.2.0 but we need newer versions)
echo "Installing moshi from git (with --no-deps to bypass version conflicts)..."
pip install --no-deps git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi

echo "Installation complete!"
echo "Run with: python streaming_server.py"
