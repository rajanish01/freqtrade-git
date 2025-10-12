#!/bin/bash

echo "🔧 Starting entrypoint script..."
echo "📋 Current user: $(whoami)"
echo "📂 Current directory: $(pwd)"

echo "🔍 Checking freqtrade installation..."
if command -v freqtrade >/dev/null 2>&1; then
    echo "✅ freqtrade command found at: $(which freqtrade)"
    freqtrade --version || echo "❌ freqtrade --version failed"
else
    echo "❌ freqtrade command not found"
    echo "🔍 Checking Python path..."
    python3 -c "import sys; print('Python path:', sys.path)"
    echo "🔍 Checking installed packages..."
    pip list | grep freqtrade || echo "❌ freqtrade not in pip list"
    exit 1
fi

echo "🔧 Generating config from environment..."
python3 /freqtrade/generate-config.py || {
    echo "❌ Failed to generate config"
    exit 1
}

echo "📄 Checking generated config..."
if [ -f "/freqtrade/user_data/config.json" ]; then
    echo "✅ Config file exists"
else
    echo "❌ Config file missing"
    exit 1
fi

echo "🚀 Starting Freqtrade with args: $@"
exec freqtrade "$@"