#!/bin/bash
# Simple startup script for the RAG Chat System

echo "🚀 Starting RAG Chat System..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3.10 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "📋 Loading environment variables..."
    export $(cat .env | xargs)
fi

# Check if packages are installed
if ! python -c "import sentence_transformers" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements_simple.txt
    pip install "numpy<2" --force-reinstall
    pip install requests beautifulsoup4 selenium webdriver-manager
    python -c "import nltk; nltk.download('punkt_tab')"
fi

echo "🌐 Starting chat server..."
echo "📱 Your browser should open automatically at http://127.0.0.1:8000"
echo "🛑 Press Ctrl+C to stop"

# Set Python path and start server
export PYTHONPATH="/Users/eriklindberg/Documents/nupo-ai/src"
python scripts/start_chat_safe.py
