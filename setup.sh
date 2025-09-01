#!/bin/bash

# Nupo AI RAG System Setup Script
# Production-Ready RAG with OpenAI Embeddings

set -e

echo "🚀 Setting up Nupo AI RAG System..."
echo "=================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/vector_db
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p models/fine_tuned

# Copy environment file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment file..."
    cp env.example .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
else
    echo "✅ Environment file already exists"
fi

# Set permissions
echo "🔒 Setting permissions..."
chmod +x start_rag_chat.sh
chmod +x scripts/start_chat_safe.py

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env and add your OPENAI_API_KEY"
echo "2. Run: ./start_rag_chat.sh"
echo "3. Open http://127.0.0.1:8000 in your browser"
echo ""
echo "🎯 Ready for production deployment!"
