#!/bin/bash

# Fly.io deployment script for Nupo RAG Server
set -e

echo "🚀 Starting Fly.io deployment for Nupo RAG Server..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first:"
    echo "curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if user is logged in
if ! flyctl auth whoami &> /dev/null; then
    echo "🔐 Please log in to Fly.io first:"
    echo "flyctl auth login"
    exit 1
fi

# Create app if it doesn't exist
echo "📦 Creating Fly.io app..."
if ! flyctl apps list | grep -q "nupo-rag-server"; then
    flyctl apps create nupo-rag-server --org personal
    echo "✅ App created: nupo-rag-server"
else
    echo "✅ App already exists: nupo-rag-server"
fi

# Create volumes if they don't exist
echo "💾 Setting up persistent volumes..."

# Check and create data volume
if ! flyctl volumes list -a nupo-rag-server | grep -q "nupo_rag_data"; then
    flyctl volumes create nupo_rag_data --region ams --size 10 -a nupo-rag-server
    echo "✅ Created data volume (10GB)"
else
    echo "✅ Data volume already exists"
fi

# Check and create config volume
if ! flyctl volumes list -a nupo-rag-server | grep -q "nupo_rag_config"; then
    flyctl volumes create nupo_rag_config --region ams --size 1 -a nupo-rag-server
    echo "✅ Created config volume (1GB)"
else
    echo "✅ Config volume already exists"
fi

# Set secrets (if provided)
echo "🔑 Setting up environment secrets..."
if [ ! -z "$OPENAI_API_KEY" ]; then
    flyctl secrets set OPENAI_API_KEY="$OPENAI_API_KEY" -a nupo-rag-server
    echo "✅ OpenAI API key set"
else
    echo "ℹ️  OPENAI_API_KEY not provided - can be set via UI after deployment"
fi

# Deploy the application
echo "🚀 Deploying application..."
flyctl deploy -a nupo-rag-server

# Show deployment status
echo "📊 Deployment status:"
flyctl status -a nupo-rag-server

# Show app URL
echo "🌐 Your RAG server is deployed at:"
echo "https://nupo-rag-server.fly.dev"

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Visit https://nupo-rag-server.fly.dev to access your RAG server"
echo "2. Set your OpenAI API key via the web interface"
echo "3. The scraped data and vector database are persisted on volumes"
echo "4. The server will stay alive with health checks"
echo ""
echo "🔧 Useful commands:"
echo "flyctl logs -a nupo-rag-server          # View logs"
echo "flyctl ssh console -a nupo-rag-server   # SSH into container"
echo "flyctl scale count 1 -a nupo-rag-server # Ensure 1 instance running"
