# Fly.io Deployment Guide

## 🚀 Quick Deploy

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Login to Fly.io
flyctl auth login

# 3. Deploy (with optional OpenAI API key)
OPENAI_API_KEY=your_key_here ./scripts/deploy.sh
```

Your RAG server will be live at: `https://nupo-rag-server.fly.dev`

## 📋 Detailed Setup

### Prerequisites

1. **Fly.io Account**: Sign up at https://fly.io
2. **flyctl CLI**: Install from https://fly.io/docs/hands-on/install-flyctl/
3. **OpenAI API Key**: Get from https://platform.openai.com/api-keys

### Step-by-Step Deployment

#### 1. Install flyctl
```bash
# macOS/Linux
curl -L https://fly.io/install.sh | sh

# Or via Homebrew
brew install flyctl
```

#### 2. Login to Fly.io
```bash
flyctl auth login
```

#### 3. Create App and Volumes
```bash
# Create the app
flyctl apps create nupo-rag-server --org personal

# Create persistent volumes
flyctl volumes create nupo_rag_data --region ams --size 10 -a nupo-rag-server
flyctl volumes create nupo_rag_config --region ams --size 1 -a nupo-rag-server
```

#### 4. Set Environment Variables
```bash
# Set OpenAI API key (optional - can be set via UI)
flyctl secrets set OPENAI_API_KEY="your_openai_api_key_here" -a nupo-rag-server
```

#### 5. Deploy
```bash
flyctl deploy -a nupo-rag-server
```

## 🏗️ Architecture

### Persistent Storage
- **Data Volume** (10GB): `/app/data` - Vector database and scraped content
- **Config Volume** (1GB): `/app/config` - API keys and configuration

### Auto-Scaling
- **Min Instances**: 1 (always running)
- **Max Instances**: 3 (scales under load)
- **Auto-stop**: Disabled (keeps server alive)

### Health Monitoring
- **Health Check**: `/api/health` every 10 seconds
- **Keep-alive**: `/api/keep-alive` endpoint
- **Status Check**: `/api/status` for system status

## 🔧 Configuration

### Environment Variables
```bash
PORT=8000                    # Server port
HOST=0.0.0.0                # Bind to all interfaces
PYTHONUNBUFFERED=1          # Real-time logs
FLY_KEEP_ALIVE=1            # Prevent sleeping
```

### Fly.io Settings
```toml
# fly.toml
auto_stop_machines = false   # Keep running
auto_start_machines = true   # Auto-restart if crashed
min_machines_running = 1     # Always 1 instance
```

## 📊 Monitoring

### Check Status
```bash
# App status
flyctl status -a nupo-rag-server

# View logs
flyctl logs -a nupo-rag-server

# SSH into container
flyctl ssh console -a nupo-rag-server
```

### Health Endpoints
- `GET /api/health` - Detailed health check
- `GET /api/status` - RAG system status
- `GET /api/keep-alive` - Keep server awake

## 💾 Data Persistence

### What's Persisted
✅ **Vector Database** - FAISS index files  
✅ **Scraped Content** - All website data  
✅ **API Keys** - Encrypted storage  
✅ **Configuration** - System settings  

### What's NOT Persisted
❌ **Logs** - Use `flyctl logs` to view  
❌ **Temporary Files** - Cleaned on restart  
❌ **Cache** - Rebuilt automatically  

### Backup Data
```bash
# SSH into container
flyctl ssh console -a nupo-rag-server

# Create backup
tar -czf /tmp/backup.tar.gz /app/data /app/config

# Download backup (from local machine)
flyctl ssh sftp get /tmp/backup.tar.gz . -a nupo-rag-server
```

## 🔄 Updates & Maintenance

### Deploy Updates
```bash
# Deploy latest changes
flyctl deploy -a nupo-rag-server

# Force rebuild
flyctl deploy --build-only -a nupo-rag-server
```

### Scale Resources
```bash
# Scale memory/CPU
flyctl scale vm shared-cpu-2x --memory 4096 -a nupo-rag-server

# Scale instances
flyctl scale count 2 -a nupo-rag-server
```

### Update Environment
```bash
# Update API key
flyctl secrets set OPENAI_API_KEY="new_key" -a nupo-rag-server

# List secrets
flyctl secrets list -a nupo-rag-server
```

## 🐛 Troubleshooting

### Common Issues

#### App Won't Start
```bash
# Check logs
flyctl logs -a nupo-rag-server

# Check app status
flyctl status -a nupo-rag-server

# Restart app
flyctl apps restart nupo-rag-server
```

#### Volume Issues
```bash
# List volumes
flyctl volumes list -a nupo-rag-server

# Check volume usage
flyctl ssh console -a nupo-rag-server
df -h
```

#### Performance Issues
```bash
# Check resource usage
flyctl ssh console -a nupo-rag-server
top
free -h

# Scale up if needed
flyctl scale vm shared-cpu-4x --memory 8192 -a nupo-rag-server
```

### Debug Commands
```bash
# View real-time logs
flyctl logs -f -a nupo-rag-server

# SSH into running container
flyctl ssh console -a nupo-rag-server

# Check environment variables
flyctl ssh console -a nupo-rag-server -C "env | grep -E '(PORT|HOST|OPENAI)'"

# Test health endpoint
curl https://nupo-rag-server.fly.dev/api/health
```

## 💰 Cost Optimization

### Resource Usage
- **CPU**: shared-cpu-2x (~$10-15/month)
- **Memory**: 2GB RAM
- **Storage**: 11GB volumes (~$2-3/month)
- **Bandwidth**: First 160GB free

### Cost Reduction Tips
1. **Use shared CPU** instead of dedicated
2. **Monitor usage** with `flyctl status`
3. **Scale down** during low usage periods
4. **Optimize Docker image** size

## 🔐 Security

### Best Practices
✅ **API keys encrypted** at rest  
✅ **HTTPS enforced** for all traffic  
✅ **Non-root user** in container  
✅ **Minimal dependencies** in production  
✅ **Regular security updates**  

### Access Control
- **Admin access**: Via Fly.io dashboard
- **SSH access**: `flyctl ssh console`
- **API access**: Public endpoints (consider auth)

## 📈 Performance

### Expected Performance
- **Response Time**: < 2 seconds for queries
- **Throughput**: 50+ concurrent requests
- **Uptime**: 99.9% with health checks
- **Memory Usage**: ~1.5GB under load

### Optimization Tips
1. **Enable caching** for frequent queries
2. **Use CDN** for static assets
3. **Monitor logs** for bottlenecks
4. **Scale horizontally** if needed

---

## 🆘 Quick Reference

### Essential Commands
```bash
flyctl deploy -a nupo-rag-server              # Deploy
flyctl logs -a nupo-rag-server                # View logs
flyctl ssh console -a nupo-rag-server         # SSH access
flyctl status -a nupo-rag-server              # Check status
flyctl scale count 1 -a nupo-rag-server       # Ensure running
```

### Important URLs
- **App**: https://nupo-rag-server.fly.dev
- **Health**: https://nupo-rag-server.fly.dev/api/health
- **Status**: https://nupo-rag-server.fly.dev/api/status
- **Dashboard**: https://fly.io/apps/nupo-rag-server

**Need help?** Check logs first, then visit https://fly.io/docs/