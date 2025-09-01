#!/usr/bin/env python3
"""
Simple setup script to configure OpenAI API key for the RAG chat system.
"""

import os
import sys
from pathlib import Path

def setup_openai_api_key():
    """Setup OpenAI API key in environment"""
    print("🚀 OpenAI API Key Setup")
    print("=" * 50)
    
    # Check if API key is already set
    current_key = os.getenv('OPENAI_API_KEY')
    if current_key:
        print(f"✅ OpenAI API key is already configured (ends with: ...{current_key[-8:]})")
        response = input("Do you want to update it? (y/N): ").lower().strip()
        if response != 'y':
            print("Keeping existing API key.")
            return
    
    print("\n📝 To get your OpenAI API key:")
    print("1. Go to: https://platform.openai.com/api-keys")
    print("2. Sign in to your OpenAI account")
    print("3. Click 'Create new secret key'")
    print("4. Copy the key (starts with 'sk-')")
    print()
    
    api_key = input("🔑 Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key entered. Exiting.")
        return
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key should start with 'sk-'")
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            return
    
    # Method 1: Set for current session
    os.environ['OPENAI_API_KEY'] = api_key
    print("✅ API key set for current session")
    
    # Method 2: Create .env file for persistence
    env_file = Path('.env')
    env_content = f"OPENAI_API_KEY={api_key}\n"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            existing_content = f.read()
        
        # Replace existing key or add new one
        lines = existing_content.split('\n')
        updated_lines = []
        key_found = False
        
        for line in lines:
            if line.startswith('OPENAI_API_KEY='):
                updated_lines.append(f"OPENAI_API_KEY={api_key}")
                key_found = True
            else:
                updated_lines.append(line)
        
        if not key_found:
            updated_lines.append(f"OPENAI_API_KEY={api_key}")
        
        env_content = '\n'.join(updated_lines)
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ API key saved to {env_file}")
    print()
    print("🎉 Setup complete!")
    print()
    print("💡 Next steps:")
    print("1. Restart your RAG chat server: ./start_rag_chat.sh")
    print("2. Check the 'Settings' section for API status")
    print("3. Try asking: 'can i use nupo when im pregnant'")
    print()
    print("💰 Cost info:")
    print("- GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens")
    print("- Typical question: ~$0.001-0.005 per response")

if __name__ == "__main__":
    try:
        setup_openai_api_key()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)
