#!/usr/bin/env python3
"""
Simple launcher for the RAG Chat System.
This makes it super easy to start your chat interface!
"""

import sys
import webbrowser
import time
import signal
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.chat_api import run_server
from loguru import logger


def main():
    """Launch the RAG chat system"""
    print("🚀 Starting RAG Chat System...")
    print("=" * 50)
    
    # Configure nice logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    
    print(f"🌐 Chat interface will be available at: {url}")
    print("📚 Upload your documents and start chatting!")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)  # Wait for server to start
        try:
            webbrowser.open(url)
            logger.info("🌐 Opened browser automatically")
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
            logger.info(f"Please open {url} in your browser manually")
    
    # Start browser opening in background
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the server
    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        print("\n👋 Goodbye! RAG Chat System stopped.")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        print("\n❌ Failed to start the chat system.")
        print("Make sure you have installed all requirements:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
