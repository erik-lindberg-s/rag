#!/usr/bin/env python3
"""
Safe launcher for the RAG Chat System that prevents segmentation faults.
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


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal, exiting cleanly...")
    os._exit(0)  # Force exit without cleanup to avoid segfault


def main():
    """Launch the RAG chat system"""
    # Set up signal handlers to prevent segfaults
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting RAG Chat System...")
    print("=" * 50)
    
    # Configure nice logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    logger.info("Starting RAG Chat Server on http://127.0.0.1:8000")
    
    try:
        # Start the server
        run_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        os._exit(0)  # Force clean exit
    except Exception as e:
        logger.error(f"Server error: {e}")
        os._exit(1)


if __name__ == "__main__":
    main()

