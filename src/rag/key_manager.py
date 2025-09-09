"""
Secure API key management for persisting OpenAI API keys.
"""

import os
import json
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from loguru import logger
from typing import Optional


class APIKeyManager:
    """Manages secure storage and retrieval of API keys."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize key storage, preferring persistent volume if available.
        
        On Fly.io we mount a volume at /app/persistent. We store keys under
        /app/persistent/config so keys survive restarts and deploys.
        """
        # Determine base config directory
        try:
            persistent_root = os.getenv('PERSISTENT_DIR', '/app/persistent')
            if storage_path:
                base_dir = Path(storage_path).parent
            elif os.path.isdir(persistent_root):
                base_dir = Path(persistent_root) / 'config'
            else:
                base_dir = Path('config')
        except Exception:
            base_dir = Path('config')

        base_dir.mkdir(parents=True, exist_ok=True)

        # Files
        self.storage_path = base_dir / 'api_keys.json'
        self.key_file = base_dir / '.key'

        # Generate or load encryption key
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create a new one."""
        try:
            if self.key_file.exists():
                with open(self.key_file, 'rb') as f:
                    return f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                # Make key file read-only for security
                os.chmod(self.key_file, 0o600)
                logger.info("Generated new encryption key for API key storage")
                return key
        except Exception as e:
            logger.error(f"Error managing encryption key: {e}")
            # Fallback to session-only storage
            return Fernet.generate_key()
    
    def save_openai_key(self, api_key: str) -> bool:
        """
        Securely save OpenAI API key.
        
        Args:
            api_key: The OpenAI API key to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Encrypt the API key
            encrypted_key = self.cipher.encrypt(api_key.encode())
            
            # Load existing data or create new
            data = {}
            if self.storage_path.exists():
                try:
                    with open(self.storage_path, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Corrupted API key storage, creating new")
                    data = {}
            
            # Save encrypted key
            data['openai_api_key'] = base64.b64encode(encrypted_key).decode()
            data['updated_at'] = str(datetime.now())
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Make file read-only for security
            os.chmod(self.storage_path, 0o600)
            
            logger.info("OpenAI API key saved securely")
            return True
            
        except Exception as e:
            logger.error(f"Error saving API key: {e}")
            return False
    
    def get_openai_key(self) -> Optional[str]:
        """
        Retrieve the saved OpenAI API key.
        
        Returns:
            The decrypted API key if found, None otherwise
        """
        try:
            if not self.storage_path.exists():
                return None
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            encrypted_key_b64 = data.get('openai_api_key')
            if not encrypted_key_b64:
                return None
            
            # Decrypt the key
            encrypted_key = base64.b64decode(encrypted_key_b64)
            decrypted_key = self.cipher.decrypt(encrypted_key).decode()
            
            logger.info("OpenAI API key retrieved from secure storage")
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Error retrieving API key: {e}")
            return None
    
    def delete_openai_key(self) -> bool:
        """
        Delete the saved OpenAI API key.
        
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if self.storage_path.exists():
                self.storage_path.unlink()
                logger.info("OpenAI API key deleted from storage")
            return True
        except Exception as e:
            logger.error(f"Error deleting API key: {e}")
            return False
    
    def has_openai_key(self) -> bool:
        """Check if an OpenAI API key is stored."""
        return self.get_openai_key() is not None


# Add missing import
from datetime import datetime
