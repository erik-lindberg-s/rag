"""
Data loader for various file formats containing proprietary company data.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from loguru import logger
import PyPDF2
from docx import Document


class DataLoader:
    """Load and parse various data formats from company sources."""
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.txt', '.json', '.csv', '.pdf', '.docx', '.md']
        
    def load_text_file(self, file_path: Union[str, Path]) -> str:
        """Load content from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def load_json_file(self, file_path: Union[str, Path]) -> Union[Dict, List]:
        """Load content from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_csv_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load content from a CSV file."""
        return pd.read_csv(file_path)
    
    def load_pdf_file(self, file_path: Union[str, Path]) -> str:
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"Error reading PDF {file_path}: {e}")
        return text
    
    def load_docx_file(self, file_path: Union[str, Path]) -> str:
        """Extract text content from a DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.warning(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def load_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a single file and return its content with metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported file format: {file_ext}")
            return {}
        
        # Load content based on file type
        content = ""
        if file_ext in ['.txt', '.md']:
            content = self.load_text_file(file_path)
        elif file_ext == '.json':
            content = self.load_json_file(file_path)
        elif file_ext == '.csv':
            content = self.load_csv_file(file_path)
        elif file_ext == '.pdf':
            content = self.load_pdf_file(file_path)
        elif file_ext == '.docx':
            content = self.load_docx_file(file_path)
        
        # Get file metadata
        stat = file_path.stat()
        
        return {
            'content': content,
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_type': file_ext,
            'file_size': stat.st_size,
            'modified_time': stat.st_mtime,
            'created_time': stat.st_ctime
        }
    
    def load_directory(self, directory_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to directory (defaults to self.data_dir)
            
        Returns:
            List of dictionaries containing file contents and metadata
        """
        if directory_path is None:
            directory_path = self.data_dir
        else:
            directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        
        # Recursively find all supported files
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.load_file(file_path)
                    if doc:  # Only add if content was successfully loaded
                        documents.append(doc)
                        logger.info(f"Loaded: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def load_from_config(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Load documents from multiple sources defined in configuration.
        
        Args:
            sources: List of source configurations
            
        Returns:
            List of loaded documents
        """
        all_documents = []
        
        for source in sources:
            source_type = source.get('type')
            path = source.get('path')
            
            if source_type == 'directory':
                documents = self.load_directory(path)
                all_documents.extend(documents)
            elif source_type == 'file':
                document = self.load_file(path)
                if document:
                    all_documents.append(document)
            else:
                logger.warning(f"Unknown source type: {source_type}")
        
        return all_documents

