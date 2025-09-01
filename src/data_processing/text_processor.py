"""
Text processing utilities for cleaning and preparing company data.
"""

import re
import string
from typing import List, Dict, Any, Optional
from loguru import logger
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    """Process and clean text data for training and RAG."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize TextProcessor.
        
        Args:
            language: Language code for processing
        """
        self.language = language
        self.setup_nltk()
        self.setup_spacy()
        
    def setup_nltk(self):
        """Download and setup NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def setup_spacy(self):
        """Setup spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!\?]{2,}', '!', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_pii(self, text: str) -> str:
        """
        Remove potential PII (Personal Identifiable Information) from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with PII removed/masked
        """
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers (various formats)
        text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '[PHONE]', text)
        
        # Social Security Numbers
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Credit card numbers (simple pattern)
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Text to process
            
        Returns:
            List of entities with their types and positions
        """
        if not self.nlp:
            logger.warning("spaCy model not available for entity extraction")
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for RAG processing.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks with metadata
        """
        if not text:
            return []
        
        # Split into sentences first
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        sentence_start_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(word_tokenize(sentence))
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'token_count': current_tokens,
                    'sentence_start': sentence_start_idx,
                    'sentence_end': i - 1,
                    'chunk_id': len(chunks)
                })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                # Add previous sentences for overlap
                for j in range(i - 1, -1, -1):
                    sent_tokens = len(word_tokenize(sentences[j]))
                    if overlap_tokens + sent_tokens <= overlap:
                        overlap_sentences.insert(0, sentences[j])
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_chunk = ' '.join(overlap_sentences)
                current_tokens = overlap_tokens
                sentence_start_idx = i - len(overlap_sentences)
            
            current_chunk += ' ' + sentence if current_chunk else sentence
            current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'sentence_start': sentence_start_idx,
                'sentence_end': len(sentences) - 1,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def preprocess_for_training(self, text: str, max_length: int = 2048) -> str:
        """
        Preprocess text specifically for model training.
        
        Args:
            text: Raw text
            max_length: Maximum length in tokens
            
        Returns:
            Preprocessed text ready for training
        """
        # Clean the text
        text = self.clean_text(text)
        
        # Remove PII for privacy
        text = self.remove_pii(text)
        
        # Truncate if too long (rough token estimation: 1 token ≈ 4 characters)
        if len(text) > max_length * 4:
            text = text[:max_length * 4]
            # Try to end at a sentence boundary
            last_period = text.rfind('.')
            if last_period > len(text) * 0.8:  # If period is in last 20%
                text = text[:last_period + 1]
        
        return text
    
    def process_document(self, document: Dict[str, Any], 
                        chunk_for_rag: bool = True,
                        chunk_size: int = 512,
                        overlap: int = 50) -> Dict[str, Any]:
        """
        Process a complete document for both RAG and training.
        
        Args:
            document: Document dictionary with content and metadata
            chunk_for_rag: Whether to create chunks for RAG
            chunk_size: Size of chunks for RAG
            overlap: Overlap between chunks
            
        Returns:
            Processed document with cleaned text and optional chunks
        """
        content = document.get('content', '')
        
        # Handle different content types
        if isinstance(content, dict):
            # JSON content - extract text fields
            text_content = self._extract_text_from_dict(content)
        elif isinstance(content, list):
            # List content - join elements
            text_content = ' '.join(str(item) for item in content)
        else:
            text_content = str(content)
        
        # Clean and preprocess
        cleaned_text = self.clean_text(text_content)
        processed_text = self.preprocess_for_training(cleaned_text)
        
        # Extract entities
        entities = self.extract_entities(processed_text)
        
        result = {
            **document,  # Keep original metadata
            'processed_content': processed_text,
            'entities': entities,
            'word_count': len(word_tokenize(processed_text)),
            'char_count': len(processed_text)
        }
        
        # Create chunks for RAG if requested
        if chunk_for_rag and processed_text:
            chunks = self.chunk_text(processed_text, chunk_size, overlap)
            result['chunks'] = chunks
        
        return result
    
    def _extract_text_from_dict(self, data: Dict[str, Any], 
                               text_fields: Optional[List[str]] = None) -> str:
        """Extract text content from dictionary data."""
        if text_fields is None:
            # Common text field names
            text_fields = ['text', 'content', 'body', 'description', 'title', 
                          'summary', 'abstract', 'message', 'comment']
        
        extracted_text = []
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in text_fields and isinstance(value, str):
                        extracted_text.append(value)
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, f"{prefix}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        extract_recursive(item, prefix)
                    elif isinstance(item, str):
                        extracted_text.append(item)
        
        extract_recursive(data)
        return ' '.join(extracted_text)

