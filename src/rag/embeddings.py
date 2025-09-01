"""
Embedding generation using OpenAI API for stable, crash-free semantic search.
NO FALLBACKS - ONLY PERFECT SEMANTIC SEARCH.
"""

import numpy as np
import openai
import os
from typing import List, Dict, Any, Optional, Union
from loguru import logger

class EmbeddingGenerator:
    """Generate embeddings using OpenAI API - stable, no crashes."""
    
    def __init__(self, 
                 model_name: str = "openai",
                 device: Optional[str] = None,
                 batch_size: int = 20):
        """
        Initialize OpenAI embedding generator.
        
        Args:
            model_name: Must be "openai" 
            device: Ignored (OpenAI API)
            batch_size: Batch size for OpenAI API calls
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = 1536  # OpenAI text-embedding-3-small dimension
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise Exception("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        
        logger.info("✅ Using OpenAI embeddings for stable, crash-free semantic search")
        logger.info("🚀 NO LOCAL PYTORCH/TRANSFORMERS - NO CRASHES!")
        logger.info(f"📏 Embedding dimension: {self.embedding_dim}")
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            
            # Extract embeddings
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"OpenAI embedding API error: {e}")
            raise Exception(f"Failed to generate OpenAI embeddings: {e}")
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s) using OpenAI API.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        logger.debug(f"Generating OpenAI embeddings for {len(texts)} texts")
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_openai_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.zeros((len(texts), self.embedding_dim))
        
        logger.debug(f"✅ Generated {embeddings.shape[0]} OpenAI embeddings of dimension {embeddings.shape[1]}")
        return embeddings
    
    def generate_document_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents using OpenAI API.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Documents with embeddings added
        """
        if not documents:
            return []
        
        # Extract texts from documents
        texts = []
        for doc in documents:
            if 'chunks' in doc:
                # Use chunks if available
                chunk_texts = [chunk['text'] for chunk in doc['chunks']]
                texts.extend(chunk_texts)
            else:
                # Use processed content
                content = doc.get('processed_content', doc.get('content', ''))
                if content:
                    texts.append(str(content))
        
        if not texts:
            logger.warning("No text content found in documents")
            return documents
        
        logger.info(f"🔥 Generating OpenAI embeddings for {len(texts)} text segments...")
        
        try:
            # Generate all embeddings using OpenAI API
            embeddings = self.generate_embeddings(texts)
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise Exception(f"Cannot generate embeddings: {e}")
        
        # Assign embeddings back to documents
        embedding_idx = 0
        processed_docs = []
        
        for doc in documents:
            doc_copy = doc.copy()
            
            if 'chunks' in doc:
                # Add embeddings to chunks
                for chunk in doc_copy['chunks']:
                    if embedding_idx < len(embeddings):
                        chunk['embedding'] = embeddings[embedding_idx]
                        embedding_idx += 1
            else:
                # Add embedding to document
                content = doc.get('processed_content', doc.get('content', ''))
                if content and embedding_idx < len(embeddings):
                    doc_copy['embedding'] = embeddings[embedding_idx]
                    embedding_idx += 1
            
            processed_docs.append(doc_copy)
        
        logger.info(f"✅ Generated OpenAI embeddings for {len(processed_docs)} documents")
        return processed_docs
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Matrix of document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, doc_norms.T).flatten()
        
        return similarities
    
    def find_similar_documents(self, query: str, 
                              document_embeddings: List[np.ndarray],
                              documents: List[Dict[str, Any]],
                              top_k: int = 5,
                              threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find documents most similar to a query using OpenAI embeddings.
        
        Args:
            query: Query text
            document_embeddings: List of document embeddings
            documents: List of document dictionaries
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar documents with similarity scores
        """
        if not document_embeddings or not documents:
            return []
        
        # Generate query embedding using OpenAI
        query_embedding = self.generate_embeddings(query)
        
        # Stack document embeddings
        doc_emb_matrix = np.vstack(document_embeddings)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, doc_emb_matrix)
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                doc = documents[idx].copy()
                doc['similarity_score'] = float(similarity)
                results.append(doc)
        
        return results