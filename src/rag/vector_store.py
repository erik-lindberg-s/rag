"""
Vector store implementation for RAG system using FAISS and ChromaDB.
"""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import numpy as np
import faiss
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Only FAISS will be supported.")


class VectorStore:
    """Vector store for efficient similarity search and document retrieval."""
    
    def __init__(self, 
                 store_type: str = "faiss",
                 dimension: int = 384,
                 index_type: str = "IndexFlatL2",
                 persist_path: Optional[str] = None,
                 collection_name: str = "documents"):
        """
        Initialize vector store.
        
        Args:
            store_type: Type of vector store ("faiss" or "chromadb")
            dimension: Embedding dimension
            index_type: FAISS index type (for FAISS store)
            persist_path: Path to persist the index
            collection_name: Name of the collection (for ChromaDB)
        """
        self.store_type = store_type.lower()
        self.dimension = dimension
        self.index_type = index_type
        self.persist_path = Path(persist_path) if persist_path else None
        self.collection_name = collection_name
        
        # Initialize metadata storage
        self.documents = []  # Store document metadata
        self.id_to_index = {}  # Map document IDs to index positions
        
        if self.store_type == "faiss":
            self._init_faiss()
        elif self.store_type == "chromadb":
            if not CHROMADB_AVAILABLE:
                raise ImportError("ChromaDB is not installed. Please install with: pip install chromadb")
            self._init_chromadb()
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
        
        # Load existing data if persist path exists
        if self.persist_path and self.persist_path.exists():
            self.load()
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        if self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        elif self.index_type == "IndexIVFFlat":
            # For larger datasets, use IVF index
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}")
        
        logger.info(f"Initialized FAISS index: {self.index_type} with dimension {self.dimension}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        if self.persist_path:
            # Persistent ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            # In-memory ChromaDB
            self.chroma_client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]], 
                     embeddings: np.ndarray,
                     batch_size: int = 1000) -> List[str]:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries
            embeddings: Numpy array of embeddings
            batch_size: Batch size for adding documents
            
        Returns:
            List of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        document_ids = []
        
        if self.store_type == "faiss":
            document_ids = self._add_documents_faiss(documents, embeddings, batch_size)
        elif self.store_type == "chromadb":
            document_ids = self._add_documents_chromadb(documents, embeddings, batch_size)
        
        logger.info(f"Added {len(documents)} documents to {self.store_type} vector store")
        return document_ids
    
    def _add_documents_faiss(self, documents: List[Dict[str, Any]], 
                            embeddings: np.ndarray, batch_size: int) -> List[str]:
        """Add documents to FAISS index."""
        document_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Generate document IDs
            batch_ids = [f"doc_{len(self.documents) + j}" for j in range(len(batch_docs))]
            
            # Add embeddings to FAISS index
            if self.index_type == "IndexIVFFlat" and not self.index.is_trained:
                # Train the index if it's IVF and not trained yet
                if len(self.documents) == 0:  # First batch
                    self.index.train(batch_embeddings.astype(np.float32))
            
            self.index.add(batch_embeddings.astype(np.float32))
            
            # Store document metadata
            for j, (doc_id, doc) in enumerate(zip(batch_ids, batch_docs)):
                doc_with_id = {**doc, 'id': doc_id}
                self.documents.append(doc_with_id)
                self.id_to_index[doc_id] = len(self.documents) - 1
                document_ids.append(doc_id)
        
        return document_ids
    
    def _add_documents_chromadb(self, documents: List[Dict[str, Any]], 
                               embeddings: np.ndarray, batch_size: int) -> List[str]:
        """Add documents to ChromaDB collection."""
        document_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Generate document IDs
            batch_ids = [f"doc_{len(self.documents) + j}" for j in range(len(batch_docs))]
            
            # Prepare documents for ChromaDB
            batch_texts = []
            batch_metadatas = []
            
            for doc_id, doc in zip(batch_ids, batch_docs):
                # Use processed content or content as text
                text = doc.get('processed_content', doc.get('content', ''))
                if isinstance(text, (dict, list)):
                    text = str(text)
                batch_texts.append(text)
                
                # Prepare metadata (ChromaDB requires string values)
                metadata = {
                    'id': doc_id,
                    'filename': str(doc.get('filename', '')),
                    'file_type': str(doc.get('file_type', '')),
                    'file_path': str(doc.get('file_path', '')),
                    'word_count': str(doc.get('word_count', 0)),
                    'char_count': str(doc.get('char_count', 0))
                }
                batch_metadatas.append(metadata)
                
                # Store full document data separately
                doc_with_id = {**doc, 'id': doc_id}
                self.documents.append(doc_with_id)
                self.id_to_index[doc_id] = len(self.documents) - 1
                document_ids.append(doc_id)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        return document_ids
    
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 5, 
               threshold: float = 0.0,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            filter_metadata: Metadata filters (ChromaDB only)
            
        Returns:
            List of similar documents with scores
        """
        if self.store_type == "faiss":
            return self._search_faiss(query_embedding, top_k, threshold)
        elif self.store_type == "chromadb":
            return self._search_chromadb(query_embedding, top_k, threshold, filter_metadata)
    
    def _search_faiss(self, query_embedding: np.ndarray, 
                     top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        if self.index.ntotal == 0:
            return []
        
        # Ensure query embedding is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            # Convert distance to similarity score
            if self.index_type == "IndexFlatIP":
                similarity = float(distance)  # Inner product is already similarity
            else:
                # For L2 distance, convert to similarity (higher is better)
                similarity = 1.0 / (1.0 + float(distance))
            
            if similarity >= threshold:
                doc = self.documents[idx].copy()
                doc['similarity_score'] = similarity
                doc['distance'] = float(distance)
                doc['rank'] = i + 1
                results.append(doc)
        
        return results
    
    def _search_chromadb(self, query_embedding: np.ndarray, 
                        top_k: int, threshold: float,
                        filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search using ChromaDB collection."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Build where clause for filtering
        where_clause = None
        if filter_metadata:
            where_clause = {k: {"$eq": str(v)} for k, v in filter_metadata.items()}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where_clause
        )
        
        # Process results
        processed_results = []
        if results['ids'] and results['ids'][0]:
            for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = 1.0 - float(distance)
                
                if similarity >= threshold:
                    # Get full document data
                    doc_idx = self.id_to_index.get(doc_id)
                    if doc_idx is not None:
                        doc = self.documents[doc_idx].copy()
                        doc['similarity_score'] = similarity
                        doc['distance'] = float(distance)
                        doc['rank'] = i + 1
                        processed_results.append(doc)
        
        return processed_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        doc_idx = self.id_to_index.get(doc_id)
        if doc_idx is not None:
            return self.documents[doc_idx]
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID (ChromaDB only)."""
        if self.store_type == "chromadb":
            try:
                self.collection.delete(ids=[doc_id])
                # Remove from local storage
                doc_idx = self.id_to_index.get(doc_id)
                if doc_idx is not None:
                    del self.documents[doc_idx]
                    del self.id_to_index[doc_id]
                    # Update indices
                    for key, idx in self.id_to_index.items():
                        if idx > doc_idx:
                            self.id_to_index[key] = idx - 1
                return True
            except Exception as e:
                logger.error(f"Error deleting document {doc_id}: {e}")
                return False
        else:
            logger.warning("Document deletion not supported for FAISS store")
            return False
    
    def save(self, path: Optional[str] = None):
        """Save the vector store to disk."""
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            logger.warning("No save path specified")
            return
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.store_type == "faiss":
            # Save FAISS index
            faiss.write_index(self.index, str(save_path / "index.faiss"))
            
            # Save documents metadata
            with open(save_path / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(save_path / "id_to_index.pkl", 'wb') as f:
                pickle.dump(self.id_to_index, f)
            
            # Save configuration
            config = {
                'store_type': self.store_type,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'collection_name': self.collection_name
            }
            with open(save_path / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved FAISS vector store to {save_path}")
        
        elif self.store_type == "chromadb":
            # ChromaDB is automatically persisted if using PersistentClient
            logger.info("ChromaDB collection automatically persisted")
    
    def load(self, path: Optional[str] = None):
        """Load the vector store from disk."""
        load_path = Path(path) if path else self.persist_path
        if not load_path or not load_path.exists():
            logger.warning("No valid load path found")
            return
        
        if self.store_type == "faiss":
            # Load FAISS index
            index_path = load_path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                
                # Load documents metadata
                docs_path = load_path / "documents.pkl"
                if docs_path.exists():
                    with open(docs_path, 'rb') as f:
                        self.documents = pickle.load(f)
                
                id_index_path = load_path / "id_to_index.pkl"
                if id_index_path.exists():
                    with open(id_index_path, 'rb') as f:
                        self.id_to_index = pickle.load(f)
                
                logger.info(f"Loaded FAISS vector store from {load_path}")
                logger.info(f"Index contains {self.index.ntotal} vectors")
        
        elif self.store_type == "chromadb":
            # ChromaDB collection is automatically loaded if using PersistentClient
            logger.info("ChromaDB collection automatically loaded")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.store_type == "faiss":
            return {
                'store_type': self.store_type,
                'total_documents': len(self.documents),
                'index_total': self.index.ntotal,
                'dimension': self.dimension,
                'index_type': self.index_type
            }
        elif self.store_type == "chromadb":
            count = self.collection.count()
            return {
                'store_type': self.store_type,
                'total_documents': count,
                'collection_name': self.collection_name,
                'dimension': self.dimension
            }
    
    def get_existing_urls(self) -> Set[str]:
        """Get set of URLs that have already been indexed."""
        existing_urls = set()
        
        if self.store_type == "faiss":
            for doc in self.documents:
                if 'url' in doc:
                    existing_urls.add(doc['url'])
        elif self.store_type == "chromadb":
            # Get all documents with their metadata
            try:
                results = self.collection.get()
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata and 'url' in metadata:
                            existing_urls.add(metadata['url'])
            except Exception as e:
                logger.warning(f"Error getting existing URLs from ChromaDB: {e}")
        
        logger.info(f"Found {len(existing_urls)} existing URLs in vector store")
        return existing_urls
    
    def reset(self):
        """Reset the vector store (remove all documents)."""
        if self.store_type == "faiss":
            self.index.reset()
            self.documents = []
            self.id_to_index = {}
        elif self.store_type == "chromadb":
            # Delete and recreate collection
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.documents = []
            self.id_to_index = {}
        
        logger.info("Vector store reset successfully")


