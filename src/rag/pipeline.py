"""
Complete RAG pipeline that integrates all components.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from loguru import logger

from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingGenerator
from rag.retriever import DocumentRetriever
from data_processing import DataLoader, TextProcessor, WebScraper


class RAGPipeline:
    """Complete RAG pipeline for document ingestion, indexing, and retrieval."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.components = {}
        self._initialize_components()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = "config/model_config.yaml"
        
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('rag', self._get_default_config())
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'vector_db': {
                'type': 'faiss',
                'dimension': 1536,  # OpenAI text-embedding-3-small
                'index_type': 'IndexFlatL2'
            },
            'embedding_model': 'openai',
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.1,
                'chunk_size': 512,
                'chunk_overlap': 50
            },
            'document_processing': {
                'supported_formats': ['txt', 'pdf', 'docx', 'md', 'json'],
                'max_document_length': 10000
            }
        }
    
    def _initialize_components(self):
        """Initialize all RAG components."""
        logger.info("Initializing RAG pipeline components...")
        
        # Initialize data loader
        self.components['data_loader'] = DataLoader()
        
        # Initialize text processor
        self.components['text_processor'] = TextProcessor()
        
        # Initialize web scraper
        self.components['web_scraper'] = WebScraper()
        
        # Initialize embedding generator
        embedding_model = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L12-v2')
        self.components['embedding_generator'] = EmbeddingGenerator(
            model_name=embedding_model
        )
        
        # Initialize vector store
        vector_config = self.config.get('vector_db', {})
        self.components['vector_store'] = VectorStore(
            store_type=vector_config.get('type', 'faiss'),
            dimension=self.components['embedding_generator'].embedding_dim,
            index_type=vector_config.get('index_type', 'IndexFlatL2'),
            persist_path=vector_config.get('persist_path', './data/vector_db')
        )
        
        # Initialize document retriever
        self.components['retriever'] = DocumentRetriever(
            vector_store=self.components['vector_store'],
            embedding_generator=self.components['embedding_generator'],
            default_top_k=self.config.get('retrieval', {}).get('top_k', 5),
            default_threshold=self.config.get('retrieval', {}).get('similarity_threshold', 0.1)
        )
        
        logger.info("RAG pipeline components initialized successfully")
    
    def ingest_documents(self, 
                        data_sources: Union[str, List[str], List[Dict[str, Any]]],
                        chunk_documents: bool = True,
                        batch_size: int = 100) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            data_sources: Path(s) to data or list of source configurations
            chunk_documents: Whether to chunk documents for RAG
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("Starting document ingestion...")
        
        # Load documents
        if isinstance(data_sources, str):
            # Single directory or file
            if Path(data_sources).is_dir():
                documents = self.components['data_loader'].load_directory(data_sources)
            else:
                documents = [self.components['data_loader'].load_file(data_sources)]
        elif isinstance(data_sources, list):
            if all(isinstance(source, str) for source in data_sources):
                # List of file paths
                documents = []
                for source in data_sources:
                    if Path(source).is_dir():
                        documents.extend(self.components['data_loader'].load_directory(source))
                    else:
                        documents.append(self.components['data_loader'].load_file(source))
            else:
                # List of source configurations
                documents = self.components['data_loader'].load_from_config(data_sources)
        else:
            raise ValueError("Invalid data_sources format")
        
        if not documents:
            logger.warning("No documents found to ingest")
            return {'total_documents': 0, 'total_chunks': 0, 'success': False}
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Process documents
        processed_documents = []
        total_chunks = 0
        
        retrieval_config = self.config.get('retrieval', {})
        chunk_size = retrieval_config.get('chunk_size', 512)
        chunk_overlap = retrieval_config.get('chunk_overlap', 50)
        
        for doc in documents:
            try:
                processed_doc = self.components['text_processor'].process_document(
                    doc, 
                    chunk_for_rag=chunk_documents,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                processed_documents.append(processed_doc)
                
                if chunk_documents and 'chunks' in processed_doc:
                    total_chunks += len(processed_doc['chunks'])
                else:
                    total_chunks += 1
                    
            except Exception as e:
                logger.error(f"Error processing document {doc.get('filename', 'unknown')}: {e}")
        
        if not processed_documents:
            logger.error("No documents were successfully processed")
            return {'total_documents': 0, 'total_chunks': 0, 'success': False}
        
        logger.info(f"Processed {len(processed_documents)} documents into {total_chunks} chunks")
        
        # Generate embeddings and index documents
        return self._index_documents(processed_documents, batch_size, chunk_documents)
    
    def _index_documents(self, 
                        documents: List[Dict[str, Any]], 
                        batch_size: int,
                        chunk_documents: bool) -> Dict[str, Any]:
        """Index processed documents in the vector store."""
        logger.info("Generating embeddings and indexing documents...")
        
        # Prepare documents for embedding
        if chunk_documents:
            # Use chunks as individual documents
            embedding_docs = []
            for doc in documents:
                if 'chunks' in doc:
                    for chunk in doc['chunks']:
                        # Create a document entry for each chunk
                        chunk_doc = {
                            **doc,  # Include original metadata
                            'content': chunk['text'],
                            'processed_content': chunk['text'],
                            'chunk_id': chunk['chunk_id'],
                            'token_count': chunk['token_count'],
                            'is_chunk': True,
                            'parent_doc_id': doc.get('id', doc.get('filename', 'unknown'))
                        }
                        # Remove chunks array to avoid confusion
                        chunk_doc.pop('chunks', None)
                        embedding_docs.append(chunk_doc)
                else:
                    # Document without chunks
                    doc['is_chunk'] = False
                    embedding_docs.append(doc)
        else:
            # Use full documents
            for doc in documents:
                doc['is_chunk'] = False
            embedding_docs = documents
        
        if not embedding_docs:
            logger.error("No documents prepared for embedding")
            return {'total_documents': 0, 'total_chunks': 0, 'success': False}
        
        # Generate embeddings in batches
        total_indexed = 0
        for i in range(0, len(embedding_docs), batch_size):
            batch = embedding_docs[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
                batch_with_embeddings = self.components['embedding_generator'].generate_document_embeddings(batch)
                
                # Extract embeddings and documents
                embeddings = []
                clean_docs = []
                
                for doc in batch_with_embeddings:
                    if 'embedding' in doc:
                        embeddings.append(doc['embedding'])
                        # Remove embedding from doc before storing (will be stored separately)
                        clean_doc = doc.copy()
                        clean_doc.pop('embedding', None)
                        clean_docs.append(clean_doc)
                
                if embeddings and clean_docs:
                    # Add to vector store
                    import numpy as np
                    embeddings_array = np.vstack(embeddings)
                    
                    doc_ids = self.components['vector_store'].add_documents(
                        clean_docs, embeddings_array
                    )
                    total_indexed += len(doc_ids)
                    
                    logger.info(f"Indexed batch {i//batch_size + 1}: {len(doc_ids)} documents")
                
            except Exception as e:
                logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
        
        # Save vector store
        try:
            self.components['vector_store'].save()
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.warning(f"Error saving vector store: {e}")
        
        success = total_indexed > 0
        result = {
            'total_documents': len(documents),
            'total_indexed': total_indexed,
            'total_chunks': total_indexed if chunk_documents else len(documents),
            'success': success
        }
        
        if success:
            logger.info(f"Successfully indexed {total_indexed} document chunks")
        else:
            logger.error("Failed to index any documents")
        
        return result
    
    def search(self, 
              query: str,
              top_k: int = 5,
              threshold: float = 0.1,
              filters: Optional[Dict[str, Any]] = None,
              rerank: bool = True,
              include_explanation: bool = False) -> Dict[str, Any]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Metadata filters
            rerank: Whether to apply re-ranking
            include_explanation: Whether to include explanation for results
            
        Returns:
            Dictionary with search results and metadata
        """
        if not query.strip():
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'search_time': 0,
                'success': False,
                'error': 'Empty query'
            }
        
        import time
        start_time = time.time()
        
        try:
            # Retrieve documents
            results = self.components['retriever'].retrieve(
                query=query,
                top_k=top_k,
                threshold=threshold,
                filters=filters,
                rerank=rerank
            )
            
            search_time = time.time() - start_time
            
            # Add explanations if requested
            if include_explanation and results:
                for result in results:
                    doc_id = result.get('id')
                    if doc_id:
                        explanation = self.components['retriever'].explain_retrieval(query, doc_id)
                        result['explanation'] = explanation
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'search_time': search_time,
                'success': True,
                'parameters': {
                    'top_k': top_k,
                    'threshold': threshold,
                    'filters': filters,
                    'rerank': rerank
                }
            }
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Error during search: {e}")
            
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'search_time': search_time,
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        vector_stats = self.components['vector_store'].get_stats()
        
        return {
            'vector_store': vector_stats,
            'embedding_model': self.config.get('embedding_model'),
            'embedding_dimension': self.components['embedding_generator'].embedding_dim,
            'configuration': self.config
        }
    
    def reset(self):
        """Reset the RAG system (remove all indexed documents)."""
        logger.warning("Resetting RAG system - all indexed documents will be removed")
        self.components['vector_store'].reset()
        logger.info("RAG system reset completed")
    
    def save(self, path: Optional[str] = None):
        """Save the RAG system state."""
        self.components['vector_store'].save(path)
        logger.info("RAG system saved")
    
    def load(self, path: Optional[str] = None):
        """Load the RAG system state."""
        self.components['vector_store'].load(path)
        logger.info("RAG system loaded")
    
    def add_documents_from_text(self, 
                               texts: List[str],
                               metadata_list: Optional[List[Dict[str, Any]]] = None,
                               chunk_documents: bool = True) -> Dict[str, Any]:
        """
        Add documents directly from text strings.
        
        Args:
            texts: List of text strings
            metadata_list: Optional list of metadata for each text
            chunk_documents: Whether to chunk the texts
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not texts:
            return {'total_documents': 0, 'total_chunks': 0, 'success': False}
        
        # Create document objects
        documents = []
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            doc = {
                'content': text,
                'filename': metadata.get('filename', f'text_document_{i}'),
                'file_type': '.txt',
                'file_path': metadata.get('file_path', f'memory://text_{i}'),
                'file_size': len(text),
                **metadata
            }
            documents.append(doc)
        
        # Process and index documents
        return self._process_and_index_documents(documents, chunk_documents)
    
    def _process_and_index_documents(self, 
                                   documents: List[Dict[str, Any]], 
                                   chunk_documents: bool) -> Dict[str, Any]:
        """Process and index a list of documents."""
        # Process documents
        processed_documents = []
        retrieval_config = self.config.get('retrieval', {})
        chunk_size = retrieval_config.get('chunk_size', 512)
        chunk_overlap = retrieval_config.get('chunk_overlap', 50)
        
        for doc in documents:
            try:
                processed_doc = self.components['text_processor'].process_document(
                    doc, 
                    chunk_for_rag=chunk_documents,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                processed_documents.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        if not processed_documents:
            return {'total_documents': 0, 'total_chunks': 0, 'success': False}
        
        # Index documents
        return self._index_documents(processed_documents, batch_size=100, chunk_documents=chunk_documents)
    
    def ingest_websites(self, 
                       urls: List[str],
                       max_depth: int = 2,
                       max_pages_per_site: int = 50,
                       include_patterns: Optional[List[str]] = None,
                       exclude_patterns: Optional[List[str]] = None,
                       chunk_documents: bool = True) -> Dict[str, Any]:
        """
        Scrape websites and ingest them into the RAG system.
        
        Args:
            urls: List of website URLs to scrape
            max_depth: Maximum depth to crawl (0 = only main page, 1 = main + linked pages, etc.)
            max_pages_per_site: Maximum pages to scrape per website
            include_patterns: URL patterns to include (regex)
            exclude_patterns: URL patterns to exclude (regex)
            chunk_documents: Whether to chunk documents for RAG
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting website ingestion for {len(urls)} URLs...")
        
        # Configure web scraper
        self.components['web_scraper'].max_pages = max_pages_per_site
        
        try:
            # Get existing URLs to avoid re-scraping
            existing_urls = self.components['vector_store'].get_existing_urls()
            
            # Scrape websites
            documents = self.components['web_scraper'].scrape_multiple_websites(
                urls=urls,
                max_depth=max_depth,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                existing_urls=existing_urls
            )
            
            if not documents:
                logger.warning("No documents found from website scraping")
                return {'total_documents': 0, 'total_chunks': 0, 'success': False}
            
            logger.info(f"Scraped {len(documents)} pages from {len(urls)} websites")
            
            # Process and index documents
            return self._process_and_index_documents(documents, chunk_documents)
            
        except Exception as e:
            logger.error(f"Error during website ingestion: {e}")
            return {'total_documents': 0, 'total_chunks': 0, 'success': False, 'error': str(e)}

