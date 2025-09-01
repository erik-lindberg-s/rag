"""
Document retriever for RAG system.
Combines vector search with metadata filtering and re-ranking.
"""

import re
from typing import List, Dict, Any, Optional, Union, Callable
from loguru import logger
import numpy as np
from datetime import datetime

from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingGenerator


class DocumentRetriever:
    """Advanced document retrieval system for RAG."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_generator: EmbeddingGenerator,
                 default_top_k: int = 10,
                 default_threshold: float = 0.1):
        """
        Initialize document retriever.
        
        Args:
            vector_store: Vector store for similarity search
            embedding_generator: Embedding generator for queries
            default_top_k: Default number of results to retrieve
            default_threshold: Default similarity threshold
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold
        
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms for better TF-IDF matching"""
        query_lower = query.lower()
        
        # Define semantic expansions
        expansions = {
            'snack': ['snacking', 'snacks', 'treats', 'bites', 'cravings'],
            'cant stop': ['addictive', 'compulsive', 'control', 'habits'],
            'getting fat': ['weight gain', 'obesity', 'overweight', 'calories'],
            'psychology': ['psychological', 'mental', 'brain', 'mind', 'behavior'],
            'emotional': ['emotions', 'stress', 'comfort', 'feelings'],
            'why': ['reason', 'cause', 'explanation', 'understanding'],
        }
        
        # Add related terms to query
        expanded_terms = [query]
        
        for trigger, related_terms in expansions.items():
            if trigger in query_lower:
                expanded_terms.extend(related_terms)
        
        expanded_query = ' '.join(expanded_terms)
        logger.debug(f"Expanded query: '{query}' → '{expanded_query}'")
        
        return expanded_query

    def retrieve(self, 
                query: str,
                top_k: Optional[int] = None,
                threshold: Optional[float] = None,
                filters: Optional[Dict[str, Any]] = None,
                rerank: bool = True,
                include_chunks: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Metadata filters
            rerank: Whether to apply re-ranking
            include_chunks: Whether to include chunk-level results
            
        Returns:
            List of retrieved documents with relevance scores
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        top_k = top_k or self.default_top_k
        threshold = threshold or self.default_threshold
        
        logger.info(f"Retrieving documents for query: '{query[:100]}...'")
        
        # Expand query for better TF-IDF matching
        expanded_query = self._expand_query(query)
        
        # Generate query embedding using expanded query
        query_embedding = self.embedding_generator.generate_embeddings(expanded_query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results for filtering/reranking
            threshold=threshold,
            filter_metadata=filters
        )
        
        if not results:
            logger.info("No documents found matching the query")
            return []
        
        # Process results
        processed_results = self._process_results(results, query, include_chunks)
        
        # Apply filters
        if filters:
            processed_results = self._apply_filters(processed_results, filters)
        
        # Re-rank results
        if rerank and len(processed_results) > 1:
            processed_results = self._rerank_results(processed_results, query)
        
        # TODO: Replace with proper semantic embeddings instead of hardcoded boosts
        
        # Return top-k results
        final_results = processed_results[:top_k]
        
        logger.info(f"Retrieved {len(final_results)} documents")
        return final_results
    

    
    def _process_results(self, results: List[Dict[str, Any]], 
                        query: str, include_chunks: bool) -> List[Dict[str, Any]]:
        """Process raw search results."""
        processed = []
        
        for result in results:
            # Add query relevance information
            result['query'] = query
            result['retrieved_at'] = datetime.now().isoformat()
            
            # Extract and highlight relevant snippets
            content = result.get('processed_content', result.get('content', ''))
            if content:
                result['snippet'] = self._extract_snippet(content, query)
                result['highlighted_snippet'] = self._highlight_query_terms(
                    result['snippet'], query
                )
            
            # Handle chunks if present
            if include_chunks and 'chunks' in result:
                relevant_chunks = self._find_relevant_chunks(result['chunks'], query)
                result['relevant_chunks'] = relevant_chunks
            
            processed.append(result)
        
        return processed
    
    def _extract_snippet(self, content: str, query: str, 
                        snippet_length: int = 200) -> str:
        """Extract relevant snippet from document content."""
        if not content or not query:
            return content[:snippet_length] if content else ""
        
        # Find the best snippet containing query terms
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        if not query_terms:
            return content[:snippet_length]
        
        # Find positions of query terms
        term_positions = []
        content_lower = content.lower()
        
        for term in query_terms:
            positions = [m.start() for m in re.finditer(re.escape(term), content_lower)]
            term_positions.extend(positions)
        
        if not term_positions:
            return content[:snippet_length]
        
        # Find the position with the most query terms nearby
        best_pos = min(term_positions)
        
        # Extract snippet around the best position
        start = max(0, best_pos - snippet_length // 2)
        end = min(len(content), start + snippet_length)
        
        # Adjust to word boundaries
        if start > 0:
            # Find the next word boundary
            while start < len(content) and content[start] not in ' \n\t':
                start += 1
        
        if end < len(content):
            # Find the previous word boundary
            while end > start and content[end] not in ' \n\t':
                end -= 1
        
        snippet = content[start:end].strip()
        
        # Add ellipsis if needed
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _highlight_query_terms(self, text: str, query: str) -> str:
        """Highlight query terms in text."""
        if not text or not query:
            return text
        
        query_terms = [term for term in query.split() if len(term) > 2]
        highlighted = text
        
        for term in query_terms:
            # Case-insensitive highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term.upper()}**", highlighted)
        
        return highlighted
    
    def _find_relevant_chunks(self, chunks: List[Dict[str, Any]], 
                            query: str, max_chunks: int = 3) -> List[Dict[str, Any]]:
        """Find the most relevant chunks within a document."""
        if not chunks:
            return []
        
        query_terms = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            chunk_words = set(chunk_text.split())
            
            # Calculate relevance score
            overlap = len(query_terms.intersection(chunk_words))
            total_query_terms = len(query_terms)
            
            if total_query_terms > 0:
                relevance_score = overlap / total_query_terms
            else:
                relevance_score = 0
            
            # Add embedding similarity if available
            if 'embedding' in chunk:
                query_embedding = self.embedding_generator.generate_embeddings(query)
                chunk_embedding = chunk['embedding']
                
                if isinstance(chunk_embedding, np.ndarray):
                    similarity = self.embedding_generator.compute_similarity(
                        query_embedding, chunk_embedding.reshape(1, -1)
                    )[0]
                    relevance_score = (relevance_score + similarity) / 2
            
            scored_chunks.append({
                **chunk,
                'relevance_score': relevance_score,
                'query_term_overlap': overlap
            })
        
        # Sort by relevance and return top chunks
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_chunks[:max_chunks]
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata filters to results."""
        filtered_results = []
        
        for result in results:
            include = True
            
            for key, value in filters.items():
                result_value = result.get(key)
                
                # Handle different filter types
                if isinstance(value, dict):
                    # Range or comparison filters
                    if '$gt' in value and result_value <= value['$gt']:
                        include = False
                        break
                    if '$lt' in value and result_value >= value['$lt']:
                        include = False
                        break
                    if '$gte' in value and result_value < value['$gte']:
                        include = False
                        break
                    if '$lte' in value and result_value > value['$lte']:
                        include = False
                        break
                    if '$eq' in value and result_value != value['$eq']:
                        include = False
                        break
                    if '$ne' in value and result_value == value['$ne']:
                        include = False
                        break
                    if '$in' in value and result_value not in value['$in']:
                        include = False
                        break
                elif isinstance(value, list):
                    # List of acceptable values
                    if result_value not in value:
                        include = False
                        break
                else:
                    # Exact match
                    if result_value != value:
                        include = False
                        break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _rerank_results(self, results: List[Dict[str, Any]], 
                       query: str) -> List[Dict[str, Any]]:
        """Re-rank results using multiple signals."""
        if len(results) <= 1:
            return results
        
        # Calculate additional ranking signals
        for result in results:
            signals = self._calculate_ranking_signals(result, query)
            result['ranking_signals'] = signals
            result['combined_score'] = self._combine_ranking_signals(signals)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def _calculate_ranking_signals(self, result: Dict[str, Any], 
                                 query: str) -> Dict[str, float]:
        """Calculate various ranking signals."""
        signals = {}
        
        # Semantic similarity (from vector search)
        signals['semantic_similarity'] = result.get('similarity_score', 0.0)
        
        # Query term frequency
        content = result.get('processed_content', result.get('content', ''))
        if content:
            query_terms = query.lower().split()
            content_lower = content.lower()
            
            term_freq = sum(content_lower.count(term) for term in query_terms)
            signals['term_frequency'] = min(term_freq / len(query_terms), 1.0)
            
            # Document length penalty (prefer concise, relevant documents)
            doc_length = len(content.split())
            signals['length_penalty'] = 1.0 / (1.0 + doc_length / 1000)
        else:
            signals['term_frequency'] = 0.0
            signals['length_penalty'] = 0.0
        
        # Recency (if timestamp available)
        if 'modified_time' in result:
            try:
                # Assume modified_time is a timestamp
                age_days = (datetime.now().timestamp() - result['modified_time']) / 86400
                signals['recency'] = max(0.0, 1.0 - age_days / 365)  # Decay over a year
            except (ValueError, TypeError):
                signals['recency'] = 0.5  # Neutral score
        else:
            signals['recency'] = 0.5
        
        # File type preference (you can customize this)
        file_type_scores = {
            '.md': 1.0,
            '.txt': 0.9,
            '.pdf': 0.8,
            '.docx': 0.7,
            '.json': 0.6,
            '.csv': 0.5
        }
        file_type = result.get('file_type', '')
        signals['file_type_score'] = file_type_scores.get(file_type, 0.5)
        
        return signals
    
    def _combine_ranking_signals(self, signals: Dict[str, float]) -> float:
        """Combine ranking signals into a final score."""
        # Weighted combination of signals
        weights = {
            'semantic_similarity': 0.4,
            'term_frequency': 0.3,
            'recency': 0.1,
            'file_type_score': 0.1,
            'length_penalty': 0.1
        }
        
        combined_score = 0.0
        for signal, weight in weights.items():
            combined_score += signals.get(signal, 0.0) * weight
        
        return combined_score
    
    def retrieve_by_metadata(self, 
                           filters: Dict[str, Any],
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents by metadata filters only."""
        # Get all documents from vector store
        all_docs = self.vector_store.documents
        
        # Apply filters
        filtered_docs = self._apply_filters(all_docs, filters)
        
        # Sort by relevance (using file modification time as fallback)
        filtered_docs.sort(
            key=lambda x: x.get('modified_time', 0), 
            reverse=True
        )
        
        return filtered_docs[:top_k]
    
    def get_similar_documents(self, 
                            document_id: str,
                            top_k: int = 5,
                            threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        # Get the source document
        source_doc = self.vector_store.get_document_by_id(document_id)
        if not source_doc:
            logger.warning(f"Document not found: {document_id}")
            return []
        
        # Use document content as query
        content = source_doc.get('processed_content', source_doc.get('content', ''))
        if not content:
            logger.warning(f"No content found for document: {document_id}")
            return []
        
        # Retrieve similar documents
        results = self.retrieve(
            query=content[:500],  # Use first 500 chars as query
            top_k=top_k + 1,  # +1 to account for the source document
            threshold=threshold,
            rerank=True
        )
        
        # Remove the source document from results
        filtered_results = [
            result for result in results 
            if result.get('id') != document_id
        ]
        
        return filtered_results[:top_k]
    
    def explain_retrieval(self, query: str, document_id: str) -> Dict[str, Any]:
        """Explain why a document was retrieved for a query."""
        # Get the document
        doc = self.vector_store.get_document_by_id(document_id)
        if not doc:
            return {'error': f'Document {document_id} not found'}
        
        # Generate embeddings
        query_embedding = self.embedding_generator.generate_embeddings(query)
        
        # Calculate similarity if document has embedding
        similarity = 0.0
        if 'embedding' in doc:
            doc_embedding = doc['embedding']
            if isinstance(doc_embedding, np.ndarray):
                similarity = self.embedding_generator.compute_similarity(
                    query_embedding, doc_embedding.reshape(1, -1)
                )[0]
        
        # Calculate ranking signals
        signals = self._calculate_ranking_signals(doc, query)
        
        # Find matching terms
        query_terms = set(query.lower().split())
        content = doc.get('processed_content', doc.get('content', ''))
        content_words = set(content.lower().split()) if content else set()
        matching_terms = list(query_terms.intersection(content_words))
        
        return {
            'document_id': document_id,
            'query': query,
            'similarity_score': float(similarity),
            'ranking_signals': signals,
            'matching_terms': matching_terms,
            'snippet': self._extract_snippet(content, query) if content else '',
            'metadata': {
                'filename': doc.get('filename', ''),
                'file_type': doc.get('file_type', ''),
                'word_count': doc.get('word_count', 0)
            }
        }

