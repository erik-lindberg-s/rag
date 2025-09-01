"""
Dataset builder for training data preparation.
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class DatasetBuilder:
    """Build datasets for training from processed documents."""
    
    def __init__(self):
        """Initialize dataset builder."""
        pass
    
    def build_training_dataset(self, 
                              documents: List[Dict[str, Any]], 
                              format_type: str = "instruction") -> List[Dict[str, Any]]:
        """
        Build a training dataset from processed documents.
        
        Args:
            documents: List of processed documents
            format_type: Type of dataset format ("instruction", "completion", etc.)
            
        Returns:
            List of training examples
        """
        training_examples = []
        
        for doc in documents:
            content = doc.get('processed_content', doc.get('content', ''))
            if not content:
                continue
            
            if format_type == "instruction":
                # Create instruction-following examples
                example = {
                    'instruction': f"Based on the document '{doc.get('filename', 'unknown')}', answer the following question:",
                    'input': "What is this document about?",
                    'output': content[:500] + "..." if len(content) > 500 else content,
                    'metadata': {
                        'source_file': doc.get('filename', ''),
                        'file_type': doc.get('file_type', ''),
                        'word_count': doc.get('word_count', 0)
                    }
                }
                training_examples.append(example)
        
        logger.info(f"Built {len(training_examples)} training examples")
        return training_examples
