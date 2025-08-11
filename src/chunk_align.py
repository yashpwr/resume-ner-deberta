"""
Tokenization and chunking for resume NER with DeBERTa-v3-small.
Handles wordpiece alignment and BIO label preservation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)


class TokenizerChunker:
    """Handles tokenization and chunking for DeBERTa model."""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-small", max_length: int = 256, stride: int = 32):
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized tokenizer for {model_name}")
        logger.info(f"Max length: {max_length}, Stride: {stride}")
    
    def tokenize_and_align(self, tokens: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Tokenize tokens and align BIO labels to wordpieces.
        
        Args:
            tokens: List of word tokens
            labels: List of BIO labels corresponding to tokens
            
        Returns:
            Dict with tokenized inputs and aligned labels
        """
        # Tokenize with is_split_into_words=True
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=False,
            padding=False,
            return_offsets_mapping=True
        )
        
        # Get word IDs for each token
        word_ids = tokenized.word_ids()
        
        # Align labels to wordpieces
        aligned_labels = []
        label_to_id = {}  # We'll need this for training
        
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                # Special tokens (CLS, SEP, etc.) get -100
                aligned_labels.append(-100)
            else:
                # Get the label for this word
                label = labels[word_id]
                
                # Convert label to ID if needed
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                
                # Only label the first subword of each word
                if i == 0 or word_ids[i-1] != word_id:
                    aligned_labels.append(label_to_id[label])
                else:
                    # Subsequent subwords get -100
                    aligned_labels.append(-100)
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': aligned_labels,
            'word_ids': word_ids,
            'label_to_id': label_to_id
        }
    
    def chunk_sequence(self, input_ids: List[int], attention_mask: List[int], 
                      labels: List[int], word_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Chunk a sequence into overlapping chunks of max_length.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Aligned labels
            word_ids: Word IDs for each token
            
        Returns:
            List of chunked sequences
        """
        chunks = []
        
        # Calculate chunk boundaries
        total_length = len(input_ids)
        
        if total_length <= self.max_length:
            # No need to chunk
            chunks.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'word_ids': word_ids
            })
            return chunks
        
        # Create overlapping chunks
        for start in range(0, total_length, self.max_length - self.stride):
            end = start + self.max_length
            
            # Extract chunk
            chunk_input_ids = input_ids[start:end]
            chunk_attention_mask = attention_mask[start:end]
            chunk_labels = labels[start:end]
            chunk_word_ids = word_ids[start:end]
            
            # Adjust word_ids for the chunk
            chunk_word_ids = [wid - start if wid is not None else None for wid in chunk_word_ids]
            
            chunks.append({
                'input_ids': chunk_input_ids,
                'attention_mask': chunk_attention_mask,
                'labels': chunk_labels,
                'word_ids': chunk_word_ids
            })
        
        logger.debug(f"Chunked sequence of length {total_length} into {len(chunks)} chunks")
        return chunks
    
    def process_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single sample through tokenization and chunking.
        
        Args:
            sample: Sample with tokens and labels
            
        Returns:
            List of processed chunks
        """
        tokens = sample.get('tokens', [])
        labels = sample.get('labels', [])
        
        if not tokens or not labels:
            logger.warning("Sample missing tokens or labels")
            return []
        
        if len(tokens) != len(labels):
            logger.warning(f"Token/label length mismatch: {len(tokens)} vs {len(labels)}")
            return []
        
        # Tokenize and align
        tokenized = self.tokenize_and_align(tokens, labels)
        
        # Chunk if necessary
        chunks = self.chunk_sequence(
            tokenized['input_ids'],
            tokenized['attention_mask'],
            tokenized['labels'],
            tokenized['word_ids']
        )
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['sample_id'] = sample.get('id', f"sample_{i}")
            chunk['source'] = sample.get('source', 'unknown')
            chunk['original_length'] = len(tokens)
            chunk['chunk_id'] = i
        
        return chunks
    
    def process_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples.
        
        Args:
            samples: List of samples
            
        Returns:
            List of all processed chunks
        """
        all_chunks = []
        
        for sample in samples:
            try:
                chunks = self.process_sample(sample)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
        
        logger.info(f"Processed {len(samples)} samples into {len(all_chunks)} chunks")
        return all_chunks
    
    def get_label_mapping(self, samples: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Build label to ID mapping from all samples.
        
        Args:
            samples: List of samples
            
        Returns:
            Label to ID mapping
        """
        label_set = set()
        
        for sample in samples:
            labels = sample.get('labels', [])
            for label in labels:
                if label and label != 'O':
                    label_set.add(label)
        
        # Add O label
        label_set.add('O')
        
        # Sort labels (O first, then alphabetically)
        sorted_labels = sorted(label_set, key=lambda x: (x != 'O', x))
        
        # Create mapping
        label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        
        logger.info(f"Created label mapping: {label_to_id}")
        return label_to_id
    
    def convert_labels_to_ids(self, labels: List[str], label_to_id: Dict[str, int]) -> List[int]:
        """
        Convert string labels to IDs.
        
        Args:
            labels: List of string labels
            label_to_id: Label to ID mapping
            
        Returns:
            List of label IDs
        """
        return [label_to_id.get(label, 0) for label in labels]  # 0 is usually O
    
    def prepare_for_training(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Prepare samples for training by converting labels to IDs and chunking.
        
        Args:
            samples: List of samples
            
        Returns:
            Tuple of (processed_chunks, label_to_id)
        """
        # Get label mapping
        label_to_id = self.get_label_mapping(samples)
        
        # Convert labels to IDs in samples
        processed_samples = []
        for sample in samples:
            processed_sample = sample.copy()
            labels = sample.get('labels', [])
            processed_sample['labels'] = self.convert_labels_to_ids(labels, label_to_id)
            processed_samples.append(processed_sample)
        
        # Process through tokenization and chunking
        chunks = self.process_batch(processed_samples)
        
        return chunks, label_to_id


def create_tokenizer_chunker(model_name: str = "microsoft/deberta-v3-small", 
                           max_length: int = 256, stride: int = 32) -> TokenizerChunker:
    """Factory function to create a TokenizerChunker instance."""
    return TokenizerChunker(model_name, max_length, stride)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the tokenizer
    chunker = create_tokenizer_chunker()
    
    # Test sample
    test_sample = {
        'tokens': ['John', 'Smith', 'works', 'at', 'Google', 'as', 'a', 'Software', 'Engineer'],
        'labels': ['B-NAME', 'I-NAME', 'O', 'O', 'B-COMPANY', 'O', 'O', 'B-TITLE', 'I-TITLE']
    }
    
    chunks, label_mapping = chunker.prepare_for_training([test_sample])
    print(f"Created {len(chunks)} chunks")
    print(f"Label mapping: {label_mapping}")
