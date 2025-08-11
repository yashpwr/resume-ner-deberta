"""
Label space management for resume NER.
Builds union label set, applies normalization rules, and enforces BIO validity.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class LabelSpaceManager:
    """Manages label space for resume NER datasets."""
    
    def __init__(self, synonyms_file: str = "configs/label_synonyms.json"):
        self.synonyms_file = Path(synonyms_file)
        self.synonyms = self._load_synonyms()
        self.label2id = {}
        self.id2label = {}
        self.bio_labels = []
    
    def _load_synonyms(self) -> Dict[str, str]:
        """Load label synonyms from JSON file."""
        if not self.synonyms_file.exists():
            logger.warning(f"Synonyms file not found: {self.synonyms_file}")
            return {}
        
        try:
            with open(self.synonyms_file, 'r', encoding='utf-8') as f:
                synonyms = json.load(f)
            logger.info(f"Loaded {len(synonyms)} label synonyms")
            return synonyms
        except Exception as e:
            logger.error(f"Error loading synonyms file: {e}")
            return {}
    
    def normalize_label(self, label: str) -> str:
        """Normalize a label using synonyms."""
        # Remove BIO prefix if present
        bio_prefix = ""
        if label.startswith(('B-', 'I-')):
            bio_prefix = label[:2]
            label = label[2:]
        
        # Apply synonym mapping
        normalized = self.synonyms.get(label, label)
        
        # Restore BIO prefix
        return f"{bio_prefix}{normalized}" if bio_prefix else normalized
    
    def build_label_space(self, samples: List[Dict]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        """Build complete label space from samples."""
        logger.info("Building label space from samples...")
        
        # Collect all unique labels
        all_labels = set()
        for sample in samples:
            labels = sample.get('labels', [])
            if isinstance(labels, list):
                for label in labels:
                    if label and label != 'O':
                        all_labels.add(label)
        
        logger.info(f"Found {len(all_labels)} unique labels")
        
        # Normalize labels
        normalized_labels = set()
        for label in all_labels:
            normalized = self.normalize_label(label)
            normalized_labels.add(normalized)
        
        # Add O label (always present)
        normalized_labels.add('O')
        
        # Sort labels (O first, then alphabetically)
        sorted_labels = sorted(normalized_labels, key=lambda x: (x != 'O', x))
        
        # Create ID mappings
        self.label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id2label = {idx: label for idx, label in enumerate(sorted_labels)}
        
        # Create BIO labels
        self.bio_labels = []
        for label in sorted_labels:
            if label == 'O':
                self.bio_labels.append('O')
            else:
                self.bio_labels.extend([f'B-{label}', f'I-{label}'])
        
        logger.info(f"Final label space: {len(sorted_labels)} base labels, {len(self.bio_labels)} BIO labels")
        logger.info(f"Base labels: {sorted_labels}")
        
        return sorted_labels, self.label2id, self.id2label
    
    def convert_to_ids(self, labels: List[str]) -> List[int]:
        """Convert string labels to IDs."""
        return [self.label2id.get(label, 0) for label in labels]  # 0 is usually O
    
    def convert_to_bio(self, labels: List[str]) -> List[str]:
        """Convert labels to BIO format."""
        bio_labels = []
        
        for i, label in enumerate(labels):
            if label == 'O':
                bio_labels.append('O')
            else:
                # Check if this is the start of a new entity
                if i == 0 or labels[i-1] != label:
                    bio_labels.append(f'B-{label}')
                else:
                    bio_labels.append(f'I-{label}')
        
        return bio_labels
    
    def fix_bio_validity(self, labels: List[str]) -> List[str]:
        """Fix BIO validity issues in labels."""
        fixed_labels = labels.copy()
        
        for i in range(len(fixed_labels)):
            label = fixed_labels[i]
            
            if label.startswith('I-'):
                # Check if there's a preceding B- or I- with the same entity type
                entity_type = label[2:]
                has_preceding = False
                
                # Look backwards for B- or I- of same type
                for j in range(i-1, -1, -1):
                    prev_label = fixed_labels[j]
                    if prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}':
                        has_preceding = True
                        break
                    elif prev_label == 'O':
                        break
                
                if not has_preceding:
                    # Convert I- to B- if no preceding label
                    fixed_labels[i] = f'B-{entity_type}'
                    logger.debug(f"Fixed I- without preceding B- at position {i}: {label} -> B-{entity_type}")
        
        return fixed_labels
    
    def apply_normalization(self, samples: List[Dict]) -> List[Dict]:
        """Apply label normalization to all samples."""
        logger.info("Applying label normalization...")
        
        normalized_samples = []
        for sample in samples:
            labels = sample.get('labels', [])
            if not labels:
                continue
            
            # Normalize labels
            normalized_labels = [self.normalize_label(label) for label in labels]
            
            # Convert to BIO if not already
            if not any(label.startswith(('B-', 'I-')) for label in normalized_labels):
                normalized_labels = self.convert_to_bio(normalized_labels)
            
            # Fix BIO validity
            normalized_labels = self.fix_bio_validity(normalized_labels)
            
            # Create normalized sample
            normalized_sample = sample.copy()
            normalized_sample['labels'] = normalized_labels
            normalized_sample['ner_tags'] = normalized_labels  # Keep both for compatibility
            
            normalized_samples.append(normalized_sample)
        
        logger.info(f"Normalized {len(normalized_samples)} samples")
        return normalized_samples
    
    def save_label_mappings(self, output_dir: str = "artifacts"):
        """Save label mappings to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save base label mappings
        label2id_file = output_path / "label2id.json"
        id2label_file = output_path / "id2label.json"
        
        with open(label2id_file, 'w', encoding='utf-8') as f:
            json.dump(self.label2id, f, indent=2, ensure_ascii=False)
        
        with open(id2label_file, 'w', encoding='utf-8') as f:
            json.dump(self.id2label, f, indent=2, ensure_ascii=False)
        
        # Save BIO label list
        bio_labels_file = output_path / "bio_labels.json"
        with open(bio_labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.bio_labels, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved label mappings to {output_path}")
        logger.info(f"Base labels: {list(self.label2id.keys())}")
        logger.info(f"BIO labels: {self.bio_labels}")
    
    def get_statistics(self, samples: List[Dict]) -> Dict[str, Any]:
        """Get statistics about label distribution."""
        stats = defaultdict(int)
        total_samples = len(samples)
        
        for sample in samples:
            labels = sample.get('labels', [])
            for label in labels:
                stats[label] += 1
        
        # Calculate percentages
        label_percentages = {}
        total_labels = sum(stats.values())
        for label, count in stats.items():
            label_percentages[label] = (count / total_labels * 100) if total_labels > 0 else 0
        
        return {
            'total_samples': total_samples,
            'total_labels': total_labels,
            'label_counts': dict(stats),
            'label_percentages': label_percentages,
            'unique_labels': len(stats),
            'most_common_labels': sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]
        }


def process_label_space(samples: List[Dict], output_dir: str = "artifacts") -> Tuple[LabelSpaceManager, List[Dict]]:
    """Process label space and return manager and normalized samples."""
    manager = LabelSpaceManager()
    
    # Build label space
    base_labels, label2id, id2label = manager.build_label_space(samples)
    
    # Apply normalization
    normalized_samples = manager.apply_normalization(samples)
    
    # Save mappings
    manager.save_label_mappings(output_dir)
    
    # Print statistics
    stats = manager.get_statistics(normalized_samples)
    logger.info("Label space statistics:")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Total labels: {stats['total_labels']}")
    logger.info(f"Unique labels: {stats['unique_labels']}")
    logger.info(f"Most common labels: {stats['most_common_labels'][:5]}")
    
    return manager, normalized_samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    test_samples = [
        {'labels': ['PERSON', 'O', 'O', 'COMPANY'], 'text': 'John works at Google'},
        {'labels': ['O', 'O', 'EMAIL', 'O'], 'text': 'Contact me at john@email.com'},
        {'labels': ['PERSON', 'O', 'TITLE'], 'text': 'Jane is a Developer'}
    ]
    
    manager, normalized = process_label_space(test_samples)
    print("Label space processing completed!")
