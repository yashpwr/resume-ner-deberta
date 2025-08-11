#!/usr/bin/env python3
"""
Dataset merging script for resume NER.
Merges all datasets and creates train/val/test splits.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import hashlib
from sklearn.model_selection import train_test_split
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_io import load_all_datasets
from label_space import process_label_space

logger = logging.getLogger(__name__)


def deduplicate_samples(samples):
    """Remove duplicate samples based on text hash."""
    unique_samples = {}
    duplicates = 0
    
    for sample in samples:
        # Create hash of text content
        text = sample.get('text', '')
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash not in unique_samples:
            unique_samples[text_hash] = sample
        else:
            duplicates += 1
    
    logger.info(f"Removed {duplicates} duplicate samples")
    return list(unique_samples.values())


def create_stratified_split(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Create stratified train/val/test split based on key labels."""
    np.random.seed(seed)
    
    # Create stratification feature based on presence of key labels
    key_labels = ['NAME', 'EMAIL', 'COMPANY', 'TITLE']
    
    def get_strata(sample):
        labels = sample.get('labels', [])
        # Count how many key labels are present
        key_count = sum(1 for label in labels if any(key in label for key in key_labels))
        # Create strata: 0, 1, 2, 3, 4+
        return min(key_count, 4)
    
    # Get strata for each sample
    strata = [get_strata(sample) for sample in samples]
    
    # First split: train vs temp
    train_samples, temp_samples, train_strata, temp_strata = train_test_split(
        samples, strata, 
        test_size=(val_ratio + test_ratio), 
        random_state=seed,
        stratify=strata
    )
    
    # Second split: val vs test
    val_samples, test_samples, val_strata, test_strata = train_test_split(
        temp_samples, temp_strata,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=temp_strata
    )
    
    logger.info(f"Split created: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    
    # Log strata distribution
    for split_name, split_samples, split_strata in [("train", train_samples, train_strata), 
                                                   ("val", val_samples, val_strata), 
                                                   ("test", test_samples, test_strata)]:
        strata_counts = {}
        for stratum in split_strata:
            strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
        logger.info(f"{split_name} strata distribution: {strata_counts}")
    
    return train_samples, val_samples, test_samples


def save_jsonl(samples, output_file):
    """Save samples to JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main function for dataset merging."""
    parser = argparse.ArgumentParser(description="Merge resume NER datasets and create splits")
    parser.add_argument("--input_dir", type=str, default="data", help="Input directory with datasets")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for merged datasets")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logger.error("Ratios must sum to 1.0")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting dataset merge process...")
    
    try:
        # Load all datasets
        logger.info("Loading datasets...")
        samples = load_all_datasets(args.input_dir)
        
        if not samples:
            logger.error("No samples loaded")
            sys.exit(1)
        
        logger.info(f"Loaded {len(samples)} raw samples")
        
        # Process label space
        logger.info("Processing label space...")
        label_manager, normalized_samples = process_label_space(samples)
        
        logger.info(f"Label space: {len(label_manager.label2id)} labels")
        logger.info(f"Labels: {list(label_manager.label2id.keys())}")
        
        # Deduplicate samples
        logger.info("Deduplicating samples...")
        unique_samples = deduplicate_samples(normalized_samples)
        
        # Create splits
        logger.info("Creating train/val/test splits...")
        train_samples, val_samples, test_samples = create_stratified_split(
            unique_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        # Save splits
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Saving dataset splits...")
        save_jsonl(train_samples, output_dir / "train.jsonl")
        save_jsonl(val_samples, output_dir / "val.jsonl")
        save_jsonl(test_samples, output_dir / "test.jsonl")
        
        # Save dataset statistics
        stats = {
            'total_samples': len(unique_samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'label_space': list(label_manager.label2id.keys()),
            'label_counts': label_manager.get_statistics(unique_samples)
        }
        
        stats_file = output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Dataset statistics saved to {stats_file}")
        logger.info("Dataset merge completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
