#!/usr/bin/env python3
"""
Data fetching script for resume NER datasets.
Downloads/clones all required datasets from various sources.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_io import load_all_datasets

logger = logging.getLogger(__name__)


def main():
    """Main function for data fetching."""
    parser = argparse.ArgumentParser(description="Fetch all resume NER datasets")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for datasets")
    parser.add_argument("--skip_kaggle", action="store_true", help="Skip Kaggle dataset")
    parser.add_argument("--skip_github", action="store_true", help="Skip GitHub datasets")
    parser.add_argument("--skip_hf", action="store_true", help="Skip HuggingFace dataset")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting data fetch process...")
    
    try:
        # Load all datasets
        samples = load_all_datasets(args.output_dir)
        
        if samples:
            logger.info(f"Successfully fetched {len(samples)} samples")
            logger.info("Data fetch completed successfully!")
        else:
            logger.warning("No samples were fetched")
            
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
