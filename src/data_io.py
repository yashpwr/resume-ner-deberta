"""
Data loading and preprocessing for resume NER datasets.
Handles HuggingFace, Kaggle, and GitHub datasets with Doccano format conversion.
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import pandas as pd
import requests
from datasets import load_dataset
from bs4 import BeautifulSoup
import zipfile

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load(self) -> List[Dict[str, Any]]:
        """Load and return dataset samples."""
        raise NotImplementedError
    
    def save_samples(self, samples: List[Dict[str, Any]], filename: str):
        """Save samples to JSONL file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(samples)} samples to {output_file}")


class HuggingFaceLoader(DatasetLoader):
    """Loader for HuggingFace datasets."""
    
    def __init__(self, dataset_name: str, output_dir: str = "data"):
        super().__init__(output_dir)
        self.dataset_name = dataset_name
    
    def load(self) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace."""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)
            
            samples = []
            for split in dataset.keys():
                for item in dataset[split]:
                    sample = self._convert_item(item)
                    if sample:
                        sample['source'] = f"hf_{self.dataset_name}_{split}"
                        samples.append(sample)
            
            logger.info(f"Loaded {len(samples)} samples from {self.dataset_name}")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading dataset {self.dataset_name}: {e}")
            return []
    
    def _convert_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert HuggingFace item to standard format."""
        # Handle different possible field names
        tokens = item.get('tokens') or item.get('words') or []
        ner_tags = item.get('ner_tags') or item.get('labels') or []
        
        if not tokens or not ner_tags:
            return None
        
        # Ensure tokens and tags have same length
        if len(tokens) != len(ner_tags):
            logger.warning(f"Token/tag length mismatch: {len(tokens)} vs {len(ner_tags)}")
            return None
        
        return {
            'tokens': tokens,
            'ner_tags': ner_tags,
            'labels': ner_tags,  # Keep original labels for now
            'text': ' '.join(tokens)
        }


class KaggleLoader(DatasetLoader):
    """Loader for Kaggle datasets."""
    
    def __init__(self, dataset_name: str, output_dir: str = "data"):
        super().__init__(output_dir)
        self.dataset_name = dataset_name
        self.kaggle_dir = self.output_dir / "kaggle_ats"
    
    def load(self) -> List[Dict[str, Any]]:
        """Load Kaggle dataset if credentials available."""
        # Check if Kaggle credentials exist
        if not self._check_kaggle_creds():
            logger.warning("Kaggle credentials not found. Skipping Kaggle dataset.")
            logger.info("To use Kaggle dataset, set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
            return []
        
        try:
            self._download_dataset()
            return self._parse_dataset()
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {e}")
            return []
    
    def _check_kaggle_creds(self) -> bool:
        """Check if Kaggle credentials are available."""
        from dotenv import load_dotenv
        load_dotenv()
        
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')
        
        if not username or not key:
            return False
        
        # Set environment variables for Kaggle API
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        return True
    
    def _download_dataset(self):
        """Download dataset using Kaggle API."""
        self.kaggle_dir.mkdir(exist_ok=True)
        
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=self.kaggle_dir,
                unzip=True
            )
            logger.info(f"Downloaded dataset to {self.kaggle_dir}")
        except Exception as e:
            logger.error(f"Failed to download via Kaggle API: {e}")
            raise
    
    def _parse_dataset(self) -> List[Dict[str, Any]]:
        """Parse downloaded Kaggle dataset."""
        samples = []
        
        # Look for CSV files
        csv_files = list(self.kaggle_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in Kaggle dataset")
            return samples
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Parsing {csv_file} with columns: {list(df.columns)}")
                
                # This dataset doesn't have NER labels, so we'll skip it for training
                # but keep the structure for future use
                for _, row in df.iterrows():
                    # Extract text from resume if available
                    text = self._extract_text_from_row(row)
                    if text:
                        tokens = text.split()
                        # Create dummy labels (all O) since this dataset has no NER annotations
                        ner_tags = ['O'] * len(tokens)
                        
                        samples.append({
                            'tokens': tokens,
                            'ner_tags': ner_tags,
                            'labels': ner_tags,
                            'text': text,
                            'note': 'No NER labels - for future pretraining only'
                        })
                
            except Exception as e:
                logger.error(f"Error parsing {csv_file}: {e}")
                continue
        
        logger.info(f"Parsed {len(samples)} samples from Kaggle dataset")
        return samples
    
    def _extract_text_from_row(self, row: pd.Series) -> str:
        """Extract text content from a dataset row."""
        # Try different possible column names
        text_columns = ['resume', 'text', 'content', 'description']
        
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                text = str(row[col])
                # Basic cleaning
                if text.strip():
                    return text.strip()
        
        return ""


class GitHubLoader(DatasetLoader):
    """Loader for GitHub datasets."""
    
    def __init__(self, repo_url: str, output_dir: str = "data"):
        super().__init__(output_dir)
        self.repo_url = repo_url
        self.repo_name = repo_url.split('/')[-1]
        self.repo_dir = self.output_dir / f"github_{self.repo_name}"
    
    def load(self) -> List[Dict[str, Any]]:
        """Clone and load GitHub dataset."""
        try:
            self._clone_repo()
            return self._parse_repo()
        except Exception as e:
            logger.error(f"Error loading GitHub dataset {self.repo_url}: {e}")
            return []
    
    def _clone_repo(self):
        """Clone GitHub repository."""
        if self.repo_dir.exists():
            logger.info(f"Repository already exists at {self.repo_dir}")
            return
        
        logger.info(f"Cloning {self.repo_url} to {self.repo_dir}")
        subprocess.run([
            'git', 'clone', '--depth', '1', self.repo_url, str(self.repo_dir)
        ], check=True)
    
    def _parse_repo(self) -> List[Dict[str, Any]]:
        """Parse repository content based on its structure."""
        if 'Resume-Corpus-Dataset' in self.repo_name:
            return self._parse_resume_corpus()
        elif 'resume-dataset' in self.repo_name:
            return self._parse_doccano_resume()
        else:
            logger.warning(f"Unknown repository structure for {self.repo_name}")
            return []
    
    def _parse_resume_corpus(self) -> List[Dict[str, Any]]:
        """Parse Resume-Corpus-Dataset format."""
        samples = []
        
        # Look for JSON files
        json_files = list(self.repo_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = [data]
                else:
                    continue
                
                for item in items:
                    sample = self._convert_resume_corpus_item(item)
                    if sample:
                        sample['source'] = f"github_resume_corpus_{json_file.stem}"
                        samples.append(sample)
                        
            except Exception as e:
                logger.error(f"Error parsing {json_file}: {e}")
                continue
        
        return samples
    
    def _convert_resume_corpus_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Resume-Corpus-Dataset item to standard format."""
        # This dataset may have different structures, so we need to be flexible
        if 'text' in item:
            text = item['text']
        elif 'content' in item:
            text = item['content']
        elif 'resume' in item:
            text = item['resume']
        else:
            return None
        
        if not text or not isinstance(text, str):
            return None
        
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 10:  # Skip very short texts
            return None
        
        tokens = text.split()
        
        # Check if there are any annotations
        if 'annotations' in item or 'labels' in item or 'entities' in item:
            # Try to extract NER labels
            ner_tags = self._extract_ner_tags(item, tokens)
        else:
            # No annotations available
            ner_tags = ['O'] * len(tokens)
        
        return {
            'tokens': tokens,
            'ner_tags': ner_tags,
            'labels': ner_tags,
            'text': text
        }
    
    def _extract_ner_tags(self, item: Dict[str, Any], tokens: List[str]) -> List[str]:
        """Extract NER tags from item annotations."""
        # This is a simplified implementation
        # In practice, you'd need to handle the specific annotation format
        ner_tags = ['O'] * len(tokens)
        
        # Look for entity annotations
        entities = item.get('entities') or item.get('annotations') or item.get('labels') or []
        
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict):
                    label = entity.get('label', 'O')
                    start = entity.get('start', 0)
                    end = entity.get('end', 0)
                    
                    # Convert character positions to token positions
                    # This is a simplified approach
                    if start == 0 and end > 0:
                        ner_tags[0] = f"B-{label}"
                        for i in range(1, min(len(ner_tags), 3)):  # Assume max 3 tokens
                            ner_tags[i] = f"I-{label}"
        
        return ner_tags
    
    def _parse_doccano_resume(self) -> List[Dict[str, Any]]:
        """Parse Doccano format resume dataset."""
        samples = []
        
        # Look for JSONL files (Doccano format)
        jsonl_files = list(self.repo_dir.rglob("*.jsonl"))
        if not jsonl_files:
            # Also check for regular JSON files
            jsonl_files = list(self.repo_dir.rglob("*.json"))
        
        logger.info(f"Found {len(jsonl_files)} potential Doccano files")
        
        for file_path in jsonl_files:
            try:
                samples.extend(self._parse_doccano_file(file_path))
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                continue
        
        return samples
    
    def _parse_doccano_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single Doccano format file."""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line.strip())
                    sample = self._convert_doccano_item(data)
                    if sample:
                        sample['source'] = f"github_doccano_{file_path.stem}_{line_num}"
                        samples.append(sample)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num} in {file_path}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num} in {file_path}: {e}")
                    continue
        
        return samples
    
    def _convert_doccano_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Doccano item to standard format."""
        # Doccano format: {"text": "...", "labels": [[start, end, label], ...]}
        text = item.get('text', '')
        labels = item.get('labels', [])
        
        if not text or not isinstance(text, str):
            return None
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 10:
            return None
        
        # Convert character spans to BIO tags
        tokens, ner_tags = self._spans_to_bio(text, labels)
        
        if not tokens:
            return None
        
        return {
            'tokens': tokens,
            'ner_tags': ner_tags,
            'labels': ner_tags,
            'text': text
        }
    
    def _spans_to_bio(self, text: str, spans: List[List]) -> Tuple[List[str], List[str]]:
        """Convert character spans to BIO tags."""
        tokens = text.split()
        ner_tags = ['O'] * len(tokens)
        
        # Create a mapping from character positions to token positions
        char_to_token = []
        token_start = 0
        
        for i, token in enumerate(tokens):
            token_end = token_start + len(token)
            char_to_token.extend([i] * len(token))
            token_start = token_end + 1  # +1 for space
        
        # Apply spans
        for span in spans:
            if len(span) >= 3:
                start_char, end_char, label = span[0], span[1], span[2]
                
                # Find tokens that overlap with this span
                span_tokens = set()
                for char_pos in range(start_char, end_char):
                    if char_pos < len(char_to_token):
                        span_tokens.add(char_to_token[char_pos])
                
                if span_tokens:
                    span_tokens = sorted(span_tokens)
                    # Mark first token as B-, rest as I-
                    ner_tags[span_tokens[0]] = f"B-{label}"
                    for token_idx in span_tokens[1:]:
                        ner_tags[token_idx] = f"I-{label}"
        
        return tokens, ner_tags


def load_all_datasets(output_dir: str = "data") -> List[Dict[str, Any]]:
    """Load all available datasets."""
    loaders = [
        HuggingFaceLoader("Mehyaar/Annotated_NER_PDF_Resumes", output_dir),
        KaggleLoader("mgmitesh/ats-scoring-dataset", output_dir),
        GitHubLoader("https://github.com/vrundag91/Resume-Corpus-Dataset", output_dir),
        GitHubLoader("https://github.com/juanfpinzon/resume-dataset", output_dir)
    ]
    
    all_samples = []
    
    for loader in loaders:
        try:
            samples = loader.load()
            all_samples.extend(samples)
            logger.info(f"Loaded {len(samples)} samples from {loader.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to load dataset with {loader.__class__.__name__}: {e}")
            continue
    
    logger.info(f"Total samples loaded: {len(all_samples)}")
    return all_samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = load_all_datasets()
    print(f"Loaded {len(samples)} total samples")
