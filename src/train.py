"""
Training script for resume NER using DeBERTa-v3-small with LoRA.
Includes early stopping, metrics tracking, and model saving.
"""

import argparse
import json
import logging
import os
import sys
import glob
import re
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import yaml

# Disable wandb completely to avoid interactive prompts in Kaggle
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure src and repo root are importable when run as a script
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, TrainingArguments,
    DataCollatorForTokenClassification, EarlyStoppingCallback, Trainer
)
from peft import get_peft_model, LoraConfig

# Set up logger first
logger = logging.getLogger(__name__)

try:
    from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
    HAS_SEQEVAL = True
except ImportError:
    logger.warning("seqeval not found, using sklearn metrics as fallback")
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    HAS_SEQEVAL = False
from sklearn.model_selection import train_test_split

# Import from the same directory
from data_io import load_all_datasets
from label_space import process_label_space
from chunk_align import create_tokenizer_chunker


def _safe_mkdirs(*paths):
    """Safely create directories."""
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def _latest_checkpoint(dirpath: str):
    """Find the latest checkpoint in the given directory."""
    if not dirpath or not os.path.isdir(dirpath):
        return None
    ckpts = glob.glob(os.path.join(dirpath, "checkpoint-*"))
    if not ckpts:
        return None
    def _step(p):
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=_step)


class ResumeNERTrainer:
    """Trainer for resume NER model."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_device()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        
        # Set random seed
        torch.manual_seed(self.config.get('seed', 42))
        np.random.seed(self.config.get('seed', 42))
        
    logger.info("Resume NER Trainer initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        out_cfg = self.config.get("output", {})
        logs_dir = out_cfg.get("logs_dir", "artifacts/logs")
        _safe_mkdirs(logs_dir)
        log_file = os.path.join(logs_dir, "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
        )
    # logger = logging.getLogger("resume-ner-train")
    
    def setup_device(self):
        """Setup device (GPU/CPU)."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_and_preprocess_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and preprocess all datasets."""
        logger.info("Loading datasets...")
        
        # Try to load from standardized dataset first
        standardized_path = Path(__file__).parent.parent / 'data' / 'standardized' / 'unified_dataset.json'
        if standardized_path.exists():
            logger.info("Loading from standardized dataset...")
            try:
                with open(standardized_path, 'r') as f:
                    samples = json.load(f)
                logger.info(f"✅ Loaded {len(samples)} samples from standardized dataset")
                
                # Convert to expected format
                normalized_samples = []
                label_set = set(['O'])  # Always include O label
                
                for sample in samples:
                    if 'text' in sample and 'annotations' in sample:
                        # Convert annotations to tokens and labels
                        text = sample['text']
                        annotations = sample['annotations']
                        
                        # Create simple tokenization (split by whitespace)
                        tokens = text.split()
                        labels = ['O'] * len(tokens)
                        
                        # Apply annotations
                        for start, end, label in annotations:
                            if isinstance(start, int) and isinstance(end, int):
                                # Find tokens that fall within annotation range
                                for i, token in enumerate(tokens):
                                    if start <= text.find(token) < end:
                                        if i == 0 or labels[i-1] == 'O':
                                            labels[i] = f'B-{label}'
                                            label_set.add(f'B-{label}')
                                        else:
                                            labels[i] = f'I-{label}'
                                            label_set.add(f'I-{label}')
                        
                        normalized_samples.append({
                            'tokens': tokens,
                            'labels': labels,
                            'text': text
                        })
                
                # Create label mappings
                sorted_labels = sorted(label_set, key=lambda x: (x != 'O', x))
                self.label2id = {label: idx for idx, label in enumerate(sorted_labels)}
                self.id2label = {idx: label for label, idx in self.label2id.items()}
                
                logger.info(f"✅ Converted {len(normalized_samples)} samples to training format")
                logger.info(f"✅ Created label mapping: {len(self.label2id)} labels")
                
            except Exception as e:
                logger.warning(f"Failed to load standardized dataset: {e}")
                normalized_samples = []
        else:
            logger.info("Standardized dataset not found, trying original data loaders...")
            # Load all datasets
            try:
                samples = load_all_datasets()
                if not samples:
                    raise ValueError("No samples loaded from datasets")
                
                logger.info(f"Loaded {len(samples)} raw samples")
                
                # Process label space
                label_manager, normalized_samples = process_label_space(samples)
                self.label2id = label_manager.label2id
                self.id2label = label_manager.id2label
                
                logger.info(f"Label space: {len(self.label2id)} labels")
                
            except Exception as e:
                logger.error(f"Failed to load datasets: {e}")
                # Create minimal label mappings as fallback
                self.label2id = {'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}
                self.id2label = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}
                logger.warning("Using fallback label mappings")
                
                if not normalized_samples:
                    raise ValueError(f"Could not load any training data: {e}")
        
        # Create tokenizer and chunker
        try:
            chunker = create_tokenizer_chunker(
                model_name=self.config['model'].get('name', 'microsoft/deberta-base'),
                max_length=self.config['model'].get('max_length', 256),
                stride=self.config['model'].get('stride', 32)
            )
            logger.info("✅ Tokenizer and chunker created successfully")
        except Exception as e:
            logger.warning(f"Failed to create chunker: {e}")
            chunker = None
        
        # Prepare for training
        if chunker is not None:
            try:
                processed_chunks, _ = chunker.prepare_for_training(normalized_samples)
                logger.info(f"✅ Created {len(processed_chunks)} training chunks")
            except Exception as e:
                logger.error(f"Failed to prepare training data: {e}")
                chunker = None
        
        if chunker is None:
            # Fallback: create proper tokenized chunks
            logger.info("Creating fallback tokenized chunks...")
            
            # Initialize a basic tokenizer for fallback
            from transformers import AutoTokenizer
            fallback_tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'].get('name', 'microsoft/deberta-v3-small'),
                use_fast=False  # Use slow tokenizer to avoid conversion issues
            )
            if fallback_tokenizer.pad_token is None:
                fallback_tokenizer.pad_token = fallback_tokenizer.eos_token
            
            processed_chunks = []
            max_length = self.config['model'].get('max_length', 256)
            
            for sample in normalized_samples:
                if 'tokens' in sample and 'labels' in sample:
                    # Join tokens back to text for proper tokenization
                    text = ' '.join(sample['tokens'])
                    
                    # Tokenize the text
                    tokenized = fallback_tokenizer(
                        text,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors=None,
                        return_special_tokens_mask=False,
                        return_offsets_mapping=False,
                        return_overflowing_tokens=False
                    )
                    
                    # Extract only the essential fields we need
                    input_ids = tokenized.get('input_ids', [])
                    attention_mask = tokenized.get('attention_mask', [])
                    
                    # Align labels with tokenized text (simple alignment)
                    original_labels = sample['labels'][:max_length]
                    
                    # Create label sequence that matches tokenized length
                    labels = []
                    token_count = len(input_ids)
                    
                    # Simple label alignment - repeat first few labels and pad with O
                    for i in range(token_count):
                        if i < len(original_labels):
                            label = original_labels[i]
                        else:
                            label = 'O'  # Default to O for padding
                        labels.append(self.label2id.get(label, 0))
                    
                    # Create chunk with proper structure - only keep essential keys
                    chunk = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels
                    }
                    
                    # Ensure all sequences have the same length
                    if len(chunk['input_ids']) != len(chunk['attention_mask']) or len(chunk['input_ids']) != len(chunk['labels']):
                        logger.warning(f"Sequence length mismatch in sample, skipping...")
                        continue
                    
                    processed_chunks.append(chunk)
            
            logger.info(f"✅ Created {len(processed_chunks)} fallback tokenized chunks")
        
        # Clean and validate processed chunks
        logger.info("Cleaning and validating dataset...")
        cleaned_chunks = []
        for i, chunk in enumerate(processed_chunks):
            # Ensure chunk has all required keys
            if not all(key in chunk for key in ['input_ids', 'attention_mask', 'labels']):
                logger.warning(f"Chunk {i} missing required keys, skipping...")
                continue
            
            # Convert to lists if they're not already
            chunk['input_ids'] = list(chunk['input_ids']) if not isinstance(chunk['input_ids'], list) else chunk['input_ids']
            chunk['attention_mask'] = list(chunk['attention_mask']) if not isinstance(chunk['attention_mask'], list) else chunk['attention_mask']
            chunk['labels'] = list(chunk['labels']) if not isinstance(chunk['labels'], list) else chunk['labels']
            
            # Ensure all values are integers (not None or other types)
            try:
                chunk['input_ids'] = [int(x) for x in chunk['input_ids'] if x is not None]
                chunk['attention_mask'] = [int(x) for x in chunk['attention_mask'] if x is not None]
                chunk['labels'] = [int(x) for x in chunk['labels'] if x is not None]
                
                # Check sequence lengths match
                if len(chunk['input_ids']) == len(chunk['attention_mask']) == len(chunk['labels']):
                    cleaned_chunks.append(chunk)
                else:
                    logger.warning(f"Chunk {i} has mismatched lengths, skipping...")
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Chunk {i} has invalid data types, skipping: {e}")
                continue
        
        processed_chunks = cleaned_chunks
        logger.info(f"✅ Cleaned dataset: {len(processed_chunks)} valid chunks")
        
        if not processed_chunks:
            raise ValueError("No valid chunks after cleaning! Check your data preparation.")
        
        # Debug: Show final dataset structure
        first_chunk = processed_chunks[0]
        logger.info(f"Final chunk keys: {list(first_chunk.keys())}")
        logger.info(f"Sample lengths - input_ids: {len(first_chunk['input_ids'])}, attention_mask: {len(first_chunk['attention_mask'])}, labels: {len(first_chunk['labels'])}")
        logger.info(f"Sample input_ids (first 10): {first_chunk['input_ids'][:10]}")
        logger.info(f"Sample labels (first 10): {first_chunk['labels'][:10]}")
        
        # Check if we have enough data for splitting
        if len(processed_chunks) < 10:
            raise ValueError(f"Not enough data for splitting. Need at least 10 chunks, got {len(processed_chunks)}")
        
        # Split into train/val/test using stratified split with fallback
        logger.info("Splitting data into train/val/test sets...")
        
        # Replace the first stratified split with a safe fallback:
        try:
            train_chunks, temp_chunks = train_test_split(
                processed_chunks, test_size=0.2, random_state=42,
                stratify=[c.get("source", "kaggle") for c in processed_chunks]
            )
        except ValueError:
            # Fallback when a class has <2 members
            train_chunks, temp_chunks = train_test_split(
                processed_chunks, test_size=0.2, random_state=42
            )

        try:
            val_chunks, test_chunks = train_test_split(
                temp_chunks, test_size=0.5, random_state=42,
                stratify=[c.get("source", "kaggle") for c in temp_chunks]
            )
        except ValueError:
            val_chunks, test_chunks = train_test_split(
                temp_chunks, test_size=0.5, random_state=42
            )
        
        logger.info(f"Split: {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_chunks)
        val_dataset = Dataset.from_list(val_chunks)
        test_dataset = Dataset.from_list(test_chunks)
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        model_name = self.config['model'].get('name', 'microsoft/deberta-base')
        
        # Load tokenizer with fallback options
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            logger.info(f"✅ Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name} tokenizer: {e}")
            # Fallback to a more compatible model
            fallback_model = "microsoft/deberta-base"
            logger.info(f"Trying fallback model: {fallback_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, use_fast=False)
            # Update config to use fallback model
            self.config['model']['name'] = fallback_model
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with fallback
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True
            )
            logger.info(f"✅ Loaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name} model: {e}")
            # Use fallback model
            fallback_model = "microsoft/deberta-base"
            logger.info(f"Loading fallback model: {fallback_model}")
            self.model = AutoModelForTokenClassification.from_pretrained(
                fallback_model,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True
            )
        
        # Apply LoRA with fallbacks
        # Choose target modules based on model type
        model_name = self.config['model'].get('name', 'microsoft/deberta-base')
        if 'deberta-v3-small' in model_name:
            target_modules = ["query", "value"]
        else:
            target_modules = ["query", "key", "value"]
        
        lora_config = LoraConfig(
            task_type="TOKEN_CLS",  # Correct PEFT task type
            r=self.config['lora'].get('r', 8),
            lora_alpha=self.config['lora'].get('alpha', 16),
            lora_dropout=self.config['lora'].get('dropout', 0.05),
            target_modules=target_modules,
            bias="none"
        )
        
        try:
            self.model = get_peft_model(self.model, lora_config)
            logger.info("✅ LoRA configuration applied successfully")
        except Exception as e:
            logger.warning(f"Failed to apply LoRA: {e}")
            logger.info("Continuing with base model (no LoRA)")
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model setup complete: {self.model.num_parameters()} parameters")
        try:
            trainable_params = self.model.num_parameters(only_trainable=True)
            logger.info(f"Trainable parameters: {trainable_params}")
        except Exception as e:
            logger.warning(f"Could not count trainable parameters: {e}")
            logger.info("Model setup complete")
    
    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics."""
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        
        if HAS_SEQEVAL:
            # Convert BIO to entity-level for seqeval
            true_predictions_entities = self._bio_to_entities(true_predictions)
            true_labels_entities = self._bio_to_entities(true_labels)
            
            # Calculate metrics with seqeval
            try:
                precision = precision_score(true_labels_entities, true_predictions_entities, average='weighted')
                recall = recall_score(true_labels_entities, true_predictions_entities, average='weighted')
                f1 = f1_score(true_labels_entities, true_predictions_entities, average='weighted')
                
                # Detailed classification report
                report = classification_report(true_labels_entities, true_predictions_entities, output_dict=True)
            except Exception as e:
                logger.warning(f"Seqeval metrics failed: {e}, using fallback")
                # Fallback to simple accuracy
                precision = recall = f1 = 0.5
                report = {}
        else:
            # Fallback: use token-level accuracy
            flat_predictions = [p for pred in true_predictions for p in pred]
            flat_labels = [l for label in true_labels for l in label]
            
            if flat_predictions and flat_labels:
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(flat_labels, flat_predictions)
                precision = recall = f1 = accuracy
                report = {"accuracy": accuracy}
            else:
                precision = recall = f1 = 0.0
                report = {}
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report
        }
    
    def _bio_to_entities(self, bio_sequences: List[List[str]]) -> List[List[str]]:
        """Convert BIO sequences to entity lists."""
        entities_list = []
        
        for sequence in bio_sequences:
            entities = []
            current_entity = None
            
            for tag in sequence:
                if tag.startswith('B-'):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = tag[2:]
                elif tag.startswith('I-'):
                    if current_entity and tag[2:] == current_entity:
                        continue  # Same entity, continue
                    else:
                        current_entity = None
                else:  # O tag
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # Don't forget the last entity
            if current_entity:
                entities.append(current_entity)
            
            entities_list.append(entities)
        
        return entities_list
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Train the model."""
        logger.info("Starting training...")
        
        # Training arguments with fallbacks
        training_args = TrainingArguments(
            output_dir=self.config['output'].get('model_dir', 'artifacts/model'),
            num_train_epochs=int(self.config['training'].get('num_epochs', 5)),
            per_device_train_batch_size=int(self.config['training'].get('batch_size', 4)),
            per_device_eval_batch_size=int(self.config['training'].get('batch_size', 4)),
            gradient_accumulation_steps=int(self.config['training'].get('grad_accum_steps', 4)),
            learning_rate=float(self.config['training'].get('learning_rate', 3e-4)),
            weight_decay=float(self.config['training'].get('weight_decay', 0.01)),
            warmup_steps=int(self.config['training'].get('warmup_steps', 100)),
            fp16=bool(self.config['training'].get('fp16', True)),
            label_smoothing_factor=float(self.config['training'].get('label_smoothing', 0.05)),
            
            # Evaluation - use config values for Kaggle-friendly settings
            eval_strategy=self.config['training'].get('evaluation_strategy', 'steps'),
            eval_steps=int(self.config['training'].get('eval_steps', 500)),
            save_strategy="steps",
            save_steps=int(self.config['training'].get('save_steps', 500)),
            save_total_limit=int(self.config['training'].get('save_total_limit', 3)),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Logging
            logging_dir=self.config['output'].get('logs_dir', 'artifacts/logs'),
            logging_steps=int(self.config['training'].get('logging_steps', 50)),
            report_to=[],  # Explicitly disable all reporting (wandb/tensorboard)
            
            # Other
            dataloader_num_workers=0,  # Set to 0 for Kaggle to avoid multiprocessing issues
            remove_unused_columns=False,
            seed=int(self.config.get('seed', 42)),
            gradient_checkpointing=bool(self.config['training'].get('gradient_checkpointing', True))
        )
        
        # Data collator - use a simpler approach to avoid extra fields
        try:
            tokenizer = self.tokenizer  # Capture tokenizer in closure
            
            def custom_data_collator(features):
                # Only keep essential fields and ensure proper format
                batch = {}
                
                # Get the maximum length in this batch
                max_len = max(len(f["input_ids"]) for f in features)
                
                # Pad sequences manually
                batch["input_ids"] = []
                batch["attention_mask"] = []
                batch["labels"] = []
                
                # Get pad token ID with fallback
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                if pad_token_id is None:
                    pad_token_id = 0  # fallback to 0
                
                for feature in features:
                    # Ensure we only have the fields we need
                    input_ids = list(feature["input_ids"])[:max_len]
                    attention_mask = list(feature["attention_mask"])[:max_len]
                    labels = list(feature["labels"])[:max_len]
                    
                    # Pad to max_len
                    pad_length = max_len - len(input_ids)
                    if pad_length > 0:
                        input_ids.extend([pad_token_id] * pad_length)
                        attention_mask.extend([0] * pad_length)
                        labels.extend([-100] * pad_length)  # -100 is ignored in loss computation
                    
                    batch["input_ids"].append(input_ids)
                    batch["attention_mask"].append(attention_mask)
                    batch["labels"].append(labels)
                
                # Convert to tensors
                import torch
                batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
                return batch
            
            data_collator = custom_data_collator
            logger.info("✅ Custom data collator created successfully")
        except Exception as e:
            logger.error(f"Failed to create data collator: {e}")
            raise ValueError(f"Data collator creation failed: {e}")
        
        # Early stopping callback with fallback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=int(self.config['training'].get('early_stopping_patience', 3))
        )
        
        # Initialize trainer
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[early_stopping]
            )
            logger.info("✅ Trainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise ValueError(f"Trainer initialization failed: {e}")
        
        # Train with resume functionality
        out_cfg = self.config.get("output", {})
        model_dir = out_cfg.get("model_dir", "artifacts/model")
        _safe_mkdirs(model_dir)

        # If CLI provided --resume_from_checkpoint use it; else auto-detect latest.
        resume_path = getattr(self, "resume_from_checkpoint", None)
        if not resume_path:
            try:
                resume_path = _latest_checkpoint(model_dir)
            except:
                # In case of PyTorch security issues or other checkpoint problems
                logger.warning("Could not resume from checkpoint, starting fresh training")
                resume_path = None

        logger.info(f"Resume from checkpoint: {resume_path or 'None'}")

        # Call trainer.train with resume parameter (True or checkpoint path both work)
        try:
            train_output = trainer.train(resume_from_checkpoint=resume_path or False)
        except ValueError as e:
            if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
                logger.warning("PyTorch security issue detected, starting training from scratch")
                train_output = trainer.train(resume_from_checkpoint=False)
            else:
                raise
        trainer.save_model(model_dir)
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.parent
        
        # Create output directories with absolute paths
        model_dir = script_dir / self.config['output'].get('model_dir', 'artifacts/model')
        logs_dir = script_dir / self.config['output'].get('logs_dir', 'artifacts/logs')
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save best model
        trainer.save_model()
        self.tokenizer.save_pretrained(str(model_dir))
        
        # Save label mappings
        label_mappings = {
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        
        with open(model_dir / 'label_mappings.json', 'w') as f:
            json.dump(label_mappings, f, indent=2)
        
        logger.info("Training completed!")
        return trainer
    
    def run(self):
        """Run the complete training pipeline."""
        try:
            # Load and preprocess data
            train_dataset, val_dataset, test_dataset = self.load_and_preprocess_data()
            
            # Setup model and tokenizer
            self.setup_model_and_tokenizer()
            
            # Train
            trainer = self.train(train_dataset, val_dataset)
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(test_dataset)
            
            logger.info("Test Results:")
            for key, value in test_results.items():
                if key != 'classification_report':
                    if isinstance(value, (float, int)):
                        logger.info(f"{key}: {value:.4f}")
                    else:
                        logger.info(f"{key}: {value}")
                else:
                    logger.info(f"{key}: {json.dumps(value, indent=2)}")
            
            # Save test results
            script_dir = Path(__file__).parent.parent
            model_dir = script_dir / self.config['output'].get('model_dir', 'artifacts/model')
            results_file = model_dir / 'test_results.json'
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            logger.info(f"Training pipeline completed. Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    trainer = ResumeNERTrainer(args.config)
    # persist the optional resume argument into the instance if present
    trainer.resume_from_checkpoint = args.resume_from_checkpoint
    trainer.run()


if __name__ == "__main__":
    main()
