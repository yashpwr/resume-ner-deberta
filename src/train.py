"""
Training script for resume NER using DeBERTa-v3-small with LoRA.
Includes early stopping, metrics tracking, and model saving.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import yaml

# Add the src directory to Python path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, TrainingArguments,
    DataCollatorForTokenClassification, EarlyStoppingCallback, Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import evaluate

# Import from the same directory
from data_io import load_all_datasets
from label_space import process_label_space
from chunk_align import create_tokenizer_chunker

logger = logging.getLogger(__name__)


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
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.parent
        artifacts_dir = script_dir / 'artifacts'
        
        # Create artifacts directory if it doesn't exist
        os.makedirs(artifacts_dir, exist_ok=True)
        
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(artifacts_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_device(self):
        """Setup device (GPU/CPU)."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_and_preprocess_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and preprocess all datasets."""
        logger.info("Loading datasets...")
        
        # Load all datasets
        samples = load_all_datasets()
        
        if not samples:
            raise ValueError("No samples loaded from datasets")
        
        logger.info(f"Loaded {len(samples)} raw samples")
        
        # Process label space
        label_manager, normalized_samples = process_label_space(samples)
        self.label2id = label_manager.label2id
        self.id2label = label_manager.id2label
        
        logger.info(f"Label space: {len(self.label2id)} labels")
        
        # Create tokenizer and chunker
        chunker = create_tokenizer_chunker(
            model_name=self.config['model']['name'],
            max_length=self.config['model']['max_length'],
            stride=self.config['model']['stride']
        )
        
        # Prepare for training
        processed_chunks, _ = chunker.prepare_for_training(normalized_samples)
        
        logger.info(f"Created {len(processed_chunks)} training chunks")
        
        # Split into train/val/test
        train_chunks, temp_chunks = train_test_split(
            processed_chunks, 
            test_size=0.2, 
            random_state=self.config.get('seed', 42),
            stratify=[chunk.get('source', 'unknown') for chunk in processed_chunks]
        )
        
        val_chunks, test_chunks = train_test_split(
            temp_chunks,
            test_size=0.5,
            random_state=self.config.get('seed', 42),
            stratify=[chunk.get('source', 'unknown') for chunk in temp_chunks]
        )
        
        logger.info(f"Split: {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_chunks)
        val_dataset = Dataset.from_list(val_chunks)
        test_dataset = Dataset.from_list(test_chunks)
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        model_name = self.config['model']['name']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLASSIFICATION,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model setup complete: {self.model.num_parameters()} parameters")
        logger.info(f"Trainable parameters: {self.model.num_parameters(only_trainable=True)}")
    
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
        
        # Convert BIO to entity-level for seqeval
        true_predictions_entities = self._bio_to_entities(true_predictions)
        true_labels_entities = self._bio_to_entities(true_labels)
        
        # Calculate metrics
        precision = precision_score(true_labels_entities, true_predictions_entities, average='weighted')
        recall = recall_score(true_labels_entities, true_predictions_entities, average='weighted')
        f1 = f1_score(true_labels_entities, true_predictions_entities, average='weighted')
        
        # Detailed classification report
        report = classification_report(true_labels_entities, true_predictions_entities, output_dict=True)
        
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
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output']['model_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['grad_accum_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            fp16=self.config['training']['fp16'],
            label_smoothing_factor=self.config['training']['label_smoothing'],
            
            # Evaluation
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Logging
            logging_dir=self.config['output']['logs_dir'],
            logging_steps=10,
            report_to=None,  # Disable wandb/tensorboard for now
            
            # Save
            save_total_limit=3,
            
            # Other
            dataloader_num_workers=4,
            remove_unused_columns=False,
            seed=self.config.get('seed', 42)
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            return_tensors="pt"
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config['training']['early_stopping_patience']
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Train
        trainer.train()
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.parent
        
        # Create output directories with absolute paths
        model_dir = script_dir / self.config['output']['model_dir']
        logs_dir = script_dir / self.config['output']['logs_dir']
        
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
                    logger.info(f"{key}: {value:.4f}")
            
            # Save test results
            results_file = model_dir / 'test_results.json'
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            logger.info(f"Training pipeline completed. Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train resume NER model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = ResumeNERTrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
