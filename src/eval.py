"""
Evaluation script for resume NER model.
Computes seqeval metrics and provides detailed per-label analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
from seqeval.metrics import (
    precision_score, recall_score, f1_score, classification_report,
    accuracy_score, performance_measure
)
from datasets import Dataset
import pandas as pd

from data_io import load_all_datasets
from label_space import process_label_space
from chunk_align import create_tokenizer_chunker

logger = logging.getLogger(__name__)


class ResumeNEREvaluator:
    """Evaluator for resume NER model."""
    
    def __init__(self, model_dir: str = "artifacts/model"):
        self.model_dir = Path(model_dir)
        self.setup_logging()
        self.setup_device()
        
        # Load model components
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        
        logger.info("Resume NER Evaluator initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_device(self):
        """Setup device (GPU/CPU)."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_dir}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Load base model
        base_model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        
        # Load label mappings
        label_mappings_file = self.model_dir / 'label_mappings.json'
        if label_mappings_file.exists():
            with open(label_mappings_file, 'r') as f:
                mappings = json.load(f)
                self.label2id = mappings['label2id']
                self.id2label = mappings['id2label']
        else:
            # Fallback to model config
            self.label2id = base_model.config.label2id
            self.id2label = base_model.config.id2label
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded with {len(self.label2id)} labels")
        logger.info(f"Labels: {list(self.label2id.keys())}")
    
    def load_test_data(self) -> Dataset:
        """Load test dataset."""
        logger.info("Loading test data...")
        
        # Load all datasets
        samples = load_all_datasets()
        
        if not samples:
            raise ValueError("No samples loaded from datasets")
        
        # Process label space
        label_manager, normalized_samples = process_label_space(samples)
        
        # Create tokenizer and chunker
        chunker = create_tokenizer_chunker(
            model_name="microsoft/deberta-v3-small",
            max_length=256,
            stride=32
        )
        
        # Prepare for evaluation
        processed_chunks, _ = chunker.prepare_for_training(normalized_samples)
        
        # Use a subset for evaluation (last 20%)
        eval_size = max(1, len(processed_chunks) // 5)
        eval_chunks = processed_chunks[-eval_size:]
        
        logger.info(f"Using {len(eval_chunks)} chunks for evaluation")
        
        return Dataset.from_list(eval_chunks)
    
    def predict(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Run predictions on dataset."""
        logger.info("Running predictions...")
        
        predictions = []
        
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(dataset)} samples")
            
            # Prepare input
            input_ids = torch.tensor([sample['input_ids']]).to(self.device)
            attention_mask = torch.tensor([sample['attention_mask']]).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Get predictions
            pred_ids = torch.argmax(logits, dim=2)[0].cpu().numpy()
            
            # Convert to labels
            pred_labels = [self.id2label.get(pid, 'O') for pid in pred_ids]
            
            # Get true labels
            true_labels = [self.id2label.get(lid, 'O') for lid in sample['labels'] if lid != -100]
            
            # Align predictions with true labels (remove special tokens)
            word_ids = sample['word_ids']
            aligned_preds = []
            aligned_trues = []
            
            for j, word_id in enumerate(word_ids):
                if word_id is not None:  # Not a special token
                    aligned_preds.append(pred_labels[j])
                    aligned_trues.append(true_labels[j])
            
            predictions.append({
                'sample_id': sample.get('sample_id', f'sample_{i}'),
                'source': sample.get('source', 'unknown'),
                'predictions': aligned_preds,
                'true_labels': aligned_trues,
                'text': ' '.join(sample.get('tokens', []))
            })
        
        logger.info(f"Completed predictions for {len(predictions)} samples")
        return predictions
    
    def compute_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        logger.info("Computing metrics...")
        
        # Extract predictions and true labels
        all_preds = []
        all_trues = []
        
        for pred in predictions:
            all_preds.append(pred['predictions'])
            all_trues.append(pred['true_labels'])
        
        # Convert BIO to entity-level for seqeval
        pred_entities = self._bio_to_entities(all_preds)
        true_entities = self._bio_to_entities(all_trues)
        
        # Overall metrics
        precision = precision_score(true_entities, pred_entities, average='weighted')
        recall = recall_score(true_entities, pred_entities, average='weighted')
        f1 = f1_score(true_entities, pred_entities, average='weighted')
        accuracy = accuracy_score(true_entities, pred_entities)
        
        # Detailed classification report
        class_report = classification_report(true_entities, pred_entities, output_dict=True)
        
        # Performance measures (confusion matrix style)
        perf_measures = performance_measure(true_entities, pred_entities)
        
        # Per-label metrics
        per_label_metrics = self._compute_per_label_metrics(all_preds, all_trues)
        
        # Error analysis
        error_analysis = self._analyze_errors(predictions)
        
        metrics = {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            },
            'classification_report': class_report,
            'performance_measures': perf_measures,
            'per_label': per_label_metrics,
            'error_analysis': error_analysis
        }
        
        return metrics
    
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
    
    def _compute_per_label_metrics(self, predictions: List[List[str]], 
                                 true_labels: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each label."""
        per_label = {}
        
        # Get all unique labels
        all_labels = set()
        for seq in predictions + true_labels:
            for label in seq:
                if label != 'O':
                    all_labels.add(label)
        
        for label in all_labels:
            # Convert to binary classification for this label
            binary_preds = []
            binary_trues = []
            
            for pred_seq, true_seq in zip(predictions, true_labels):
                for pred, true in zip(pred_seq, true_seq):
                    binary_preds.append(1 if pred == label else 0)
                    binary_trues.append(1 if true == label else 0)
            
            # Calculate metrics
            tp = sum(1 for p, t in zip(binary_preds, binary_trues) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(binary_preds, binary_trues) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(binary_preds, binary_trues) if p == 0 and t == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_label[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        
        return per_label
    
    def _analyze_errors(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction errors."""
        error_counts = {}
        confusion_matrix = {}
        
        for pred in predictions:
            pred_labels = pred['predictions']
            true_labels = pred['true_labels']
            
            for pred_label, true_label in zip(pred_labels, true_labels):
                if pred_label != true_label:
                    # Count error types
                    error_key = f"{true_label}->{pred_label}"
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
                    
                    # Build confusion matrix
                    if true_label not in confusion_matrix:
                        confusion_matrix[true_label] = {}
                    if pred_label not in confusion_matrix[true_label]:
                        confusion_matrix[true_label][pred_label] = 0
                    confusion_matrix[true_label][pred_label] += 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'error_counts': dict(sorted_errors[:20]),  # Top 20 errors
            'confusion_matrix': confusion_matrix,
            'total_errors': sum(error_counts.values())
        }
    
    def save_results(self, metrics: Dict[str, Any], output_file: str = "artifacts/eval_results.json"):
        """Save evaluation results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        overall = metrics['overall']
        
        print("\n" + "="*50)
        print("RESUME NER EVALUATION RESULTS")
        print("="*50)
        
        print(f"\nOverall Metrics:")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1:        {overall['f1']:.4f}")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        
        print(f"\nPer-Label Metrics:")
        per_label = metrics['per_label']
        for label, scores in sorted(per_label.items()):
            print(f"  {label:15} - P: {scores['precision']:.3f}, R: {scores['recall']:.3f}, F1: {scores['f1']:.3f}")
        
        print(f"\nTop Error Patterns:")
        error_analysis = metrics['error_analysis']
        for error, count in list(error_analysis['error_counts'].items())[:10]:
            print(f"  {error}: {count}")
        
        print(f"\nTotal Errors: {error_analysis['total_errors']}")
        print("="*50)
    
    def run(self):
        """Run complete evaluation pipeline."""
        try:
            # Load model
            self.load_model()
            
            # Load test data
            test_dataset = self.load_test_data()
            
            # Run predictions
            predictions = self.predict(test_dataset)
            
            # Compute metrics
            metrics = self.compute_metrics(predictions)
            
            # Print summary
            self.print_summary(metrics)
            
            # Save results
            self.save_results(metrics)
            
            logger.info("Evaluation completed successfully!")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate resume NER model")
    parser.add_argument("--model_dir", type=str, default="artifacts/model", help="Path to trained model")
    parser.add_argument("--output", type=str, default="artifacts/eval_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = ResumeNEREvaluator(args.model_dir)
    evaluator.run()


if __name__ == "__main__":
    main()
