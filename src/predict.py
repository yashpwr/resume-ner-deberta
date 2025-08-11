"""
CLI prediction script for resume NER model.
Supports text input and file input with JSON output.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

from postprocess import create_post_processor

logger = logging.getLogger(__name__)


class ResumeNERPredictor:
    """Predictor for resume NER model."""
    
    def __init__(self, model_dir: str = "artifacts/model"):
        self.model_dir = Path(model_dir)
        self.setup_logging()
        self.setup_device()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.post_processor = create_post_processor()
        
        logger.info("Resume NER Predictor initialized")
    
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
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
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
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess input text into tokens."""
        # Basic text cleaning
        text = text.strip()
        
        # Split into tokens (simple word-based tokenization)
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Run prediction on input text."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        if not tokens:
            return {
                'entities': [],
                'entity_counts': {},
                'normalized': {},
                'total_entities': 0,
                'text': text,
                'tokens': tokens
            }
        
        # Tokenize with model tokenizer
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get predictions
        pred_ids = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        # Convert to labels
        pred_labels = []
        word_ids = tokenized.word_ids()
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None:  # Not a special token
                pred_id = pred_ids[i]
                label = self.id2label.get(pred_id, 'O')
                pred_labels.append(label)
        
        # Post-process predictions
        results = self.post_processor.postprocess_predictions(tokens, pred_labels)
        
        # Add input information
        results['text'] = text
        results['tokens'] = tokens
        results['raw_predictions'] = pred_labels
        
        return results
    
    def predict_file(self, file_path: str) -> Dict[str, Any]:
        """Run prediction on text file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        # Run prediction
        results = self.predict(text)
        results['file_path'] = str(file_path)
        
        return results
    
    def format_output(self, results: Dict[str, Any], format_type: str = "json") -> str:
        """Format results for output."""
        if format_type == "json":
            return json.dumps(results, indent=2, ensure_ascii=False, default=str)
        elif format_type == "pretty":
            return self._format_pretty(results)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _format_pretty(self, results: Dict[str, Any]) -> str:
        """Format results in a human-readable format."""
        output = []
        
        # Header
        output.append("=" * 60)
        output.append("RESUME NER PREDICTION RESULTS")
        output.append("=" * 60)
        
        # Input text (truncated if too long)
        text = results.get('text', '')
        if len(text) > 100:
            text = text[:100] + "..."
        output.append(f"\nInput Text: {text}")
        
        # Entity summary
        entity_counts = results.get('entity_counts', {})
        if entity_counts:
            output.append(f"\nEntity Summary:")
            for entity_type, count in sorted(entity_counts.items()):
                output.append(f"  {entity_type}: {count}")
        
        # Detailed entities
        entities = results.get('entities', [])
        if entities:
            output.append(f"\nDetailed Entities:")
            for i, entity in enumerate(entities, 1):
                output.append(f"  {i}. {entity['label']}: '{entity['text']}'")
                if 'confidence' in entity:
                    output.append(f"     Confidence: {entity['confidence']:.3f}")
                
                # Show normalization if available
                normalized_fields = []
                for key, value in entity.items():
                    if key not in ['label', 'text', 'start', 'end', 'tokens', 'confidence']:
                        if value is not None and value != '':
                            normalized_fields.append(f"{key}: {value}")
                
                if normalized_fields:
                    output.append(f"     Normalized: {', '.join(normalized_fields)}")
        
        # Normalized values
        normalized = results.get('normalized', {})
        if normalized:
            output.append(f"\nNormalized Values:")
            for entity_type, values in normalized.items():
                if values:
                    output.append(f"  {entity_type.title()}: {', '.join(values)}")
        
        output.append("\n" + "=" * 60)
        
        return "\n".join(output)
    
    def run_prediction(self, input_text: Optional[str] = None, 
                      input_file: Optional[str] = None,
                      output_format: str = "json") -> str:
        """Run prediction and return formatted output."""
        try:
            # Load model if not already loaded
            if not self.model:
                self.load_model()
            
            # Run prediction
            if input_text:
                results = self.predict(input_text)
            elif input_file:
                results = self.predict_file(input_file)
            else:
                raise ValueError("Either input_text or input_file must be provided")
            
            # Format output
            return self.format_output(results, output_format)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            error_result = {
                'error': str(e),
                'status': 'failed'
            }
            return json.dumps(error_result, indent=2)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Resume NER Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from text
  python -m src.predict --text "John Smith works at Google as a Software Engineer"
  
  # Predict from file
  python -m src.predict --file resume.txt
  
  # Pretty output format
  python -m src.predict --text "Contact me at john@email.com" --format pretty
  
  # Custom model directory
  python -m src.predict --text "..." --model_dir /path/to/model
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Input text for prediction')
    input_group.add_argument('--file', type=str, help='Input file path for prediction')
    
    # Output options
    parser.add_argument('--format', choices=['json', 'pretty'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--model_dir', type=str, default='artifacts/model',
                       help='Path to trained model directory (default: artifacts/model)')
    parser.add_argument('--output', type=str, help='Output file path (default: stdout)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = ResumeNERPredictor(args.model_dir)
    
    # Run prediction
    try:
        output = predictor.run_prediction(
            input_text=args.text,
            input_file=args.file,
            output_format=args.format
        )
        
        # Write output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Results saved to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
