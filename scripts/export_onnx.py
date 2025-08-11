#!/usr/bin/env python3
"""
ONNX export script for resume NER model.
Exports trained PyTorch model to ONNX format for faster inference.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_dir: str):
    """Load trained model and tokenizer."""
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load base model
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        torch_dtype=torch.float32  # Use float32 for ONNX export
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    # Load label mappings
    label_mappings_file = model_dir / 'label_mappings.json'
    if label_mappings_file.exists():
        with open(label_mappings_file, 'r') as f:
            mappings = json.load(f)
            label2id = mappings['label2id']
            id2label = mappings['id2label']
    else:
        # Fallback to model config
        label2id = base_model.config.label2id
        id2label = base_model.config.id2label
    
    return model, tokenizer, label2id, id2label


def create_dummy_input(tokenizer, max_length: int = 256):
    """Create dummy input for ONNX export."""
    # Create dummy text
    dummy_text = "John Smith works at Google as a Software Engineer"
    tokens = dummy_text.split()
    
    # Tokenize
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    return tokenized


def export_to_onnx(model, tokenizer, output_path: str, max_length: int = 256):
    """Export PyTorch model to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = create_dummy_input(tokenizer, max_length)
    
    # Export to ONNX
    logger.info("Exporting model to ONNX...")
    
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        verbose=False
    )
    
    logger.info(f"Model exported to {output_path}")


def validate_onnx_model(onnx_path: str, tokenizer, max_length: int = 256):
    """Validate exported ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
        
        # Test inference
        session = ort.InferenceSession(str(onnx_path))
        
        # Create test input
        test_text = "Jane Doe is a Data Scientist"
        test_tokens = test_text.split()
        
        test_input = tokenizer(
            test_tokens,
            is_split_into_words=True,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Run inference
        outputs = session.run(
            ['logits'],
            {
                'input_ids': test_input['input_ids'],
                'attention_mask': test_input['attention_mask']
            }
        )
        
        logits = outputs[0]
        predictions = np.argmax(logits, axis=2)
        
        logger.info(f"ONNX inference test passed. Output shape: {logits.shape}")
        logger.info(f"Sample predictions: {predictions[0][:10]}")
        
        return True
        
    except ImportError:
        logger.warning("ONNX runtime not available, skipping validation")
        return False
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def main():
    """Main function for ONNX export."""
    parser = argparse.ArgumentParser(description="Export resume NER model to ONNX")
    parser.add_argument("--model_dir", type=str, default="artifacts/model", 
                       help="Path to trained model directory")
    parser.add_argument("--output_dir", type=str, default="artifacts/onnx",
                       help="Output directory for ONNX model")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length for ONNX export")
    parser.add_argument("--validate", action="store_true",
                       help="Validate exported ONNX model")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting ONNX export process...")
    
    try:
        # Load model and tokenizer
        logger.info("Loading trained model...")
        model, tokenizer, label2id, id2label = load_model_and_tokenizer(args.model_dir)
        
        logger.info(f"Model loaded with {len(label2id)} labels")
        
        # Export to ONNX
        output_path = Path(args.output_dir) / "model.onnx"
        export_to_onnx(model, tokenizer, output_path, args.max_length)
        
        # Copy tokenizer and label mappings to ONNX directory
        onnx_dir = Path(args.output_dir)
        onnx_dir.mkdir(exist_ok=True)
        
        # Save tokenizer
        tokenizer.save_pretrained(onnx_dir)
        
        # Save label mappings
        label_mappings = {
            'label2id': label2id,
            'id2label': id2label
        }
        
        with open(onnx_dir / 'label_mappings.json', 'w') as f:
            json.dump(label_mappings, f, indent=2)
        
        # Validate if requested
        if args.validate:
            logger.info("Validating ONNX model...")
            validation_success = validate_onnx_model(output_path, tokenizer, args.max_length)
            
            if validation_success:
                logger.info("ONNX export and validation completed successfully!")
            else:
                logger.warning("ONNX export completed but validation failed")
        else:
            logger.info("ONNX export completed successfully!")
        
        # Save export info
        export_info = {
            'model_dir': str(args.model_dir),
            'max_length': args.max_length,
            'labels': list(label2id.keys()),
            'export_timestamp': str(torch.tensor(0).device)  # Placeholder
        }
        
        with open(onnx_dir / 'export_info.json', 'w') as f:
            json.dump(export_info, f, indent=2)
        
        logger.info(f"ONNX model and artifacts saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
