# Resume NER Training in Kaggle

## Quick Setup Commands for Kaggle

Run these commands in a Kaggle notebook cell:

### 1. Clone Repository
```bash
!cd /kaggle/working && git clone https://github.com/yashpwr/resume-ner-deberta.git
!cd /kaggle/working/resume-ner-deberta
```

### 2. Install Required Packages
```bash
!pip install seqeval>=1.2.2 peft>=0.6.0 accelerate>=0.24.0 transformers>=4.35.0 datasets>=2.14.0 rich>=13.0.0 pyyaml>=6.0
```

### 3. Run Training
```bash
!cd /kaggle/working/resume-ner-deberta && python -m src.train --config configs/train.yaml
```

## Alternative: Use Setup Script

```python
# Run the setup script
!cd /kaggle/working/resume-ner-deberta && python setup_kaggle.py

# Then run training
!cd /kaggle/working/resume-ner-deberta && python -m src.train --config configs/train.yaml
```

## Expected Output Location

The trained model will be saved to:
```
/kaggle/working/resume-ner-deberta/artifacts/model/
```

Files generated:
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration  
- `tokenizer.json` - Tokenizer files
- `label_mappings.json` - Label mappings
- `test_results.json` - Evaluation results
- `training.log` - Training logs

## Configuration

The training uses BERT-base-uncased model with:
- Max sequence length: 192
- Batch size: 4  
- Learning rate: 3e-4
- LoRA fine-tuning for efficiency
- 5 epochs

## Troubleshooting

If you get package import errors:
1. Install missing packages with `!pip install <package_name>`
2. Restart the notebook kernel
3. Re-run the training command

The script includes fallback mechanisms for missing packages and should work even if some optional dependencies are unavailable.
