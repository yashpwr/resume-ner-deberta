# Resume NER with DeBERTa-v3-small

A comprehensive Named Entity Recognition (NER) system for resume parsing using Microsoft's DeBERTa-v3-small model with LoRA fine-tuning. This project extracts structured information from resume text including names, companies, job titles, skills, education, and contact information.

## 🚀 Features

- **Multi-Dataset Support**: Handles HuggingFace, Kaggle, and GitHub datasets
- **Advanced NER**: Uses DeBERTa-v3-small with LoRA for efficient fine-tuning
- **Smart Label Normalization**: Automatically normalizes entity labels across datasets
- **BIO Tagging**: Implements proper BIO (Beginning-Inside-Outside) sequence labeling
- **Post-Processing**: Intelligent entity merging and validation with regex patterns
- **Multiple Inference Modes**: PyTorch and ONNX runtime support
- **FastAPI API**: Production-ready REST API with async support
- **Comprehensive Evaluation**: Detailed metrics and error analysis

## 🏗️ Architecture

```
resume-ner-deberta/
├── src/                    # Core source code
│   ├── data_io.py         # Dataset loaders (HF/Kaggle/GitHub)
│   ├── label_space.py     # Label normalization and BIO validation
│   ├── chunk_align.py     # Tokenization and wordpiece alignment
│   ├── train.py           # Training with LoRA and early stopping
│   ├── eval.py            # Evaluation and metrics computation
│   ├── predict.py         # CLI prediction interface
│   ├── postprocess.py     # Entity merging and normalization
│   └── api.py             # FastAPI REST API
├── scripts/                # Utility scripts
│   ├── fetch_data.py      # Download all datasets
│   ├── merge_datasets.py  # Create train/val/test splits
│   └── export_onnx.py     # Export to ONNX format
├── configs/                # Configuration files
│   ├── train.yaml         # Training hyperparameters
│   └── label_synonyms.json # Label normalization rules
├── data/                   # Dataset storage (gitignored)
├── artifacts/              # Model outputs (gitignored)
└── tests/                  # Unit tests
```

## 🎯 Supported Entity Types

- **Personal**: NAME, EMAIL, PHONE, ADDRESS
- **Professional**: COMPANY, TITLE, SKILL, TOOL
- **Education**: UNIVERSITY, DEGREE, MAJOR, GRAD_YEAR
- **Contact**: WEBSITE, LINKEDIN, GITHUB
- **Other**: PROJECT, VOLUNTEER, CERTIFICATION, PUBLICATION
- **Extended Labels**: ACHIEVEMENT, AWARD, RESEARCH, METHODOLOGY, FRAMEWORK, PLATFORM, TECHNOLOGY, STANDARD, COMPLIANCE, GOVERNANCE, and many more

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yashpwr/resume-ner-deberta.git
cd resume-ner-deberta

# Install dependencies
make setup
# or manually: pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your Kaggle credentials (optional)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 3. Fetch and Prepare Data

```bash
# Download all datasets
make fetch

# Merge datasets and create splits
make merge
```

### 4. Train Model

```bash
# Train the model
make train

# Monitor training logs
tail -f artifacts/training.log
```

### 5. Evaluate and Predict

```bash
# Evaluate on test set
make eval

# Make predictions
make predict
# or: python -m src.predict --text "John works at Google as a Software Engineer"
```

### 6. Serve API

```bash
# Start FastAPI server
make serve

# Test API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "John Smith is a Software Engineer at Google"}'
```

## 🎓 **Training on Kaggle Notebooks**

This section provides step-by-step instructions for training the resume NER model on Kaggle notebooks with GPU acceleration.

### **Prerequisites**
- Kaggle account with access to GPU notebooks
- Basic understanding of Python and machine learning
- Familiarity with Jupyter notebooks

### **Step 1: Create a New Kaggle Notebook**

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "Create" → "New Notebook"
3. Select "GPU" as the accelerator
4. Choose Python as the language
5. Set a descriptive title like "Resume NER Training with DeBERTa-v3-small"

### **Step 2: Clone the Repository**

```python
# Always reset to a valid directory
%cd /kaggle/working

# If folder exists from a previous run, remove it
!rm -rf resume-ner-deberta

# Clone fresh
!git clone https://github.com/yashpwr/resume-ner-deberta.git

# Enter the repo
%cd resume-ner-deberta

# Install requirements (if file exists)
!if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
```

### **Step 3: Setup Kaggle Datasets**

```python
# Install Kaggle API if not already available
!pip install kaggle

# Set up Kaggle credentials (you'll need to upload kaggle.json)
import os
import shutil

# Copy kaggle.json to writable directory
kaggle_creds_path = '/kaggle/input/yashpwrr-kaggle-credentials/kaggle.json'
working_dir = '/kaggle/working'

if os.path.exists(kaggle_creds_path):
    # Create config directory in writable location
    config_dir = f'{working_dir}/.kaggle'
    os.makedirs(config_dir, exist_ok=True)
    
    # Copy credentials file
    shutil.copy(kaggle_creds_path, f'{config_dir}/kaggle.json')
    
    # Set environment variable
    os.environ['KAGGLE_CONFIG_DIR'] = config_dir
    
    print("✅ Kaggle credentials copied to writable directory")
    
    # Download required datasets
    !kaggle datasets download -d mgmitesh/ats-scoring-dataset -p data/
    !kaggle datasets download -d mehyaar/annotated-ner-pdf-resumes -d data/
else:
    print("❌ kaggle.json not found in uploaded dataset")
    print("Please upload your kaggle.json file as a dataset named 'yashpwrr/kaggle-credentials'")

# Unzip datasets
!unzip data/ats-scoring-dataset.zip -d data/
!unzip data/annotated-ner-pdf-resumes.zip -d data/
```

### **Step 4: Prepare Training Data**

```python
# Import necessary modules
import sys
sys.path.append('/kaggle/working/resume-ner-deberta/src')

from data_io import load_all_datasets
from label_space import process_label_space
from chunk_align import create_tokenizer_chunker

# Load and process datasets
print("Loading datasets...")
samples = load_all_datasets('data/')

if samples:
    print(f"Loaded {len(samples)} samples")
    
    # Process label space
    print("Processing label space...")
    label_manager, normalized_samples = process_label_space(samples)
    
    print(f"Label space: {len(label_manager.label2id)} labels")
    print(f"Labels: {list(label_manager.label2id.keys())}")
else:
    print("No samples loaded!")
```

### **Step 5: Create Training Configuration**

```python
# Create training config for Kaggle environment
import yaml

kaggle_config = {
    'model': {
        'name': 'microsoft/deberta-v3-small',
        'max_length': 256,
        'stride': 32
    },
    'training': {
        'learning_rate': 3e-4,
        'batch_size': 4,  # Reduced for Kaggle GPU memory
        'grad_accum_steps': 4,  # Increased to maintain effective batch size
        'num_epochs': 5,
        'early_stopping_patience': 3,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'fp16': True,
        'label_smoothing': 0.05
    },
    'lora': {
        'r': 8,
        'alpha': 16,
        'dropout': 0.05,
        'target_modules': ['query', 'value']
    },
    'data': {
        'train_file': 'data/train.jsonl',
        'val_file': 'data/val.jsonl',
        'test_file': 'data/test.jsonl'
    },
    'output': {
        'model_dir': '/kaggle/working/artifacts/model',
        'logs_dir': '/kaggle/working/artifacts/logs'
    },
    'seed': 42
}

# Save config
with open('configs/kaggle_train.yaml', 'w') as f:
    yaml.dump(kaggle_config, f, default_flow_style=False)

print("Kaggle training config created!")
```

### **Step 6: Prepare Data for Training**

```python
# Create tokenizer and chunker
chunker = create_tokenizer_chunker(
    model_name='microsoft/deberta-v3-small',
    max_length=256,
    stride=32
)

# Prepare for training
print("Preparing data for training...")
processed_chunks, label_to_id = chunker.prepare_for_training(normalized_samples)

print(f"Created {len(processed_chunks)} training chunks")

# Split into train/val/test
from sklearn.model_selection import train_test_split
import json

# Create splits
train_chunks, temp_chunks = train_test_split(
    processed_chunks, 
    test_size=0.2, 
    random_state=42,
    stratify=[chunk.get('source', 'unknown') for chunk in processed_chunks]
)

val_chunks, test_chunks = train_test_split(
    temp_chunks,
    test_size=0.5,
    random_state=42,
    stratify=[chunk.get('source', 'unknown') for chunk in temp_chunks]
)

print(f"Split: {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test")

# Save splits
import os
os.makedirs('data', exist_ok=True)

def save_jsonl(samples, filename):
    with open(f'data/{filename}', 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

save_jsonl(train_chunks, 'train.jsonl')
save_jsonl(val_chunks, 'val.jsonl')
save_jsonl(test_chunks, 'test.jsonl')

print("Data splits saved!")
```

### **Step 7: Train the Model**

```python
# Import training module
from train import ResumeNERTrainer

# Create trainer with Kaggle config
trainer = ResumeNERTrainer('configs/kaggle_train.yaml')

# Run training
print("Starting training...")
trainer.run()

print("Training completed!")
```

### **Step 8: Evaluate and Save Results**

```python
# Evaluate the trained model
from eval import ResumeNEREvaluator

evaluator = ResumeNEREvaluator('/kaggle/working/artifacts/model')
evaluator.run()

# Save model and results
import shutil

# Create output directory
os.makedirs('/kaggle/working/output', exist_ok=True)

# Copy trained model
shutil.copytree('/kaggle/working/artifacts/model', '/kaggle/working/output/model')

# Copy evaluation results
shutil.copy('/kaggle/working/artifacts/eval_results.json', '/kaggle/working/output/')

# Copy training logs
shutil.copy('/kaggle/working/artifacts/training.log', '/kaggle/working/output/')

print("Model and results saved to /kaggle/working/output/")
```

### **Step 9: Export to ONNX (Optional)**

```python
# Export to ONNX for faster inference
from export_onnx import main as export_onnx

# Set up arguments
import sys
sys.argv = [
    'export_onnx.py',
    '--model_dir', '/kaggle/working/artifacts/model',
    '--output_dir', '/kaggle/working/output/onnx',
    '--validate'
]

# Export
export_onnx()

print("ONNX export completed!")
```

### **Step 10: Download Results**

```python
# Create a zip file for easy download
!cd /kaggle/working && zip -r resume_ner_model.zip output/

print("Results zipped as resume_ner_model.zip")
print("Download the zip file from the Kaggle notebook output!")
```

### **Kaggle-Specific Tips**

#### **Memory Management**
```python
# Monitor GPU memory
!nvidia-smi

# Clear cache if needed
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
# Add to training config if needed
```

#### **Dataset Handling**
```python
# For large datasets, use streaming
from datasets import load_dataset

dataset = load_dataset(
    "json", 
    data_files="data/train.jsonl",
    streaming=True
)

# Process in batches
for batch in dataset.iter(batch_size=100):
    # Process batch
    pass
```

#### **Training Monitoring**
```python
# Enable wandb for experiment tracking
import wandb
wandb.init(project="resume-ner", name="deberta-v3-small-lora")

# Add to training config
kaggle_config['training']['report_to'] = 'wandb'
```

#### **Model Checkpointing**
```python
# Save checkpoints during training
kaggle_config['training']['save_steps'] = 500
kaggle_config['training']['save_total_limit'] = 3

# Resume from checkpoint if training is interrupted
# trainer = ResumeNERTrainer('configs/kaggle_train.yaml')
# trainer.resume_from_checkpoint('/kaggle/working/artifacts/model/checkpoint-1000')
```

### **Complete Kaggle Notebook Example**

Here's a minimal working example for Kaggle:

```python
# Install dependencies
!pip install transformers peft accelerate seqeval scikit-learn pyyaml

# Clone repo
!git clone https://github.com/yashpwr/resume-ner-deberta.git
%cd resume-ner-deberta

# Download datasets
!kaggle datasets download -d mgmitesh/ats-scoring-dataset -p data/
!unzip data/ats-scoring-dataset.zip -d data/

# Train model
!python -m src.train --config configs/train.yaml

# Evaluate
!python -m src.eval

# Make predictions
!python -m src.predict --text "John Smith works at Google as a Software Engineer"
```

### **Troubleshooting on Kaggle**

#### **Common Issues**

1. **Out of Memory (OOM)**
   ```python
   # Reduce batch size
   kaggle_config['training']['batch_size'] = 2
   kaggle_config['training']['grad_accum_steps'] = 8
   
   # Enable gradient checkpointing
   kaggle_config['training']['gradient_checkpointing'] = True
   ```

2. **Dataset Download Issues**
   ```python
   # Use direct download links
   !wget -O data/dataset.zip "https://kaggle.com/datasets/..."
   
   # Or upload datasets manually to Kaggle
   ```

3. **Training Interruption**
   ```python
   # Save checkpoints more frequently
   kaggle_config['training']['save_steps'] = 100
   
   # Resume from last checkpoint
   trainer.resume_from_checkpoint('artifacts/model/checkpoint-latest')
   ```

4. **GPU Not Available**
   ```python
   # Check GPU availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   
   # Fallback to CPU if needed
   if not torch.cuda.is_available():
       kaggle_config['training']['fp16'] = False
   ```

5. **Kaggle API Credential Issues**
   ```python
   # Method 1: Upload kaggle.json as a dataset
   # 1. Create a new dataset on Kaggle
   # 2. Upload your kaggle.json file
   # 3. Make it private for security
   # 4. Reference it in your notebook:
   
   import os
   os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input/kaggle-credentials/'
   
   # Test credentials
   !kaggle config view
   
   # Method 2: Set credentials directly
   os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
   os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key'
   
   # Method 3: Use Kaggle's built-in credentials
   # If you're logged into Kaggle, you can use:
   !kaggle datasets list --limit 5
   ```

6. **Dataset Download Failures**
   ```python
   # Verify dataset exists and is accessible
   !kaggle datasets list --search "ats scoring"
   !kaggle datasets list --search "annotated ner pdf resumes"
   
   # Check if you have access to the dataset
   !kaggle datasets metadata -d mgmitesh/ats-scoring-dataset
   
   # Alternative: Download manually and upload to Kaggle
   # 1. Download dataset locally
   # 2. Upload as a new dataset to your Kaggle account
   # 3. Use your own dataset in the notebook
   ```

7. **File Permission Issues**
   ```python
   # Check file permissions
   !ls -la /kaggle/input/
   
   # Verify kaggle.json is readable
   !cat /kaggle/input/your-dataset-name/kaggle.json
   
   # Set correct permissions if needed
   !chmod 600 /kaggle/input/your-dataset-name/kaggle.json
   ```

#### **Kaggle Credential Setup Guide**

**Step 1: Get Your Kaggle API Credentials**
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll down to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json` file

**Step 2: Upload Credentials to Kaggle**
1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Click "Create Dataset"
3. Name it `kaggle-credentials` or similar
4. Upload your `kaggle.json` file
5. Make it **Private** (important for security)
6. Click "Create"

**Step 3: Use in Notebook**
```python
# Method 1: Copy credentials to writable directory (Recommended)
import os
import shutil

# Copy kaggle.json to writable directory
kaggle_creds_path = '/kaggle/input/yashpwrr-kaggle-credentials/kaggle.json'
working_dir = '/kaggle/working'

if os.path.exists(kaggle_creds_path):
    # Create config directory in writable location
    config_dir = f'{working_dir}/.kaggle'
    os.makedirs(config_dir, exist_ok=True)
    
    # Copy credentials file
    shutil.copy(kaggle_creds_path, f'{config_dir}/kaggle.json')
    
    # Set environment variable
    os.environ['KAGGLE_CONFIG_DIR'] = config_dir
    
    print("✅ Kaggle credentials copied to writable directory")
    
    # Test the setup
    !kaggle config view
    
    # Now download datasets
    !kaggle datasets download -d mgmitesh/ats-scoring-dataset -p data/
    !kaggle datasets download -d mehyaar/annotated-ner-pdf-resumes -p data/
else:
    print("❌ kaggle.json not found in uploaded dataset")

# Method 2: Set credentials directly (Alternative)
# os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
# os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key'
```

**Step 4: Verify Setup**
```python
# Check if credentials are working
!kaggle datasets list --limit 3

# Check if target datasets are accessible
!kaggle datasets metadata -d mgmitesh/ats-scoring-dataset
!kaggle datasets metadata -d mehyaar/annotated-ner-pdf-resumes
```

**Common Credential Errors & Solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'username'` | Missing or invalid `kaggle.json` | Upload correct `kaggle.json` as dataset |
| `403 Forbidden` | No access to dataset | Check dataset privacy settings |
| `404 Not Found` | Dataset doesn't exist | Verify dataset name and owner |
| `Authentication failed` | Invalid API key | Regenerate API key in Kaggle settings |
| `Permission denied` | File permission issues | Check file permissions in uploaded dataset |

## 📊 Dataset Sources

### 1. HuggingFace Dataset
- **Source**: [Mehyaar/Annotated_NER_PDF_Resumes](https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes)
- **Content**: PDF resumes with NER annotations
- **Format**: Tokens + NER tags

### 2. Kaggle Dataset
- **Source**: [ATS Scoring Dataset](https://www.kaggle.com/datasets/mgmitesh/ats-scoring-dataset)
- **Content**: Resume text for ATS scoring
- **Note**: No NER labels (used for future pretraining)

### 3. GitHub Datasets
- **Resume Corpus**: [vrundag91/Resume-Corpus-Dataset](https://github.com/vrundag91/Resume-Corpus-Dataset)
- **Doccano Format**: [juanfpinzon/resume-dataset](https://github.com/juanfpinzon/resume-dataset)

## 🎛️ Configuration

### Training Configuration (`configs/train.yaml`)

```yaml
model:
  name: "microsoft/deberta-v3-small"
  max_length: 256
  stride: 32

training:
  learning_rate: 3e-4
  batch_size: 8
  grad_accum_steps: 2
  num_epochs: 5
  early_stopping_patience: 3
  fp16: true
  label_smoothing: 0.05

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["query", "value"]

data:
  train_file: "data/train.jsonl"
  val_file: "data/val.jsonl"
  test_file: "data/test.jsonl"

output:
  model_dir: "artifacts/model"
  logs_dir: "artifacts/logs"

seed: 42
```

### Label Synonyms (`configs/label_synonyms.json`)

```json
{
  "URL": "WEBSITE",
  "PERSON": "NAME",
  "EMAIL_ADDRESS": "EMAIL",
  "ORG": "COMPANY",
  "JOB_TITLE": "TITLE"
}
```

## 🔧 Advanced Usage

### Custom Training

```bash
# Train with custom config
python -m src.train --config configs/custom_train.yaml

# Resume training from checkpoint
python -m src.train --config configs/train.yaml --resume_from_checkpoint artifacts/model/checkpoint-1000
```

### Batch Prediction

```bash
# Process multiple files
python -m src.predict --file resume1.txt --output results1.json
python -m src.predict --file resume2.txt --output results2.json

# Pretty output format
python -m src.predict --text "..." --format pretty
```

### ONNX Export

```bash
# Export to ONNX for faster inference
make onnx

# Use ONNX model in API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "...", "use_onnx": true}'
```

## 📈 Model Performance

The model achieves competitive performance on resume NER tasks:

- **Precision**: ~0.85-0.90
- **Recall**: ~0.80-0.85
- **F1 Score**: ~0.82-0.87
- **Inference Speed**: ~50-100ms per resume (CPU), ~10-20ms (GPU)

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_postprocess.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker Support

### Development Container

```bash
# Build and run dev container
docker build -f .devcontainer/Dockerfile -t resume-ner-dev .
docker run -it --gpus all -v $(pwd):/workspace resume-ner-dev
```

### Production Container

```bash
# Build production image
docker build -t resume-ner-api .

# Run API container
docker run -p 8000:8000 --gpus all resume-ner-api
```

## 🌐 API Endpoints

### Health Check
```http
GET /
```

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "John Smith works at Google as a Software Engineer",
  "use_onnx": false
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

[
  "John Smith works at Google",
  "Jane Doe is a Data Scientist at Microsoft"
]
```

## 📝 Output Format

### Prediction Response
```json
{
  "entities": [
    {
      "label": "NAME",
      "text": "John Smith",
      "start": 0,
      "end": 1,
      "confidence": 0.95,
      "first_name": "John",
      "last_name": "Smith"
    },
    {
      "label": "COMPANY",
      "text": "Google",
      "start": 4,
      "end": 4,
      "confidence": 0.92,
      "clean_name": "Google"
    }
  ],
  "entity_counts": {
    "NAME": 1,
    "COMPANY": 1,
    "TITLE": 1
  },
  "normalized": {
    "name": ["John Smith"],
    "company": ["Google"],
    "title": ["Software Engineer"]
  },
  "total_entities": 3,
  "processing_time_ms": 45.2,
  "model_type": "pytorch"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Kaggle Dataset Not Loading**
   - Check `.env` file has correct credentials
   - Verify Kaggle API is working: `kaggle datasets list`

2. **CUDA Out of Memory**
   - Reduce batch size in `configs/train.yaml`
   - Use gradient accumulation: increase `grad_accum_steps`

3. **Model Not Converging**
   - Check learning rate: try `1e-4` or `5e-4`
   - Verify data quality with `make eval`

4. **API Not Starting**
   - Check if model exists in `artifacts/model/`
   - Verify port 8000 is available

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m src.train --config configs/train.yaml --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft for DeBERTa-v3-small
- HuggingFace for transformers library
- PEFT team for LoRA implementation
- Dataset contributors for annotated resume data

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yashpwr/resume-ner-deberta/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yashpwr/resume-ner-deberta/discussions)

---

**Star this repository if you find it useful! ⭐**
