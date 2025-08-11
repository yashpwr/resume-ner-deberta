"""
FastAPI API for resume NER predictions.
Supports both PyTorch and ONNX inference modes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Try to import ONNX runtime for faster inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Import PyTorch components
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from postprocess import create_post_processor

logger = logging.getLogger(__name__)


# Pydantic models for API
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for NER entities", min_length=1)
    use_onnx: bool = Field(False, description="Use ONNX runtime for inference if available")


class PredictionResponse(BaseModel):
    entities: List[Dict[str, Any]] = Field(..., description="List of detected entities")
    entity_counts: Dict[str, int] = Field(..., description="Count of entities by type")
    normalized: Dict[str, List[str]] = Field(..., description="Normalized entity values")
    total_entities: int = Field(..., description="Total number of entities found")
    text: str = Field(..., description="Input text")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_type: str = Field(..., description="Model type used (pytorch/onnx)")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    onnx_available: bool = Field(..., description="Whether ONNX runtime is available")
    torch_available: bool = Field(..., description="Whether PyTorch is available")


class ResumeNERAPI:
    """FastAPI application for resume NER."""
    
    def __init__(self, model_dir: str = "artifacts/model"):
        self.model_dir = Path(model_dir)
        self.setup_logging()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.post_processor = create_post_processor()
        
        # ONNX session
        self.onnx_session = None
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Resume NER API",
            description="API for extracting named entities from resume text using DeBERTa-v3-small",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self.setup_routes()
        
        logger.info("Resume NER API initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None or self.onnx_session is not None,
                onnx_available=ONNX_AVAILABLE,
                torch_available=TORCH_AVAILABLE
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_entities(request: PredictionRequest):
            """Predict entities from text."""
            try:
                start_time = time.time()
                
                # Check if model is loaded
                if not self.model and not self.onnx_session:
                    await self.load_model()
                
                # Run prediction
                if request.use_onnx and self.onnx_session:
                    results = await self.predict_onnx(request.text)
                    model_type = "onnx"
                else:
                    results = await self.predict_pytorch(request.text)
                    model_type = "pytorch"
                
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Add metadata
                results['processing_time_ms'] = round(processing_time, 2)
                results['model_type'] = model_type
                
                return PredictionResponse(**results)
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/batch")
        async def predict_batch(texts: List[str]):
            """Predict entities from multiple texts (batch processing)."""
            try:
                if not self.model and not self.onnx_session:
                    await self.load_model()
                
                results = []
                for text in texts:
                    try:
                        if self.onnx_session:
                            result = await self.predict_onnx(text)
                        else:
                            result = await self.predict_pytorch(text)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'error': str(e),
                            'text': text,
                            'entities': [],
                            'entity_counts': {},
                            'normalized': {},
                            'total_entities': 0
                        })
                
                return {"results": results}
                
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_dir}")
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Try to load ONNX model first
        onnx_path = self.model_dir / "model.onnx"
        if onnx_path.exists() and ONNX_AVAILABLE:
            await self.load_onnx_model(onnx_path)
        elif TORCH_AVAILABLE:
            await self.load_pytorch_model()
        else:
            raise RuntimeError("Neither ONNX nor PyTorch models are available")
    
    async def load_onnx_model(self, onnx_path: Path):
        """Load ONNX model."""
        try:
            # Create ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            
            self.onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=providers
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Load label mappings
            label_mappings_file = self.model_dir / 'label_mappings.json'
            if label_mappings_file.exists():
                with open(label_mappings_file, 'r') as f:
                    mappings = json.load(f)
                    self.label2id = mappings['label2id']
                    self.id2label = mappings['id2label']
            
            logger.info("ONNX model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.onnx_session = None
            raise
    
    async def load_pytorch_model(self):
        """Load PyTorch model."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Load base model
            base_model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
            
            # Move to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            
            logger.info("PyTorch model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            self.model = None
            raise
    
    async def predict_onnx(self, text: str) -> Dict[str, Any]:
        """Run prediction using ONNX model."""
        if not self.onnx_session:
            raise RuntimeError("ONNX model not loaded")
        
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        if not tokens:
            return self._empty_result(text, tokens)
        
        # Tokenize
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Run ONNX inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        
        outputs = self.onnx_session.run(
            [output_name],
            {input_name: tokenized['input_ids']}
        )
        
        # Process outputs
        logits = outputs[0]
        pred_ids = logits.argmax(axis=2)[0]
        
        # Convert to labels
        pred_labels = self._convert_predictions_to_labels(pred_ids, tokenized)
        
        # Post-process
        results = self.post_processor.postprocess_predictions(tokens, pred_labels)
        results['text'] = text
        results['tokens'] = tokens
        
        return results
    
    async def predict_pytorch(self, text: str) -> Dict[str, Any]:
        """Run prediction using PyTorch model."""
        if not self.model:
            raise RuntimeError("PyTorch model not loaded")
        
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        if not tokens:
            return self._empty_result(text, tokens)
        
        # Tokenize
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get predictions
        pred_ids = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        # Convert to labels
        pred_labels = self._convert_predictions_to_labels(pred_ids, tokenized)
        
        # Post-process
        results = self.post_processor.postprocess_predictions(tokens, pred_labels)
        results['text'] = text
        results['tokens'] = tokens
        
        return results
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess input text into tokens."""
        text = text.strip()
        tokens = text.split()
        return [token for token in tokens if token.strip()]
    
    def _convert_predictions_to_labels(self, pred_ids: List[int], tokenized: Dict) -> List[str]:
        """Convert prediction IDs to labels."""
        pred_labels = []
        word_ids = tokenized.word_ids()
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None:  # Not a special token
                pred_id = pred_ids[i]
                label = self.id2label.get(pred_id, 'O')
                pred_labels.append(label)
        
        return pred_labels
    
    def _empty_result(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Return empty result for empty text."""
        return {
            'entities': [],
            'entity_counts': {},
            'normalized': {},
            'total_entities': 0,
            'text': text,
            'tokens': tokens
        }
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


def create_api(model_dir: str = "artifacts/model") -> ResumeNERAPI:
    """Factory function to create API instance."""
    return ResumeNERAPI(model_dir)


# Global API instance
api_instance = None


def get_app() -> FastAPI:
    """Get FastAPI app instance for uvicorn."""
    global api_instance
    if api_instance is None:
        api_instance = create_api()
    return api_instance.get_app()


if __name__ == "__main__":
    # Create and run API
    api = create_api()
    app = api.get_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
