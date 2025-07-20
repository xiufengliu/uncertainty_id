"""
FastAPI server for Uncertainty-Aware Intrusion Detection System.

This module implements a production-ready REST API server with comprehensive
monitoring, logging, and error handling capabilities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime, timedelta
import psutil
import os
from pathlib import Path

from .models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthResponse, MetricsResponse, 
    ModelInfoResponse, ErrorResponse
)
from .monitoring import APIMonitor, PerformanceTracker
from ..models import BayesianEnsembleIDS
from ..data import NetworkDataProcessor

logger = logging.getLogger(__name__)


class UncertaintyIDSAPI:
    """
    Main API class for uncertainty-aware intrusion detection.
    
    Provides real-time inference capabilities with comprehensive
    uncertainty quantification and monitoring.
    """
    
    def __init__(self, model_path: str, processor_path: str, 
                 config: Optional[Dict] = None):
        """
        Initialize the API with model and processor.
        
        Args:
            model_path: Path to trained model checkpoint
            processor_path: Path to data processor
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.model_path = model_path
        self.processor_path = processor_path
        
        # Initialize components
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Monitoring and metrics
        self.monitor = APIMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # Service metadata
        self.start_time = datetime.utcnow()
        self.version = "1.0.0"
        self.model_version = "unknown"
        
        # Load model and processor
        self._load_model()
        self._load_processor()
        
        logger.info(f"UncertaintyIDSAPI initialized with model from {model_path}")
    
    def _load_model(self):
        """Load the trained model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            model_config = checkpoint.get('model_config', {})
            
            # Create model instance
            self.model = BayesianEnsembleIDS(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Extract model version
            self.model_version = checkpoint.get('version', 'unknown')
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _load_processor(self):
        """Load the data processor."""
        try:
            self.processor = NetworkDataProcessor()
            self.processor.load_preprocessors(self.processor_path)
            
            logger.info("Data processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise RuntimeError(f"Processor loading failed: {e}")
    
    async def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make a single prediction with uncertainty quantification.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response with uncertainty estimates
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            current_flow = request.current_flow.to_array()
            
            # Handle historical flows
            if request.historical_flows and request.historical_flows.flows:
                historical_flows = request.historical_flows.to_array()
                
                # Ensure we have the right sequence length
                seq_len = self.model.max_seq_len
                if len(historical_flows) > seq_len:
                    historical_flows = historical_flows[-seq_len:]
                elif len(historical_flows) < seq_len:
                    # Pad with zeros or repeat last flow
                    padding_needed = seq_len - len(historical_flows)
                    if len(historical_flows) > 0:
                        padding = np.tile(historical_flows[-1], (padding_needed, 1))
                    else:
                        padding = np.zeros((padding_needed, current_flow.shape[0]))
                    historical_flows = np.vstack([padding, historical_flows])
            else:
                # Create dummy historical flows
                seq_len = self.model.max_seq_len
                historical_flows = np.tile(current_flow, (seq_len, 1))
            
            # Convert to tensors
            historical_tensor = torch.FloatTensor(historical_flows).unsqueeze(0).to(self.device)
            current_tensor = torch.FloatTensor(current_flow).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                results = self.model.predict_with_uncertainty(
                    historical_tensor, current_tensor
                )
            
            # Extract results
            prediction = results['predictions'].item()
            confidence = results['confidence'].item()
            
            # Prepare response
            response_data = {
                'prediction': prediction,
                'prediction_label': 'attack' if prediction == 1 else 'normal',
                'confidence': confidence,
                'model_version': self.model_version,
                'processing_time_ms': (time.time() - start_time) * 1000,
            }
            
            # Add uncertainty if requested
            if request.return_uncertainty:
                response_data.update({
                    'uncertainty': results['total_uncertainty'].item(),
                    'epistemic_uncertainty': results['epistemic_uncertainty'].item(),
                    'aleatoric_uncertainty': results['aleatoric_uncertainty'].item(),
                })
                
                # Determine if review is required
                uncertainty_threshold = request.uncertainty_threshold or 0.2
                response_data['requires_review'] = results['total_uncertainty'].item() > uncertainty_threshold
            else:
                response_data['requires_review'] = False
            
            # Add probabilities if requested
            if request.return_probabilities:
                probs = results['probabilities'].squeeze().cpu().numpy()
                response_data['probabilities'] = {
                    'normal': float(probs[0]),
                    'attack': float(probs[1])
                }
            
            # Add explanation if requested
            if request.return_explanation:
                # Simple feature importance based on attention weights
                if 'attention_weights' in results:
                    attention = results['attention_weights'].squeeze().cpu().numpy()
                    # Use last row (query attention) as feature importance
                    feature_importance = attention[-1, :-1]  # Exclude self-attention
                    top_features = np.argsort(feature_importance)[-5:][::-1]
                    
                    response_data['explanation'] = {
                        'top_features': [int(idx) for idx in top_features],
                        'feature_importance': [float(feature_importance[idx]) for idx in top_features],
                        'method': 'attention_weights'
                    }
            
            # Update monitoring
            self.monitor.record_prediction(prediction, confidence, 
                                         results.get('total_uncertainty', torch.tensor(0)).item())
            self.performance_tracker.record_request_time(time.time() - start_time)
            
            return PredictionResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Make batch predictions.
        
        Args:
            request: Batch prediction request
            
        Returns:
            Batch prediction response
        """
        start_time = time.time()
        
        try:
            # Process each request
            predictions = []
            for single_request in request.flows:
                prediction = await self.predict_single(single_request)
                predictions.append(prediction)
            
            # Prepare response
            response_data = {
                'predictions': predictions,
                'total_samples': len(predictions),
                'processing_time_ms': (time.time() - start_time) * 1000,
            }
            
            # Add batch summary if requested
            if request.return_batch_summary:
                attack_count = sum(1 for p in predictions if p.prediction == 1)
                avg_confidence = np.mean([p.confidence for p in predictions])
                
                if predictions[0].uncertainty is not None:
                    avg_uncertainty = np.mean([p.uncertainty for p in predictions])
                    high_uncertainty_count = sum(1 for p in predictions if p.uncertainty > 0.2)
                else:
                    avg_uncertainty = None
                    high_uncertainty_count = 0
                
                response_data['batch_summary'] = {
                    'attack_rate': attack_count / len(predictions),
                    'avg_confidence': avg_confidence,
                    'avg_uncertainty': avg_uncertainty,
                    'high_uncertainty_rate': high_uncertainty_count / len(predictions),
                    'requires_review_count': sum(1 for p in predictions if p.requires_review),
                }
            
            return BatchPredictionResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def get_health(self) -> HealthResponse:
        """Get service health status."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return HealthResponse(
            status="healthy" if self.model is not None else "unhealthy",
            version=self.version,
            model_loaded=self.model is not None,
            uptime_seconds=uptime
        )
    
    def get_metrics(self) -> MetricsResponse:
        """Get service metrics."""
        # Get system metrics
        memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent()
        
        # Get API metrics
        api_metrics = self.monitor.get_metrics()
        perf_metrics = self.performance_tracker.get_metrics()
        
        return MetricsResponse(
            total_requests=api_metrics['total_requests'],
            requests_per_second=api_metrics['requests_per_second'],
            avg_processing_time_ms=perf_metrics['avg_time_ms'],
            p95_processing_time_ms=perf_metrics['p95_time_ms'],
            p99_processing_time_ms=perf_metrics['p99_time_ms'],
            total_predictions=api_metrics['total_predictions'],
            attack_rate=api_metrics['attack_rate'],
            avg_uncertainty=api_metrics['avg_uncertainty'],
            high_uncertainty_rate=api_metrics['high_uncertainty_rate'],
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )
    
    def get_model_info(self) -> ModelInfoResponse:
        """Get model information."""
        model_info = self.model.get_model_info() if self.model else {}
        
        return ModelInfoResponse(
            model_name="BayesianEnsembleIDS",
            model_version=self.model_version,
            model_type=model_info.get('model_type', 'BayesianEnsembleIDS'),
            ensemble_size=model_info.get('n_ensemble', None),
            sequence_length=model_info.get('max_seq_len', 50),
            n_features=41,
            n_classes=model_info.get('n_classes', 2),
            api_version=self.version
        )


def create_app(model_path: str, processor_path: str, 
               config: Optional[Dict] = None) -> FastAPI:
    """
    Create FastAPI application instance.
    
    Args:
        model_path: Path to trained model
        processor_path: Path to data processor
        config: Optional configuration
        
    Returns:
        FastAPI application instance
    """
    # Initialize API
    api = UncertaintyIDSAPI(model_path, processor_path, config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Uncertainty-Aware Intrusion Detection API",
        description="Production-ready API for real-time network intrusion detection with uncertainty quantification",
        version=api.version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Exception handler
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                details={"exception": str(exc)},
                request_id=str(uuid.uuid4())
            ).dict()
        )
    
    # Routes
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make a single prediction with uncertainty quantification."""
        return await api.predict_single(request)
    
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        """Make batch predictions."""
        return await api.predict_batch(request)
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return api.get_health()
    
    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics():
        """Get service metrics."""
        return api.get_metrics()
    
    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info():
        """Get model information."""
        return api.get_model_info()
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "Uncertainty-Aware Intrusion Detection API",
            "version": api.version,
            "status": "running",
            "docs": "/docs"
        }
    
    return app


# For running with uvicorn
def create_app_from_env() -> FastAPI:
    """Create app using environment variables."""
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    processor_path = os.getenv("PROCESSOR_PATH", "preprocessors/")
    
    return create_app(model_path, processor_path)
