"""
REST API module for Uncertainty-Aware Intrusion Detection System.

This module provides a production-ready REST API for real-time inference
with comprehensive monitoring, logging, and deployment capabilities.
"""

from .server import UncertaintyIDSAPI, create_app
from .models import (
    NetworkFlowRequest, PredictionResponse, BatchPredictionRequest,
    HealthResponse, MetricsResponse, ModelInfoResponse
)
from .middleware import (
    RequestLoggingMiddleware, MetricsMiddleware, RateLimitMiddleware
)
from .monitoring import APIMonitor, PerformanceTracker

__all__ = [
    # Main API
    'UncertaintyIDSAPI',
    'create_app',
    
    # Request/Response models
    'NetworkFlowRequest',
    'PredictionResponse', 
    'BatchPredictionRequest',
    'HealthResponse',
    'MetricsResponse',
    'ModelInfoResponse',
    
    # Middleware
    'RequestLoggingMiddleware',
    'MetricsMiddleware',
    'RateLimitMiddleware',
    
    # Monitoring
    'APIMonitor',
    'PerformanceTracker',
]
