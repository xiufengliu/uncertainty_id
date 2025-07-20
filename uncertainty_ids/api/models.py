"""
Pydantic models for API request/response validation.

This module defines the data models used for API communication,
including request validation and response formatting.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import numpy as np


class NetworkFlowRequest(BaseModel):
    """
    Request model for single network flow prediction.
    
    Contains the 41 standard network intrusion detection features.
    """
    # Basic connection features
    duration: float = Field(..., ge=0, description="Connection duration in seconds")
    protocol_type: Union[str, int] = Field(..., description="Protocol type (tcp, udp, icmp)")
    service: Union[str, int] = Field(..., description="Network service type")
    flag: Union[str, int] = Field(..., description="Connection flag")
    src_bytes: int = Field(..., ge=0, description="Bytes sent from source")
    dst_bytes: int = Field(..., ge=0, description="Bytes sent to destination")
    
    # Content features
    land: int = Field(..., ge=0, le=1, description="1 if connection is from/to same host/port")
    wrong_fragment: int = Field(..., ge=0, description="Number of wrong fragments")
    urgent: int = Field(..., ge=0, description="Number of urgent packets")
    
    # Traffic features
    hot: int = Field(..., ge=0, description="Number of hot indicators")
    num_failed_logins: int = Field(..., ge=0, description="Number of failed login attempts")
    logged_in: int = Field(..., ge=0, le=1, description="1 if successfully logged in")
    num_compromised: int = Field(..., ge=0, description="Number of compromised conditions")
    root_shell: int = Field(..., ge=0, le=1, description="1 if root shell obtained")
    su_attempted: int = Field(..., ge=0, le=1, description="1 if su root command attempted")
    num_root: int = Field(..., ge=0, description="Number of root accesses")
    num_file_creations: int = Field(..., ge=0, description="Number of file creation operations")
    num_shells: int = Field(..., ge=0, description="Number of shell prompts")
    num_access_files: int = Field(..., ge=0, description="Number of operations on access control files")
    num_outbound_cmds: int = Field(..., ge=0, description="Number of outbound commands")
    is_host_login: int = Field(..., ge=0, le=1, description="1 if host login")
    is_guest_login: int = Field(..., ge=0, le=1, description="1 if guest login")
    
    # Time-based traffic features
    count: int = Field(..., ge=0, description="Number of connections to same host in past 2 seconds")
    srv_count: int = Field(..., ge=0, description="Number of connections to same service in past 2 seconds")
    serror_rate: float = Field(..., ge=0, le=1, description="% of connections with SYN errors")
    srv_serror_rate: float = Field(..., ge=0, le=1, description="% of connections to same service with SYN errors")
    rerror_rate: float = Field(..., ge=0, le=1, description="% of connections with REJ errors")
    srv_rerror_rate: float = Field(..., ge=0, le=1, description="% of connections to same service with REJ errors")
    same_srv_rate: float = Field(..., ge=0, le=1, description="% of connections to same service")
    diff_srv_rate: float = Field(..., ge=0, le=1, description="% of connections to different services")
    srv_diff_host_rate: float = Field(..., ge=0, le=1, description="% of connections to different hosts")
    
    # Host-based traffic features
    dst_host_count: int = Field(..., ge=0, description="Count of connections having same destination host")
    dst_host_srv_count: int = Field(..., ge=0, description="Count of connections having same destination host and service")
    dst_host_same_srv_rate: float = Field(..., ge=0, le=1, description="% of connections having same destination host and service")
    dst_host_diff_srv_rate: float = Field(..., ge=0, le=1, description="% of different services on destination host")
    dst_host_same_src_port_rate: float = Field(..., ge=0, le=1, description="% of connections having same source port")
    dst_host_srv_diff_host_rate: float = Field(..., ge=0, le=1, description="% of connections having same service and different destination host")
    dst_host_serror_rate: float = Field(..., ge=0, le=1, description="% of connections having SYN error")
    dst_host_srv_serror_rate: float = Field(..., ge=0, le=1, description="% of connections having same service and SYN error")
    dst_host_rerror_rate: float = Field(..., ge=0, le=1, description="% of connections having REJ error")
    dst_host_srv_rerror_rate: float = Field(..., ge=0, le=1, description="% of connections having same service and REJ error")
    
    # Optional metadata
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the network flow")
    source_ip: Optional[str] = Field(default=None, description="Source IP address")
    destination_ip: Optional[str] = Field(default=None, description="Destination IP address")
    
    @validator('protocol_type')
    def validate_protocol_type(cls, v):
        """Validate and convert protocol type."""
        if isinstance(v, str):
            protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
            return protocol_map.get(v.lower(), 3)  # 3 for unknown
        return v
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        features = [
            self.duration, self.protocol_type, self.service, self.flag,
            self.src_bytes, self.dst_bytes, self.land, self.wrong_fragment,
            self.urgent, self.hot, self.num_failed_logins, self.logged_in,
            self.num_compromised, self.root_shell, self.su_attempted,
            self.num_root, self.num_file_creations, self.num_shells,
            self.num_access_files, self.num_outbound_cmds, self.is_host_login,
            self.is_guest_login, self.count, self.srv_count, self.serror_rate,
            self.srv_serror_rate, self.rerror_rate, self.srv_rerror_rate,
            self.same_srv_rate, self.diff_srv_rate, self.srv_diff_host_rate,
            self.dst_host_count, self.dst_host_srv_count, self.dst_host_same_srv_rate,
            self.dst_host_diff_srv_rate, self.dst_host_same_src_port_rate,
            self.dst_host_srv_diff_host_rate, self.dst_host_serror_rate,
            self.dst_host_srv_serror_rate, self.dst_host_rerror_rate,
            self.dst_host_srv_rerror_rate
        ]
        return np.array(features, dtype=np.float32)


class HistoricalFlowsRequest(BaseModel):
    """Request model for historical network flows (for transformer context)."""
    flows: List[NetworkFlowRequest] = Field(..., min_items=1, max_items=100, 
                                          description="Historical network flows")
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([flow.to_array() for flow in self.flows], dtype=np.float32)


class PredictionRequest(BaseModel):
    """Complete prediction request with historical context and current flow."""
    historical_flows: Optional[HistoricalFlowsRequest] = Field(
        default=None, description="Historical network flows for context"
    )
    current_flow: NetworkFlowRequest = Field(..., description="Current flow to classify")
    
    # Prediction options
    return_probabilities: bool = Field(default=True, description="Return class probabilities")
    return_uncertainty: bool = Field(default=True, description="Return uncertainty estimates")
    return_explanation: bool = Field(default=False, description="Return prediction explanation")
    uncertainty_threshold: Optional[float] = Field(default=None, ge=0, le=1, 
                                                  description="Custom uncertainty threshold")


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    # Core prediction
    prediction: int = Field(..., description="Predicted class (0=normal, 1=attack)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    
    # Confidence and uncertainty
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    uncertainty: Optional[float] = Field(default=None, ge=0, description="Total uncertainty")
    epistemic_uncertainty: Optional[float] = Field(default=None, ge=0, description="Model uncertainty")
    aleatoric_uncertainty: Optional[float] = Field(default=None, ge=0, description="Data uncertainty")
    
    # Probabilities
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Class probabilities")
    
    # Decision support
    requires_review: bool = Field(..., description="Whether prediction requires human review")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    
    # Explanation (optional)
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Prediction explanation")
    
    # Metadata
    model_version: str = Field(..., description="Model version used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    @validator('prediction_label', pre=True, always=True)
    def set_prediction_label(cls, v, values):
        """Set human-readable prediction label."""
        if 'prediction' in values:
            return 'attack' if values['prediction'] == 1 else 'normal'
        return v
    
    @validator('risk_level', pre=True, always=True)
    def set_risk_level(cls, v, values):
        """Set risk level based on prediction and uncertainty."""
        if 'prediction' in values and 'uncertainty' in values:
            prediction = values['prediction']
            uncertainty = values.get('uncertainty', 0)
            
            if prediction == 1:  # Attack predicted
                if uncertainty > 0.3:
                    return 'high'
                else:
                    return 'medium'
            else:  # Normal predicted
                if uncertainty > 0.5:
                    return 'medium'
                else:
                    return 'low'
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    flows: List[PredictionRequest] = Field(..., min_items=1, max_items=1000,
                                         description="Batch of prediction requests")
    
    # Batch options
    return_individual_results: bool = Field(default=True, description="Return individual results")
    return_batch_summary: bool = Field(default=True, description="Return batch summary statistics")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="Individual predictions")
    
    # Batch summary
    batch_summary: Optional[Dict[str, Any]] = Field(default=None, description="Batch summary statistics")
    
    # Metadata
    total_samples: int = Field(..., description="Total number of samples processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")


class MetricsResponse(BaseModel):
    """Metrics response model."""
    # Request metrics
    total_requests: int = Field(..., description="Total number of requests")
    requests_per_second: float = Field(..., description="Current requests per second")
    
    # Performance metrics
    avg_processing_time_ms: float = Field(..., description="Average processing time")
    p95_processing_time_ms: float = Field(..., description="95th percentile processing time")
    p99_processing_time_ms: float = Field(..., description="99th percentile processing time")
    
    # Prediction metrics
    total_predictions: int = Field(..., description="Total predictions made")
    attack_rate: float = Field(..., description="Proportion of attack predictions")
    avg_uncertainty: float = Field(..., description="Average uncertainty")
    high_uncertainty_rate: float = Field(..., description="Proportion of high uncertainty predictions")
    
    # System metrics
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type")
    
    # Model configuration
    ensemble_size: Optional[int] = Field(default=None, description="Number of ensemble members")
    sequence_length: int = Field(..., description="Input sequence length")
    n_features: int = Field(..., description="Number of input features")
    n_classes: int = Field(..., description="Number of output classes")
    
    # Training information
    training_dataset: Optional[str] = Field(default=None, description="Training dataset")
    training_date: Optional[datetime] = Field(default=None, description="Training completion date")
    
    # Performance metrics
    validation_accuracy: Optional[float] = Field(default=None, description="Validation accuracy")
    validation_f1_score: Optional[float] = Field(default=None, description="Validation F1 score")
    calibration_error: Optional[float] = Field(default=None, description="Expected calibration error")
    
    # Deployment information
    deployment_date: datetime = Field(default_factory=datetime.utcnow, description="Deployment date")
    api_version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
