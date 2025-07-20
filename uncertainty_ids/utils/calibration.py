"""
Uncertainty calibration utilities.

This module provides methods for calibrating uncertainty estimates
to ensure they are well-calibrated and reliable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for uncertainty calibration.
    
    Applies a learned temperature parameter to logits to improve
    calibration without changing the model's accuracy.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(self, logits: torch.Tensor, targets: torch.Tensor, 
            max_iter: int = 50, lr: float = 0.01) -> float:
        """
        Fit temperature parameter using validation data.
        
        Args:
            logits: Model logits (batch_size, n_classes)
            targets: True targets (batch_size,)
            max_iter: Maximum optimization iterations
            lr: Learning rate
            
        Returns:
            Final temperature value
        """
        # Initialize temperature
        self.temperature.data.fill_(1.0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            loss = F.cross_entropy(self.forward(logits), targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        # Ensure temperature is positive
        self.temperature.data = torch.clamp(self.temperature.data, min=1e-3)
        
        logger.info(f"Temperature scaling fitted: T = {self.temperature.item():.4f}")
        return self.temperature.item()


class PlattScaling:
    """
    Platt scaling for binary classification calibration.
    
    Fits a sigmoid function to map prediction scores to calibrated probabilities.
    """
    
    def __init__(self):
        self.calibrator = LogisticRegression()
        self.fitted = False
    
    def fit(self, scores: np.ndarray, targets: np.ndarray):
        """
        Fit Platt scaling parameters.
        
        Args:
            scores: Prediction scores (n_samples,)
            targets: True binary targets (n_samples,)
        """
        scores = scores.reshape(-1, 1)
        self.calibrator.fit(scores, targets)
        self.fitted = True
        
        logger.info("Platt scaling fitted successfully")
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.
        
        Args:
            scores: Prediction scores (n_samples,)
            
        Returns:
            Calibrated probabilities (n_samples, 2)
        """
        if not self.fitted:
            raise ValueError("Platt scaling must be fitted before prediction")
        
        scores = scores.reshape(-1, 1)
        return self.calibrator.predict_proba(scores)


class IsotonicCalibration:
    """
    Isotonic regression for calibration.
    
    Non-parametric method that fits a monotonic function
    to map prediction scores to calibrated probabilities.
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False
    
    def fit(self, scores: np.ndarray, targets: np.ndarray):
        """
        Fit isotonic regression.
        
        Args:
            scores: Prediction scores (n_samples,)
            targets: True binary targets (n_samples,)
        """
        self.calibrator.fit(scores, targets)
        self.fitted = True
        
        logger.info("Isotonic calibration fitted successfully")
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.
        
        Args:
            scores: Prediction scores (n_samples,)
            
        Returns:
            Calibrated probabilities (n_samples,)
        """
        if not self.fitted:
            raise ValueError("Isotonic calibration must be fitted before prediction")
        
        calibrated_probs = self.calibrator.predict(scores)
        return calibrated_probs


class UncertaintyCalibrator:
    """
    Comprehensive uncertainty calibration framework.
    
    Combines multiple calibration methods and provides
    utilities for evaluating calibration quality.
    """
    
    def __init__(self, method: str = 'temperature'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('temperature', 'platt', 'isotonic')
        """
        self.method = method
        self.calibrator = self._create_calibrator(method)
        self.fitted = False
    
    def _create_calibrator(self, method: str):
        """Create calibrator based on method."""
        if method == 'temperature':
            return TemperatureScaling()
        elif method == 'platt':
            return PlattScaling()
        elif method == 'isotonic':
            return IsotonicCalibration()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor, 
            uncertainties: Optional[torch.Tensor] = None):
        """
        Fit calibration parameters.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: True targets
            uncertainties: Uncertainty estimates (optional)
        """
        if self.method == 'temperature':
            if predictions.dim() == 1:
                # Convert to logits format
                predictions = torch.stack([1 - predictions, predictions], dim=1)
            self.calibrator.fit(predictions, targets)
        else:
            # Convert to numpy for sklearn-based methods
            if isinstance(predictions, torch.Tensor):
                if predictions.dim() > 1:
                    predictions = predictions[:, 1]  # Use positive class probability
                predictions = predictions.cpu().numpy()
            
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            
            self.calibrator.fit(predictions, targets)
        
        self.fitted = True
    
    def calibrate(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        if self.method == 'temperature':
            return F.softmax(self.calibrator(predictions), dim=-1)
        else:
            # Handle sklearn-based methods
            if isinstance(predictions, torch.Tensor):
                if predictions.dim() > 1:
                    predictions = predictions[:, 1]
                predictions_np = predictions.cpu().numpy()
            else:
                predictions_np = predictions
            
            if self.method == 'platt':
                calibrated = self.calibrator.predict_proba(predictions_np)
                return torch.tensor(calibrated, dtype=torch.float32)
            else:  # isotonic
                calibrated = self.calibrator.predict_proba(predictions_np)
                # Convert to binary probability format
                calibrated_binary = np.stack([1 - calibrated, calibrated], axis=1)
                return torch.tensor(calibrated_binary, dtype=torch.float32)
    
    def compute_calibration_error(self, confidences: np.ndarray, 
                                 accuracies: np.ndarray, 
                                 n_bins: int = 10) -> Dict[str, float]:
        """
        Compute calibration error metrics.
        
        Args:
            confidences: Confidence scores
            accuracies: Binary accuracy indicators
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with calibration metrics
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0  # Expected Calibration Error
        mce = 0.0  # Maximum Calibration Error
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'n_bins': n_bins
        }
    
    def reliability_diagram_data(self, confidences: np.ndarray, 
                                accuracies: np.ndarray, 
                                n_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate data for reliability diagram.
        
        Args:
            confidences: Confidence scores
            accuracies: Binary accuracy indicators
            n_bins: Number of bins
            
        Returns:
            Dictionary with reliability diagram data
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_count = in_bin.sum()
            
            if bin_count > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(bin_count)
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        return {
            'bin_centers': np.array(bin_centers),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts),
            'bin_boundaries': bin_boundaries
        }
    
    def save(self, filepath: str):
        """Save calibrator to file."""
        if self.method == 'temperature':
            torch.save({
                'method': self.method,
                'state_dict': self.calibrator.state_dict(),
                'fitted': self.fitted
            }, filepath)
        else:
            import joblib
            joblib.dump({
                'method': self.method,
                'calibrator': self.calibrator,
                'fitted': self.fitted
            }, filepath)
        
        logger.info(f"Calibrator saved to {filepath}")
    
    def load(self, filepath: str):
        """Load calibrator from file."""
        if self.method == 'temperature':
            checkpoint = torch.load(filepath, map_location='cpu')
            self.method = checkpoint['method']
            self.calibrator = self._create_calibrator(self.method)
            self.calibrator.load_state_dict(checkpoint['state_dict'])
            self.fitted = checkpoint['fitted']
        else:
            import joblib
            checkpoint = joblib.load(filepath)
            self.method = checkpoint['method']
            self.calibrator = checkpoint['calibrator']
            self.fitted = checkpoint['fitted']
        
        logger.info(f"Calibrator loaded from {filepath}")


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, 
                        n_bins: int = 10) -> Dict[str, float]:
    """
    Evaluate calibration quality of predictions.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for evaluation
        
    Returns:
        Dictionary with calibration metrics
    """
    calibrator = UncertaintyCalibrator(method='isotonic')  # Dummy for evaluation
    
    # Convert probabilities to confidences (max probability)
    if y_prob.ndim > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = np.maximum(y_prob, 1 - y_prob)
        predictions = (y_prob > 0.5).astype(int)
    
    # Compute accuracy indicators
    accuracies = (predictions == y_true).astype(float)
    
    # Compute calibration metrics
    return calibrator.compute_calibration_error(confidences, accuracies, n_bins)
