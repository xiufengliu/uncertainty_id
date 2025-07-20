"""
Comprehensive evaluation metrics for uncertainty-aware intrusion detection.

This module implements metrics for assessing both detection performance
and uncertainty quality in intrusion detection systems.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    detection_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    calibration_metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    roc_data: Optional[Dict[str, np.ndarray]] = None
    pr_data: Optional[Dict[str, np.ndarray]] = None
    reliability_data: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict] = None


class DetectionMetrics:
    """
    Standard intrusion detection performance metrics.
    
    Implements comprehensive evaluation metrics specifically designed
    for intrusion detection systems where false positives are critical.
    """
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute comprehensive detection performance metrics.
        
        Args:
            y_true: True binary labels (0=normal, 1=attack)
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all detection metrics
        """
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # IDS-specific metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        
        # Balanced metrics
        balanced_accuracy = (detection_rate + true_negative_rate) / 2
        
        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom != 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': false_positive_rate,
            'true_negative_rate': true_negative_rate,
            'detection_rate': detection_rate,
            'balanced_accuracy': balanced_accuracy,
            'matthews_correlation': mcc,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
        }
        
        # Add probability-based metrics if available
        if y_prob is not None:
            try:
                # Handle both binary probabilities and probability arrays
                if y_prob.ndim == 1:
                    prob_positive = y_prob
                else:
                    prob_positive = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
                
                auc_roc = roc_auc_score(y_true, prob_positive)
                auc_pr = average_precision_score(y_true, prob_positive)
                
                metrics.update({
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr,
                })
            except Exception as e:
                logger.warning(f"Could not compute probability-based metrics: {e}")
                metrics.update({
                    'auc_roc': 0.0,
                    'auc_pr': 0.0,
                })
        
        return metrics
    
    @staticmethod
    def compute_cost_sensitive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                     fp_cost: float = 1.0, fn_cost: float = 10.0) -> Dict[str, float]:
        """
        Compute cost-sensitive metrics for intrusion detection.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            fp_cost: Cost of false positive (investigating benign traffic)
            fn_cost: Cost of false negative (missing real attack)
            
        Returns:
            Dictionary with cost-sensitive metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = fp * fp_cost + fn * fn_cost
        total_samples = len(y_true)
        avg_cost_per_sample = total_cost / total_samples
        
        # Cost-sensitive accuracy (minimize expected cost)
        baseline_cost = np.sum(y_true) * fn_cost  # Cost of predicting all normal
        cost_reduction = (baseline_cost - total_cost) / baseline_cost if baseline_cost > 0 else 0
        
        return {
            'total_cost': total_cost,
            'avg_cost_per_sample': avg_cost_per_sample,
            'cost_reduction': cost_reduction,
            'fp_cost_contribution': fp * fp_cost,
            'fn_cost_contribution': fn * fn_cost,
        }


class UncertaintyMetrics:
    """
    Metrics for evaluating uncertainty quality in predictions.
    
    Assesses how well uncertainty estimates correlate with prediction
    correctness and provide useful information for decision making.
    """
    
    @staticmethod
    def compute_uncertainty_accuracy_correlation(uncertainties: np.ndarray, 
                                               correctness: np.ndarray) -> float:
        """
        Compute correlation between uncertainty and prediction correctness.
        
        Higher uncertainty should correlate with incorrect predictions.
        
        Args:
            uncertainties: Uncertainty estimates
            correctness: Binary correctness indicators (1=correct, 0=incorrect)
            
        Returns:
            Correlation coefficient (higher is better)
        """
        # We want high uncertainty for incorrect predictions
        incorrectness = 1 - correctness.astype(float)
        correlation = np.corrcoef(uncertainties, incorrectness)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def compute_uncertainty_rejection_curve(uncertainties: np.ndarray, 
                                          correctness: np.ndarray,
                                          n_points: int = 20) -> Dict[str, np.ndarray]:
        """
        Compute accuracy vs rejection rate curve based on uncertainty.
        
        Shows how accuracy improves when rejecting high-uncertainty predictions.
        
        Args:
            uncertainties: Uncertainty estimates
            correctness: Binary correctness indicators
            n_points: Number of points in the curve
            
        Returns:
            Dictionary with rejection curve data
        """
        # Sort by uncertainty (ascending)
        sorted_indices = np.argsort(uncertainties)
        sorted_correctness = correctness[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]
        
        # Compute curve points
        n_samples = len(uncertainties)
        rejection_rates = np.linspace(0, 0.9, n_points)  # Don't reject everything
        accuracies = []
        thresholds = []
        
        for rejection_rate in rejection_rates:
            n_keep = int(n_samples * (1 - rejection_rate))
            if n_keep > 0:
                kept_correctness = sorted_correctness[:n_keep]
                accuracy = np.mean(kept_correctness)
                threshold = sorted_uncertainties[n_keep - 1] if n_keep < n_samples else np.max(sorted_uncertainties)
            else:
                accuracy = 0.0
                threshold = np.min(sorted_uncertainties)
            
            accuracies.append(accuracy)
            thresholds.append(threshold)
        
        return {
            'rejection_rates': rejection_rates,
            'accuracies': np.array(accuracies),
            'thresholds': np.array(thresholds)
        }
    
    @staticmethod
    def compute_area_under_rejection_curve(uncertainties: np.ndarray, 
                                         correctness: np.ndarray) -> float:
        """
        Compute area under the uncertainty-rejection curve (AURC).
        
        Higher values indicate better uncertainty quality.
        """
        curve_data = UncertaintyMetrics.compute_uncertainty_rejection_curve(
            uncertainties, correctness
        )
        
        # Compute area using trapezoidal rule
        aurc = np.trapz(curve_data['accuracies'], curve_data['rejection_rates'])
        return aurc
    
    @staticmethod
    def compute_uncertainty_based_metrics(uncertainties: np.ndarray, 
                                        correctness: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive uncertainty-based metrics.
        
        Args:
            uncertainties: Uncertainty estimates
            correctness: Binary correctness indicators
            
        Returns:
            Dictionary with uncertainty quality metrics
        """
        # Basic statistics
        mean_uncertainty = np.mean(uncertainties)
        std_uncertainty = np.std(uncertainties)
        
        # Uncertainty-accuracy correlation
        correlation = UncertaintyMetrics.compute_uncertainty_accuracy_correlation(
            uncertainties, correctness
        )
        
        # Area under rejection curve
        aurc = UncertaintyMetrics.compute_area_under_rejection_curve(
            uncertainties, correctness
        )
        
        # Uncertainty distribution for correct vs incorrect predictions
        correct_mask = correctness.astype(bool)
        incorrect_mask = ~correct_mask
        
        if np.any(correct_mask) and np.any(incorrect_mask):
            mean_uncertainty_correct = np.mean(uncertainties[correct_mask])
            mean_uncertainty_incorrect = np.mean(uncertainties[incorrect_mask])
            uncertainty_separation = mean_uncertainty_incorrect - mean_uncertainty_correct
        else:
            mean_uncertainty_correct = mean_uncertainty
            mean_uncertainty_incorrect = mean_uncertainty
            uncertainty_separation = 0.0
        
        return {
            'mean_uncertainty': mean_uncertainty,
            'std_uncertainty': std_uncertainty,
            'uncertainty_accuracy_correlation': correlation,
            'area_under_rejection_curve': aurc,
            'mean_uncertainty_correct': mean_uncertainty_correct,
            'mean_uncertainty_incorrect': mean_uncertainty_incorrect,
            'uncertainty_separation': uncertainty_separation,
        }


class CalibrationMetrics:
    """
    Metrics for evaluating uncertainty calibration quality.
    
    Assesses how well predicted confidence scores match actual accuracy.
    """
    
    @staticmethod
    def compute_expected_calibration_error(confidences: np.ndarray, 
                                         correctness: np.ndarray,
                                         n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Measures the difference between predicted confidence and actual accuracy.
        
        Args:
            confidences: Confidence scores (0-1)
            correctness: Binary correctness indicators
            n_bins: Number of bins for calibration
            
        Returns:
            Expected calibration error (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(correctness[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def compute_maximum_calibration_error(confidences: np.ndarray, 
                                        correctness: np.ndarray,
                                        n_bins: int = 10) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        Maximum difference between confidence and accuracy across all bins.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(correctness[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    @staticmethod
    def compute_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Brier Score for probability predictions.
        
        Measures the mean squared difference between predicted probabilities
        and actual outcomes.
        
        Args:
            probabilities: Predicted probabilities
            targets: True binary targets
            
        Returns:
            Brier score (lower is better)
        """
        if probabilities.ndim > 1:
            # Multi-class case - use sklearn implementation
            return brier_score_loss(targets, probabilities[:, 1])
        else:
            # Binary case
            return np.mean((probabilities - targets) ** 2)
    
    @staticmethod
    def generate_reliability_diagram_data(confidences: np.ndarray, 
                                        correctness: np.ndarray,
                                        n_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate data for reliability diagram (calibration plot).
        
        Args:
            confidences: Confidence scores
            correctness: Binary correctness indicators
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
            bin_count = np.sum(in_bin)
            
            if bin_count > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(np.mean(correctness[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
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
    
    @staticmethod
    def compute_calibration_metrics(confidences: np.ndarray, 
                                  correctness: np.ndarray,
                                  probabilities: Optional[np.ndarray] = None,
                                  targets: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute comprehensive calibration metrics.
        
        Args:
            confidences: Confidence scores
            correctness: Binary correctness indicators
            probabilities: Predicted probabilities (optional)
            targets: True targets (optional, for Brier score)
            
        Returns:
            Dictionary with calibration metrics
        """
        metrics = {
            'expected_calibration_error': CalibrationMetrics.compute_expected_calibration_error(
                confidences, correctness
            ),
            'maximum_calibration_error': CalibrationMetrics.compute_maximum_calibration_error(
                confidences, correctness
            ),
        }
        
        # Add Brier score if probabilities and targets are provided
        if probabilities is not None and targets is not None:
            metrics['brier_score'] = CalibrationMetrics.compute_brier_score(
                probabilities, targets
            )
        
        return metrics


class ComprehensiveEvaluator:
    """
    Main evaluator class that combines all evaluation metrics.
    
    Provides a unified interface for comprehensive evaluation of
    uncertainty-aware intrusion detection models.
    """
    
    def __init__(self):
        self.detection_metrics = DetectionMetrics()
        self.uncertainty_metrics = UncertaintyMetrics()
        self.calibration_metrics = CalibrationMetrics()
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray] = None,
                      uncertainties: Optional[np.ndarray] = None,
                      confidences: Optional[np.ndarray] = None,
                      include_curves: bool = True) -> EvaluationResults:
        """
        Comprehensive evaluation of uncertainty-aware IDS model.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            uncertainties: Uncertainty estimates (optional)
            confidences: Confidence scores (optional)
            include_curves: Whether to compute ROC/PR curves
            
        Returns:
            EvaluationResults object with all metrics
        """
        # Compute correctness for uncertainty evaluation
        correctness = (y_true == y_pred).astype(float)
        
        # Detection performance metrics
        detection_results = self.detection_metrics.compute_all_metrics(
            y_true, y_pred, y_prob
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Initialize results containers
        uncertainty_results = {}
        calibration_results = {}
        roc_data = None
        pr_data = None
        reliability_data = None
        
        # Uncertainty quality metrics (if uncertainties provided)
        if uncertainties is not None:
            uncertainty_results = self.uncertainty_metrics.compute_uncertainty_based_metrics(
                uncertainties, correctness
            )
        
        # Calibration metrics (if confidences provided)
        if confidences is not None:
            calibration_results = self.calibration_metrics.compute_calibration_metrics(
                confidences, correctness, y_prob, y_true
            )
            
            # Reliability diagram data
            reliability_data = self.calibration_metrics.generate_reliability_diagram_data(
                confidences, correctness
            )
        
        # ROC and PR curves (if probabilities provided)
        if y_prob is not None and include_curves:
            try:
                prob_positive = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                
                fpr, tpr, roc_thresholds = roc_curve(y_true, prob_positive)
                precision, recall, pr_thresholds = precision_recall_curve(y_true, prob_positive)
                
                roc_data = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': roc_thresholds
                }
                
                pr_data = {
                    'precision': precision,
                    'recall': recall,
                    'thresholds': pr_thresholds
                }
            except Exception as e:
                logger.warning(f"Could not compute ROC/PR curves: {e}")
        
        # Create metadata
        metadata = {
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true),
            'n_negative': len(y_true) - np.sum(y_true),
            'has_uncertainties': uncertainties is not None,
            'has_confidences': confidences is not None,
            'has_probabilities': y_prob is not None,
        }
        
        return EvaluationResults(
            detection_metrics=detection_results,
            uncertainty_metrics=uncertainty_results,
            calibration_metrics=calibration_results,
            confusion_matrix=cm,
            roc_data=roc_data,
            pr_data=pr_data,
            reliability_data=reliability_data,
            metadata=metadata
        )
    
    def compare_models(self, results_list: List[EvaluationResults], 
                      model_names: List[str]) -> Dict[str, Dict]:
        """
        Compare multiple model evaluation results.
        
        Args:
            results_list: List of EvaluationResults objects
            model_names: Names of the models
            
        Returns:
            Dictionary with comparison results
        """
        if len(results_list) != len(model_names):
            raise ValueError("Number of results must match number of model names")
        
        comparison = {
            'detection_metrics': {},
            'uncertainty_metrics': {},
            'calibration_metrics': {},
            'rankings': {}
        }
        
        # Collect metrics for each model
        for metric_type in ['detection_metrics', 'uncertainty_metrics', 'calibration_metrics']:
            comparison[metric_type] = {}
            
            # Get all metric names from first model
            if hasattr(results_list[0], metric_type):
                metric_dict = getattr(results_list[0], metric_type)
                for metric_name in metric_dict.keys():
                    comparison[metric_type][metric_name] = {}
                    
                    for i, (result, model_name) in enumerate(zip(results_list, model_names)):
                        model_metric_dict = getattr(result, metric_type)
                        if metric_name in model_metric_dict:
                            comparison[metric_type][metric_name][model_name] = model_metric_dict[metric_name]
        
        # Compute rankings for key metrics
        key_metrics = ['f1_score', 'auc_roc', 'false_positive_rate']
        for metric in key_metrics:
            if metric in comparison['detection_metrics']:
                values = comparison['detection_metrics'][metric]
                # Sort by value (descending for most metrics, ascending for FPR)
                reverse = metric != 'false_positive_rate'
                sorted_models = sorted(values.items(), key=lambda x: x[1], reverse=reverse)
                comparison['rankings'][metric] = [model for model, _ in sorted_models]
        
        return comparison
