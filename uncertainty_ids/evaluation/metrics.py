"""
Evaluation metrics for uncertainty-aware intrusion detection.
Based on the evaluation metrics described in Section 4 of the paper.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy.stats import pearsonr
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt


class ClassificationMetrics:
    """
    Standard classification metrics for intrusion detection.
    """
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')

        Returns:
            Dictionary of metrics
        """
        # Determine if binary or multi-class
        n_classes = len(np.unique(y_true))
        if n_classes > 2 and average == 'binary':
            average = 'macro'  # Default to macro for multi-class

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        # Compute FPR and TPR for binary classification
        if n_classes == 2:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                metrics['fpr'] = 0.0
                metrics['tpr'] = 0.0
        else:
            # For multi-class, compute macro-averaged FPR and TPR
            cm = confusion_matrix(y_true, y_pred)
            fprs = []
            tprs = []
            for i in range(n_classes):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp

                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                fprs.append(fpr)
                tprs.append(tpr)

            metrics['fpr'] = np.mean(fprs)
            metrics['tpr'] = np.mean(tprs)

        # AUC if probabilities provided
        if y_proba is not None:
            try:
                if n_classes == 2:
                    # Binary classification
                    if y_proba.ndim == 1:
                        metrics['auc'] = roc_auc_score(y_true, y_proba)
                    else:
                        metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except (ValueError, IndexError):
                metrics['auc'] = 0.0

        return metrics


class UncertaintyMetrics:
    """
    Uncertainty-specific evaluation metrics.
    Based on Section 4.2 Uncertainty Quality Analysis in the paper.
    """
    
    @staticmethod
    def uncertainty_accuracy_correlation(
        uncertainties: np.ndarray,
        correctness: np.ndarray
    ) -> float:
        """
        Compute correlation between uncertainty and prediction correctness.
        
        Based on paper: "strong negative correlation of -0.78 ± 0.03"
        
        Args:
            uncertainties: Uncertainty estimates
            correctness: Binary correctness (1 if correct, 0 if incorrect)
            
        Returns:
            Pearson correlation coefficient
        """
        correlation, _ = pearsonr(uncertainties, correctness)
        return correlation
    
    @staticmethod
    def mutual_information(
        uncertainties: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute mutual information between uncertainty and correctness.

        Based on paper: "measured mutual information of 0.34 bits"

        Args:
            uncertainties: Uncertainty estimates
            correctness: Binary correctness
            n_bins: Number of bins for discretization

        Returns:
            Mutual information in bits
        """
        # Normalize uncertainties to [0, 1] range
        if uncertainties.max() > uncertainties.min():
            uncertainties_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
        else:
            uncertainties_norm = np.zeros_like(uncertainties)

        # Create bin edges
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_edges[-1] += 1e-10  # Ensure last bin includes maximum value

        # Discretize uncertainties
        uncertainty_bins = np.digitize(uncertainties_norm, bin_edges) - 1
        uncertainty_bins = np.clip(uncertainty_bins, 0, n_bins - 1)

        # Compute joint and marginal probabilities
        joint_counts = np.zeros((n_bins, 2))
        for i in range(len(uncertainties)):
            joint_counts[uncertainty_bins[i], int(correctness[i])] += 1

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        joint_probs = (joint_counts + epsilon) / (len(uncertainties) + 2 * n_bins * epsilon)
        marginal_uncertainty = joint_probs.sum(axis=1)
        marginal_correctness = joint_probs.sum(axis=0)

        # Compute mutual information with numerical stability
        mi = 0.0
        for i in range(n_bins):
            for j in range(2):
                if joint_probs[i, j] > epsilon and marginal_uncertainty[i] > epsilon and marginal_correctness[j] > epsilon:
                    mi += joint_probs[i, j] * np.log2(
                        joint_probs[i, j] / (marginal_uncertainty[i] * marginal_correctness[j])
                    )

        return max(0.0, mi)  # Ensure non-negative MI

    @staticmethod
    def area_under_risk_coverage_curve(
        uncertainties: np.ndarray,
        correctness: np.ndarray,
        n_points: int = 100
    ) -> float:
        """
        Compute Area Under Risk-Coverage Curve (AURC).

        Args:
            uncertainties: Uncertainty estimates
            correctness: Binary correctness
            n_points: Number of points for curve approximation

        Returns:
            AURC value (lower is better)
        """
        # Sort by uncertainty (descending - highest uncertainty first)
        sorted_indices = np.argsort(-uncertainties)
        sorted_correctness = correctness[sorted_indices]

        n_samples = len(uncertainties)
        rejection_rates = np.linspace(0, 1, n_points)
        error_rates = []

        for rejection_rate in rejection_rates:
            n_keep = int((1 - rejection_rate) * n_samples)
            if n_keep > 0:
                kept_correctness = sorted_correctness[:n_keep]
                error_rate = 1 - kept_correctness.mean()
            else:
                error_rate = 0.0
            error_rates.append(error_rate)

        # Compute area under curve
        aurc = np.trapz(error_rates, rejection_rates)
        return aurc
    
    @staticmethod
    def area_under_rejection_curve(
        uncertainties: np.ndarray,
        correctness: np.ndarray,
        n_thresholds: int = 100
    ) -> float:
        """
        Compute Area Under Rejection Curve (AURC).
        
        Based on paper: "AURC of 0.92 (averaged across datasets)"
        
        Args:
            uncertainties: Uncertainty estimates
            correctness: Binary correctness
            n_thresholds: Number of rejection thresholds
            
        Returns:
            AURC value
        """
        # Sort by uncertainty (descending)
        sorted_indices = np.argsort(-uncertainties)
        sorted_correctness = correctness[sorted_indices]
        
        # Compute rejection curve
        n_samples = len(uncertainties)
        rejection_rates = np.linspace(0, 1, n_thresholds)
        error_rates = []
        
        for rejection_rate in rejection_rates:
            n_keep = int((1 - rejection_rate) * n_samples)
            if n_keep > 0:
                kept_correctness = sorted_correctness[:n_keep]
                error_rate = 1 - kept_correctness.mean()
            else:
                error_rate = 0.0
            error_rates.append(error_rate)
        
        # Compute area under curve
        aurc = np.trapz(error_rates, rejection_rates)
        return aurc


class CalibrationMetrics:
    """
    Calibration metrics for uncertainty estimates.
    Based on the calibration analysis in Section 4.2 of the paper.
    """
    
    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Based on paper ECE values: 0.0008 (CICIDS2017), 0.2022 (NSL-KDD)
        
        Args:
            confidences: Confidence scores (max probability)
            correctness: Binary correctness
            n_bins: Number of calibration bins
            
        Returns:
            ECE value
        """
        # Ensure confidences are in [0, 1] range
        confidences = np.clip(confidences, 0.0, 1.0)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(confidences)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin (inclusive of upper bound for last bin)
            if bin_upper == 1.0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

            n_samples_in_bin = in_bin.sum()

            if n_samples_in_bin > 0:
                prop_in_bin = n_samples_in_bin / total_samples
                accuracy_in_bin = correctness[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    
    @staticmethod
    def reliability_diagram_data(
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram.
        
        Args:
            confidences: Confidence scores
            correctness: Binary correctness
            n_bins: Number of bins
            
        Returns:
            bin_centers: Center of each bin
            bin_accuracies: Accuracy in each bin
            bin_confidences: Average confidence in each bin
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Handle last bin inclusively
            if i == n_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

            if in_bin.sum() > 0:
                bin_accuracies.append(correctness[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
            else:
                # For empty bins, use NaN to indicate no data
                bin_accuracies.append(np.nan)
                bin_confidences.append(np.nan)
        
        return bin_centers, np.array(bin_accuracies), np.array(bin_confidences)
    
    @staticmethod
    def brier_score(
        probabilities: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute Brier score for calibration assessment.
        
        Args:
            probabilities: Predicted probabilities
            targets: True binary labels
            
        Returns:
            Brier score (lower is better)
        """
        return np.mean((probabilities - targets) ** 2)


class RobustnessMetrics:
    """
    Robustness evaluation metrics.
    Based on Section 4.3 Robustness Analysis in the paper.
    """
    
    @staticmethod
    def adversarial_robustness_ratio(
        clean_accuracy: float,
        adversarial_accuracy: float
    ) -> float:
        """
        Compute robustness ratio.
        
        Based on paper: "robustness ratio of 0.920" for PGD attacks
        
        Args:
            clean_accuracy: Accuracy on clean samples
            adversarial_accuracy: Accuracy on adversarial samples
            
        Returns:
            Robustness ratio (adversarial_accuracy / clean_accuracy)
        """
        return adversarial_accuracy / clean_accuracy if clean_accuracy > 0 else 0.0
    
    @staticmethod
    def uncertainty_increase_under_attack(
        clean_uncertainties: np.ndarray,
        adversarial_uncertainties: np.ndarray
    ) -> float:
        """
        Compute mean increase in uncertainty under adversarial attack.
        
        Based on paper: "mean increase of 0.23 ± 0.04"
        
        Args:
            clean_uncertainties: Uncertainties on clean samples
            adversarial_uncertainties: Uncertainties on adversarial samples
            
        Returns:
            Mean uncertainty increase
        """
        return np.mean(adversarial_uncertainties - clean_uncertainties)
