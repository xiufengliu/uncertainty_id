"""
Comprehensive Evaluation Framework for Uncertainty-Aware Intrusion Detection
Implements evaluation metrics for both detection performance and uncertainty calibration
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    detection_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    calibration_metrics: Dict[str, float]
    reliability_diagram: Optional[Dict] = None
    roc_data: Optional[Dict] = None
    pr_data: Optional[Dict] = None

class DetectionMetrics:
    """Standard intrusion detection performance metrics"""
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive detection metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC
        try:
            auc_roc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        except:
            auc_roc = 0.0
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # False positive rate (critical for IDS)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # True negative rate (specificity)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Detection rate (same as recall/sensitivity)
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Balanced accuracy
        balanced_accuracy = (detection_rate + tnr) / 2
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'false_positive_rate': fpr,
            'true_negative_rate': tnr,
            'detection_rate': detection_rate,
            'balanced_accuracy': balanced_accuracy,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

class UncertaintyMetrics:
    """Metrics for evaluating uncertainty quality"""
    
    @staticmethod
    def compute_uncertainty_accuracy_correlation(uncertainties: np.ndarray, 
                                               correctness: np.ndarray) -> float:
        """
        Compute correlation between uncertainty and prediction correctness
        Higher uncertainty should correlate with incorrect predictions
        """
        return np.corrcoef(uncertainties, 1 - correctness.astype(float))[0, 1]
    
    @staticmethod
    def compute_uncertainty_rejection_curve(uncertainties: np.ndarray, 
                                          correctness: np.ndarray, 
                                          n_points: int = 20) -> Dict[str, np.ndarray]:
        """
        Compute accuracy vs rejection rate curve
        Shows how accuracy improves when rejecting high-uncertainty predictions
        """
        thresholds = np.linspace(0, 1, n_points)
        accuracies = []
        rejection_rates = []
        
        for threshold in thresholds:
            # Keep predictions with uncertainty below threshold
            keep_mask = uncertainties <= threshold
            
            if keep_mask.sum() == 0:
                accuracies.append(0.0)
                rejection_rates.append(1.0)
            else:
                accuracy = correctness[keep_mask].mean()
                rejection_rate = 1 - keep_mask.mean()
                
                accuracies.append(accuracy)
                rejection_rates.append(rejection_rate)
        
        return {
            'thresholds': thresholds,
            'accuracies': np.array(accuracies),
            'rejection_rates': np.array(rejection_rates)
        }
    
    @staticmethod
    def compute_area_under_rejection_curve(uncertainties: np.ndarray, 
                                         correctness: np.ndarray) -> float:
        """Compute area under the uncertainty-rejection curve"""
        curve_data = UncertaintyMetrics.compute_uncertainty_rejection_curve(
            uncertainties, correctness
        )
        
        # Compute area using trapezoidal rule
        return np.trapz(curve_data['accuracies'], curve_data['rejection_rates'])

class CalibrationMetrics:
    """Metrics for evaluating uncertainty calibration quality"""
    
    @staticmethod
    def compute_expected_calibration_error(confidences: np.ndarray, 
                                         correctness: np.ndarray, 
                                         n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        Measures the difference between predicted confidence and actual accuracy
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correctness[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def compute_maximum_calibration_error(confidences: np.ndarray, 
                                        correctness: np.ndarray, 
                                        n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = correctness[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    @staticmethod
    def compute_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Brier Score - measures the mean squared difference between 
        predicted probabilities and actual outcomes
        """
        if probabilities.ndim > 1:
            # Multi-class case
            n_classes = probabilities.shape[1]
            targets_one_hot = np.eye(n_classes)[targets]
            return np.mean(np.sum((probabilities - targets_one_hot) ** 2, axis=1))
        else:
            # Binary case
            return np.mean((probabilities - targets) ** 2)
    
    @staticmethod
    def generate_reliability_diagram(confidences: np.ndarray, 
                                   correctness: np.ndarray, 
                                   n_bins: int = 10) -> Dict:
        """Generate data for reliability diagram"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(correctness[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        return {
            'bin_centers': np.array(bin_centers),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts)
        }

class ComprehensiveEvaluator:
    """Main evaluation class that combines all metrics"""
    
    def __init__(self):
        self.detection_metrics = DetectionMetrics()
        self.uncertainty_metrics = UncertaintyMetrics()
        self.calibration_metrics = CalibrationMetrics()
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prob: np.ndarray, uncertainties: np.ndarray, 
                      confidences: np.ndarray) -> EvaluationResults:
        """
        Comprehensive evaluation of uncertainty-aware IDS model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            uncertainties: Uncertainty estimates
            confidences: Confidence scores
        """
        
        # Compute correctness for uncertainty evaluation
        correctness = (y_true == y_pred).astype(float)
        
        # Detection performance metrics
        detection_results = self.detection_metrics.compute_all_metrics(
            y_true, y_pred, y_prob
        )
        
        # Uncertainty quality metrics
        uncertainty_results = {
            'uncertainty_accuracy_correlation': 
                self.uncertainty_metrics.compute_uncertainty_accuracy_correlation(
                    uncertainties, correctness
                ),
            'area_under_rejection_curve': 
                self.uncertainty_metrics.compute_area_under_rejection_curve(
                    uncertainties, correctness
                )
        }
        
        # Calibration metrics
        calibration_results = {
            'expected_calibration_error': 
                self.calibration_metrics.compute_expected_calibration_error(
                    confidences, correctness
                ),
            'maximum_calibration_error': 
                self.calibration_metrics.compute_maximum_calibration_error(
                    confidences, correctness
                ),
            'brier_score': 
                self.calibration_metrics.compute_brier_score(y_prob, y_true)
        }
        
        # Generate additional diagnostic data
        reliability_data = self.calibration_metrics.generate_reliability_diagram(
            confidences, correctness
        )
        
        rejection_curve_data = self.uncertainty_metrics.compute_uncertainty_rejection_curve(
            uncertainties, correctness
        )
        
        # ROC and PR curve data
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        )
        
        roc_data = {'fpr': fpr, 'tpr': tpr}
        pr_data = {'precision': precision_curve, 'recall': recall_curve}
        
        return EvaluationResults(
            detection_metrics=detection_results,
            uncertainty_metrics=uncertainty_results,
            calibration_metrics=calibration_results,
            reliability_diagram=reliability_data,
            roc_data=roc_data,
            pr_data=pr_data
        )
    
    def plot_evaluation_results(self, results: EvaluationResults, 
                              save_path: Optional[str] = None):
        """Generate comprehensive evaluation plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Reliability Diagram
        rel_data = results.reliability_diagram
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        axes[0, 0].bar(rel_data['bin_centers'], rel_data['bin_accuracies'], 
                      width=0.08, alpha=0.7, label='Accuracy')
        axes[0, 0].plot(rel_data['bin_centers'], rel_data['bin_confidences'], 
                       'ro-', label='Confidence')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Reliability Diagram')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curve
        roc_data = results.roc_data
        auc_score = results.detection_metrics['auc_roc']
        axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'], 
                       label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        pr_data = results.pr_data
        axes[0, 2].plot(pr_data['recall'], pr_data['precision'])
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Detection Metrics Bar Plot
        det_metrics = results.detection_metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
        metric_values = [det_metrics[key] for key in key_metrics]
        
        axes[1, 0].bar(key_metrics, metric_values, alpha=0.7)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Detection Performance Metrics')
        axes[1, 0].set_ylim(0, 1)
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
        
        # 5. Calibration Metrics
        cal_metrics = results.calibration_metrics
        cal_names = ['ECE', 'MCE', 'Brier Score']
        cal_values = [
            cal_metrics['expected_calibration_error'],
            cal_metrics['maximum_calibration_error'],
            cal_metrics['brier_score']
        ]
        
        axes[1, 1].bar(cal_names, cal_values, alpha=0.7, color='orange')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Calibration Metrics (Lower is Better)')
        
        # 6. Uncertainty Metrics
        unc_metrics = results.uncertainty_metrics
        unc_names = ['Uncertainty-Accuracy\nCorrelation', 'Area Under\nRejection Curve']
        unc_values = [
            unc_metrics['uncertainty_accuracy_correlation'],
            unc_metrics['area_under_rejection_curve']
        ]
        
        axes[1, 2].bar(unc_names, unc_values, alpha=0.7, color='green')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Uncertainty Quality Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, results: EvaluationResults) -> str:
        """Generate a comprehensive text report"""
        
        report = "=" * 60 + "\n"
        report += "UNCERTAINTY-AWARE INTRUSION DETECTION EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Detection Performance
        report += "DETECTION PERFORMANCE METRICS:\n"
        report += "-" * 30 + "\n"
        det_metrics = results.detection_metrics
        report += f"Accuracy: {det_metrics['accuracy']:.4f}\n"
        report += f"Precision: {det_metrics['precision']:.4f}\n"
        report += f"Recall (Detection Rate): {det_metrics['recall']:.4f}\n"
        report += f"F1-Score: {det_metrics['f1_score']:.4f}\n"
        report += f"AUC-ROC: {det_metrics['auc_roc']:.4f}\n"
        report += f"False Positive Rate: {det_metrics['false_positive_rate']:.4f}\n"
        report += f"Balanced Accuracy: {det_metrics['balanced_accuracy']:.4f}\n\n"
        
        # Confusion Matrix
        report += "CONFUSION MATRIX:\n"
        report += "-" * 15 + "\n"
        report += f"True Positives: {det_metrics['true_positives']}\n"
        report += f"False Positives: {det_metrics['false_positives']}\n"
        report += f"True Negatives: {det_metrics['true_negatives']}\n"
        report += f"False Negatives: {det_metrics['false_negatives']}\n\n"
        
        # Uncertainty Quality
        report += "UNCERTAINTY QUALITY METRICS:\n"
        report += "-" * 30 + "\n"
        unc_metrics = results.uncertainty_metrics
        report += f"Uncertainty-Accuracy Correlation: {unc_metrics['uncertainty_accuracy_correlation']:.4f}\n"
        report += f"Area Under Rejection Curve: {unc_metrics['area_under_rejection_curve']:.4f}\n\n"
        
        # Calibration Quality
        report += "CALIBRATION QUALITY METRICS:\n"
        report += "-" * 30 + "\n"
        cal_metrics = results.calibration_metrics
        report += f"Expected Calibration Error: {cal_metrics['expected_calibration_error']:.4f}\n"
        report += f"Maximum Calibration Error: {cal_metrics['maximum_calibration_error']:.4f}\n"
        report += f"Brier Score: {cal_metrics['brier_score']:.4f}\n\n"
        
        # Interpretation
        report += "INTERPRETATION:\n"
        report += "-" * 15 + "\n"
        
        if cal_metrics['expected_calibration_error'] < 0.05:
            report += "✓ Model is well-calibrated (ECE < 0.05)\n"
        elif cal_metrics['expected_calibration_error'] < 0.10:
            report += "⚠ Model calibration is acceptable (ECE < 0.10)\n"
        else:
            report += "✗ Model calibration needs improvement (ECE > 0.10)\n"
        
        if unc_metrics['uncertainty_accuracy_correlation'] > 0.3:
            report += "✓ Uncertainty estimates are informative (correlation > 0.3)\n"
        else:
            report += "⚠ Uncertainty estimates may need improvement\n"
        
        if det_metrics['false_positive_rate'] < 0.01:
            report += "✓ Low false positive rate suitable for production\n"
        elif det_metrics['false_positive_rate'] < 0.05:
            report += "⚠ Moderate false positive rate - monitor in production\n"
        else:
            report += "✗ High false positive rate - not suitable for production\n"
        
        return report

# Example usage
if __name__ == "__main__":
    # Generate sample evaluation data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.binomial(1, 0.1, n_samples)  # 10% attack rate
    y_pred = np.random.binomial(1, 0.12, n_samples)  # Slightly higher prediction rate
    y_prob = np.random.beta(2, 8, (n_samples, 2))  # Sample probabilities
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    uncertainties = np.random.beta(2, 5, n_samples)  # Sample uncertainties
    confidences = 1 - uncertainties  # Simple confidence calculation
    
    # Evaluate
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_model(y_true, y_pred, y_prob, uncertainties, confidences)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    # Plot results
    evaluator.plot_evaluation_results(results)
