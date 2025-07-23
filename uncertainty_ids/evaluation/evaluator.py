"""
Model evaluator for comprehensive evaluation.
Based on the experimental evaluation in Section 4 of the paper.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from pathlib import Path

from ..models.transformer import BayesianEnsembleTransformer
from ..models.uncertainty import UncertaintyQuantifier
from .metrics import ClassificationMetrics, UncertaintyMetrics, CalibrationMetrics, RobustnessMetrics
from ..utils.visualization import (
    plot_uncertainty_distribution, plot_calibration_diagram,
    plot_confusion_matrix, save_figure
)


class ModelEvaluator:
    """
    Comprehensive model evaluator for uncertainty-aware intrusion detection.
    
    Implements all evaluation metrics reported in the paper:
    - Classification performance (accuracy, F1-score, FPR)
    - Uncertainty quality (correlation, mutual information, AURC)
    - Calibration quality (ECE, reliability diagrams)
    - Robustness analysis
    """
    
    def __init__(
        self,
        model: BayesianEnsembleTransformer,
        uncertainty_quantifier: UncertaintyQuantifier,
        device: torch.device
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Bayesian ensemble transformer
            uncertainty_quantifier: Uncertainty quantification module
            device: Evaluation device
        """
        self.model = model.to(device)
        self.uncertainty_quantifier = uncertainty_quantifier.to(device)
        self.device = device
        
        # Metrics calculators
        self.classification_metrics = ClassificationMetrics()
        self.uncertainty_metrics = UncertaintyMetrics()
        self.calibration_metrics = CalibrationMetrics()
        self.robustness_metrics = RobustnessMetrics()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def evaluate_dataset(
        self,
        test_loader: DataLoader,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on a dataset.
        
        Args:
            test_loader: Test data loader
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary of all evaluation metrics
        """
        self.logger.info(f"Evaluating on {dataset_name} dataset...")
        
        self.model.eval()
        self.uncertainty_quantifier.eval()
        
        # Collect predictions and uncertainties
        all_predictions = []
        all_labels = []
        all_probs = []
        all_epistemic_unc = []
        all_aleatoric_unc = []
        all_total_unc = []
        
        with torch.no_grad():
            for cont_features, cat_features, labels in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
                # Move to device
                cont_features = cont_features.to(self.device)
                cat_features = cat_features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                ensemble_logits, _, individual_logits = self.model(
                    cont_features, cat_features, return_individual=True
                )
                
                # Uncertainty quantification
                predictions, epistemic_unc, aleatoric_unc, total_unc, ensemble_probs = \
                    self.uncertainty_quantifier(ensemble_logits, individual_logits)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(ensemble_probs.cpu().numpy())
                all_epistemic_unc.extend(epistemic_unc.cpu().numpy())
                all_aleatoric_unc.extend(aleatoric_unc.cpu().numpy())
                all_total_unc.extend(total_unc.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_epistemic_unc = np.array(all_epistemic_unc)
        all_aleatoric_unc = np.array(all_aleatoric_unc)
        all_total_unc = np.array(all_total_unc)
        
        # Get confidence scores (max probability)
        all_confidences = np.max(all_probs, axis=1)
        all_pos_probs = all_probs[:, 1]  # Positive class probabilities
        
        # Compute correctness
        correctness = (all_predictions == all_labels).astype(float)
        
        # 1. Classification Metrics
        classification_results = self.classification_metrics.compute_metrics(
            all_labels, all_predictions, all_pos_probs
        )
        
        # 2. Uncertainty Quality Metrics
        uncertainty_results = {
            'uncertainty_accuracy_correlation': self.uncertainty_metrics.uncertainty_accuracy_correlation(
                all_total_unc, correctness
            ),
            'mutual_information': self.uncertainty_metrics.mutual_information(
                all_total_unc, correctness
            ),
            'aurc': self.uncertainty_metrics.area_under_rejection_curve(
                all_total_unc, correctness
            )
        }
        
        # 3. Calibration Metrics
        calibration_results = {
            'ece': self.calibration_metrics.expected_calibration_error(
                all_confidences, correctness
            ),
            'brier_score': self.calibration_metrics.brier_score(
                all_pos_probs, all_labels
            )
        }
        
        # 4. Uncertainty Decomposition Analysis
        uncertainty_decomposition = {
            'mean_epistemic_uncertainty': np.mean(all_epistemic_unc),
            'std_epistemic_uncertainty': np.std(all_epistemic_unc),
            'mean_aleatoric_uncertainty': np.mean(all_aleatoric_unc),
            'std_aleatoric_uncertainty': np.std(all_aleatoric_unc),
            'mean_total_uncertainty': np.mean(all_total_unc),
            'std_total_uncertainty': np.std(all_total_unc)
        }
        
        # Combine all results
        results = {
            **classification_results,
            **uncertainty_results,
            **calibration_results,
            **uncertainty_decomposition
        }
        
        # Log key results
        self.logger.info(f"{dataset_name} Results:")
        self.logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        self.logger.info(f"  F1-Score: {results['f1_score']:.4f}")
        self.logger.info(f"  FPR: {results['fpr']:.4f}")
        self.logger.info(f"  ECE: {results['ece']:.4f}")
        self.logger.info(f"  AURC: {results['aurc']:.4f}")
        self.logger.info(f"  Uncertainty-Accuracy Correlation: {results['uncertainty_accuracy_correlation']:.4f}")
        
        return results
    
    def evaluate_icl_performance(
        self,
        icl_loader,
        test_families: Dict[str, Dict[str, torch.Tensor]],
        k_shots: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate In-Context Learning performance.
        
        Based on Section 4.3 ICL evaluation in the paper.
        
        Args:
            icl_loader: ICL data loader
            test_families: Meta-test attack families
            k_shots: List of k-shot settings to evaluate
            
        Returns:
            ICL performance results
        """
        self.logger.info("Evaluating ICL performance...")
        
        from ..models.icl import ICLEnabledTransformer
        
        # Convert model to ICL-enabled if needed
        if not isinstance(self.model, ICLEnabledTransformer):
            self.logger.warning("Model is not ICL-enabled. Creating ICL wrapper...")
            # This would require implementing an ICL wrapper
            return {}
        
        self.model.eval()
        
        icl_results = {}
        
        for k in k_shots:
            self.logger.info(f"Evaluating {k}-shot ICL...")
            
            family_results = {}
            
            for family_name, family_data in test_families.items():
                # Sample episodes for this family
                n_episodes = 10  # Number of episodes to average over
                episode_accuracies = []
                
                for episode in range(n_episodes):
                    # Sample k context examples and queries
                    total_samples = len(family_data['cont'])
                    if total_samples < k + 5:  # Need at least k+5 samples
                        continue
                    
                    indices = torch.randperm(total_samples)
                    context_indices = indices[:k]
                    query_indices = indices[k:k+5]  # 5 query examples
                    
                    context_cont = family_data['cont'][context_indices].unsqueeze(0)
                    context_cat = family_data['cat'][context_indices].unsqueeze(0)
                    context_labels = family_data['labels'][context_indices].unsqueeze(0)
                    
                    query_cont = family_data['cont'][query_indices]
                    query_cat = family_data['cat'][query_indices]
                    query_labels = family_data['labels'][query_indices]
                    
                    # ICL predictions
                    episode_predictions = []
                    with torch.no_grad():
                        for q_idx in range(len(query_indices)):
                            logits, _ = self.model(
                                context_cont.to(self.device),
                                context_cat.to(self.device),
                                context_labels.to(self.device),
                                query_cont[q_idx:q_idx+1].to(self.device),
                                query_cat[q_idx:q_idx+1].to(self.device)
                            )
                            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                            episode_predictions.append(pred)
                    
                    # Compute episode accuracy
                    episode_accuracy = (np.array(episode_predictions) == query_labels.numpy()).mean()
                    episode_accuracies.append(episode_accuracy)
                
                if episode_accuracies:
                    family_results[family_name] = {
                        'accuracy': np.mean(episode_accuracies),
                        'std': np.std(episode_accuracies)
                    }
            
            icl_results[f'{k}_shot'] = family_results
        
        return icl_results
    
    def compare_with_baselines(
        self,
        test_loader: DataLoader,
        baseline_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model performance with baseline methods.
        
        Args:
            test_loader: Test data loader
            baseline_results: Dictionary of baseline method results
            
        Returns:
            Comparison results
        """
        # Evaluate our model
        our_results = self.evaluate_dataset(test_loader, "Comparison")
        
        # Create comparison table
        comparison = {
            'Ours (Bayesian Ensemble Transformer)': our_results,
            **baseline_results
        }
        
        # Log comparison
        self.logger.info("Baseline Comparison:")
        for method, results in comparison.items():
            self.logger.info(f"  {method}:")
            self.logger.info(f"    Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            self.logger.info(f"    F1-Score: {results.get('f1_score', 'N/A'):.4f}")
            self.logger.info(f"    ECE: {results.get('ece', 'N/A'):.4f}")
        
        return comparison

    def generate_visualizations(self,
                              test_loader: DataLoader,
                              output_dir: str = "results/figures",
                              dataset_name: str = "dataset") -> None:
        """
        Generate comprehensive visualizations for the evaluation results.
        All plots are saved in PDF format for publication quality.

        Args:
            test_loader: Test data loader
            output_dir: Directory to save visualizations
            dataset_name: Name of the dataset for plot titles
        """
        self.logger.info("Generating visualizations...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get predictions and uncertainties
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collecting predictions for visualization"):
                x_cont, x_cat, labels = batch
                x_cont = x_cont.to(self.device)
                x_cat = x_cat.to(self.device)
                labels = labels.to(self.device)

                # Get ensemble predictions
                ensemble_logits, _, individual_preds = self.model(x_cont, x_cat, return_individual=True)

                # Calculate uncertainties
                uncertainties = self.uncertainty_quantifier.calculate_uncertainty(
                    individual_preds, ensemble_logits
                )

                # Get probabilities and predictions
                probabilities = torch.softmax(ensemble_logits, dim=-1)
                predictions = torch.argmax(ensemble_logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(uncertainties['total_uncertainty'].cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_uncertainties = np.array(all_uncertainties)
        all_probabilities = np.array(all_probabilities)

        # Calculate correctness
        correctness = (all_predictions == all_labels).astype(int)

        # Generate visualizations

        # 1. Uncertainty distribution
        plot_uncertainty_distribution(
            all_uncertainties,
            correctness,
            output_path / f"{dataset_name}_uncertainty_distribution.pdf"
        )

        # 2. Calibration diagram
        plot_calibration_diagram(
            all_labels,
            all_probabilities,
            output_path=output_path / f"{dataset_name}_calibration_diagram.pdf"
        )

        # 3. Confusion matrix
        class_names = ['Normal', 'Attack'] if len(np.unique(all_labels)) == 2 else None
        plot_confusion_matrix(
            all_labels,
            all_predictions,
            class_names=class_names,
            output_path=output_path / f"{dataset_name}_confusion_matrix.pdf"
        )

        self.logger.info(f"Visualizations saved to {output_path}")

        # Return statistics for further analysis
        return {
            'uncertainty_stats': {
                'mean_uncertainty_correct': np.mean(all_uncertainties[correctness == 1]),
                'mean_uncertainty_incorrect': np.mean(all_uncertainties[correctness == 0]),
                'std_uncertainty_correct': np.std(all_uncertainties[correctness == 1]),
                'std_uncertainty_incorrect': np.std(all_uncertainties[correctness == 0])
            },
            'calibration_stats': {
                'mean_confidence': np.mean(all_probabilities),
                'std_confidence': np.std(all_probabilities)
            }
        }
