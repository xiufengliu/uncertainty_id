#!/usr/bin/env python3
"""
Experimental runner for reproducing research results.

This script runs comprehensive experiments to reproduce the results
from the research paper and compare different model configurations.
"""

import sys
import os
import logging
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncertainty_ids import BayesianEnsembleIDS, SingleLayerTransformerIDS
from uncertainty_ids.data import NetworkDataProcessor, SyntheticIDSDataset, create_data_loaders
from uncertainty_ids.training import UncertaintyIDSTrainer, TrainingConfig
from uncertainty_ids.evaluation import ComprehensiveEvaluator
from uncertainty_ids.utils import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs comprehensive experiments for research reproducibility.
    """
    
    def __init__(self, output_dir: str = './experiment_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.evaluator = ComprehensiveEvaluator()
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def run_ensemble_size_experiment(self, dataset, ensemble_sizes: List[int] = [1, 3, 5, 10, 15, 20]):
        """
        Experiment: Effect of ensemble size on performance and uncertainty quality.
        """
        logger.info("🔬 Running ensemble size experiment...")
        
        results = {}
        
        for ensemble_size in ensemble_sizes:
            logger.info(f"Training with ensemble size: {ensemble_size}")
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15
            )
            
            # Configure model
            model_config = {
                'n_ensemble': ensemble_size,
                'd_model': 64,  # Smaller for faster experiments
                'max_seq_len': 20,
                'n_classes': 2,
                'dropout_rate': 0.1
            }
            
            training_config = TrainingConfig(
                model_type='bayesian_ensemble',
                model_params=model_config,
                batch_size=32,
                n_epochs=20,  # Fewer epochs for experiments
                learning_rate=1e-3,
                early_stopping_patience=5,
                log_every=5,
                checkpoint_dir=str(self.output_dir / f'ensemble_{ensemble_size}')
            )
            
            # Train model
            trainer = UncertaintyIDSTrainer(training_config)
            history = trainer.train(train_loader, val_loader)
            
            # Evaluate
            evaluation_results = trainer.evaluate(test_loader)
            
            # Store results
            results[ensemble_size] = {
                'detection_metrics': evaluation_results.detection_metrics,
                'uncertainty_metrics': evaluation_results.uncertainty_metrics,
                'calibration_metrics': evaluation_results.calibration_metrics,
                'training_time': sum(history.get('epoch_times', [0])),
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
            }
            
            logger.info(f"Ensemble {ensemble_size} - Accuracy: {evaluation_results.detection_metrics['accuracy']:.4f}")
        
        # Save results
        with open(self.output_dir / 'ensemble_size_experiment.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.results['ensemble_size'] = results
        return results
    
    def run_model_dimension_experiment(self, dataset, model_dims: List[int] = [32, 64, 128, 256]):
        """
        Experiment: Effect of model dimension on performance.
        """
        logger.info("🔬 Running model dimension experiment...")
        
        results = {}
        
        for d_model in model_dims:
            logger.info(f"Training with model dimension: {d_model}")
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15
            )
            
            # Configure model
            model_config = {
                'n_ensemble': 5,  # Fixed ensemble size
                'd_model': d_model,
                'max_seq_len': 20,
                'n_classes': 2,
                'dropout_rate': 0.1
            }
            
            training_config = TrainingConfig(
                model_type='bayesian_ensemble',
                model_params=model_config,
                batch_size=32,
                n_epochs=20,
                learning_rate=1e-3,
                early_stopping_patience=5,
                checkpoint_dir=str(self.output_dir / f'dim_{d_model}')
            )
            
            # Train model
            trainer = UncertaintyIDSTrainer(training_config)
            history = trainer.train(train_loader, val_loader)
            
            # Evaluate
            evaluation_results = trainer.evaluate(test_loader)
            
            # Count parameters
            param_count = sum(p.numel() for p in trainer.model.parameters())
            
            # Store results
            results[d_model] = {
                'detection_metrics': evaluation_results.detection_metrics,
                'uncertainty_metrics': evaluation_results.uncertainty_metrics,
                'calibration_metrics': evaluation_results.calibration_metrics,
                'parameter_count': param_count,
                'training_time': sum(history.get('epoch_times', [0])),
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
            }
            
            logger.info(f"Dim {d_model} - Accuracy: {evaluation_results.detection_metrics['accuracy']:.4f}, Params: {param_count:,}")
        
        # Save results
        with open(self.output_dir / 'model_dimension_experiment.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.results['model_dimension'] = results
        return results
    
    def run_uncertainty_calibration_experiment(self, dataset, calibration_methods: List[str] = ['temperature', 'platt', 'isotonic']):
        """
        Experiment: Comparison of uncertainty calibration methods.
        """
        logger.info("🔬 Running uncertainty calibration experiment...")
        
        results = {}
        
        for method in calibration_methods:
            logger.info(f"Training with calibration method: {method}")
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15
            )
            
            # Configure model
            model_config = {
                'n_ensemble': 5,
                'd_model': 64,
                'max_seq_len': 20,
                'n_classes': 2,
                'dropout_rate': 0.1
            }
            
            training_config = TrainingConfig(
                model_type='bayesian_ensemble',
                model_params=model_config,
                batch_size=32,
                n_epochs=20,
                learning_rate=1e-3,
                calibrate_uncertainty=True,
                calibration_method=method,
                checkpoint_dir=str(self.output_dir / f'calib_{method}')
            )
            
            # Train model
            trainer = UncertaintyIDSTrainer(training_config)
            history = trainer.train(train_loader, val_loader)
            
            # Evaluate
            evaluation_results = trainer.evaluate(test_loader)
            
            # Store results
            results[method] = {
                'detection_metrics': evaluation_results.detection_metrics,
                'uncertainty_metrics': evaluation_results.uncertainty_metrics,
                'calibration_metrics': evaluation_results.calibration_metrics,
                'training_time': sum(history.get('epoch_times', [0])),
            }
            
            logger.info(f"Calibration {method} - ECE: {evaluation_results.calibration_metrics.get('expected_calibration_error', 'N/A')}")
        
        # Save results
        with open(self.output_dir / 'calibration_experiment.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.results['calibration'] = results
        return results
    
    def run_baseline_comparison(self, dataset):
        """
        Experiment: Compare with baseline methods (single model, no uncertainty).
        """
        logger.info("🔬 Running baseline comparison...")
        
        results = {}
        
        # Baseline 1: Single transformer (no ensemble)
        logger.info("Training single transformer baseline...")
        
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15
        )
        
        single_config = TrainingConfig(
            model_type='single_transformer',
            model_params={
                'd_model': 64,
                'max_seq_len': 20,
                'n_classes': 2,
                'dropout_rate': 0.1
            },
            batch_size=32,
            n_epochs=20,
            learning_rate=1e-3,
            calibrate_uncertainty=False,
            checkpoint_dir=str(self.output_dir / 'single_baseline')
        )
        
        trainer = UncertaintyIDSTrainer(single_config)
        history = trainer.train(train_loader, val_loader)
        evaluation_results = trainer.evaluate(test_loader)
        
        results['single_transformer'] = {
            'detection_metrics': evaluation_results.detection_metrics,
            'uncertainty_metrics': evaluation_results.uncertainty_metrics,
            'calibration_metrics': evaluation_results.calibration_metrics,
            'training_time': sum(history.get('epoch_times', [0])),
        }
        
        # Baseline 2: Bayesian ensemble (our method)
        logger.info("Training Bayesian ensemble...")
        
        ensemble_config = TrainingConfig(
            model_type='bayesian_ensemble',
            model_params={
                'n_ensemble': 10,
                'd_model': 64,
                'max_seq_len': 20,
                'n_classes': 2,
                'dropout_rate': 0.1
            },
            batch_size=32,
            n_epochs=20,
            learning_rate=1e-3,
            calibrate_uncertainty=True,
            checkpoint_dir=str(self.output_dir / 'ensemble_baseline')
        )
        
        trainer = UncertaintyIDSTrainer(ensemble_config)
        history = trainer.train(train_loader, val_loader)
        evaluation_results = trainer.evaluate(test_loader)
        
        results['bayesian_ensemble'] = {
            'detection_metrics': evaluation_results.detection_metrics,
            'uncertainty_metrics': evaluation_results.uncertainty_metrics,
            'calibration_metrics': evaluation_results.calibration_metrics,
            'training_time': sum(history.get('epoch_times', [0])),
        }
        
        # Save results
        with open(self.output_dir / 'baseline_comparison.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.results['baseline_comparison'] = results
        return results
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report of all experiments."""
        logger.info("📊 Generating summary report...")
        
        report = {
            'experiment_date': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': self.results
        }
        
        # Ensemble size summary
        if 'ensemble_size' in self.results:
            ensemble_results = self.results['ensemble_size']
            best_ensemble_size = max(
                ensemble_results.keys(),
                key=lambda k: ensemble_results[k]['detection_metrics']['f1_score']
            )
            
            report['summary']['ensemble_size'] = {
                'best_size': best_ensemble_size,
                'best_f1_score': ensemble_results[best_ensemble_size]['detection_metrics']['f1_score'],
                'performance_trend': 'Analyzed ensemble sizes from 1 to 20'
            }
        
        # Model dimension summary
        if 'model_dimension' in self.results:
            dim_results = self.results['model_dimension']
            best_dim = max(
                dim_results.keys(),
                key=lambda k: dim_results[k]['detection_metrics']['f1_score']
            )
            
            report['summary']['model_dimension'] = {
                'best_dimension': best_dim,
                'best_f1_score': dim_results[best_dim]['detection_metrics']['f1_score'],
                'parameter_count': dim_results[best_dim]['parameter_count']
            }
        
        # Calibration summary
        if 'calibration' in self.results:
            calib_results = self.results['calibration']
            best_method = min(
                calib_results.keys(),
                key=lambda k: calib_results[k]['calibration_metrics'].get('expected_calibration_error', float('inf'))
            )
            
            report['summary']['calibration'] = {
                'best_method': best_method,
                'best_ece': calib_results[best_method]['calibration_metrics'].get('expected_calibration_error', 'N/A')
            }
        
        # Baseline comparison summary
        if 'baseline_comparison' in self.results:
            baseline_results = self.results['baseline_comparison']
            
            report['summary']['baseline_comparison'] = {
                'single_transformer_f1': baseline_results['single_transformer']['detection_metrics']['f1_score'],
                'bayesian_ensemble_f1': baseline_results['bayesian_ensemble']['detection_metrics']['f1_score'],
                'improvement': baseline_results['bayesian_ensemble']['detection_metrics']['f1_score'] - 
                              baseline_results['single_transformer']['detection_metrics']['f1_score']
            }
        
        # Save report
        with open(self.output_dir / 'experiment_summary.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        logger.info(f"📋 Summary report saved to: {self.output_dir}")
        return report
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate a markdown version of the report."""
        md_content = f"""# Uncertainty-Aware IDS Experiment Results

**Experiment Date:** {report['experiment_date']}

## Summary

"""
        
        if 'ensemble_size' in report['summary']:
            es = report['summary']['ensemble_size']
            md_content += f"""### Ensemble Size Experiment
- **Best ensemble size:** {es['best_size']}
- **Best F1-score:** {es['best_f1_score']:.4f}
- **Analysis:** {es['performance_trend']}

"""
        
        if 'model_dimension' in report['summary']:
            md = report['summary']['model_dimension']
            md_content += f"""### Model Dimension Experiment
- **Best dimension:** {md['best_dimension']}
- **Best F1-score:** {md['best_f1_score']:.4f}
- **Parameter count:** {md['parameter_count']:,}

"""
        
        if 'calibration' in report['summary']:
            cal = report['summary']['calibration']
            md_content += f"""### Calibration Method Experiment
- **Best method:** {cal['best_method']}
- **Best ECE:** {cal['best_ece']}

"""
        
        if 'baseline_comparison' in report['summary']:
            bc = report['summary']['baseline_comparison']
            md_content += f"""### Baseline Comparison
- **Single Transformer F1:** {bc['single_transformer_f1']:.4f}
- **Bayesian Ensemble F1:** {bc['bayesian_ensemble_f1']:.4f}
- **Improvement:** {bc['improvement']:.4f}

"""
        
        md_content += """## Detailed Results

See `experiment_summary.json` for complete numerical results.

## Reproducibility

All experiments were run with:
- Random seed: 42
- PyTorch version: Latest
- Hardware: [Add your hardware details]

## Next Steps

1. Run experiments on real datasets (NSL-KDD, CICIDS2017, UNSW-NB15)
2. Compare with additional baseline methods
3. Analyze computational efficiency
4. Investigate uncertainty quality in more detail
"""
        
        with open(self.output_dir / 'experiment_report.md', 'w') as f:
            f.write(md_content)


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive experiments')
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                       help='Output directory for results')
    parser.add_argument('--dataset-size', type=int, default=5000,
                       help='Size of synthetic dataset')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['ensemble_size', 'model_dimension', 'calibration', 'baseline'],
                       default=['ensemble_size', 'model_dimension', 'calibration', 'baseline'],
                       help='Experiments to run')
    
    args = parser.parse_args()
    
    print("🧪 Starting Comprehensive Experiments")
    print("=" * 50)
    
    # Create synthetic dataset
    logger.info(f"Creating synthetic dataset with {args.dataset_size} samples...")
    dataset = SyntheticIDSDataset.create_synthetic(
        n_samples=args.dataset_size,
        sequence_length=20,
        attack_rate=0.15,
        random_state=42
    )
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.output_dir)
    
    # Run selected experiments
    if 'ensemble_size' in args.experiments:
        runner.run_ensemble_size_experiment(dataset)
    
    if 'model_dimension' in args.experiments:
        runner.run_model_dimension_experiment(dataset)
    
    if 'calibration' in args.experiments:
        runner.run_uncertainty_calibration_experiment(dataset)
    
    if 'baseline' in args.experiments:
        runner.run_baseline_comparison(dataset)
    
    # Generate summary report
    runner.generate_summary_report()
    
    print("🎉 All experiments completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
