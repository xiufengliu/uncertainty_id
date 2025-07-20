"""
Evaluation module for Uncertainty-Aware Intrusion Detection System.

This module provides comprehensive evaluation capabilities including:
- Detection performance metrics
- Uncertainty quality assessment
- Calibration evaluation
- Visualization tools
- Benchmark comparisons
"""

from .metrics import (
    DetectionMetrics, UncertaintyMetrics, CalibrationMetrics,
    ComprehensiveEvaluator
)
from .visualizer import (
    UncertaintyVisualizer, CalibrationPlotter, PerformancePlotter
)
from .benchmarks import BenchmarkSuite, ModelComparator
from .reports import EvaluationReporter, ResultsAnalyzer

__all__ = [
    # Metrics
    'DetectionMetrics',
    'UncertaintyMetrics', 
    'CalibrationMetrics',
    'ComprehensiveEvaluator',
    
    # Visualization
    'UncertaintyVisualizer',
    'CalibrationPlotter',
    'PerformancePlotter',
    
    # Benchmarking
    'BenchmarkSuite',
    'ModelComparator',
    
    # Reporting
    'EvaluationReporter',
    'ResultsAnalyzer',
]
