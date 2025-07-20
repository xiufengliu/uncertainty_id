"""
Monitoring and metrics collection for the API server.

This module provides comprehensive monitoring capabilities including
performance tracking, metrics collection, and alerting.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import numpy as np
import logging
from datetime import datetime, timedelta
import psutil
import os

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks API performance metrics including response times,
    throughput, and resource usage.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.request_times = deque(maxlen=window_size)
        self.request_timestamps = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        # Counters
        self.total_requests = 0
        self.error_count = 0
        
        # Start time
        self.start_time = time.time()
    
    def record_request_time(self, duration: float):
        """Record a request processing time."""
        with self.lock:
            self.request_times.append(duration)
            self.request_timestamps.append(time.time())
            self.total_requests += 1
    
    def record_error(self):
        """Record an error occurrence."""
        with self.lock:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        with self.lock:
            if not self.request_times:
                return {
                    'avg_time_ms': 0.0,
                    'p50_time_ms': 0.0,
                    'p95_time_ms': 0.0,
                    'p99_time_ms': 0.0,
                    'requests_per_second': 0.0,
                    'error_rate': 0.0,
                    'total_requests': self.total_requests,
                    'total_errors': self.error_count
                }
            
            times_ms = [t * 1000 for t in self.request_times]
            
            # Calculate percentiles
            p50 = np.percentile(times_ms, 50)
            p95 = np.percentile(times_ms, 95)
            p99 = np.percentile(times_ms, 99)
            avg_time = np.mean(times_ms)
            
            # Calculate requests per second (last minute)
            current_time = time.time()
            recent_requests = sum(
                1 for ts in self.request_timestamps 
                if current_time - ts <= 60
            )
            rps = recent_requests / 60.0
            
            # Error rate
            error_rate = self.error_count / max(self.total_requests, 1)
            
            return {
                'avg_time_ms': avg_time,
                'p50_time_ms': p50,
                'p95_time_ms': p95,
                'p99_time_ms': p99,
                'requests_per_second': rps,
                'error_rate': error_rate,
                'total_requests': self.total_requests,
                'total_errors': self.error_count
            }
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.request_times.clear()
            self.request_timestamps.clear()
            self.total_requests = 0
            self.error_count = 0
            self.start_time = time.time()


class APIMonitor:
    """
    Comprehensive API monitoring including prediction tracking,
    uncertainty analysis, and system metrics.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize API monitor.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.lock = threading.Lock()
        
        # Prediction tracking
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.uncertainties = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Counters
        self.total_predictions = 0
        self.attack_predictions = 0
        self.high_uncertainty_predictions = 0
        self.review_required_predictions = 0
        
        # Thresholds
        self.uncertainty_threshold = 0.2
        self.confidence_threshold = 0.8
        
        # System monitoring
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
    
    def record_prediction(self, prediction: int, confidence: float, 
                         uncertainty: Optional[float] = None):
        """
        Record a prediction with its metadata.
        
        Args:
            prediction: Predicted class (0=normal, 1=attack)
            confidence: Prediction confidence
            uncertainty: Uncertainty estimate (optional)
        """
        with self.lock:
            self.predictions.append(prediction)
            self.confidences.append(confidence)
            self.timestamps.append(time.time())
            
            if uncertainty is not None:
                self.uncertainties.append(uncertainty)
                
                # Update counters
                if uncertainty > self.uncertainty_threshold:
                    self.high_uncertainty_predictions += 1
                
                if uncertainty > self.uncertainty_threshold or confidence < self.confidence_threshold:
                    self.review_required_predictions += 1
            
            self.total_predictions += 1
            if prediction == 1:
                self.attack_predictions += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics."""
        with self.lock:
            # Basic prediction metrics
            attack_rate = self.attack_predictions / max(self.total_predictions, 1)
            
            # Confidence metrics
            if self.confidences:
                avg_confidence = np.mean(list(self.confidences))
                min_confidence = np.min(list(self.confidences))
                max_confidence = np.max(list(self.confidences))
            else:
                avg_confidence = min_confidence = max_confidence = 0.0
            
            # Uncertainty metrics
            if self.uncertainties:
                avg_uncertainty = np.mean(list(self.uncertainties))
                high_uncertainty_rate = self.high_uncertainty_predictions / max(self.total_predictions, 1)
            else:
                avg_uncertainty = 0.0
                high_uncertainty_rate = 0.0
            
            # Review rate
            review_rate = self.review_required_predictions / max(self.total_predictions, 1)
            
            # Requests per second (last minute)
            current_time = time.time()
            recent_predictions = sum(
                1 for ts in self.timestamps 
                if current_time - ts <= 60
            )
            predictions_per_second = recent_predictions / 60.0
            
            return {
                'total_predictions': self.total_predictions,
                'attack_predictions': self.attack_predictions,
                'attack_rate': attack_rate,
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'avg_uncertainty': avg_uncertainty,
                'high_uncertainty_predictions': self.high_uncertainty_predictions,
                'high_uncertainty_rate': high_uncertainty_rate,
                'review_required_predictions': self.review_required_predictions,
                'review_rate': review_rate,
                'predictions_per_second': predictions_per_second,
                'requests_per_second': predictions_per_second,  # Alias for compatibility
            }
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system resource metrics."""
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # System-wide metrics
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent()
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            
            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'process_memory_percent': memory_percent,
                'process_cpu_percent': cpu_percent,
                'system_memory_percent': system_memory.percent,
                'system_cpu_percent': system_cpu,
                'disk_usage_percent': disk_usage.percent,
                'uptime_seconds': time.time() - self.start_time,
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {}
    
    def get_uncertainty_distribution(self) -> Dict[str, Any]:
        """Get uncertainty distribution statistics."""
        with self.lock:
            if not self.uncertainties:
                return {}
            
            uncertainties = list(self.uncertainties)
            
            return {
                'mean': np.mean(uncertainties),
                'std': np.std(uncertainties),
                'min': np.min(uncertainties),
                'max': np.max(uncertainties),
                'p25': np.percentile(uncertainties, 25),
                'p50': np.percentile(uncertainties, 50),
                'p75': np.percentile(uncertainties, 75),
                'p95': np.percentile(uncertainties, 95),
                'p99': np.percentile(uncertainties, 99),
            }
    
    def get_confidence_distribution(self) -> Dict[str, Any]:
        """Get confidence distribution statistics."""
        with self.lock:
            if not self.confidences:
                return {}
            
            confidences = list(self.confidences)
            
            return {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'p25': np.percentile(confidences, 25),
                'p50': np.percentile(confidences, 50),
                'p75': np.percentile(confidences, 75),
                'p95': np.percentile(confidences, 95),
                'p99': np.percentile(confidences, 99),
            }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        metrics = self.get_metrics()
        system_metrics = self.get_system_metrics()
        
        # High attack rate alert
        if metrics['attack_rate'] > 0.5:
            alerts.append({
                'type': 'high_attack_rate',
                'severity': 'warning',
                'message': f"High attack rate detected: {metrics['attack_rate']:.2%}",
                'value': metrics['attack_rate'],
                'threshold': 0.5
            })
        
        # High uncertainty rate alert
        if metrics['high_uncertainty_rate'] > 0.3:
            alerts.append({
                'type': 'high_uncertainty_rate',
                'severity': 'warning',
                'message': f"High uncertainty rate: {metrics['high_uncertainty_rate']:.2%}",
                'value': metrics['high_uncertainty_rate'],
                'threshold': 0.3
            })
        
        # Low confidence alert
        if metrics['avg_confidence'] < 0.6:
            alerts.append({
                'type': 'low_confidence',
                'severity': 'warning',
                'message': f"Low average confidence: {metrics['avg_confidence']:.3f}",
                'value': metrics['avg_confidence'],
                'threshold': 0.6
            })
        
        # High memory usage alert
        if system_metrics.get('process_memory_percent', 0) > 80:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'critical',
                'message': f"High memory usage: {system_metrics['process_memory_percent']:.1f}%",
                'value': system_metrics['process_memory_percent'],
                'threshold': 80
            })
        
        # High CPU usage alert
        if system_metrics.get('process_cpu_percent', 0) > 90:
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'critical',
                'message': f"High CPU usage: {system_metrics['process_cpu_percent']:.1f}%",
                'value': system_metrics['process_cpu_percent'],
                'threshold': 90
            })
        
        return alerts
    
    def reset(self):
        """Reset all monitoring data."""
        with self.lock:
            self.predictions.clear()
            self.confidences.clear()
            self.uncertainties.clear()
            self.timestamps.clear()
            
            self.total_predictions = 0
            self.attack_predictions = 0
            self.high_uncertainty_predictions = 0
            self.review_required_predictions = 0
            
            self.start_time = time.time()


class AlertManager:
    """
    Manages alerts and notifications for the monitoring system.
    """
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def process_alerts(self, alerts: List[Dict[str, Any]]):
        """Process new alerts and manage alert lifecycle."""
        with self.lock:
            current_time = datetime.utcnow()
            
            # Track current alert types
            current_alert_types = {alert['type'] for alert in alerts}
            
            # Add new alerts
            for alert in alerts:
                alert_type = alert['type']
                
                if alert_type not in self.active_alerts:
                    # New alert
                    alert['first_seen'] = current_time
                    alert['last_seen'] = current_time
                    alert['count'] = 1
                    
                    self.active_alerts[alert_type] = alert
                    self.alert_history.append(alert.copy())
                    
                    logger.warning(f"New alert: {alert['message']}")
                else:
                    # Update existing alert
                    self.active_alerts[alert_type]['last_seen'] = current_time
                    self.active_alerts[alert_type]['count'] += 1
                    self.active_alerts[alert_type]['value'] = alert['value']
            
            # Clear resolved alerts
            resolved_alerts = []
            for alert_type in list(self.active_alerts.keys()):
                if alert_type not in current_alert_types:
                    resolved_alert = self.active_alerts.pop(alert_type)
                    resolved_alert['resolved_at'] = current_time
                    resolved_alerts.append(resolved_alert)
                    
                    logger.info(f"Alert resolved: {resolved_alert['message']}")
            
            return resolved_alerts
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        with self.lock:
            return list(self.alert_history)[-limit:]
    
    def clear_alerts(self):
        """Clear all active alerts."""
        with self.lock:
            self.active_alerts.clear()


class HealthChecker:
    """
    Performs health checks on the API service and its dependencies.
    """
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = None
    
    def add_check(self, name: str, check_func, timeout: float = 5.0):
        """Add a health check function."""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout,
            'last_result': None,
            'last_check': None
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_config in self.checks.items():
            try:
                start_time = time.time()
                result = check_config['func']()
                duration = time.time() - start_time
                
                check_result = {
                    'healthy': True,
                    'response_time': duration,
                    'details': result,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                if duration > check_config['timeout']:
                    check_result['healthy'] = False
                    check_result['error'] = f"Check timeout ({duration:.2f}s > {check_config['timeout']}s)"
                    overall_healthy = False
                
            except Exception as e:
                check_result = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                overall_healthy = False
            
            results[name] = check_result
            self.checks[name]['last_result'] = check_result
            self.checks[name]['last_check'] = time.time()
        
        self.last_check_time = time.time()
        
        return {
            'healthy': overall_healthy,
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if self.last_check_time is None:
            return self.run_checks()
        
        # Return cached results if recent
        if time.time() - self.last_check_time < 30:  # 30 seconds cache
            results = {}
            overall_healthy = True
            
            for name, check_config in self.checks.items():
                if check_config['last_result']:
                    results[name] = check_config['last_result']
                    if not check_config['last_result']['healthy']:
                        overall_healthy = False
            
            return {
                'healthy': overall_healthy,
                'checks': results,
                'timestamp': datetime.utcfromtimestamp(self.last_check_time).isoformat(),
                'cached': True
            }
        
        return self.run_checks()
