import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DashboardService:
    """Service untuk aggregate data dan prepare dashboard widgets"""
    
    def __init__(self):
        self.widget_types = [
            'summary_stats', 'performance_metrics', 'data_quality',
            'recent_analyses', 'trend_analysis', 'comparison_chart'
        ]
    
    def prepare_dashboard_data(self, analysis_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare comprehensive dashboard data"""
        logger.info("Preparing dashboard data")
        
        try:
            dashboard_data = {
                'summary': self._generate_summary_stats(analysis_history),
                'widgets': {},
                'charts': {},
                'insights': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # Generate widgets
            for widget_type in self.widget_types:
                dashboard_data['widgets'][widget_type] = self._generate_widget_data(
                    widget_type, analysis_history
                )
            
            # Generate charts
            dashboard_data['charts'] = self._prepare_dashboard_charts(analysis_history)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error preparing dashboard data: {str(e)}")
            raise
    
    def aggregate_analysis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple analysis results"""
        logger.info("Aggregating analysis results")
        
        try:
            aggregated = {
                'total_analyses': len(results),
                'analysis_types': {},
                'average_performance': {},
                'trends': {},
                'summary_metrics': {}
            }
            
            # Count analysis types
            for result in results:
                model_type = result.get('model_type', 'unknown')
                aggregated['analysis_types'][model_type] = aggregated['analysis_types'].get(model_type, 0) + 1
            
            # Calculate average performance
            prediction_scores = []
            anomaly_percentages = []
            clustering_scores = []
            
            for result in results:
                if 'metrics' in result:
                    metrics = result['metrics']
                    if 'r2_score' in metrics:
                        prediction_scores.append(metrics['r2_score'])
                    elif 'accuracy' in metrics:
                        prediction_scores.append(metrics['accuracy'])
                
                if 'anomaly_percentage' in result:
                    anomaly_percentages.append(result['anomaly_percentage'])
                
                if 'evaluation' in result and 'silhouette_score' in result['evaluation']:
                    clustering_scores.append(result['evaluation']['silhouette_score'])
            
            if prediction_scores:
                aggregated['average_performance']['prediction'] = np.mean(prediction_scores)
            if anomaly_percentages:
                aggregated['average_performance']['anomaly_detection'] = np.mean(anomaly_percentages)
            if clustering_scores:
                aggregated['average_performance']['clustering'] = np.mean(clustering_scores)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating results: {str(e)}")
            raise
    
    def generate_performance_summary(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary untuk current analysis"""
        logger.info("Generating performance summary")
        
        try:
            summary = {
                'overall_grade': 'Unknown',
                'key_metrics': {},
                'performance_indicators': {},
                'recommendations': []
            }
            
            model_type = current_results.get('model_type', '')
            
            if 'prediction' in model_type.lower():
                summary.update(self._summarize_prediction_performance(current_results))
            elif 'anomaly' in model_type.lower():
                summary.update(self._summarize_anomaly_performance(current_results))
            elif 'cluster' in model_type.lower():
                summary.update(self._summarize_clustering_performance(current_results))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            raise
    
    def _generate_summary_stats(self, analysis_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_analyses': len(analysis_history),
            'successful_analyses': len([r for r in analysis_history if r.get('status') == 'success']),
            'most_used_model': self._get_most_used_model(analysis_history),
            'average_accuracy': self._calculate_average_accuracy(analysis_history)
        }
    
    def _generate_widget_data(self, widget_type: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data untuk specific widget type"""
        if widget_type == 'summary_stats':
            return self._generate_summary_stats(data)
        elif widget_type == 'performance_metrics':
            return self._generate_performance_widget(data)
        elif widget_type == 'data_quality':
            return self._generate_data_quality_widget(data)
        elif widget_type == 'recent_analyses':
            return self._generate_recent_analyses_widget(data)
        else:
            return {}
    
    def _prepare_dashboard_charts(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare chart data untuk dashboard"""
        return {
            'performance_trend': self._prepare_performance_trend_chart(data),
            'model_usage': self._prepare_model_usage_chart(data),
            'accuracy_distribution': self._prepare_accuracy_distribution_chart(data)
        }
    
    def _get_most_used_model(self, data: List[Dict[str, Any]]) -> str:
        """Get most frequently used model"""
        model_counts = {}
        for result in data:
            model = result.get('model_type', 'unknown')
            model_counts[model] = model_counts.get(model, 0) + 1
        
        return max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else 'None'
    
    def _calculate_average_accuracy(self, data: List[Dict[str, Any]]) -> float:
        """Calculate average accuracy across all analyses"""
        accuracies = []
        for result in data:
            if 'metrics' in result:
                metrics = result['metrics']
                if 'accuracy' in metrics:
                    accuracies.append(metrics['accuracy'])
                elif 'r2_score' in metrics:
                    accuracies.append(metrics['r2_score'])
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _summarize_prediction_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize prediction model performance"""
        metrics = results.get('metrics', {})
        
        if 'r2_score' in metrics:
            score = metrics['r2_score']
            grade = 'Excellent' if score >= 0.9 else 'Good' if score >= 0.8 else 'Fair' if score >= 0.7 else 'Poor'
        elif 'accuracy' in metrics:
            score = metrics['accuracy']
            grade = 'Excellent' if score >= 0.9 else 'Good' if score >= 0.8 else 'Fair' if score >= 0.7 else 'Poor'
        else:
            score = 0
            grade = 'Unknown'
        
        return {
            'overall_grade': grade,
            'key_metrics': {'primary_score': score},
            'performance_indicators': {
                'training_samples': results.get('training_samples', 0),
                'test_samples': results.get('test_samples', 0)
            }
        }
    
    def _summarize_anomaly_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize anomaly detection performance"""
        anomaly_percentage = results.get('anomaly_percentage', 0)
        
        if anomaly_percentage < 5:
            grade = 'Normal'
        elif anomaly_percentage < 15:
            grade = 'Moderate'
        else:
            grade = 'High'
        
        return {
            'overall_grade': grade,
            'key_metrics': {'anomaly_percentage': anomaly_percentage},
            'performance_indicators': {
                'total_samples': results.get('total_samples', 0),
                'anomalies_detected': results.get('num_anomalies', 0)
            }
        }
    
    def _summarize_clustering_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize clustering performance"""
        evaluation = results.get('evaluation', {})
        silhouette_score = evaluation.get('silhouette_score', 0)
        
        if silhouette_score > 0.7:
            grade = 'Excellent'
        elif silhouette_score > 0.5:
            grade = 'Good'
        elif silhouette_score > 0.3:
            grade = 'Fair'
        else:
            grade = 'Poor'
        
        return {
            'overall_grade': grade,
            'key_metrics': {'silhouette_score': silhouette_score},
            'performance_indicators': {
                'num_clusters': evaluation.get('num_clusters', 0),
                'total_samples': results.get('total_samples', 0)
            }
        }
