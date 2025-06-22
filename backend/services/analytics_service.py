import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service untuk generate insights dan analytics dari hasil ML"""
    
    def __init__(self):
        self.insight_templates = {
            'prediction': {
                'excellent': "Model menunjukkan performa sangat baik dengan akurasi tinggi.",
                'good': "Model memiliki performa yang baik dan dapat diandalkan.",
                'fair': "Model memiliki performa cukup, pertimbangkan tuning parameter.",
                'poor': "Model memerlukan perbaikan signifikan."
            },
            'anomaly': {
                'low': "Dataset menunjukkan pola normal dengan sedikit anomali.",
                'medium': "Terdapat beberapa anomali yang perlu diperhatikan.",
                'high': "Dataset memiliki banyak anomali, perlu investigasi lebih lanjut."
            },
            'clustering': {
                'good': "Clustering menghasilkan segmentasi yang jelas dan bermakna.",
                'fair': "Clustering cukup baik, pertimbangkan jumlah cluster yang berbeda.",
                'poor': "Clustering kurang optimal, data mungkin tidak cocok untuk segmentasi."
            }
        }
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automated insights dari ML results"""
        logger.info("Generating automated insights")
        
        try:
            insights = {
                'summary': '',
                'recommendations': [],
                'key_findings': [],
                'data_quality': {},
                'model_performance': {},
                'generated_at': datetime.now().isoformat()
            }
            
            model_type = analysis_results.get('model_type', '')
            
            # Prediction insights
            if 'prediction' in model_type.lower():
                insights.update(self._generate_prediction_insights(analysis_results))
            
            # Anomaly detection insights
            elif 'anomaly' in model_type.lower() or 'isolation' in model_type.lower():
                insights.update(self._generate_anomaly_insights(analysis_results))
            
            # Clustering insights
            elif 'kmeans' in model_type.lower() or 'cluster' in model_type.lower():
                insights.update(self._generate_clustering_insights(analysis_results))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            raise
    
    def calculate_data_quality_score(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics"""
        logger.info("Calculating data quality score")
        
        try:
            total_cells = dataframe.shape[0] * dataframe.shape[1]
            missing_cells = dataframe.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            # Completeness score
            completeness_score = max(0, 100 - missing_percentage)
            
            # Consistency score (based on data types)
            numeric_cols = dataframe.select_dtypes(include=['number']).columns
            consistency_score = (len(numeric_cols) / len(dataframe.columns)) * 100
            
            # Uniqueness score (average uniqueness across columns)
            uniqueness_scores = []
            for col in dataframe.columns:
                unique_ratio = dataframe[col].nunique() / len(dataframe)
                uniqueness_scores.append(min(unique_ratio * 100, 100))
            
            uniqueness_score = np.mean(uniqueness_scores)
            
            # Overall quality score
            overall_score = (completeness_score * 0.4 + 
                           consistency_score * 0.3 + 
                           uniqueness_score * 0.3)
            
            return {
                'overall_score': round(overall_score, 2),
                'completeness_score': round(completeness_score, 2),
                'consistency_score': round(consistency_score, 2),
                'uniqueness_score': round(uniqueness_score, 2),
                'missing_percentage': round(missing_percentage, 2),
                'total_rows': dataframe.shape[0],
                'total_columns': dataframe.shape[1],
                'numeric_columns': len(numeric_cols),
                'grade': self._get_quality_grade(overall_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {str(e)}")
            raise
    
    def recommend_parameters(self, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal parameters untuk analysis"""
        logger.info("Generating parameter recommendations")
        
        try:
            recommendations = {
                'preprocessing': {},
                'model_selection': {},
                'parameters': {},
                'reasoning': []
            }
            
            rows = data_characteristics.get('rows', 0)
            columns = data_characteristics.get('columns', 0)
            missing_percentage = data_characteristics.get('missing_percentage', 0)
            
            # Preprocessing recommendations
            if missing_percentage > 20:
                recommendations['preprocessing']['missing_strategy'] = 'drop'
                recommendations['reasoning'].append("High missing values detected, recommend dropping incomplete rows")
            elif missing_percentage > 5:
                recommendations['preprocessing']['missing_strategy'] = 'mean'
                recommendations['reasoning'].append("Moderate missing values, recommend mean imputation")
            else:
                recommendations['preprocessing']['missing_strategy'] = 'none'
            
            # Normalization recommendations
            if columns > 10:
                recommendations['preprocessing']['normalization'] = 'standard'
                recommendations['reasoning'].append("Multiple features detected, recommend standardization")
            
            # Model selection recommendations
            if rows < 1000:
                recommendations['model_selection']['prediction'] = 'random_forest'
                recommendations['reasoning'].append("Small dataset, Random Forest recommended for stability")
            else:
                recommendations['model_selection']['prediction'] = 'mlp'
                recommendations['reasoning'].append("Large dataset, Neural Network can capture complex patterns")
            
            # Clustering parameters
            optimal_clusters = min(int(np.sqrt(rows / 2)), 10)
            recommendations['parameters']['n_clusters'] = max(2, optimal_clusters)
            recommendations['reasoning'].append(f"Optimal cluster count estimated: {optimal_clusters}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def _generate_prediction_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights untuk prediction results"""
        metrics = results.get('metrics', {})
        
        if 'r2_score' in metrics:
            score = metrics['r2_score']
            grade = 'excellent' if score >= 0.9 else 'good' if score >= 0.8 else 'fair' if score >= 0.7 else 'poor'
        elif 'accuracy' in metrics:
            score = metrics['accuracy']
            grade = 'excellent' if score >= 0.9 else 'good' if score >= 0.8 else 'fair' if score >= 0.7 else 'poor'
        else:
            grade = 'fair'
        
        return {
            'summary': self.insight_templates['prediction'][grade],
            'model_performance': {
                'grade': grade,
                'score': score if 'score' in locals() else 0
            },
            'recommendations': [
                "Consider feature engineering for better performance" if grade in ['fair', 'poor'] else "Model is performing well",
                "Validate with additional test data" if grade == 'excellent' else "Consider hyperparameter tuning"
            ]
        }
    
    def _generate_anomaly_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights untuk anomaly detection"""
        anomaly_percentage = results.get('anomaly_percentage', 0)
        
        if anomaly_percentage < 5:
            level = 'low'
        elif anomaly_percentage < 15:
            level = 'medium'
        else:
            level = 'high'
        
        return {
            'summary': self.insight_templates['anomaly'][level],
            'key_findings': [
                f"Detected {results.get('num_anomalies', 0)} anomalies ({anomaly_percentage:.1f}%)",
                f"Anomaly level: {level.upper()}"
            ],
            'recommendations': [
                "Investigate anomalous data points" if level != 'low' else "Data appears normal",
                "Consider adjusting contamination parameter" if level == 'high' else "Current settings seem appropriate"
            ]
        }
    
    def _generate_clustering_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights untuk clustering"""
        evaluation = results.get('evaluation', {})
        silhouette_score = evaluation.get('silhouette_score', 0)
        
        if silhouette_score > 0.7:
            grade = 'good'
        elif silhouette_score > 0.5:
            grade = 'fair'
        else:
            grade = 'poor'
        
        return {
            'summary': self.insight_templates['clustering'][grade],
            'model_performance': {
                'silhouette_score': silhouette_score,
                'grade': grade
            },
            'recommendations': [
                "Clusters are well-separated" if grade == 'good' else "Consider different number of clusters",
                "Try different clustering algorithms" if grade == 'poor' else "Current clustering is acceptable"
            ]
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Very Poor"
