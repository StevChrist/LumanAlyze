import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import io
import base64
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ExportHandler:
    """Class untuk menangani export hasil analisis dalam berbagai format"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'excel']
    
    def export_to_csv(self, data: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
        """Export hasil analisis ke format CSV"""
        logger.info("Exporting data to CSV format")
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lumenalyze_results_{timestamp}.csv"
            
            # Convert analysis results to DataFrame
            if 'visualization_data' in data:
                viz_data = data['visualization_data']
                
                # For prediction results
                if 'actual' in viz_data and 'predicted' in viz_data:
                    df = pd.DataFrame({
                        'actual': viz_data['actual'],
                        'predicted': viz_data['predicted'],
                        'residual': [p - a for a, p in zip(viz_data['actual'], viz_data['predicted'])]
                    })
                
                # For anomaly detection results
                elif 'anomaly_indices' in viz_data:
                    data_points = np.array(viz_data.get('data_points', []))
                    anomaly_flags = [i in viz_data['anomaly_indices'] for i in range(len(data_points))]
                    
                    df_data = {
                        'is_anomaly': anomaly_flags,
                        'anomaly_score': viz_data.get('anomaly_scores', [0] * len(data_points))
                    }
                    
                    # Add feature columns if available
                    if len(data_points) > 0 and len(data_points[0]) > 0:
                        for i in range(len(data_points[0])):
                            df_data[f'feature_{i+1}'] = data_points[:, i]
                    
                    df = pd.DataFrame(df_data)
                
                # For clustering results
                elif 'cluster_labels' in viz_data:
                    data_points = np.array(viz_data.get('data_points', []))
                    
                    df_data = {
                        'cluster_label': viz_data['cluster_labels']
                    }
                    
                    # Add feature columns if available
                    if len(data_points) > 0 and len(data_points[0]) > 0:
                        for i in range(len(data_points[0])):
                            df_data[f'feature_{i+1}'] = data_points[:, i]
                    
                    df = pd.DataFrame(df_data)
                
                else:
                    # Fallback: create summary DataFrame
                    df = pd.DataFrame([data.get('metrics', {})])
            
            else:
                # Create summary DataFrame from metrics
                df = pd.DataFrame([data.get('metrics', {})])
            
            # Convert DataFrame to CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Encode to base64 for download
            csv_base64 = base64.b64encode(csv_content.encode()).decode()
            
            return {
                'status': 'success',
                'filename': filename,
                'content': csv_base64,
                'content_type': 'text/csv',
                'rows': len(df),
                'columns': len(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    def export_to_json(self, data: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
        """Export hasil analisis ke format JSON"""
        logger.info("Exporting data to JSON format")
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lumenalyze_results_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'application': 'LumenALYZE',
                    'version': '1.0.0'
                },
                'analysis_results': data
            }
            
            # Convert to JSON string
            json_content = json.dumps(export_data, indent=2, default=self._json_serializer)
            
            # Encode to base64 for download
            json_base64 = base64.b64encode(json_content.encode()).decode()
            
            return {
                'status': 'success',
                'filename': filename,
                'content': json_base64,
                'content_type': 'application/json',
                'size_kb': len(json_content) / 1024
            }
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise
    
    def export_to_excel(self, data: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
        """Export hasil analisis ke format Excel dengan multiple sheets"""
        logger.info("Exporting data to Excel format")
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lumenalyze_results_{timestamp}.xlsx"
            
            # Create Excel buffer
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Analysis Type': [data.get('model_type', 'Unknown')],
                    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Status': [data.get('status', 'Unknown')]
                }
                
                if 'metrics' in data:
                    for key, value in data['metrics'].items():
                        if isinstance(value, (int, float, str)):
                            summary_data[key.replace('_', ' ').title()] = [value]
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed results sheet
                if 'visualization_data' in data:
                    viz_data = data['visualization_data']
                    
                    # Prediction results
                    if 'actual' in viz_data and 'predicted' in viz_data:
                        results_df = pd.DataFrame({
                            'Actual': viz_data['actual'],
                            'Predicted': viz_data['predicted'],
                            'Residual': [p - a for a, p in zip(viz_data['actual'], viz_data['predicted'])]
                        })
                        results_df.to_excel(writer, sheet_name='Prediction Results', index=False)
                    
                    # Anomaly detection results
                    elif 'anomaly_indices' in viz_data:
                        data_points = np.array(viz_data.get('data_points', []))
                        anomaly_flags = [i in viz_data['anomaly_indices'] for i in range(len(data_points))]
                        
                        results_data = {
                            'Is_Anomaly': anomaly_flags,
                            'Anomaly_Score': viz_data.get('anomaly_scores', [0] * len(data_points))
                        }
                        
                        if len(data_points) > 0 and len(data_points[0]) > 0:
                            for i in range(len(data_points[0])):
                                results_data[f'Feature_{i+1}'] = data_points[:, i]
                        
                        results_df = pd.DataFrame(results_data)
                        results_df.to_excel(writer, sheet_name='Anomaly Results', index=False)
                    
                    # Clustering results
                    elif 'cluster_labels' in viz_data:
                        data_points = np.array(viz_data.get('data_points', []))
                        
                        results_data = {
                            'Cluster_Label': viz_data['cluster_labels']
                        }
                        
                        if len(data_points) > 0 and len(data_points[0]) > 0:
                            for i in range(len(data_points[0])):
                                results_data[f'Feature_{i+1}'] = data_points[:, i]
                        
                        results_df = pd.DataFrame(results_data)
                        results_df.to_excel(writer, sheet_name='Cluster Results', index=False)
            
            # Get Excel content
            excel_content = excel_buffer.getvalue()
            excel_base64 = base64.b64encode(excel_content).decode()
            
            return {
                'status': 'success',
                'filename': filename,
                'content': excel_base64,
                'content_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'size_kb': len(excel_content) / 1024
            }
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise
    
    def generate_summary_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ringkasan laporan dari hasil analisis"""
        logger.info("Generating summary report")
        
        try:
            report = {
                'title': 'LumenALYZE Analysis Report',
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_type': analysis_results.get('model_type', 'Unknown'),
                'status': analysis_results.get('status', 'Unknown')
            }
            
            # Add metrics summary
            if 'metrics' in analysis_results:
                metrics = analysis_results['metrics']
                report['performance_metrics'] = {}
                
                # Classification metrics
                if 'accuracy' in metrics:
                    report['performance_metrics']['accuracy'] = f"{metrics['accuracy']:.4f}"
                    report['performance_metrics']['accuracy_percentage'] = f"{metrics['accuracy'] * 100:.2f}%"
                
                # Regression metrics
                if 'r2_score' in metrics:
                    report['performance_metrics']['r2_score'] = f"{metrics['r2_score']:.4f}"
                    report['performance_metrics']['r2_percentage'] = f"{metrics['r2_score'] * 100:.2f}%"
                
                if 'mse' in metrics:
                    report['performance_metrics']['mse'] = f"{metrics['mse']:.4f}"
                
                if 'rmse' in metrics:
                    report['performance_metrics']['rmse'] = f"{metrics['rmse']:.4f}"
            
            # Add data summary
            if 'training_samples' in analysis_results:
                report['data_summary'] = {
                    'training_samples': analysis_results['training_samples'],
                    'test_samples': analysis_results['test_samples'],
                    'total_samples': analysis_results['training_samples'] + analysis_results['test_samples']
                }
            
            # Add anomaly summary
            if 'num_anomalies' in analysis_results:
                report['anomaly_summary'] = {
                    'anomalies_detected': analysis_results['num_anomalies'],
                    'anomaly_percentage': f"{analysis_results['anomaly_percentage']:.2f}%",
                    'total_samples': analysis_results['total_samples']
                }
            
            # Add clustering summary
            if 'evaluation' in analysis_results:
                evaluation = analysis_results['evaluation']
                report['clustering_summary'] = {
                    'num_clusters': evaluation.get('num_clusters', 0),
                    'silhouette_score': f"{evaluation.get('silhouette_score', 0):.4f}",
                    'total_samples': analysis_results.get('total_samples', 0)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            raise
    
    def _json_serializer(self, obj):
        """Custom JSON serializer untuk numpy objects"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
