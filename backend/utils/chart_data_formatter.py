import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChartDataFormatter:
    """Class untuk memformat data khusus untuk berbagai jenis chart"""
    
    def __init__(self):
        self.default_colors = ['#014F39', '#FBF7C7', '#E8E8E6', '#120F0A']
        self.chart_themes = {
            'lumenalyze': {
                'background': '#014F39',
                'grid_color': 'rgba(251, 247, 199, 0.2)',
                'text_color': '#FBF7C7',
                'font_family': 'Lora, serif'
            }
        }
    
    def format_for_plotly(self, data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """Format data khusus untuk Plotly.js charts"""
        logger.info(f"Formatting data for Plotly chart type: {chart_type}")
        
        try:
            formatted_data = {
                'data': data.get('data', []),
                'layout': self._get_plotly_layout(chart_type),
                'config': self._get_plotly_config()
            }
            
            # Apply specific formatting based on chart type
            if chart_type == 'scatter':
                formatted_data['layout'].update({
                    'xaxis': {'title': data.get('x_title', 'X Axis')},
                    'yaxis': {'title': data.get('y_title', 'Y Axis')},
                    'hovermode': 'closest'
                })
            
            elif chart_type == 'bar':
                formatted_data['layout'].update({
                    'xaxis': {'title': data.get('x_title', 'Categories')},
                    'yaxis': {'title': data.get('y_title', 'Values')},
                    'bargap': 0.2
                })
            
            elif chart_type == 'histogram':
                formatted_data['layout'].update({
                    'xaxis': {'title': data.get('x_title', 'Values')},
                    'yaxis': {'title': 'Frequency'},
                    'bargap': 0.05
                })
            
            elif chart_type == 'line':
                formatted_data['layout'].update({
                    'xaxis': {'title': data.get('x_title', 'X Axis')},
                    'yaxis': {'title': data.get('y_title', 'Y Axis')},
                    'hovermode': 'x unified'
                })
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error formatting data for Plotly: {str(e)}")
            raise
    
    def prepare_time_series_data(self, data: pd.DataFrame, 
                               timestamp_col: str, 
                               value_cols: List[str]) -> Dict[str, Any]:
        """Mempersiapkan data time series untuk visualisasi temporal"""
        logger.info("Preparing time series data")
        
        try:
            # Ensure timestamp column is datetime
            if timestamp_col in data.columns:
                data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                data = data.sort_values(timestamp_col)
            
            time_series_data = []
            
            for col in value_cols:
                if col in data.columns:
                    trace = {
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'x': data[timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                        'y': data[col].tolist(),
                        'name': col.replace('_', ' ').title(),
                        'line': {'width': 2},
                        'marker': {'size': 4}
                    }
                    time_series_data.append(trace)
            
            return {
                'data': time_series_data,
                'x_title': 'Time',
                'y_title': 'Values',
                'chart_type': 'line'
            }
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {str(e)}")
            raise
    
    def aggregate_data_for_dashboard(self, data: pd.DataFrame, 
                                   aggregation_type: str,
                                   group_by: Optional[str] = None) -> Dict[str, Any]:
        """Agregasi data untuk widget dashboard"""
        logger.info(f"Aggregating data for dashboard: {aggregation_type}")
        
        try:
            if group_by and group_by in data.columns:
                grouped_data = data.groupby(group_by)
                
                if aggregation_type == 'count':
                    result = grouped_data.size().reset_index(name='count')
                    return {
                        'type': 'bar',
                        'x': result[group_by].tolist(),
                        'y': result['count'].tolist(),
                        'chart_type': 'bar'
                    }
                
                elif aggregation_type == 'mean':
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    result = grouped_data[numeric_cols].mean().reset_index()
                    
                    return {
                        'type': 'bar',
                        'x': result[group_by].tolist(),
                        'y': result[numeric_cols[0]].tolist() if len(numeric_cols) > 0 else [],
                        'chart_type': 'bar'
                    }
                
                elif aggregation_type == 'sum':
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    result = grouped_data[numeric_cols].sum().reset_index()
                    
                    return {
                        'type': 'bar',
                        'x': result[group_by].tolist(),
                        'y': result[numeric_cols[0]].tolist() if len(numeric_cols) > 0 else [],
                        'chart_type': 'bar'
                    }
            
            else:
                # Simple aggregation without grouping
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                if aggregation_type == 'summary':
                    summary_stats = data[numeric_cols].describe()
                    return {
                        'summary_statistics': summary_stats.to_dict(),
                        'chart_type': 'table'
                    }
                
                elif aggregation_type == 'correlation':
                    correlation_matrix = data[numeric_cols].corr()
                    return {
                        'correlation_matrix': correlation_matrix.to_dict(),
                        'chart_type': 'heatmap'
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error aggregating data for dashboard: {str(e)}")
            raise
    
    def format_metrics_for_display(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format metrics untuk display yang user-friendly"""
        logger.info("Formatting metrics for display")
        
        try:
            formatted_metrics = {}
            
            for category, values in metrics.items():
                if isinstance(values, dict):
                    formatted_category = {}
                    
                    for key, value in values.items():
                        if isinstance(value, float):
                            if 'percentage' in key.lower():
                                formatted_category[key] = f"{value:.2f}%"
                            elif 'score' in key.lower():
                                formatted_category[key] = f"{value:.4f}"
                            else:
                                formatted_category[key] = f"{value:.3f}"
                        else:
                            formatted_category[key] = value
                    
                    formatted_metrics[category] = formatted_category
                else:
                    formatted_metrics[category] = values
            
            return formatted_metrics
            
        except Exception as e:
            logger.error(f"Error formatting metrics for display: {str(e)}")
            raise
    
    def _get_plotly_layout(self, chart_type: str) -> Dict[str, Any]:
        """Mendapatkan layout default untuk Plotly charts"""
        base_layout = {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {
                'family': self.chart_themes['lumenalyze']['font_family'],
                'color': self.chart_themes['lumenalyze']['text_color']
            },
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1
            },
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
        
        # Add grid styling
        grid_style = {
            'gridcolor': self.chart_themes['lumenalyze']['grid_color'],
            'gridwidth': 1,
            'zeroline': False
        }
        
        base_layout['xaxis'] = grid_style.copy()
        base_layout['yaxis'] = grid_style.copy()
        
        return base_layout
    
    def _get_plotly_config(self) -> Dict[str, Any]:
        """Mendapatkan konfigurasi default untuk Plotly charts"""
        return {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True
        }
