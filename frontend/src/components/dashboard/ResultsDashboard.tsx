'use client'

import React from 'react';
import PredictionChart from '../charts/PredictionChart';
import AnomalyChart from '../charts/AnomalyChart';
import ClusterChart from '../charts/ClusterChart';
import MetricsChart from '../charts/MetricsChart';
import StatisticsPanel from './StatisticsPanel';
import ExportButton from './ExportButton';

interface ResultsDashboardProps {
  analysisResults: Record<string, unknown>;
  analysisType: string;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ 
  analysisResults, 
  analysisType 
}) => {
  // Type guard untuk visualization_data
  const vizData = typeof analysisResults.visualization_data === 'object' && analysisResults.visualization_data !== null
    ? analysisResults.visualization_data as Record<string, unknown>
    : {};

  const renderChart = () => {
    if (!vizData) return null;

    switch (analysisType) {
      case 'prediction':
        return (
          <PredictionChart
            actual={Array.isArray(vizData.actual) ? vizData.actual as number[] : []}
            predicted={Array.isArray(vizData.predicted) ? vizData.predicted as number[] : []}
            title="Prediction Results"
          />
        );
      
      case 'anomaly':
        return (
          <AnomalyChart
            dataPoints={Array.isArray(vizData.data_points) ? vizData.data_points as number[][] : []}
            anomalyIndices={Array.isArray(vizData.anomaly_indices) ? vizData.anomaly_indices as number[] : []}
            title="Anomaly Detection Results"
          />
        );
      
      case 'segmentation':
        return (
          <ClusterChart
            dataPoints={Array.isArray(vizData.data_points) ? vizData.data_points as number[][] : []}
            clusterLabels={Array.isArray(vizData.cluster_labels) ? vizData.cluster_labels as number[] : []}
            clusterCenters={Array.isArray(vizData.cluster_centers) ? vizData.cluster_centers as number[][] : []}
            title="Clustering Results"
          />
        );
      
      default:
        return null;
    }
  };

  // Helper untuk safe access dan casting
  const getNumber = (val: unknown): number => typeof val === 'number' ? val : 0;
  const getString = (val: unknown): string => typeof val === 'string' ? val : '';
  
  return (
    <div style={{ maxWidth: '1200px', width: '100%', margin: '0 auto' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '24px'
      }}>
        <h2 style={{
          fontSize: '24px',
          color: '#FBF7C7',
          fontFamily: 'Lora, serif',
          margin: 0
        }}>
          Analysis Results - {getString(analysisResults.model_type)}
        </h2>
        
        <ExportButton 
          data={analysisResults}
          filename={`lumenalyze_${analysisType}_results`}
        />
      </div>

      {/* Statistics Panel */}
      {typeof analysisResults.metrics === 'object' && analysisResults.metrics !== null && (
        <StatisticsPanel 
          statistics={analysisResults.metrics as Record<string, unknown>}
          title="Performance Metrics"
        />
      )}

      {/* Main Chart */}
      <div style={{ marginBottom: '24px' }}>
        {renderChart()}
      </div>

      {/* Metrics Chart */}
      {typeof analysisResults.metrics === 'object' && analysisResults.metrics !== null && (
        <div style={{ marginBottom: '24px' }}>
          <MetricsChart 
            metrics={analysisResults.metrics as Record<string, number>}
            chartType="bar"
            title="Detailed Metrics"
          />
        </div>
      )}

      {/* Additional Statistics */}
      {analysisType === 'anomaly' && (
        <StatisticsPanel 
          statistics={{
            'Total Samples': getNumber(analysisResults.total_samples),
            'Anomalies Found': getNumber(analysisResults.num_anomalies),
            'Anomaly Percentage': typeof analysisResults.anomaly_percentage === 'number'
              ? `${analysisResults.anomaly_percentage.toFixed(2)}%`
              : ''
          }}
          title="Anomaly Detection Summary"
        />
      )}

      {analysisType === 'segmentation' && typeof analysisResults.evaluation === 'object' && analysisResults.evaluation !== null && (
        <StatisticsPanel 
          statistics={{
            'Number of Clusters': getNumber((analysisResults.evaluation as Record<string, unknown>).num_clusters),
            'Silhouette Score': typeof (analysisResults.evaluation as Record<string, unknown>).silhouette_score === 'number'
              ? ((analysisResults.evaluation as Record<string, unknown>).silhouette_score as number).toFixed(4)
              : '',
            'Total Samples': getNumber(analysisResults.total_samples)
          }}
          title="Clustering Summary"
        />
      )}

      {analysisType === 'prediction' && (
        <StatisticsPanel 
          statistics={{
            'Training Samples': getNumber(analysisResults.training_samples),
            'Test Samples': getNumber(analysisResults.test_samples),
            'Model Type': getString(analysisResults.model_type)
          }}
          title="Training Summary"
        />
      )}
    </div>
  );
};

export default ResultsDashboard;