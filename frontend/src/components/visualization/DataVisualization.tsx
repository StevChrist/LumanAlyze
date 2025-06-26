import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { 
  AnalysisType, 
  AnalysisResult
} from '../../types';
import LoadingOverlay from '../ui/LoadingOverlay';
import { Layout, Config, Data as PlotlyData } from 'plotly.js';

// Dynamic import untuk Plotly dengan proper sizing
const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div style={{ minHeight: '450px' }}></div>
});

interface DataVisualizationProps {
  analysisResults: AnalysisResult;
  analysisType: AnalysisType;
  isLoading?: boolean;
}

const DataVisualization: React.FC<DataVisualizationProps> = ({ 
  analysisResults, 
  analysisType, 
  isLoading = false 
}) => {
  const [plotData, setPlotData] = useState<Partial<PlotlyData>[]>([]);
  const [layout, setLayout] = useState<Partial<Layout>>({});
  const [config, setConfig] = useState<Partial<Config>>({});
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!analysisResults || !analysisResults.visualization_data) return;

    // Responsive sizing
    const updateDimensions = () => {
      if (containerRef.current) {
        setConfig({
          responsive: true,
          displayModeBar: false,
          displaylogo: false
        });
        setLayout((prevLayout: Partial<Layout>) => ({
          ...prevLayout,
          autosize: true
        }));
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [analysisResults]);

  useEffect(() => {
    if (!analysisResults || !('visualization_data' in analysisResults)) return;

    switch (analysisType) {
      case 'prediction':
        generatePredictionPlot(analysisResults);
        break;
      case 'anomaly':
        generateAnomalyPlot(analysisResults);
        break;
      case 'clustering':
        generateClusteringPlot(analysisResults);
        break;
    }
  }, [analysisResults, analysisType]);

  const generatePredictionPlot = (data: AnalysisResult) => {
    if (!('visualization_data' in data) || !('actual' in data.visualization_data)) return;
    const { actual = [], predicted = [] } = data.visualization_data;
    setPlotData([
      {
        x: actual,
        y: predicted,
        mode: 'markers',
        type: 'scatter',
        name: 'Predictions',
        marker: {
          color: '#3B82F6',
          size: 8,
          opacity: 0.7
        }
      },
      {
        x: [Math.min(...actual), Math.max(...actual)],
        y: [Math.min(...actual), Math.max(...actual)],
        mode: 'lines',
        type: 'scatter',
        name: 'Perfect Prediction',
        marker: { color: '#EF4444', size: 6 },
        line: {
          color: '#EF4444',
          dash: 'dash',
          width: 2
        }
      }
    ]);
    setLayout({
      title: {
        text: '', // HILANGKAN JUDUL DI ATAS CHART
      },
      xaxis: { title: { text: 'Actual Values' }, gridcolor: '#E5E7EB' },
      yaxis: { title: { text: 'Predicted Values' }, gridcolor: '#E5E7EB' },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      margin: { t: 20, r: 20, b: 40, l: 40 }
    });
  };

  const generateAnomalyPlot = (data: AnalysisResult) => {
    if (
      !('visualization_data' in data) ||
      !('data_points' in data.visualization_data) ||
      !('anomaly_indices' in data.visualization_data)
    ) return;
    const { data_points = [], anomaly_indices = [] } = data.visualization_data as {
      data_points: number[][];
      anomaly_indices: number[];
    };
    const normalPoints: number[][] = data_points.filter((_: number[], index: number) => !anomaly_indices.includes(index));
    const anomalyPoints: number[][] = data_points.filter((_: number[], index: number) => anomaly_indices.includes(index));
    setPlotData([
      {
        x: normalPoints.map((point: number[]) => point[0]),
        y: normalPoints.map((point: number[]) => point[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Normal',
        marker: {
          color: '#10B981',
          size: 6,
          opacity: 0.7
        }
      },
      {
        x: anomalyPoints.map((point: number[]) => point[0]),
        y: anomalyPoints.map((point: number[]) => point[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Anomaly',
        marker: {
          color: '#EF4444',
          size: 8,
          symbol: 'x',
          opacity: 0.9
        }
      }
    ]);
    setLayout({
      title: { text: '' },
      xaxis: { title: { text: 'Component 1' }, gridcolor: '#E5E7EB' },
      yaxis: { title: { text: 'Component 2' }, gridcolor: '#E5E7EB' },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      margin: { t: 20, r: 20, b: 40, l: 40 }
    });
  };

  const generateClusteringPlot = (data: AnalysisResult) => {
    if (
      !('visualization_data' in data) ||
      !('data_points' in data.visualization_data) ||
      !('cluster_labels' in data.visualization_data)
    ) return;
    const { data_points = [], cluster_labels = [], cluster_centers = [] } = data.visualization_data as {
      data_points: number[][];
      cluster_labels: (number | string)[];
      cluster_centers?: number[][];
    };
    const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899'];
    const uniqueLabels = Array.from(new Set(cluster_labels));
    const traces: Partial<PlotlyData>[] = uniqueLabels.map(label => ({
      x: data_points.filter((_: number[], index: number) => cluster_labels[index] === label).map((point: number[]) => point[0]),
      y: data_points.filter((_: number[], index: number) => cluster_labels[index] === label).map((point: number[]) => point[1]),
      mode: 'markers',
      type: 'scatter',
      name: `Cluster ${label}`,
      marker: {
        color: colors[Number(label) % colors.length],
        size: 6,
        opacity: 0.7
      }
    }));
    if (Array.isArray(cluster_centers) && cluster_centers.length > 0) {
      traces.push({
        x: cluster_centers.map((center: number[]) => center[0]),
        y: cluster_centers.map((center: number[]) => center[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Centroids',
        marker: {
          color: '#1F2937',
          size: 12,
          symbol: 'diamond',
          line: { color: '#FFFFFF', width: 2 }
        }
      });
    }
    setPlotData(traces);
    setLayout({
      title: { text: '' },
      xaxis: { title: { text: 'Component 1' }, gridcolor: '#E5E7EB' },
      yaxis: { title: { text: 'Component 2' }, gridcolor: '#E5E7EB' },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      margin: { t: 20, r: 20, b: 40, l: 40 }
    });
  };

  if (isLoading) {
    return <LoadingOverlay message="Generating visualization..." isVisible={true} />;
  }

  if (!analysisResults) {
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '2rem',
        color: '#FBF7C7',
        minHeight: '300px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <h3>No Data to Visualize</h3>
        <p>Run an analysis to see the visualization here.</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        margin: 0,
        padding: 0,
        background: 'rgba(199, 199, 199, 0.85)',
        border: 'none',
        boxShadow: 'none'
      }}
    >
      {/* Hapus judul chart, hanya tampilkan judul section */}
      <h3 className="visualization-title" style={{ marginBottom: '1.5rem' }}>
        Data Visualization
      </h3>
      <div
        style={{
          width: '100%',
          margin: 0,
          padding: 0,
          background: 'transparent',
          border: 'none',
          boxShadow: 'none',
          minHeight: 450
        }}
      >
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          style={{
            width: '100%',
            height: '100%',
            margin: 0,
            padding: 0,
            border: 'none',
            background: 'transparent'
          }}
          useResizeHandler={true}
        />
      </div>
    </div>
  );
};

export default DataVisualization;
