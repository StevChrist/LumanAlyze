'use client'

import React from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => (
    <div style={{ 
      width: '100%', 
      height: '400px', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      color: '#FBF7C7',
      backgroundColor: 'rgba(251, 247, 199, 0.05)',
      borderRadius: '8px'
    }}>
      Loading Anomaly Chart...
    </div>
  )
});

interface AnomalyChartProps {
  dataPoints: number[][];
  anomalyIndices: number[];
  title?: string;
}

const AnomalyChart: React.FC<AnomalyChartProps> = ({ 
  dataPoints, 
  anomalyIndices, 
  title = "Anomaly Detection Results" 
}) => {
  // Prepare normal points
  const normalIndices = dataPoints.map((_, index) => index)
    .filter(index => !anomalyIndices.includes(index));

  const normalData = {
    x: normalIndices.map(i => dataPoints[i][0]),
    y: normalIndices.map(i => dataPoints[i][1]),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Normal Points',
    marker: {
      color: '#014F39',
      size: 6,
      opacity: 0.7
    }
  };

  // Prepare anomaly points
  const anomalyData = {
    x: anomalyIndices.map(i => dataPoints[i][0]),
    y: anomalyIndices.map(i => dataPoints[i][1]),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Anomalies',
    marker: {
      color: '#dc3545',
      size: 10,
      symbol: 'x',
      opacity: 0.9
    }
  };

  const layout = {
    title: {
      text: title,
      font: { family: 'Lora, serif', color: '#FBF7C7' }
    },
    xaxis: {
      title: { text: 'Feature 1' },
      gridcolor: 'rgba(251, 247, 199, 0.2)',
      color: '#FBF7C7'
    },
    yaxis: {
      title: { text: 'Feature 2' },
      gridcolor: 'rgba(251, 247, 199, 0.2)',
      color: '#FBF7C7'
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#FBF7C7' },
    showlegend: true,
    legend: {
      orientation: 'h' as const,
      y: -0.2
    }
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    responsive: true
  };

  return (
    <div style={{ width: '100%', height: '400px' }}>
      <Plot
        data={[normalData, anomalyData]}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default AnomalyChart;
