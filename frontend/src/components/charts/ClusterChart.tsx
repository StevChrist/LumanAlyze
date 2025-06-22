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
      Loading Cluster Chart...
    </div>
  )
});

interface ClusterChartProps {
  dataPoints: number[][];
  clusterLabels: number[];
  clusterCenters?: number[][];
  title?: string;
}

const ClusterChart: React.FC<ClusterChartProps> = ({ 
  dataPoints, 
  clusterLabels, 
  clusterCenters = [],
  title = "Clustering Results" 
}) => {
  const colors = ['#014F39', '#FBF7C7', '#E8E8E6', '#28a745', '#ffc107', '#dc3545'];
  const uniqueLabels = [...new Set(clusterLabels)].filter(label => label !== -1);
  
  const traces = [];

  // Create traces for each cluster
  uniqueLabels.forEach((label, index) => {
    const clusterIndices = clusterLabels
      .map((l, i) => l === label ? i : -1)
      .filter(i => i !== -1);

    const trace = {
      x: clusterIndices.map(i => dataPoints[i][0]),
      y: clusterIndices.map(i => dataPoints[i][1]),
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: `Cluster ${label}`,
      marker: {
        color: colors[index % colors.length],
        size: 8,
        opacity: 0.7
      }
    };
    traces.push(trace);
  });

  // Add cluster centers if available
  if (clusterCenters.length > 0) {
    const centersTrace = {
      x: clusterCenters.map(center => center[0]),
      y: clusterCenters.map(center => center[1]),
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: 'Cluster Centers',
      marker: {
        color: '#120F0A',
        size: 15,
        symbol: 'x',
        line: { width: 2, color: 'white' }
      }
    };
    traces.push(centersTrace);
  }

   const layout = {
    title: {
      text: title,
      font: { family: 'Lora, serif', color: '#FBF7C7' }
    },
    xaxis: {
      title: { text: 'Feature 1' }, // <-- perbaikan di sini
      gridcolor: 'rgba(251, 247, 199, 0.2)',
      color: '#FBF7C7'
    },
    yaxis: {
      title: { text: 'Feature 2' }, // <-- perbaikan di sini
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
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default ClusterChart;
