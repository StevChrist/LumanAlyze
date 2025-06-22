'use client'

import React from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => (
    <div style={{ 
      width: '100%', 
      height: '300px', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      color: '#FBF7C7',
      backgroundColor: 'rgba(251, 247, 199, 0.05)',
      borderRadius: '8px'
    }}>
      Loading Metrics Chart...
    </div>
  )
});

interface MetricsChartProps {
  metrics: Record<string, number>;
  chartType?: 'bar' | 'gauge';
  title?: string;
}

const MetricsChart: React.FC<MetricsChartProps> = ({ 
  metrics, 
  chartType = 'bar',
  title = "Model Performance Metrics" 
}) => {
  const metricNames = Object.keys(metrics);
  const metricValues = Object.values(metrics);

  let data: Partial<import('plotly.js').Data>[];
  let layout: Partial<import('plotly.js').Layout>;

  if (chartType === 'bar') {
    data = [{
      x: metricNames,
      y: metricValues,
      type: 'bar' as const,
      marker: {
        color: '#014F39',
        opacity: 0.8
      }
    }];

    layout = {
      title: {
        text: title,
        font: { family: 'Lora, serif', color: '#FBF7C7' }
      },
      xaxis: {
        title: { text: 'Metrics' }, // <-- Perbaikan di sini
        gridcolor: 'rgba(251, 247, 199, 0.2)',
        color: '#FBF7C7'
      },
      yaxis: {
        title: { text: 'Values' }, // <-- Perbaikan di sini
        gridcolor: 'rgba(251, 247, 199, 0.2)',
        color: '#FBF7C7'
      },
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#FBF7C7' }
    };
  } else {
    // Gauge chart for single metric
    const primaryMetric = metricNames[0];
    const primaryValue = metricValues[0];

    data = [{
      domain: { x: [0, 1], y: [0, 1] },
      value: primaryValue,
      title: { text: primaryMetric },
      type: 'indicator' as const,
      mode: 'gauge+number' as const,
      gauge: {
        axis: { range: [null, 1] },
        bar: { color: '#014F39' },
        steps: [
          { range: [0, 0.5], color: 'lightgray' },
          { range: [0.5, 1], color: 'gray' }
        ],
        threshold: {
          line: { color: '#dc3545', width: 4 },
          thickness: 0.75,
          value: 0.9
        }
      }
    }];

    layout = {
      title: {
        text: title,
        font: { family: 'Lora, serif', color: '#FBF7C7' }
      },
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#FBF7C7' }
    };
  }

  const config = {
    displayModeBar: true,
    displaylogo: false,
    responsive: true
  };

  return (
    <div style={{ width: '100%', height: '300px' }}>
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default MetricsChart;