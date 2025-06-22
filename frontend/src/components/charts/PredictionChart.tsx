'use client'

import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div style={{ 
    width: '100%', 
    height: '400px', 
    display: 'flex', 
    alignItems: 'center', 
    justifyContent: 'center',
    color: '#FBF7C7',
    backgroundColor: 'rgba(251, 247, 199, 0.05)',
    borderRadius: '8px'
  }}>Loading Chart...</div>
});

interface PredictionChartProps {
  actual: number[];
  predicted: number[];
  confusionMatrix?: number[][];
  labels?: string[];
  title?: string;
}

const PredictionChart: React.FC<PredictionChartProps> = ({ 
  actual, 
  predicted, 
  confusionMatrix, 
  labels, 
  title = "Prediction Results" 
}) => {
  const scatterData = {
    x: actual,
    y: predicted,
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Actual vs Predicted',
    marker: {
      color: '#014F39',
      size: 8,
      opacity: 0.7
    }
  };

  const minVal = Math.min(...actual, ...predicted);
  const maxVal = Math.max(...actual, ...predicted);

  const perfectLine = {
    x: [minVal, maxVal],
    y: [minVal, maxVal],
    mode: 'lines' as const,
    type: 'scatter' as const,
    name: 'Perfect Prediction',
    line: {
      color: '#dc3545',
      dash: 'dash' as const,
      width: 2
    }
  };

  const layout = {
    title: {
      text: title,
      font: {
        family: 'Montserrat, sans-serif',
        color: '#FBF7C7'
      }
    },
    xaxis: {
      title: { text: "Actual" },
      gridcolor: '#E8E8E6',
      color: '#FBF7C7'
    },
    yaxis: {
      title: { text: "Predicted" },
      gridcolor: '#E8E8E6',
      color: '#FBF7C7'
    },
    plot_bgcolor: 'rgba(251, 247, 199, 0.05)',
    paper_bgcolor: 'rgba(251, 247, 199, 0.05)',
    font: {
      family: 'Lora, serif',
      color: '#FBF7C7'
    },
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(1, 79, 57, 0.7)',
      bordercolor: '#FBF7C7',
      borderwidth: 1,
      font: {
        color: '#FBF7C7'
      }
    }
  };

  return (
    <div>
      <Plot
        data={[scatterData, perfectLine]}
        layout={layout}
        useResizeHandler
        style={{ width: '100%', height: '400px' }}
        config={{ responsive: true, displayModeBar: false }}
      />
      {confusionMatrix && labels && (
        <div>
          <h4>Confusion Matrix</h4>
          <table>
            <thead>
              <tr>
                <th></th>
                {labels.map((l, i) => <th key={i}>{l}</th>)}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.map((row, i) => (
                <tr key={i}>
                  <td>{labels[i]}</td>
                  {row.map((val, j) => <td key={j}>{val}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default PredictionChart;