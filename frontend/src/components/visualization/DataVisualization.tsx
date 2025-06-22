'use client'

import React, { useState, useEffect } from 'react';
import ChartContainer from './ChartContainer';
import ResultsDashboard from '../dashboard/ResultsDashboard';
import type { MLResult } from '../../app/page'; // perbaiki path sesuai lokasi MLResult

interface DataVisualizationProps {
  analysisResults: MLResult;
  analysisType: string;
}

const DataVisualization: React.FC<DataVisualizationProps> = ({ 
  analysisResults, 
  analysisType 
}) => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    if (analysisResults) {
      setLoading(false);
    }
  }, [analysisResults, analysisType]);

  if (loading) {
    return (
      <ChartContainer loading={true}>{null}</ChartContainer>
    );
  }

  return (
    <div>
      <ChartContainer
        title="Analysis Visualization"
        subtitle={`Interactive visualization for ${analysisType} analysis`}
      >
        <ResultsDashboard 
          analysisResults={analysisResults}
          analysisType={analysisType}
        />
      </ChartContainer>
    </div>
  );
};

export default DataVisualization;