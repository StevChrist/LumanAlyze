// Interface untuk hasil ML
export interface MLResult {
  status: string;
  model_type: string;
  metrics: {
    r2_score?: number;
    mse?: number;
    rmse?: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
  };
  visualization_data: {
    actual: number[];
    predicted: number[];
  };
  training_samples: number;
  test_samples: number;
}

export interface AnomalyResult {
  status: string;
  model_type: string;
  num_anomalies: number;
  anomaly_percentage: number;
  visualization_data: {
    data_points: number[][];
    anomaly_indices: number[];
    anomaly_scores: number[];
  };
  total_samples: number;
}

export interface ClusterResult {
  status: string;
  model_type: string;
  evaluation: {
    silhouette_score: number;
    num_clusters: number;
  };
  visualization_data: {
    data_points: number[][];
    cluster_labels: number[];
    cluster_centers?: number[][];
  };
  total_samples: number;
}

// Union type untuk analysis type
export type AnalysisType = 'prediction' | 'anomaly' | 'clustering';

// Union type untuk semua hasil
export type AnalysisResult = MLResult | AnomalyResult | ClusterResult;

// Props interfaces
export interface LoadingOverlayProps {
  message: string;
  isVisible: boolean;
}

// Plotly data types
export interface PlotlyMarker {
  color: string;
  size: number;
  opacity: number;
  symbol?: string;
  line?: {
    color: string;
    width: number;
  };
}

export interface PlotlyTrace {
  x: number[];
  y: number[];
  mode: string;
  type: string;
  name: string;
  marker: PlotlyMarker;
  line?: {
    color: string;
    dash?: string;
    width: number;
  };
}

export interface PlotlyLayout {
  title: {
    text: string;
    font: { size: number; color: string };
  };
  xaxis: { 
    title: string;
    gridcolor: string;
  };
  yaxis: { 
    title: string;
    gridcolor: string;
  };
  plot_bgcolor: string;
  paper_bgcolor: string;
  margin: { t: number; r: number; b: number; l: number };
}
