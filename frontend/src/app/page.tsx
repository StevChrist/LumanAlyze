'use client'

import { useState, useEffect } from 'react';

interface FileMetadata {
  filename: string;
  rows: number;
  columns: number;
  numeric_columns: string[];
  column_names: string[];
  preview: Record<string, string | number | null>[];
  missing_values: { [key: string]: number };
}

interface PreprocessingResult {
  status: string;
  original_stats: {
    rows: number;
    columns: number;
    missing_values: number;
    numeric_columns: number;
  };
  processed_stats: {
    rows: number;
    columns: number;
    missing_values: number;
    numeric_columns: number;
  };
  changes_summary: {
    rows_removed: number;
    missing_values_handled: number;
    normalization_applied: boolean;
    outliers_removed: boolean;
    preprocessing_strategy: {
      missing_values: string;
      normalization: string;
      outlier_removal: string;
    };
  };
  preview: Record<string, string | number | null>[];
  column_names: string[];
}

interface PredictionMetrics {
  r2_score?: number;
  mse?: number;
  rmse?: number;
  accuracy?: number;
}

interface VisualizationData {
  actual?: number[];
  predicted?: number[];
  feature_names?: string[];
  anomaly_indices?: number[];
  anomaly_scores?: number[];
  data_points?: number[][];
  cluster_labels?: number[];
  cluster_centers?: number[][];
}

interface ClusterEvaluation {
  silhouette_score?: number;
  num_clusters?: number;
  noise_points?: number;
}

interface ClusterStatistics {
  [key: string]: {
    size: number;
    percentage: number;
    mean_values: Record<string, number>;
  };
}

interface MLResult {
  status: string;
  model_type: string;
  metrics?: PredictionMetrics;
  visualization_data?: VisualizationData;
  num_anomalies?: number;
  anomaly_percentage?: number;
  evaluation?: ClusterEvaluation;
  cluster_statistics?: ClusterStatistics;
  training_samples?: number;
  test_samples?: number;
  total_samples?: number;
}

interface ErrorState {
  hasError: boolean;
  errorMessage: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('prediction');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string>('');
  const [fileMetadata, setFileMetadata] = useState<FileMetadata | null>(null);
  const [isClient, setIsClient] = useState(false);
  
  // State untuk preprocessing
  const [showPreprocessing, setShowPreprocessing] = useState(false);
  const [isPreprocessing, setIsPreprocessing] = useState(false);
  const [preprocessingResult, setPreprocessingResult] = useState<PreprocessingResult | null>(null);
  const [preprocessingOptions, setPreprocessingOptions] = useState({
    missingStrategy: 'mean',
    normalizeMethod: 'none',
    removeOutliers: false,
    outlierMethod: 'iqr'
  });

  // State untuk ML analysis
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [mlResult, setMlResult] = useState<MLResult | null>(null);
  const [analysisOptions, setAnalysisOptions] = useState({
    targetColumn: '',
    modelType: 'random_forest',
    taskType: 'regression',
    nClusters: 3,
    contamination: 0.1
  });

  // Error state
  const [errorState, setErrorState] = useState<ErrorState>({
    hasError: false,
    errorMessage: ''
  });

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Error handling function
  const handleError = (error: unknown, context: string) => {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error(`Error in ${context}:`, errorMessage);
    
    setErrorState({
      hasError: true,
      errorMessage: `${context}: ${errorMessage}`
    });
    
    // Clear error after 5 seconds
    setTimeout(() => {
      setErrorState({ hasError: false, errorMessage: '' });
    }, 5000);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      setSelectedFile(file);
      setUploadError('');
      setFileMetadata(null);
      
      await uploadFileToBackend(file);
    }
  };

  const uploadFileToBackend = async (file: File) => {
    setIsUploading(true);
    setUploadError('');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      
      const response = await fetch('http://localhost:8000/upload-csv', {
        method: 'POST',
        body: formData,
        mode: 'cors',
        credentials: 'omit',
        signal: controller.signal,
        headers: {}
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.status === 'success') {
        setFileMetadata(result.metadata);
        if (result.metadata.numeric_columns.length > 0) {
          setAnalysisOptions(prev => ({
            ...prev,
            targetColumn: result.metadata.numeric_columns[0]
          }));
        }
      } else {
        setUploadError(result.detail || 'Error uploading file');
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      
      if (errorMessage.includes('CORS')) {
        setUploadError('CORS Error: Backend server tidak dapat diakses. Pastikan backend berjalan di http://localhost:8000');
      } else if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
        setUploadError('Network Error: Tidak dapat terhubung ke backend. Pastikan server backend berjalan.');
      } else if (errorMessage.includes('timeout') || errorMessage.includes('aborted')) {
        setUploadError('Timeout Error: Upload memakan waktu terlalu lama. Coba dengan file yang lebih kecil.');
      } else {
        setUploadError(`Upload failed: ${errorMessage}`);
      }
    } finally {
      setIsUploading(false);
    }
  };

  const handlePreprocessData = async () => {
    setIsPreprocessing(true);
    
    try {
      const params = new URLSearchParams({
        missing_strategy: preprocessingOptions.missingStrategy,
        normalize_method: preprocessingOptions.normalizeMethod,
        remove_outliers_flag: preprocessingOptions.removeOutliers.toString(),
        outlier_method: preprocessingOptions.outlierMethod
      });
      
      const response = await fetch(`http://localhost:8000/preprocess-data?${params}`, {
        method: 'POST',
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setPreprocessingResult(result);
      
    } catch (error: unknown) {
      handleError(error, 'Preprocessing');
    } finally {
      setIsPreprocessing(false);
    }
  };

  const handleExecuteAnalysis = async () => {
    if (!fileMetadata) {
      alert('Please upload data first');
      return;
    }
    
    setIsAnalyzing(true);
    
    try {
      let endpoint = '';
      let params = {};
      
      switch (selectedCategory) {
        case 'prediction':
          endpoint = 'run-prediction';
          params = {
            target_column: analysisOptions.targetColumn,
            model_type: analysisOptions.modelType,
            task_type: analysisOptions.taskType
          };
          break;
        case 'anomaly':
          endpoint = 'detect-anomaly';
          params = {
            model_type: 'isolation_forest',
            contamination: analysisOptions.contamination
          };
          break;
        case 'segmentation':
          endpoint = 'perform-segmentation';
          params = {
            model_type: 'kmeans',
            n_clusters: analysisOptions.nClusters
          };
          break;
      }
      
      const queryParams = new URLSearchParams(params);
      const response = await fetch(`http://localhost:8000/${endpoint}?${queryParams}`, {
        method: 'POST',
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setMlResult(result);
      
    } catch (error: unknown) {
      handleError(error, 'Analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSelectChange = (
    e: React.ChangeEvent<HTMLSelectElement>,
    field: keyof typeof preprocessingOptions
  ) => {
    setPreprocessingOptions(prev => ({
      ...prev,
      [field]: e.target.value
    }));
  };

  const handleNumberChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    field: keyof typeof analysisOptions
  ) => {
    const value = field === 'contamination' ? parseFloat(e.target.value) : parseInt(e.target.value);
    setAnalysisOptions(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleHowToUseClick = () => {
    alert('How To Use page will be implemented soon!');
  };

  if (!isClient) {
    return (
      <div className="container">
        <div className="main-content">
          <div className="loading-container">
            Loading...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <header className="header">
        <div className="logo">LAM.</div>
        <nav className="nav-menu">
          <a href="#" className="nav-item">Home</a>
          <a href="#" className="nav-item">About</a>
          <a href="#" className="nav-item" onClick={handleHowToUseClick}>How To Use</a>
        </nav>
      </header>

      <main className="main-content">
        <h1 className="main-title">Welcome to LumenALYZE</h1>
        
        {/* Global Error Display */}
        {errorState.hasError && (
          <div className="error-message">
            {errorState.errorMessage}
          </div>
        )}
        
        {!showPreprocessing && !showAnalysis ? (
          <>
            {/* Upload Section */}
            <div className="upload-container">
              <div className="section-title">Drop your file bellow</div>
              <div 
                className={`upload-box ${isUploading ? 'loading' : ''}`}
                onClick={() => !isUploading && document.getElementById('file-input')?.click()}
              >
                <input
                  type="file"
                  id="file-input"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                  disabled={isUploading}
                />
                <div className="upload-icon">📁</div>
                <div className="upload-text">csv/xls file</div>
                <div className="upload-subtitle">
                  {isUploading ? 'Uploading...' : 'Click to browse or drag and drop your file here'}
                </div>
              </div>
              
              {selectedFile && (
                <div className="file-status">
                  <p className="file-selected">
                    Selected: <strong>{selectedFile.name}</strong>
                  </p>
                  {isUploading && (
                    <p className="file-processing">
                      Processing file...
                    </p>
                  )}
                </div>
              )}
              
              {uploadError && (
                <div className="error-message">
                  <div>{uploadError}</div>
                  {uploadError.includes('File encoding error') && (
                    <div className="error-help">
                      Read <span 
                        className="error-link"
                        onClick={handleHowToUseClick}
                      >
                        &quot;How To Use&quot;
                      </span> to understand how this can work
                    </div>
                  )}
                </div>
              )}
              
              {fileMetadata && (
                <div className="success-message">
                  <p><strong>File uploaded successfully!</strong></p>
                  <p>Rows: {fileMetadata.rows} | Columns: {fileMetadata.columns}</p>
                  <p>Numeric columns: {fileMetadata.numeric_columns.length}</p>
                </div>
              )}
            </div>

            {/* Data Preview */}
            {fileMetadata && (
              <div className="preview-container">
                <h3 className="preview-title">Data Preview</h3>
                <div className="preview-table-wrapper">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        {fileMetadata.column_names.map((col, idx) => (
                          <th key={idx}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {fileMetadata.preview.slice(0, 3).map((row, idx) => (
                        <tr key={idx}>
                          {fileMetadata.column_names.map((col, colIdx) => (
                            <td key={colIdx}>
                              {row[col] ?? 'N/A'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Category Section */}
            <div className="category-section">
              <div className="category-title">What Category you want to analyst?</div>
              <div className="category-options">
                <label className="category-option">
                  <input
                    type="radio"
                    name="category"
                    value="prediction"
                    checked={selectedCategory === 'prediction'}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="category-radio"
                  />
                  <span className="category-label">Prediction</span>
                </label>
                <label className="category-option">
                  <input
                    type="radio"
                    name="category"
                    value="anomaly"
                    checked={selectedCategory === 'anomaly'}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="category-radio"
                  />
                  <span className="category-label">Anomaly Detection</span>
                </label>
                <label className="category-option">
                  <input
                    type="radio"
                    name="category"
                    value="segmentation"
                    checked={selectedCategory === 'segmentation'}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="category-radio"
                  />
                  <span className="category-label">Segmentation</span>
                </label>
              </div>
            </div>

            {/* Continue Button */}
            <button
              onClick={() => {
                if (!fileMetadata) {
                  alert('Please upload a file first');
                  return;
                }
                setShowPreprocessing(true);
              }}
              disabled={!fileMetadata}
              className="execute-btn"
            >
              Continue to Preprocessing
            </button>
          </>
        ) : showPreprocessing ? (
          <>
            {/* Preprocessing Section */}
            <div className="section-container">
              <h2 className="section-heading">Data Preprocessing</h2>

              <div className="options-container">
                <h3 className="options-title">Preprocessing Options</h3>

                <div className="form-group">
                  <label className="form-label">Missing Values Strategy:</label>
                  <select
                    value={preprocessingOptions.missingStrategy}
                    onChange={(e) => handleSelectChange(e, 'missingStrategy')}
                    className="form-select"
                  >
                    <option value="mean">Fill with Mean</option>
                    <option value="median">Fill with Median</option>
                    <option value="mode">Fill with Mode</option>
                    <option value="drop">Drop Rows</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Normalization:</label>
                  <select
                    value={preprocessingOptions.normalizeMethod}
                    onChange={(e) => handleSelectChange(e, 'normalizeMethod')}
                    className="form-select"
                  >
                    <option value="none">No Normalization</option>
                    <option value="standard">Standard Scaling</option>
                    <option value="minmax">Min-Max Scaling</option>
                  </select>
                </div>

                <div className="preprocessing-checkbox">
                  <input
                    type="checkbox"
                    checked={preprocessingOptions.removeOutliers}
                    onChange={(e) => setPreprocessingOptions(prev => ({
                      ...prev,
                      removeOutliers: e.target.checked
                    }))}
                    id="remove-outliers"
                  />
                  <label htmlFor="remove-outliers">
                    Remove Outliers (IQR Method)
                  </label>
                </div>

                <button
                  onClick={handlePreprocessData}
                  disabled={isPreprocessing}
                  className="action-btn"
                >
                  {isPreprocessing ? 'Processing...' : 'Apply Preprocessing'}
                </button>
              </div>

              {preprocessingResult && (
                <div className="results-container">
                  <h3 className="results-title">Preprocessing Results</h3>

                  <div className="stats-comparison">
                    <div className="stats-before">
                      <h4>Before:</h4>
                      <p>Rows: {preprocessingResult.original_stats.rows}</p>
                      <p>Missing Values: {preprocessingResult.original_stats.missing_values}</p>
                    </div>
                    <div className="stats-after">
                      <h4>After:</h4>
                      <p>Rows: {preprocessingResult.processed_stats.rows}</p>
                      <p>Missing Values: {preprocessingResult.processed_stats.missing_values}</p>
                    </div>
                  </div>

                  <button
                    onClick={() => {
                      setShowPreprocessing(false);
                      setShowAnalysis(true);
                    }}
                    className="action-btn"
                  >
                    Continue to Analysis
                  </button>
                </div>
              )}

              <button
                onClick={() => setShowPreprocessing(false)}
                className="back-btn"
              >
                ← Back to Upload
              </button>
            </div>
          </>
        ) : (
          <>
            {/* Analysis Section */}
            <div className="section-container">
              <h2 className="section-heading">Machine Learning Analysis - {selectedCategory}</h2>

              <div className="options-container">
                <h3 className="options-title">Analysis Options</h3>

                {selectedCategory === 'prediction' && fileMetadata && (
                  <>
                    <div className="form-group">
                      <label className="form-label">Target Column:</label>
                      <select
                        value={analysisOptions.targetColumn}
                        onChange={(e) => setAnalysisOptions(prev => ({
                          ...prev,
                          targetColumn: e.target.value
                        }))}
                        className="form-select"
                      >
                        {fileMetadata.numeric_columns.map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Model Type:</label>
                      <select
                        value={analysisOptions.modelType}
                        onChange={(e) => setAnalysisOptions(prev => ({
                          ...prev,
                          modelType: e.target.value
                        }))}
                        className="form-select"
                      >
                        <option value="random_forest">Random Forest</option>
                        <option value="mlp">Neural Network (MLP)</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Task Type:</label>
                      <select
                        value={analysisOptions.taskType}
                        onChange={(e) => setAnalysisOptions(prev => ({
                          ...prev,
                          taskType: e.target.value
                        }))}
                        className="form-select"
                      >
                        <option value="regression">Regression</option>
                        <option value="classification">Classification</option>
                      </select>
                    </div>
                  </>
                )}

                {selectedCategory === 'segmentation' && (
                  <div className="form-group">
                    <label className="form-label">Number of Clusters:</label>
                    <input
                      type="number"
                      min="2"
                      max="10"
                      value={analysisOptions.nClusters}
                      onChange={(e) => handleNumberChange(e, 'nClusters')}
                      className="form-input"
                    />
                  </div>
                )}

                {selectedCategory === 'anomaly' && (
                  <div className="form-group">
                    <label className="form-label">Contamination Rate:</label>
                    <input
                      type="number"
                      min="0.01"
                      max="0.5"
                      step="0.01"
                      value={analysisOptions.contamination}
                      onChange={(e) => handleNumberChange(e, 'contamination')}
                      className="form-input"
                    />
                  </div>
                )}

                <button
                  onClick={handleExecuteAnalysis}
                  disabled={isAnalyzing}
                  className="action-btn"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
                </button>
              </div>

              {/* Results Section */}
              {mlResult && (
                <div className="results-container">
                  <h3 className="results-title">Analysis Results - {mlResult.model_type}</h3>
                  
                  {selectedCategory === 'prediction' && mlResult.metrics && (
                    <div className="metrics-display">
                      <h4>Model Performance:</h4>
                      {mlResult.metrics.r2_score && (
                        <p>R² Score: {mlResult.metrics.r2_score.toFixed(4)}</p>
                      )}
                      {mlResult.metrics.accuracy && (
                        <p>Accuracy: {(mlResult.metrics.accuracy * 100).toFixed(2)}%</p>
                      )}
                      {mlResult.training_samples && (
                        <p>Training Samples: {mlResult.training_samples} | Test Samples: {mlResult.test_samples}</p>
                      )}
                    </div>
                  )}
                  
                  {selectedCategory === 'anomaly' && (
                    <div className="metrics-display">
                      <h4>Anomaly Detection:</h4>
                      <p>Anomalies Found: {mlResult.num_anomalies} ({mlResult.anomaly_percentage?.toFixed(2)}%)</p>
                      <p>Total Samples: {mlResult.total_samples}</p>
                    </div>
                  )}
                  
                  {selectedCategory === 'segmentation' && mlResult.evaluation && (
                    <div className="metrics-display">
                      <h4>Clustering Results:</h4>
                      <p>Number of Clusters: {mlResult.evaluation.num_clusters}</p>
                      <p>Silhouette Score: {mlResult.evaluation.silhouette_score?.toFixed(4)}</p>
                      <p>Total Samples: {mlResult.total_samples}</p>
                    </div>
                  )}
                </div>
              )}

              <button
                onClick={() => {
                  setShowAnalysis(false);
                  setMlResult(null);
                }}
                className="back-btn"
              >
                ← Back to Upload
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
