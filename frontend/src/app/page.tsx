'use client'

import { useState, useEffect } from 'react';
import Link from 'next/link';
import LoadingOverlay from '../components/ui/LoadingOverlay';
import DataVisualization from '../components/visualization/DataVisualization';
import { AnalysisResult } from '../types';

type AnalysisCategory = 'prediction' | 'anomaly' | 'clustering';

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

interface ErrorState {
  hasError: boolean;
  errorMessage: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<AnalysisCategory>('prediction');
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
  const [mlResult, setMlResult] = useState<AnalysisResult | null>(null);
  const [analysisOptions, setAnalysisOptions] = useState({
    targetColumn: '',
    modelType: 'random_forest',
    taskType: 'regression',
    nClusters: 3,
    contamination: 0.1
  });

  // State untuk visualization
  const [showVisualization, setShowVisualization] = useState(false);

  // Error state
  const [errorState, setErrorState] = useState<ErrorState>({
    hasError: false,
    errorMessage: ''
  });

  // Loading message state
  const [loadingMessage, setLoadingMessage] = useState('');

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
    setLoadingMessage('Uploading and processing your file...');
    
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
      setLoadingMessage('');
    }
  };

  const handlePreprocessData = async () => {
    setIsPreprocessing(true);
    setLoadingMessage('Preprocessing your data...');
    
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
      setLoadingMessage('');
    }
  };

  const handleExecuteAnalysis = async () => {
    if (!fileMetadata) {
      alert('Please upload data first');
      return;
    }
    
    setIsAnalyzing(true);
    
    // Set loading message berdasarkan kategori
    const messages: Record<AnalysisCategory, string> = {
      prediction: 'Running machine learning prediction model...',
      anomaly: 'Detecting anomalies in your data...',
      clustering: 'Performing data clustering analysis...'
    };
    setLoadingMessage(messages[selectedCategory]);
    
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
        case 'clustering':
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
      
      // Setelah ML analysis selesai, langsung show visualization
      if (result.status === 'success') {
        setShowAnalysis(false);
        setShowVisualization(true);
      }
      
    } catch (error: unknown) {
      handleError(error, 'Analysis');
    } finally {
      setIsAnalyzing(false);
      setLoadingMessage('');
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

  const handleAnalysis = (category: AnalysisCategory) => {
    setShowVisualization(false);
    setShowAnalysis(true);
    setMlResult(null);
    setSelectedCategory(category);
  };

  // Fungsi skala log untuk error metrics
  const getMSEBarWidth = (mse: number) => {
    if (mse <= 0) return '0%';
    const min = 1e-6;
    const max = 1e-1;
    const percent = 1 - (Math.log10(mse) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
    return `${Math.max(0, Math.min(1, percent)) * 100}%`;
  };

  const getRMSEBarWidth = (rmse: number) => {
    if (rmse <= 0) return '0%';
    const min = 1e-4;
    const max = 1e-1;
    const percent = 1 - (Math.log10(rmse) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
    return `${Math.max(0, Math.min(1, percent)) * 100}%`;
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
    <>
      {/* Loading Overlay */}
      <LoadingOverlay 
        isVisible={isAnalyzing || isPreprocessing || isUploading}
        message={loadingMessage}
        analysisType={isAnalyzing ? selectedCategory : isPreprocessing ? 'preprocessing' : 'upload'}
      />

      <div className="container">
        <main className="main-content">
          {/* CONDITIONAL TITLE */}
          {!mlResult && (
            <h1 className="main-title">Welcome to LumenALYZE</h1>
          )}

          {mlResult && (
            <h1 className="results-main-title">Analysis Results</h1>
          )}
          
          {/* Global Error Display */}
          {errorState.hasError && (
            <div className="error-message">
              {errorState.errorMessage}
            </div>
          )}
          
          {!showPreprocessing && !showAnalysis && !showVisualization ? (
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
                        Read <Link href="/how-to-use" className="error-link">
                          How To Use
                        </Link> to understand how this can work
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
                  <div className="data-preview-container">
                    <table className="data-preview-table">
                      <thead>
                        <tr>
                          {fileMetadata.column_names.map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {fileMetadata.preview.map((row, idx) => (
                          <tr key={idx}>
                            {fileMetadata.column_names.map((col) => (
                              <td key={col}>{row[col]}</td>
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
                      onChange={(e) => setSelectedCategory(e.target.value as AnalysisCategory)}
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
                      onChange={(e) => setSelectedCategory(e.target.value as AnalysisCategory)}
                      className="category-radio"
                    />
                    <span className="category-label">Anomaly Detection</span>
                  </label>
                  <label className="category-option">
                    <input
                      type="radio"
                      name="category"
                      value="clustering"
                      checked={selectedCategory === 'clustering'}
                      onChange={(e) => setSelectedCategory(e.target.value as AnalysisCategory)}
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
          ) : showAnalysis ? (
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

                  {selectedCategory === 'clustering' && (
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

                <button
                  onClick={() => setShowAnalysis(false)}
                  className="back-btn"
                >
                  ← Back to Preprocessing
                </button>
              </div>
            </>
          ) : showVisualization && mlResult ? (
            <>
              <div className="section-container">
                {/* DATA VISUALIZATION */}
                <div className="visualization-container">
                  <DataVisualization
                    analysisResults={mlResult}
                    analysisType={selectedCategory}
                    isLoading={isAnalyzing}
                  />
                </div>

                {/* MODEL INFORMATION */}
                {mlResult && (
                  <div className="model-info-section">
                    <h3 className="model-info-title">Model Information</h3>
                    <table className="model-info-table">
                      <tbody>
                        <tr>
                          <td className="info-label">Model</td>
                          <td className="info-value">{mlResult.model_type}</td>
                        </tr>
                        <tr>
                          <td className="info-label">Status</td>
                          <td className="info-value status-success">{mlResult.status}</td>
                        </tr>
                        {'training_samples' in mlResult && (
                          <>
                            <tr>
                              <td className="info-label">Training</td>
                              <td className="info-value">{mlResult.training_samples}</td>
                            </tr>
                            <tr>
                              <td className="info-label">Test</td>
                              <td className="info-value">{mlResult.test_samples}</td>
                            </tr>
                          </>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* METRICS BERDASARKAN KATEGORI */}
                {selectedCategory === 'prediction' && 'metrics' in mlResult && (
                  <div className="metrics-summary-container mt-8 flex flex-col items-center">
                    <h3 className="metrics-title text-lg font-semibold mb-4">Model Performance Metrics</h3>
                    
                    {/* Primary Metrics Cards untuk Prediction */}
                    <div className="metrics-cards-grid grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                      <div className="metric-card primary bg-blue-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">🎯</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">R² Score</div>
                          <div className="metric-value text-blue-700 text-xl">
                            {mlResult.metrics?.r2_score !== undefined
                              ? mlResult.metrics.r2_score.toFixed(4)
                              : '0.0000'}
                          </div>
                          <div className="metric-description text-xs text-gray-500">Model Accuracy</div>
                        </div>
                      </div>
                      
                      <div className="metric-card success bg-green-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">✅</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">MSE</div>
                          <div className="metric-value text-green-700 text-xl">
                            {mlResult.metrics?.mse !== undefined
                              ? mlResult.metrics.mse.toFixed(4)
                              : '0.0000'}
                          </div>
                          <div className="metric-description text-xs text-gray-500">Mean Squared Error</div>
                        </div>
                      </div>
                      
                      <div className="metric-card info bg-indigo-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">📊</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">RMSE</div>
                          <div className="metric-value text-indigo-700 text-xl">
                            {mlResult.metrics?.rmse !== undefined
                              ? mlResult.metrics.rmse.toFixed(4)
                              : '0.0000'}
                          </div>
                          <div className="metric-description text-xs text-gray-500">Root Mean Squared Error</div>
                        </div>
                      </div>
                    </div>

                    {/* Detailed Metrics Table */}
                    <div className="detailed-metrics-section w-full max-w-2xl mb-8">
                      <h4 className="detailed-metrics-title font-semibold mb-2">Detailed Performance Metrics</h4>
                      <div className="metrics-table-container overflow-x-auto">
                        <table className="metrics-table min-w-full border border-gray-200 rounded-lg bg-white text-sm shadow">
                          <thead>
                            <tr className="bg-gray-50">
                              <th className="px-4 py-2 text-left">Metric</th>
                              <th className="px-4 py-2 text-left">Value</th>
                              <th className="px-4 py-2 text-left">Performance</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="table-row-primary">
                              <td className="metric-name px-4 py-2 font-medium">
                                <span className="metric-dot r2 inline-block w-2 h-2 rounded-full bg-blue-500 mr-2"></span>
                                R² Score
                              </td>
                              <td className="metric-value-cell px-4 py-2">
                                {mlResult.metrics?.r2_score !== undefined
                                  ? mlResult.metrics.r2_score.toFixed(4)
                                  : '0.0000'}
                              </td>
                              <td className="performance-indicator px-4 py-2">
                                <div className="performance-bar bg-gray-200 h-2 rounded">
                                  <div 
                                    className="performance-fill excellent bg-blue-500 h-2 rounded"
                                    style={{ 
                                      width: `${Math.min(100, (mlResult.metrics?.r2_score || 0) * 100)}%` 
                                    }}
                                  ></div>
                                </div>
                                <span className="performance-text ml-2 text-xs text-blue-700">Excellent</span>
                              </td>
                            </tr>
                            
                            <tr>
                              <td className="metric-name px-4 py-2 font-medium">
                                <span className="metric-dot mse inline-block w-2 h-2 rounded-full bg-green-500 mr-2"></span>
                                Mean Squared Error
                              </td>
                              <td className="metric-value-cell px-4 py-2">
                                {mlResult.metrics?.mse !== undefined
                                  ? mlResult.metrics.mse.toFixed(4)
                                  : '0.0000'}
                              </td>
                              <td className="performance-indicator px-4 py-2">
                                <div className="performance-bar bg-gray-200 h-2 rounded">
                                  <div
                                    className="performance-fill good bg-green-500 h-2 rounded"
                                    style={{ width: getMSEBarWidth(mlResult.metrics?.mse ?? 0) }}
                                  ></div>
                                </div>
                                <span className="performance-text ml-2 text-xs text-green-700">Very Low</span>
                              </td>
                            </tr>
                            
                            <tr>
                              <td className="metric-name px-4 py-2 font-medium">
                                <span className="metric-dot rmse inline-block w-2 h-2 rounded-full bg-indigo-500 mr-2"></span>
                                Root Mean Squared Error
                              </td>
                              <td className="metric-value-cell px-4 py-2">
                                {mlResult.metrics?.rmse !== undefined
                                  ? mlResult.metrics.rmse.toFixed(4)
                                  : '0.0000'}
                              </td>
                              <td className="performance-indicator px-4 py-2">
                                <div className="performance-bar bg-gray-200 h-2 rounded">
                                  <div
                                    className="performance-fill good bg-indigo-500 h-2 rounded"
                                    style={{ width: getRMSEBarWidth(mlResult.metrics?.rmse ?? 0) }}
                                  ></div>
                                </div>
                                <span className="performance-text ml-2 text-xs text-indigo-700">Very Low</span>
                              </td>
                            </tr>

                            {/* Classification Metrics (jika ada) */}
                            {mlResult.metrics?.accuracy !== undefined && (
                              <>
                                <tr>
                                  <td className="metric-name px-4 py-2 font-medium">
                                    <span className="metric-dot accuracy inline-block w-2 h-2 rounded-full bg-yellow-500 mr-2"></span>
                                    Accuracy
                                  </td>
                                  <td className="metric-value-cell px-4 py-2">
                                    {(mlResult.metrics.accuracy * 100).toFixed(2)}%
                                  </td>
                                  <td className="performance-indicator px-4 py-2">
                                    <div className="performance-bar bg-gray-200 h-2 rounded">
                                      <div 
                                        className="performance-fill excellent bg-yellow-500 h-2 rounded"
                                        style={{ width: `${mlResult.metrics.accuracy * 100}%` }}
                                      ></div>
                                    </div>
                                    <span className="performance-text ml-2 text-xs text-yellow-700">Excellent</span>
                                  </td>
                                </tr>
                                
                                <tr>
                                  <td className="metric-name px-4 py-2 font-medium">
                                    <span className="metric-dot precision inline-block w-2 h-2 rounded-full bg-purple-500 mr-2"></span>
                                    Precision
                                  </td>
                                  <td className="metric-value-cell px-4 py-2">
                                    {(mlResult.metrics.precision! * 100).toFixed(2)}%
                                  </td>
                                  <td className="performance-indicator px-4 py-2">
                                    <div className="performance-bar bg-gray-200 h-2 rounded">
                                      <div 
                                        className="performance-fill good bg-purple-500 h-2 rounded"
                                        style={{ width: `${mlResult.metrics.precision! * 100}%` }}
                                      ></div>
                                    </div>
                                    <span className="performance-text ml-2 text-xs text-purple-700">Good</span>
                                  </td>
                                </tr>

                                <tr>
                                  <td className="metric-name px-4 py-2 font-medium">
                                    <span className="metric-dot recall inline-block w-2 h-2 rounded-full bg-orange-500 mr-2"></span>
                                    Recall
                                  </td>
                                  <td className="metric-value-cell px-4 py-2">
                                    {(mlResult.metrics.recall! * 100).toFixed(2)}%
                                  </td>
                                  <td className="performance-indicator px-4 py-2">
                                    <div className="performance-bar bg-gray-200 h-2 rounded">
                                      <div 
                                        className="performance-fill good bg-orange-500 h-2 rounded"
                                        style={{ width: `${mlResult.metrics.recall! * 100}%` }}
                                      ></div>
                                    </div>
                                    <span className="performance-text ml-2 text-xs text-orange-700">Good</span>
                                  </td>
                                </tr>

                                <tr>
                                  <td className="metric-name px-4 py-2 font-medium">
                                    <span className="metric-dot f1 inline-block w-2 h-2 rounded-full bg-pink-500 mr-2"></span>
                                    F1 Score
                                  </td>
                                  <td className="metric-value-cell px-4 py-2">
                                    {(mlResult.metrics.f1_score! * 100).toFixed(2)}%
                                  </td>
                                  <td className="performance-indicator px-4 py-2">
                                    <div className="performance-bar bg-gray-200 h-2 rounded">
                                      <div 
                                        className="performance-fill good bg-pink-500 h-2 rounded"
                                        style={{ width: `${mlResult.metrics.f1_score! * 100}%` }}
                                      ></div>
                                    </div>
                                    <span className="performance-text ml-2 text-xs text-pink-700">Good</span>
                                  </td>
                                </tr>
                              </>
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {/* METRICS UNTUK ANOMALY DETECTION */}
                {selectedCategory === 'anomaly' && 'num_anomalies' in mlResult && (
                  <div className="metrics-summary-container mt-8 flex flex-col items-center">
                    <h3 className="metrics-title text-lg font-semibold mb-4">Anomaly Detection Results</h3>
                    
                    <div className="metrics-cards-grid grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                      <div className="metric-card primary bg-red-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">🚨</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">Anomalies Found</div>
                          <div className="metric-value text-red-700 text-xl">{mlResult.num_anomalies}</div>
                          <div className="metric-description text-xs text-gray-500">Total anomalous data points</div>
                        </div>
                      </div>
                      
                      <div className="metric-card info bg-orange-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">📊</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">Anomaly Rate</div>
                          <div className="metric-value text-orange-700 text-xl">
                            {mlResult.anomaly_percentage?.toFixed(2)}%
                          </div>
                          <div className="metric-description text-xs text-gray-500">Percentage of anomalies</div>
                        </div>
                      </div>
                      
                      <div className="metric-card success bg-blue-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">🧮</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">Total Samples</div>
                          <div className="metric-value text-blue-700 text-xl">{mlResult.total_samples}</div>
                          <div className="metric-description text-xs text-gray-500">Data points analyzed</div>
                        </div>
                      </div>
                    </div>

                    {/* Detailed Metrics Table untuk Anomaly */}
                    <div className="detailed-metrics-section w-full max-w-2xl mb-8">
                      <h4 className="detailed-metrics-title font-semibold mb-2">Detailed Anomaly Detection Metrics</h4>
                      <div className="metrics-table-container overflow-x-auto">
                        <table className="metrics-table min-w-full border border-gray-200 rounded-lg bg-white text-sm shadow">
                          <thead>
                            <tr className="bg-gray-50">
                              <th className="px-4 py-2 text-left">Metric</th>
                              <th className="px-4 py-2 text-left">Value</th>
                              <th className="px-4 py-2 text-left">Performance</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td className="metric-name px-4 py-2 font-medium">
                                <span className="metric-dot inline-block w-2 h-2 rounded-full bg-red-500 mr-2"></span>
                                Anomalies Found
                              </td>
                              <td className="metric-value-cell px-4 py-2">{mlResult.num_anomalies}</td>
                              <td className="performance-indicator px-4 py-2">
                                <div className="performance-bar bg-gray-200 h-2 rounded">
                                  <div className="performance-fill good bg-red-500 h-2 rounded" style={{ width: `${Math.min(mlResult.anomaly_percentage || 0, 100)}%` }}></div>
                                </div>
                                <span className="performance-text ml-2 text-xs text-red-700">Detected</span>
                              </td>
                            </tr>
                            <tr>
                              <td className="metric-name px-4 py-2 font-medium">
                                <span className="metric-dot inline-block w-2 h-2 rounded-full bg-orange-500 mr-2"></span>
                                Anomaly Rate
                              </td>
                              <td className="metric-value-cell px-4 py-2">{mlResult.anomaly_percentage?.toFixed(2)}%</td>
                              <td className="performance-indicator px-4 py-2">
                                <div className="performance-bar bg-gray-200 h-2 rounded">
                                  <div className="performance-fill average bg-orange-500 h-2 rounded" style={{ width: `${mlResult.anomaly_percentage}%` }}></div>
                                </div>
                                <span className="performance-text ml-2 text-xs text-orange-700">Ratio</span>
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {/* METRICS UNTUK CLUSTERING */}
                {selectedCategory === 'clustering' && 'evaluation' in mlResult && (
                  <div className="metrics-summary-container mt-8 flex flex-col items-center">
                    <h3 className="metrics-title text-lg font-semibold mb-4">Clustering Analysis Results</h3>
                    
                    <div className="metrics-cards-grid grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                      <div className="metric-card primary bg-purple-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">🔢</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">Clusters</div>
                          <div className="metric-value text-purple-700 text-xl">
                            {mlResult.evaluation.num_clusters}
                          </div>
                          <div className="metric-description text-xs text-gray-500">Number of clusters formed</div>
                        </div>
                      </div>
                      
                      <div className="metric-card info bg-teal-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">📈</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">Silhouette Score</div>
                          <div className="metric-value text-teal-700 text-xl">
                            {mlResult.evaluation.silhouette_score?.toFixed(3)}
                          </div>
                          <div className="metric-description text-xs text-gray-500">Cluster separation quality</div>
                        </div>
                      </div>
                      
                      <div className="metric-card success bg-green-50 rounded-lg p-4 flex items-center shadow">
                        <div className="metric-icon text-3xl mr-4">🧮</div>
                        <div className="metric-content">
                          <div className="metric-label font-semibold">Total Samples</div>
                          <div className="metric-value text-green-700 text-xl">{mlResult.total_samples}</div>
                          <div className="metric-description text-xs text-gray-500">Data points analyzed</div>
                        </div>
                      </div>
                    </div>

                    {/* Detailed Metrics Table untuk Clustering */}
                    <div className="detailed-metrics-section w-full max-w-2xl mb-8">
                      <h4 className="detailed-metrics-title font-semibold mb-2">Detailed Clustering Metrics</h4>
                      <div className="metrics-table-container overflow-x-auto">
                        <table className="metrics-table min-w-full border border-gray-200 rounded-lg bg-white text-sm shadow">
                          <thead>
                            <tr className="bg-gray-50">
                              <th className="px-4 py-2 text-left">Metric</th>
                              <th className="px-4 py-2 text-left">Value</th>
                              <th className="px-4 py-2 text-left">Performance</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td className="metric-name px-4 py-2 font-medium">
                                <span className="metric-dot inline-block w-2 h-2 rounded-full bg-teal-500 mr-2"></span>
                                Silhouette Score
                              </td>
                              <td className="metric-value-cell px-4 py-2">{mlResult.evaluation.silhouette_score?.toFixed(3)}</td>
                              <td className="performance-indicator px-4 py-2">
                                <div className="performance-bar bg-gray-200 h-2 rounded">
                                  <div className="performance-fill good bg-teal-500 h-2 rounded" style={{ width: `${Math.max(0, Math.min(1, mlResult.evaluation.silhouette_score || 0)) * 100}%` }}></div>
                                </div>
                                <span className="performance-text ml-2 text-xs text-teal-700">{(mlResult.evaluation.silhouette_score || 0) > 0.5 ? 'Good' : 'Average'}</span>
                              </td>
                            </tr>
                            <tr>
                              <td className="metric-name px-4 py-2 font-medium">
                                <span className="metric-dot inline-block w-2 h-2 rounded-full bg-purple-500 mr-2"></span>
                                Number of Clusters
                              </td>
                              <td className="metric-value-cell px-4 py-2">{mlResult.evaluation.num_clusters}</td>
                              <td className="performance-indicator px-4 py-2">
                                <div className="performance-bar bg-gray-200 h-2 rounded">
                                  <div className="performance-fill average bg-purple-500 h-2 rounded" style={{ width: `${Math.min(mlResult.evaluation.num_clusters * 10, 100)}%` }}></div>
                                </div>
                                <span className="performance-text ml-2 text-xs text-purple-700">Count</span>
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {/* WHAT'S NEXT SECTION - UNIVERSAL */}
                <div className="whats-next-card-container" style={{
                  maxWidth: 700,
                  margin: "2rem auto",
                  background: "rgba(255,255,255,0.04)",
                  borderRadius: "18px",
                  boxShadow: "0 4px 24px 0 rgba(0,0,0,0.18)",
                  border: "2px solid #ffeaa7",
                  padding: "2.5rem 2rem",
                  textAlign: "center"
                }}>
                  <h3 className="whats-next-title" style={{
                    fontSize: "2.2rem",
                    fontWeight: 700,
                    color: "#fff59d",
                    marginBottom: "1rem"
                  }}>What is Next?</h3>
                  <p className="whats-next-subtitle" style={{
                    color: "#fffde7",
                    fontSize: "1.1rem",
                    marginBottom: "2.2rem"
                  }}>
                    Explore different analysis types or start fresh with new data
                  </p>
                  <div className="action-buttons-section" style={{
                    display: "flex",
                    gap: "1.5rem",
                    justifyContent: "center",
                    marginBottom: "2.5rem"
                  }}>
                    <button 
                      className={`btn-primary-action${selectedCategory === 'prediction' ? ' active' : ''}`} 
                      onClick={() => handleAnalysis('prediction')}
                    >
                      🎯 TRY PREDICTION
                    </button>
                    <button 
                      className={`btn-primary-action${selectedCategory === 'anomaly' ? ' active' : ''}`} 
                      onClick={() => handleAnalysis('anomaly')}
                    >
                      🔍 TRY ANOMALY DETECTION
                    </button>
                    <button 
                      className={`btn-primary-action${selectedCategory === 'clustering' ? ' active' : ''}`} 
                      onClick={() => handleAnalysis('clustering')}
                    >
                      🧬 TRY CLUSTERING
                    </button>
                  </div>
                  <div style={{
                    display: 'flex',
                    gap: '1rem',
                    justifyContent: 'center'
                  }}>
                    <button className="btn-secondary-action" onClick={() => setShowAnalysis(true)}>
                      ⬅️ Back to Analysis Options
                    </button>
                    <button className="btn-secondary-action" onClick={() => window.location.reload()}>
                      🆕 Start Analytics Again
                    </button>
                  </div>
                </div>
              </div>
            </>
          ) : null}
        </main>
      </div>
    </>
  );
}
