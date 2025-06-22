from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import numpy as np
import math
import json

# Import semua modules di bagian atas dengan urutan yang benar
from utils.data_handler import DataHandler
from utils.json_serializer import safe_json_serializer, serialize_dataframe_preview, serialize_missing_values, clean_data_for_json
from utils.chart_data_formatter import ChartDataFormatter
from utils.export_handler import ExportHandler

from models.preprocessing import DataPreprocessor
from models.prediction import PredictionModel
from models.anomaly_detection import AnomalyDetector
from models.segmentation import DataSegmentation
from models.visualization import DataVisualization

from services.analytics_service import AnalyticsService
from services.dashboard_service import DashboardService
from services.report_service import ReportService

from config.settings import settings
from config.chart_config import CHART_THEMES, CHART_LAYOUTS, DEFAULT_CHART_CONFIG

# Setup logging dengan konfigurasi yang benar
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lumenalyze.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Custom JSON encoder untuk mengatasi error serialization
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj) or np.isinf(obj) or math.isnan(float(obj)) or math.isinf(float(obj)):
                return None
            return obj.item()
        elif isinstance(obj, np.ndarray):
            # Clean array dari infinity/NaN values
            cleaned_array = np.where(np.isfinite(obj), obj, None)
            return cleaned_array.tolist()
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            try:
                return obj.item()
            except:
                return str(obj)
        return super().default(obj)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize semua handlers di satu tempat
data_handler = DataHandler()
preprocessor = DataPreprocessor()
prediction_model = PredictionModel()
anomaly_detector = AnomalyDetector()
segmentation_model = DataSegmentation()
data_visualization = DataVisualization()
chart_formatter = ChartDataFormatter()
export_handler = ExportHandler()
analytics_service = AnalyticsService()
dashboard_service = DashboardService()
report_service = ReportService()

# Helper function untuk clean data sebelum JSON response
def clean_response_data(data):
    """Clean data untuk menghindari JSON serialization errors"""
    return clean_data_for_json(data)

# Middleware untuk logging dengan error handling yang lebih baik
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"INCOMING REQUEST: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        process_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"RESPONSE: Status {response.status_code} | Time: {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"REQUEST FAILED: {str(e)}")
        raise

# Basic endpoints
@app.get("/")
def root():
    return {
        "message": f"{settings.APP_NAME} API is running",
        "status": "ok",
        "version": settings.APP_VERSION
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now().isoformat()
    }

@app.options("/upload-csv")
async def upload_csv_options():
    return {"message": "OK"}

# Upload endpoint dengan error handling yang diperbaiki
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Validate file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        df, numeric_columns = await data_handler.process_csv_upload(file)
        
        # Clean data sebelum create metadata
        metadata = {
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "numeric_columns": numeric_columns,
            "column_names": df.columns.tolist(),
            "preview": serialize_dataframe_preview(df),
            "missing_values": serialize_missing_values(df)
        }
        
        # Clean metadata untuk menghindari JSON errors
        cleaned_metadata = clean_response_data(metadata)
        
        logger.info(f"UPLOAD SUCCESSFUL: {file.filename}")
        return {"status": "success", "metadata": cleaned_metadata}
        
    except Exception as e:
        logger.error(f"UPLOAD FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

# Preprocessing endpoint dengan perbaikan JSON serialization
@app.post("/preprocess-data")
async def preprocess_data(
    missing_strategy: str = "mean",
    normalize_method: str = "none",
    remove_outliers_flag: bool = False,
    outlier_method: str = "iqr"
):
    try:
        df_original = data_handler.get_dataframe()
        
        # Apply preprocessing
        df_processed = preprocessor.preprocess_pipeline(
            df_original, missing_strategy, normalize_method, 
            remove_outliers_flag, outlier_method
        )
        
        # Store processed data
        data_handler.store_preprocessed_data(df_processed)
        
        # Generate response with statistics - dengan safe conversion
        response_data = {
            "status": "success",
            "original_stats": {
                "rows": int(len(df_original)),
                "columns": int(len(df_original.columns)),
                "missing_values": int(df_original.isnull().sum().sum()),
                "numeric_columns": len(df_original.select_dtypes(include=['number']).columns)
            },
            "processed_stats": {
                "rows": int(len(df_processed)),
                "columns": int(len(df_processed.columns)),
                "missing_values": int(df_processed.isnull().sum().sum()),
                "numeric_columns": len(df_processed.select_dtypes(include=['number']).columns)
            },
            "changes_summary": {
                "rows_removed": int(len(df_original) - len(df_processed)),
                "missing_values_handled": int(df_original.isnull().sum().sum() - df_processed.isnull().sum().sum()),
                "normalization_applied": normalize_method != "none",
                "outliers_removed": remove_outliers_flag,
                "preprocessing_strategy": {
                    "missing_values": missing_strategy,
                    "normalization": normalize_method,
                    "outlier_removal": outlier_method if remove_outliers_flag else "none"
                }
            },
            "preview": serialize_dataframe_preview(df_processed),
            "column_names": df_processed.columns.tolist()
        }
        
        # Clean response data
        cleaned_response = clean_response_data(response_data)
        
        return cleaned_response
        
    except Exception as e:
        logger.error(f"PREPROCESSING FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

# Machine Learning Endpoints dengan perbaikan JSON handling

@app.post("/run-prediction")
async def run_prediction(
    target_column: str,
    model_type: str = "random_forest",  # random_forest atau mlp
    task_type: str = "regression"       # regression atau classification
):
    try:
        # Get data
        try:
            df = data_handler.get_dataframe(preprocessed=True)
        except:
            df = data_handler.get_dataframe(preprocessed=False)

        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train & predict
        if task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            from sklearn.metrics import r2_score, mean_squared_error
            metrics = {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(mean_squared_error(y_test, y_pred, squared=False))
            }
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            }

        # Visualisasi
        viz_data = {
            "actual": [float(x) for x in y_test.tolist()],
            "predicted": [float(x) for x in y_pred.tolist()]
        }
        if task_type == "classification":
            cm = confusion_matrix(y_test, y_pred)
            viz_data["confusion_matrix"] = cm.tolist()
            viz_data["labels"] = list(set(y_test) | set(y_pred))

        return {
            "status": "success",
            "model_type": model_type,
            "metrics": metrics,
            "visualization_data": viz_data,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-anomaly")
async def detect_anomaly(
    model_type: str = "isolation_forest",
    contamination: float = 0.05
):
    try:
        try:
            df = data_handler.get_dataframe(preprocessed=True)
        except:
            df = data_handler.get_dataframe(preprocessed=False)
        from sklearn.preprocessing import StandardScaler
        X = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X_scaled)
            scores = model.decision_function(X_scaled)
        else:
            from sklearn.svm import OneClassSVM
            model = OneClassSVM(nu=contamination)
            preds = model.fit_predict(X_scaled)
            scores = model.decision_function(X_scaled)

        anomaly_indices = [i for i, v in enumerate(preds) if v == -1]
        n_anomaly = len(anomaly_indices)
        n_normal = len(preds) - n_anomaly
        anomaly_ratio = n_anomaly / len(preds) if len(preds) > 0 else 0.0

        # Visualisasi (PCA 2D)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        viz_data = {
            "data_points": X_pca.tolist(),
            "anomaly_indices": anomaly_indices,
            "anomaly_scores": scores.tolist()
        }

        return {
            "status": "success",
            "model_type": model_type,
            "num_anomalies": n_anomaly,
            "anomaly_percentage": anomaly_ratio * 100,
            "visualization_data": viz_data,
            "total_samples": int(len(X_scaled))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/perform-segmentation")
async def perform_segmentation(
    model_type: str = "kmeans",
    n_clusters: int = 3
):
    try:
        try:
            df = data_handler.get_dataframe(preprocessed=True)
        except:
            df = data_handler.get_dataframe(preprocessed=False)
        from sklearn.preprocessing import StandardScaler
        X = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == "kmeans":
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            centers = model.cluster_centers_
        else:
            from sklearn.cluster import DBSCAN
            model = DBSCAN()
            labels = model.fit_predict(X_scaled)
            centers = None

        from sklearn.metrics import silhouette_score
        sil_score = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

        # Visualisasi (PCA 2D)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        viz_data = {
            "data_points": X_pca.tolist(),
            "cluster_labels": [int(x) for x in labels.tolist()],
            "cluster_centers": centers.tolist() if centers is not None else None
        }

        return {
            "status": "success",
            "model_type": model_type,
            "evaluation": {"silhouette_score": sil_score, "num_clusters": len(set(labels))},
            "visualization_data": viz_data,
            "total_samples": int(len(X_scaled))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional utility endpoints

@app.get("/preprocessing-status")
def get_preprocessing_status():
    """Check apakah data sudah di-preprocess"""
    logger.info("Checking preprocessing status")
    
    if 'dataframe' not in data_handler.uploaded_data:
        return {"status": "no_data", "message": "No data uploaded"}
    
    preprocessing_applied = data_handler.uploaded_data.get('preprocessing_applied', False)
    
    if preprocessing_applied:
        df = data_handler.uploaded_data['preprocessed_dataframe']
        return {
            "status": "preprocessed",
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": df.columns.tolist(),
            "ready_for_analysis": True
        }
    else:
        df = data_handler.uploaded_data['dataframe']
        return {
            "status": "raw",
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": df.columns.tolist(),
            "ready_for_analysis": False
        }

@app.get("/data-info")
def get_data_info():
    """Get information about uploaded data"""
    logger.info("Data info endpoint called")
    if 'dataframe' not in data_handler.uploaded_data:
        logger.warning("No data found in memory")
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    df = data_handler.uploaded_data['dataframe']
    logger.info(f"Data info retrieved for: {data_handler.uploaded_data['filename']}")
    
    # Convert dtypes to safe format
    dtypes_dict = {}
    for col, dtype in df.dtypes.items():
        dtypes_dict[col] = str(dtype)
    
    return {
        "filename": data_handler.uploaded_data['filename'],
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "dtypes": dtypes_dict,
        "memory_usage": float(df.memory_usage(deep=True).sum() / 1024 / 1024),  # MB
        "missing_values": clean_response_data(df.isnull().sum().to_dict())
    }

# Fase 5 endpoints dengan perbaikan

@app.post("/generate-chart-data")
async def generate_chart_data(
    analysis_type: str,
    chart_type: str = "scatter"
):
    """Generate formatted data untuk frontend charts"""
    try:
        # Get latest analysis results from memory
        if analysis_type == 'prediction' and 'preprocessed_dataframe' in data_handler.uploaded_data:
            # Sample data untuk testing
            sample_data = {
                'actual': [1.0, 2.0, 3.0, 4.0, 5.0],
                'predicted': [1.1, 1.9, 3.2, 3.8, 5.1],
                'feature_names': ['feature_1', 'feature_2']
            }
            
            chart_data = data_visualization.prepare_prediction_chart_data(
                sample_data['actual'],
                sample_data['predicted'], 
                sample_data['feature_names']
            )
            
            formatted_data = chart_formatter.format_for_plotly(chart_data, chart_type)
            
            return {
                "status": "success",
                "chart_data": clean_response_data(formatted_data),
                "chart_type": chart_type
            }
        
        else:
            raise HTTPException(status_code=404, detail="No analysis results found")
            
    except Exception as e:
        logger.error(f"CHART DATA GENERATION FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart data: {str(e)}")

@app.post("/export-results")
async def export_results(
    export_format: str = "csv",
    analysis_results: dict = None
):
    """Export analysis results dalam berbagai format"""
    try:
        if export_format not in settings.EXPORT_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported export format. Supported: {settings.EXPORT_FORMATS}")
        
        if analysis_results is None:
            # Get latest results from memory (placeholder)
            analysis_results = {"status": "no_data", "message": "No analysis results available"}
        
        if export_format.lower() == 'csv':
            result = export_handler.export_to_csv(analysis_results)
        elif export_format.lower() == 'json':
            result = export_handler.export_to_json(analysis_results)
        elif export_format.lower() == 'excel':
            result = export_handler.export_to_excel(analysis_results)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        logger.info(f"EXPORT SUCCESSFUL: {export_format}")
        return clean_response_data(result)
        
    except Exception as e:
        logger.error(f"EXPORT FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during export: {str(e)}")

@app.post("/generate-pdf-report")
async def generate_pdf_report(analysis_results: dict = None):
    """Generate comprehensive PDF report"""
    try:
        if analysis_results is None:
            analysis_results = {
                "model_type": "RandomForest_regression",
                "status": "success",
                "metrics": {"r2_score": 0.85, "mse": 0.12},
                "training_samples": 800,
                "test_samples": 200
            }
        
        pdf_report = report_service.generate_pdf_report(analysis_results)
        
        logger.info("PDF REPORT GENERATED")
        return clean_response_data(pdf_report)
        
    except Exception as e:
        logger.error(f"PDF REPORT GENERATION FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

@app.get("/generate-summary-report")
async def generate_summary_report():
    """Generate comprehensive summary report"""
    try:
        # Get latest analysis results (placeholder)
        analysis_results = {
            "model_type": "RandomForest_regression",
            "status": "success",
            "metrics": {"r2_score": 0.85, "mse": 0.12},
            "training_samples": 800,
            "test_samples": 200
        }
        
        summary_report = export_handler.generate_summary_report(analysis_results)
        
        logger.info("SUMMARY REPORT GENERATED")
        return {
            "status": "success",
            "report": clean_response_data(summary_report)
        }
        
    except Exception as e:
        logger.error(f"SUMMARY REPORT GENERATION FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary report: {str(e)}")

@app.get("/analytics-insights")
async def get_analytics_insights(analysis_id: str = "latest"):
    """Get automated insights from analysis results"""
    try:
        # Placeholder analysis results
        analysis_results = {
            "model_type": "RandomForest_regression",
            "status": "success",
            "metrics": {"r2_score": 0.85, "mse": 0.12},
            "training_samples": 800,
            "test_samples": 200
        }
        
        insights = analytics_service.generate_insights(analysis_results)
        
        logger.info("ANALYTICS INSIGHTS GENERATED")
        return {
            "status": "success",
            "insights": clean_response_data(insights)
        }
        
    except Exception as e:
        logger.error(f"ANALYTICS INSIGHTS FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@app.post("/dashboard-data")
async def get_dashboard_data():
    """Get dashboard data for analytics overview"""
    try:
        # Sample analysis history
        analysis_history = [
            {
                "model_type": "RandomForest_regression",
                "status": "success",
                "metrics": {"r2_score": 0.85},
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        dashboard_data = dashboard_service.prepare_dashboard_data(analysis_history)
        
        logger.info("DASHBOARD DATA GENERATED")
        return {
            "status": "success",
            "dashboard": clean_response_data(dashboard_data)
        }
        
    except Exception as e:
        logger.error(f"DASHBOARD DATA FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating dashboard data: {str(e)}")

# Error handler untuk debugging yang diperbaiki
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL ERROR: {str(exc)}")
    logger.exception("Full traceback:")
    return {
        "error": "Internal server error", 
        "detail": str(exc),
        "path": str(request.url),
        "timestamp": datetime.now().isoformat()
    }

# Health check untuk monitoring
@app.get("/api/status")
def api_status():
    """Extended health check with system status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION,
        "app_name": settings.APP_NAME,
        "environment": "development" if settings.DEBUG else "production",
        "endpoints": {
            "upload": "/upload-csv",
            "preprocess": "/preprocess-data",
            "prediction": "/run-prediction",
            "anomaly": "/detect-anomaly",
            "segmentation": "/perform-segmentation",
            "chart_data": "/generate-chart-data",
            "export": "/export-results",
            "insights": "/analytics-insights",
            "dashboard": "/dashboard-data",
            "pdf_report": "/generate-pdf-report"
        },
        "configuration": {
            "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
            "allowed_file_types": settings.ALLOWED_FILE_TYPES,
            "export_formats": settings.EXPORT_FORMATS,
            "cors_origins": settings.CORS_ORIGINS
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
