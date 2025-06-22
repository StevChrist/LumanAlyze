from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import numpy as np
import math
import json
import pandas as pd

# Import semua modules
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

# Setup logging
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
            if np.isnan(obj) or np.isinf(obj):
                return None
            try:
                converted = float(obj)
                if math.isnan(converted) or math.isinf(converted) or abs(converted) > 1e308:
                    return None
                return converted
            except (ValueError, OverflowError):
                return None
        elif isinstance(obj, np.ndarray):
            cleaned_array = np.where(np.isfinite(obj), obj, None)
            return cleaned_array.tolist()
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj) or abs(obj) > 1e308:
                return None
            return obj
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):
            try:
                item_val = obj.item()
                if isinstance(item_val, float) and (math.isnan(item_val) or math.isinf(item_val)):
                    return None
                return item_val
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

# Initialize handlers
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

# Helper function untuk clean data
def clean_response_data(data):
    """Clean data untuk menghindari JSON serialization errors dengan validasi ketat"""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            cleaned[k] = clean_response_data(v)
        return cleaned
    elif isinstance(data, list):
        return [clean_response_data(item) for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        if np.isnan(data) or np.isinf(data):
            return None
        try:
            converted = float(data)
            if math.isnan(converted) or math.isinf(converted) or abs(converted) > 1e308:
                return None
            return converted
        except (ValueError, OverflowError):
            return None
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data) or abs(data) > 1e308:
            return None
        return data
    elif pd.isna(data):
        return None
    else:
        return data

# Middleware untuk logging
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

# Upload endpoint
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )

        df, numeric_columns = await data_handler.process_csv_upload(file)
        
        metadata = {
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "numeric_columns": numeric_columns,
            "column_names": df.columns.tolist(),
            "preview": serialize_dataframe_preview(df),
            "missing_values": serialize_missing_values(df)
        }
        
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
        
        # Apply preprocessing dengan validasi ketat
        df_processed = preprocessor.preprocess_pipeline(
            df_original, missing_strategy, normalize_method,
            remove_outliers_flag, outlier_method
        )
        
        # Store processed data
        data_handler.store_preprocessed_data(df_processed)
        
        # Generate response dengan safe conversion
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
        
        # Clean response data dengan validasi ketat
        cleaned_response = clean_response_data(response_data)
        return cleaned_response
        
    except Exception as e:
        logger.error(f"PREPROCESSING FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

# Machine Learning Endpoints dengan perbaikan JSON handling
@app.post("/run-prediction")
async def run_prediction(
    target_column: str,
    model_type: str = "random_forest",
    task_type: str = "regression"
):
    try:
        # Get data dengan validasi
        try:
            df = data_handler.get_dataframe(preprocessed=True)
            logger.info("Using preprocessed data for prediction")
        except:
            df = data_handler.get_dataframe(preprocessed=False)
            logger.info("Using raw data for prediction")
        
        # Validasi target column
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Prepare data dengan cleaning tambahan
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # VALIDASI KRITIS: Bersihkan NaN dan infinity
        mask_y = pd.isna(y) | np.isinf(y)
        if mask_y.any():
            logger.warning(f"Removing {mask_y.sum()} invalid target values")
            y = y[~mask_y]
            X = X[~mask_y]
        
        # Bersihkan features
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Validasi final
        if X_numeric.isnull().any().any():
            raise HTTPException(status_code=400, detail="Data contains invalid values after cleaning")
        
        if y.isnull().any():
            raise HTTPException(status_code=400, detail="Target variable contains invalid values")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # VALIDASI PREDICTION RESULTS
            y_pred = np.where(np.isfinite(y_pred), y_pred, 0)
            
            from sklearn.metrics import r2_score, mean_squared_error
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Validasi metrics
            metrics = {
                "r2_score": float(r2) if math.isfinite(r2) else 0.0,
                "mse": float(mse) if math.isfinite(mse) else 0.0,
                "rmse": float(rmse) if math.isfinite(rmse) else 0.0
            }
        
        else:  # classification
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            }
        
        # Prepare visualization data dengan safe conversion
        viz_data = {
            "actual": [float(x) if math.isfinite(float(x)) else 0.0 for x in y_test.tolist()],
            "predicted": [float(x) if math.isfinite(float(x)) else 0.0 for x in y_pred.tolist()]
        }
        
        # Final response dengan cleaning
        response = {
            "status": "success",
            "model_type": model_type,
            "metrics": metrics,
            "visualization_data": viz_data,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test))
        }
        
        # Clean response sebelum return
        cleaned_response = clean_response_data(response)
        
        logger.info(f"PREDICTION SUCCESSFUL: {model_type} for {task_type}")
        return cleaned_response
        
    except Exception as e:
        logger.error(f"PREDICTION FAILED: {str(e)}")
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
        
        # Clean data sebelum scaling
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
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

        # Clean visualization data
        viz_data = {
            "data_points": [[float(x) if math.isfinite(x) else 0.0 for x in point] for point in X_pca.tolist()],
            "anomaly_indices": anomaly_indices,
            "anomaly_scores": [float(s) if math.isfinite(s) else 0.0 for s in scores.tolist()]
        }

        response = {
            "status": "success",
            "model_type": model_type,
            "num_anomalies": n_anomaly,
            "anomaly_percentage": float(anomaly_ratio * 100),
            "visualization_data": viz_data,
            "total_samples": int(len(X_scaled))
        }
        
        return clean_response_data(response)

    except Exception as e:
        logger.error(f"ANOMALY DETECTION FAILED: {str(e)}")
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
        
        # Clean data sebelum scaling
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
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

        # Clean visualization data
        viz_data = {
            "data_points": [[float(x) if math.isfinite(x) else 0.0 for x in point] for point in X_pca.tolist()],
            "cluster_labels": [int(x) for x in labels.tolist()],
            "cluster_centers": [[float(x) if math.isfinite(x) else 0.0 for x in center] for center in centers.tolist()] if centers is not None else None
        }

        response = {
            "status": "success",
            "model_type": model_type,
            "evaluation": {"silhouette_score": sil_score, "num_clusters": len(set(labels))},
            "visualization_data": viz_data,
            "total_samples": int(len(X_scaled))
        }
        
        return clean_response_data(response)

    except Exception as e:
        logger.error(f"SEGMENTATION FAILED: {str(e)}")
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

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    if "JSON compliant" in str(exc):
        logger.error(f"JSON Compliance Error: {str(exc)}")
        return {
            "error": "Data processing error",
            "detail": "Invalid numeric values detected. Please check your data for NaN or infinity values.",
            "suggestion": "Try preprocessing your data first or check for missing values."
        }
    raise exc

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
        "environment": "development" if settings.DEBUG else "production"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
