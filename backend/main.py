from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import numpy as np

# Import modules yang sudah dipisah
from utils.data_handler import DataHandler
from utils.json_serializer import safe_json_serializer, serialize_dataframe_preview, serialize_missing_values
from models.preprocessing import DataPreprocessor
from models.prediction import PredictionModel
from models.anomaly_detection import AnomalyDetector
from models.segmentation import DataSegmentation

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

app = FastAPI(title="LumenALYZE API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000"
    ],
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
    return {"message": "LumenALYZE API is running", "status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok", "app": "LumenALYZE", "version": "1.0.0"}

@app.options("/upload-csv")
async def upload_csv_options():
    return {"message": "OK"}

# Upload endpoint
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
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
        
        logger.info(f"UPLOAD SUCCESSFUL: {file.filename}")
        return {"status": "success", "metadata": metadata}
        
    except Exception as e:
        logger.error(f"UPLOAD FAILED: {str(e)}")
        raise

# Preprocessing endpoint
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
        
        # Generate response with statistics
        response_data = {
            "status": "success",
            "original_stats": {
                "rows": len(df_original),
                "columns": len(df_original.columns),
                "missing_values": df_original.isnull().sum().sum()
            },
            "processed_stats": {
                "rows": len(df_processed),
                "columns": len(df_processed.columns),
                "missing_values": df_processed.isnull().sum().sum()
            },
            "preview": serialize_dataframe_preview(df_processed),
            "column_names": df_processed.columns.tolist()
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"PREPROCESSING FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

# Machine Learning Endpoints

@app.post("/run-prediction")
async def run_prediction(
    target_column: str,
    model_type: str = "random_forest",  # random_forest atau mlp
    task_type: str = "regression"       # regression atau classification
):
    try:
        # Get preprocessed data atau raw data
        try:
            df = data_handler.get_dataframe(preprocessed=True)
            logger.info("Using preprocessed data for prediction")
        except:
            df = data_handler.get_dataframe(preprocessed=False)
            logger.info("Using raw data for prediction")
        
        # Prepare data
        X_train, X_test, y_train, y_test = prediction_model.prepare_data(df, target_column)
        
        # Train model
        if model_type == "random_forest":
            model = prediction_model.train_random_forest(X_train, y_train, task_type)
        else:
            model = prediction_model.train_mlp(X_train, y_train, task_type)
        
        # Evaluate model
        metrics = prediction_model.evaluate_model(X_test, y_test)
        
        # Make predictions on test set
        predictions = prediction_model.predict(X_test)
        
        # Prepare visualization data
        viz_data = {
            "actual": y_test.tolist(),
            "predicted": predictions.tolist(),
            "feature_names": X_train.columns.tolist()
        }
        
        response_data = {
            "status": "success",
            "model_type": prediction_model.model_type,
            "metrics": metrics,
            "visualization_data": viz_data,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        logger.info(f"PREDICTION SUCCESSFUL: {model_type} for {task_type}")
        return response_data
        
    except Exception as e:
        logger.error(f"PREDICTION FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/detect-anomaly")
async def detect_anomaly(
    model_type: str = "isolation_forest",  # isolation_forest atau one_class_svm
    contamination: float = 0.1
):
    try:
        # Get data
        try:
            df = data_handler.get_dataframe(preprocessed=True)
        except:
            df = data_handler.get_dataframe(preprocessed=False)
        
        # Prepare data
        X_scaled, feature_names = anomaly_detector.prepare_data(df)
        
        # Train model
        if model_type == "isolation_forest":
            model = anomaly_detector.train_isolation_forest(X_scaled, contamination)
        else:
            model = anomaly_detector.train_one_class_svm(X_scaled, contamination)
        
        # Detect anomalies
        results = anomaly_detector.detect_anomalies(X_scaled)
        
        # Prepare visualization data
        viz_data = {
            "anomaly_indices": np.where(results["anomalies"])[0].tolist(),
            "anomaly_scores": results["anomaly_scores"].tolist(),
            "feature_names": feature_names,
            "data_points": X_scaled.tolist()
        }
        
        response_data = {
            "status": "success",
            "model_type": anomaly_detector.model_type,
            "num_anomalies": int(results["num_anomalies"]),
            "anomaly_percentage": float(results["anomaly_percentage"]),
            "visualization_data": viz_data,
            "total_samples": len(X_scaled)
        }
        
        logger.info(f"ANOMALY DETECTION SUCCESSFUL: {model_type}")
        return response_data
        
    except Exception as e:
        logger.error(f"ANOMALY DETECTION FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during anomaly detection: {str(e)}")

@app.post("/perform-segmentation")
async def perform_segmentation(
    model_type: str = "kmeans",  # kmeans atau dbscan
    n_clusters: int = 3,         # untuk kmeans
    eps: float = 0.5,           # untuk dbscan
    min_samples: int = 5        # untuk dbscan
):
    try:
        # Get data
        try:
            df = data_handler.get_dataframe(preprocessed=True)
        except:
            df = data_handler.get_dataframe(preprocessed=False)
        
        # Prepare data
        X_scaled, feature_names = segmentation_model.prepare_data(df)
        
        # Train model
        if model_type == "kmeans":
            labels = segmentation_model.train_kmeans(X_scaled, n_clusters)
        else:
            labels = segmentation_model.train_dbscan(X_scaled, eps, min_samples)
        
        # Evaluate clustering
        evaluation = segmentation_model.evaluate_clustering(X_scaled, labels)
        
        # Get cluster statistics
        cluster_stats = segmentation_model.get_cluster_statistics(df, labels)
        
        # Prepare visualization data
        viz_data = {
            "data_points": X_scaled.tolist(),
            "cluster_labels": labels.tolist(),
            "feature_names": feature_names,
            "cluster_centers": segmentation_model.model.cluster_centers_.tolist() if model_type == "kmeans" else None
        }
        
        response_data = {
            "status": "success",
            "model_type": segmentation_model.model_type,
            "evaluation": evaluation,
            "cluster_statistics": cluster_stats,
            "visualization_data": viz_data,
            "total_samples": len(X_scaled)
        }
        
        logger.info(f"SEGMENTATION SUCCESSFUL: {model_type}")
        return response_data
        
    except Exception as e:
        logger.error(f"SEGMENTATION FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")

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
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "ready_for_analysis": True
        }
    else:
        df = data_handler.uploaded_data['dataframe']
        return {
            "status": "raw",
            "shape": df.shape,
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
    return {
        "filename": data_handler.uploaded_data['filename'],
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

# Error handler untuk debugging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL ERROR: {str(exc)}")
    logger.exception("Full traceback:")
    return {"error": "Internal server error", "detail": str(exc)}

# Health check untuk monitoring
@app.get("/api/status")
def api_status():
    """Extended health check with system status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-csv",
            "preprocess": "/preprocess-data", 
            "prediction": "/run-prediction",
            "anomaly": "/detect-anomaly",
            "segmentation": "/perform-segmentation"
        }
    }
