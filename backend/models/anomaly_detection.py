import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
    
    def prepare_data(self, df):
        """Prepare data for anomaly detection"""
        logger.info("Preparing data for anomaly detection")
        
        # Select only numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        X = df[numeric_columns]
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, numeric_columns.tolist()
    
    def train_isolation_forest(self, X, contamination=0.1):
        """Train Isolation Forest model"""
        logger.info("Training Isolation Forest")
        
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.model.fit(X)
        self.model_type = "IsolationForest"
        
        return self.model
    
    def train_one_class_svm(self, X, nu=0.1):
        """Train One-Class SVM model"""
        logger.info("Training One-Class SVM")
        
        self.model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        self.model.fit(X)
        self.model_type = "OneClassSVM"
        
        return self.model
    
    def detect_anomalies(self, X):
        """Detect anomalies in data"""
        if self.model is None:
            raise ValueError("Model must be trained before detecting anomalies")
        
        # Predict (-1 for anomalies, 1 for normal)
        predictions = self.model.predict(X)
        
        # Convert to boolean (True for anomalies)
        anomalies = predictions == -1
        
        # Calculate anomaly scores
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)
        else:
            scores = self.model.score_samples(X)
        
        return {
            "anomalies": anomalies,
            "anomaly_scores": scores,
            "num_anomalies": np.sum(anomalies),
            "anomaly_percentage": (np.sum(anomalies) / len(anomalies)) * 100
        }
