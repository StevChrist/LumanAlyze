import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)

class PredictionModel:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.is_trained = False
    
    def prepare_data(self, df, target_column, test_size=0.2):
        """Prepare data for training"""
        logger.info(f"Preparing data with target column: {target_column}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Select only numeric columns for features
        numeric_columns = X.select_dtypes(include=['number']).columns
        X = X[numeric_columns]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, task_type='regression'):
        """Train Random Forest model"""
        logger.info(f"Training Random Forest for {task_type}")
        
        if task_type == 'regression':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.model_type = f"RandomForest_{task_type}"
        self.is_trained = True
        
        return self.model
    
    def train_mlp(self, X_train, y_train, task_type='regression'):
        """Train MLP model"""
        logger.info(f"Training MLP for {task_type}")
        
        if task_type == 'regression':
            self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.model_type = f"MLP_{task_type}"
        self.is_trained = True
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.model.predict(X_test)
        
        if 'regression' in self.model_type:
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            return {
                "r2_score": r2,
                "mse": mse,
                "rmse": np.sqrt(mse)
            }
        else:
            accuracy = accuracy_score(y_test, predictions)
            return {
                "accuracy": accuracy
            }
    
    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
