import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

class DataSegmentation:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
    
    def prepare_data(self, df):
        """Prepare data for clustering"""
        logger.info("Preparing data for segmentation")
        
        # Select only numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        X = df[numeric_columns]
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, numeric_columns.tolist()
    
    def train_kmeans(self, X, n_clusters=3):
        """Train K-Means clustering"""
        logger.info(f"Training K-Means with {n_clusters} clusters")
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.model.fit_predict(X)
        self.model_type = "KMeans"
        
        return cluster_labels
    
    def train_dbscan(self, X, eps=0.5, min_samples=5):
        """Train DBSCAN clustering"""
        logger.info(f"Training DBSCAN with eps={eps}, min_samples={min_samples}")
        
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.model.fit_predict(X)
        self.model_type = "DBSCAN"
        
        return cluster_labels
    
    def evaluate_clustering(self, X, labels):
        """Evaluate clustering performance"""
        if len(set(labels)) < 2:
            return {"silhouette_score": -1, "num_clusters": len(set(labels))}
        
        silhouette_avg = silhouette_score(X, labels)
        num_clusters = len(set(labels))
        
        # For DBSCAN, count noise points
        if self.model_type == "DBSCAN":
            noise_points = np.sum(labels == -1)
            return {
                "silhouette_score": silhouette_avg,
                "num_clusters": num_clusters - (1 if -1 in labels else 0),
                "noise_points": noise_points
            }
        
        return {
            "silhouette_score": silhouette_avg,
            "num_clusters": num_clusters
        }
    
    def get_cluster_statistics(self, df, labels):
        """Get statistics for each cluster"""
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        cluster_stats = {}
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            numeric_columns = cluster_data.select_dtypes(include=['number']).columns
            
            cluster_stats[f"cluster_{cluster_id}"] = {
                "size": len(cluster_data),
                "percentage": (len(cluster_data) / len(df)) * 100,
                "mean_values": cluster_data[numeric_columns].mean().to_dict()
            }
        
        return cluster_stats
