import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class EngineHealthModel:
    """
    A model for analyzing engine health based on sensor data.
    This class analyzes correlations between features and predicts engine health.
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.is_fitted = False
        self.correlation_matrix = None
        self.engine_health_scores = None
    
    def load_data(self, file_path):
        
        columns = ["id", "time"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=columns)
        return df
    
    def analyze_correlations(self, df):
        """Analyze correlations between all features"""
        # Select only sensor and operational setting columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_') or col.startswith('op_setting_')] #dont take op_settings
        self.correlation_matrix = df[sensor_cols].corr()
        return self.correlation_matrix
    
    def get_top_correlations(self, n=10):
        """Get the top n most correlated feature pairs"""
        if self.correlation_matrix is None:
            raise ValueError("Correlation analysis must be performed first")
        
        # Get upper triangle of correlation matrix
        upper_tri = self.correlation_matrix.where(np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool))
        
        # Find top correlations
        top_correlations = []
        for i in range(len(upper_tri.columns)):
            for j in range(i+1, len(upper_tri.columns)):
                col1 = upper_tri.columns[i]
                col2 = upper_tri.columns[j]
                corr_value = upper_tri.iloc[i, j]
                if not np.isnan(corr_value):
                    top_correlations.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        top_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        return top_correlations[:n]
    
    def prepare_data(self, df):
        """Prepare data for engine health prediction"""
        # Group data by engine ID
        engine_groups = df.groupby('id')
        
        # Extract features and calculate health indicators
        X = []
        engine_ids = []
        
        for engine_id, engine_data in engine_groups:
            # Calculate statistics for each sensor
            sensor_stats = []
            for col in [col for col in engine_data.columns if col.startswith('sensor_') or col.startswith('op_setting_')]:
                sensor_stats.extend([
                    engine_data[col].mean(),
                    engine_data[col].std(),
                    engine_data[col].max(),
                    engine_data[col].min()
                ])
            
            X.append(sensor_stats)
            engine_ids.append(engine_id)
        
        X = np.array(X)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        return X_pca, engine_ids
    
    def calculate_health_scores(self, X_pca):
        """Calculate health scores for each engine"""
        # Use the first principal component as a health indicator
        # Higher values indicate better health
        health_scores = -X_pca[:, 0]  # Negate to make higher values better
        
        # Normalize to 0-100 range
        health_scores = (health_scores - health_scores.min()) / (health_scores.max() - health_scores.min()) * 100
        
        self.engine_health_scores = health_scores
        return health_scores
    
    def train(self, X_pca, health_scores):
        """Train the model to predict health scores"""
        self.model.fit(X_pca, health_scores)
        self.is_fitted = True
    
    def predict_health(self, X_pca):
        """Predict health scores for engines"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X_pca)
    
    def visualize_correlations(self):
        """Visualize the correlation matrix"""
        if self.correlation_matrix is None:
            raise ValueError("Correlation analysis must be performed first")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Return the figure for display in the GUI
        return plt.gcf()
    
    def visualize_health_scores(self, engine_ids):
        """Visualize engine health scores"""
        if self.engine_health_scores is None:
            raise ValueError("Health scores must be calculated first")
        
        plt.figure(figsize=(12, 6))
        plt.bar(engine_ids, self.engine_health_scores)
        plt.xlabel('Engine ID')
        plt.ylabel('Health Score (0-100)')
        plt.title('Engine Health Scores')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Return the figure for display in the GUI
        return plt.gcf() 