import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

class TurbofanRegressionModel:
    """
    A simple regression model for turbofan engine sensor data.
    This class makes it easy to analyze and predict fan speed based on other sensor readings.
    """
    
    def __init__(self):
        self.linear_model = LinearRegression()
        self.ridge_model = Ridge(alpha=1.0)
        self.lasso_model = Lasso(alpha=1.0)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_importance = None
        self.model_comparison = None
        self.models = {
            'Linear': self.linear_model,
            'Ridge': self.ridge_model,
            'Lasso': self.lasso_model
        }
        self.results = {}
    
    def load_data(self, file_path):
        """Load turbofan dataset from NASA C-MAPSS data"""
        columns = ["id", "time"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=columns)
        return df
    
    def prepare_data(self, df, target_sensor=8, feature_sensors=None):
        """Prepare data for regression modeling"""
        # Default to using all other sensors if not specified
        if feature_sensors is None:
            feature_sensors = [i for i in range(1, 22) if i != target_sensor]
        
        # Extract features and target
        X = df[[f"sensor_{i}" for i in feature_sensors]] #x parameter (all other sensor readings)
        y = df[f"sensor_{target_sensor}"] #y parameter
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # For regression, we also scale the target variable
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        #reshape-> 2-D 
        #flatten-> back to 1-D
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_train.columns
    
    def train(self, X_train, y_train):
        """Train all regression models"""
        # Train all models
        for name, model in self.models.items():
            model.fit(X_train, y_train)
        
        self.is_fitted = True
    
    def evaluate(self, X_test, y_test):
        """Evaluate all models and return metrics"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        results = {}
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            std_dev = np.std(y_pred - y_test)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAE': mae,
                'StdDev': std_dev,
                'y_true': y_test,
                'y_pred': y_pred
            }
        
        self.results = results
        return results
    
    def get_feature_importance(self, model_type='Linear', feature_names=None):
        """Get the importance of each feature for the specified model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Select the appropriate model
        if model_type == 'Linear':
            model = self.linear_model
        elif model_type == 'Ridge':
            model = self.ridge_model
        elif model_type == 'Lasso':
            model = self.lasso_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # For linear regression, abs value of coefficients can be used as feature importance
        importance = np.abs(model.coef_)
        
        # Create a dictionary of feature names and their importance
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(len(importance))]
        
        self.feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        self.feature_importance = {k: v for k, v in sorted(
            self.feature_importance.items(), key=lambda item: item[1], reverse=True
        )}
        
        return self.feature_importance
    
    def predict(self, X_new, model_type='Linear'):
        """Make predictions on new data using the specified model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Scale the new data
        X_new_scaled = self.scaler_X.transform(X_new)
        
        # Select the appropriate model
        if model_type == 'Linear':
            model = self.linear_model
        elif model_type == 'Ridge':
            model = self.ridge_model
        elif model_type == 'Lasso':
            model = self.lasso_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Make prediction
        y_pred_scaled = model.predict(X_new_scaled)
        
        # Inverse transform to get original scale
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def compare_models(self):
        """Compare the performance of all models"""
        if not self.results:
            return {"error": "No results available. Please train and evaluate models first."}
        
        # Find best model based on different metrics
        best_model_rmse = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        best_model_r2 = max(self.results.items(), key=lambda x: x[1]['R2'])
        best_model_mae = min(self.results.items(), key=lambda x: x[1]['MAE'])
        best_model_mse = min(self.results.items(), key=lambda x: x[1]['MSE'])
        
        analysis = f"""Model Comparison Analysis:

Best Model (based on RMSE): {best_model_rmse[0]}
- RMSE: {best_model_rmse[1]['RMSE']:.4f}
- R²: {best_model_rmse[1]['R2']:.4f}
- MAE: {best_model_rmse[1]['MAE']:.4f}
- MSE: {best_model_rmse[1]['MSE']:.4f}
- StdDev: {best_model_rmse[1]['StdDev']:.4f}

Best Model (based on R²): {best_model_r2[0]}
- RMSE: {best_model_r2[1]['RMSE']:.4f}
- R²: {best_model_r2[1]['R2']:.4f}
- MAE: {best_model_r2[1]['MAE']:.4f}
- MSE: {best_model_r2[1]['MSE']:.4f}
- StdDev: {best_model_r2[1]['StdDev']:.4f}

Best Model (based on MAE): {best_model_mae[0]}
- RMSE: {best_model_mae[1]['RMSE']:.4f}
- R²: {best_model_mae[1]['R2']:.4f}
- MAE: {best_model_mae[1]['MAE']:.4f}
- MSE: {best_model_mae[1]['MSE']:.4f}
- StdDev: {best_model_mae[1]['StdDev']:.4f}

Best Model (based on MSE): {best_model_mse[0]}
- RMSE: {best_model_mse[1]['RMSE']:.4f}
- R²: {best_model_mse[1]['R2']:.4f}
- MAE: {best_model_mse[1]['MAE']:.4f}
- MSE: {best_model_mse[1]['MSE']:.4f}
- StdDev: {best_model_mse[1]['StdDev']:.4f}

Detailed Comparison:
"""
        for model_name, metrics in self.results.items():
            analysis += f"\n{model_name} Regression:"
            analysis += f"\n- RMSE: {metrics['RMSE']:.4f}"
            analysis += f"\n- R²: {metrics['R2']:.4f}"
            analysis += f"\n- MAE: {metrics['MAE']:.4f}"
            analysis += f"\n- MSE: {metrics['MSE']:.4f}"
            analysis += f"\n- StdDev: {metrics['StdDev']:.4f}"
        
        return {
            "analysis": analysis,
            "best_model": {
                "RMSE": best_model_rmse[0],
                "R2": best_model_r2[0],
                "MAE": best_model_mae[0],
                "MSE": best_model_mse[0]
            }
        }
    
    def visualize_results(self, model_type='Linear'):
        """Visualize model results for all regression types"""
        if not self.results:
            print("No results available. Please train and evaluate models first.")
            return
        
        results = self.results[model_type]
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Actual vs Predicted
        ax1.scatter(results['y_true'], results['y_pred'], alpha=0.5)
        ax1.plot([min(results['y_true']), max(results['y_true'])],
                 [min(results['y_true']), max(results['y_true'])], 'r--')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{model_type} Regression: Actual vs Predicted')
        
        # Plot 2: Error Distribution
        errors = results['y_pred'] - results['y_true']
        sns.histplot(errors, kde=True, ax=ax2)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        
        plt.tight_layout()
        plt.show()
