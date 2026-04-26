from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

class ModelEvaluator:
    def evaluate_classification_models(self, models, X_test, y_test):
        """
        Evaluate classification models
        """
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
                'recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
                'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4)
            }
            
        return results
    
    def evaluate_regression_models(self, models, X_test, y_test):
        """
        Evaluate regression models
        """
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'r2': round(r2_score(y_test, y_pred), 4),
                'mae': round(mean_absolute_error(y_test, y_pred), 4),
                'mse': round(mean_squared_error(y_test, y_pred), 4),
                'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            }
            
        return results