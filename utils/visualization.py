import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class VisualizationGenerator:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def create_confusion_matrix(self, model, X_test, y_test, user_id):
        """
        Create confusion matrix plot
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        filepath = f'static/plots/cm_{user_id}.png'
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close()
        
    def create_feature_importance(self, model, feature_names, user_id):
        """
        Create feature importance plot
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.title('Top 10 Feature Importances')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            
            filepath = f'static/plots/feature_importance_{user_id}.png'
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close()
            
    def create_residual_plot(self, model, X_test, y_test, user_id):
        """
        Create residual plot for regression models
        """
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        
        filepath = f'static/plots/residuals_{user_id}.png'
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close()
        
    def create_correlation_heatmap(self, df, user_id):
        """
        Create correlation heatmap for numerical features
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Heatmap')
            
            filepath = f'static/plots/correlation_{user_id}.png'
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close()