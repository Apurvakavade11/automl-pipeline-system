import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def preprocess(self, X, y, problem_type):
        """
        Preprocess the data for training
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=self.feature_names)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X_imputed)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X_encoded)
        
        # Handle target variable
        if problem_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders['target'] = le
        else:
            y_encoded = y.values if hasattr(y, 'values') else y
        
        # Split data
        stratify_param = y_encoded if problem_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test, {
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'imputer': self.imputer
        }
    
    def preprocess_predict(self, X):
        """
        Preprocess data for prediction
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=self.feature_names)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X_imputed)
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        return X_scaled
    
    def _encode_categorical_features(self, X):
        """
        Encode categorical features using label encoding
        """
        X_encoded = X.copy()
        
        for column in X.columns:
            if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    # Handle unseen categories
                    X_encoded[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
                else:
                    # Transform using existing encoder
                    def encode_value(val):
                        try:
                            return self.label_encoders[column].transform([str(val)])[0]
                        except:
                            return -1
                    X_encoded[column] = X[column].astype(str).apply(encode_value)
        
        return X_encodedimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def preprocess(self, X, y, problem_type):
        """
        Preprocess the data for training
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=self.feature_names)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X_imputed)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X_encoded)
        
        # Handle target variable
        if problem_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders['target'] = le
        else:
            y_encoded = y.values if hasattr(y, 'values') else y
        
        # Split data
        stratify_param = y_encoded if problem_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test, {
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'imputer': self.imputer
        }
    
    def preprocess_predict(self, X):
        """
        Preprocess data for prediction
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=self.feature_names)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X_imputed)
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        return X_scaled
    
    def _encode_categorical_features(self, X):
        """
        Encode categorical features using label encoding
        """
        X_encoded = X.copy()
        
        for column in X.columns:
            if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    # Handle unseen categories
                    X_encoded[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
                else:
                    # Transform using existing encoder
                    def encode_value(val):
                        try:
                            return self.label_encoders[column].transform([str(val)])[0]
                        except:
                            return -1
                    X_encoded[column] = X[column].astype(str).apply(encode_value)
        
        return X_encoded