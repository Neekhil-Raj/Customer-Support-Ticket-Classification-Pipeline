import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple, Any
import logging

class MLModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize different ML models"""
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }
        return models
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   model_name: str, hyperparameters: Dict = None) -> Any:
        """Train a specific model"""
        models = self.initialize_models()
        model = models[model_name]
        
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                            model_name: str) -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning"""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f"No parameter grid defined for {model_name}")
        
        models = self.initialize_models()
        model = models[model_name]
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, 
                      y_test: np.ndarray, label_encoder: Any) -> Dict:
        """Evaluate model performance"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        classification_rep = classification_report(y_test, y_pred, 
                                                 target_names=label_encoder.classes_,
                                                 output_dict=True)
        
        # ROC AUC (for multiclass)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_auc_score
                
                y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
                roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        return {
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def train_and_evaluate_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    label_encoder: Any) -> Dict:
        """Train and evaluate all models"""
        results = {}
        models = self.initialize_models()
        
        with mlflow.start_run():
            for name, model in models.items():
                self.logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test, label_encoder)
                
                # Log to MLflow
                with mlflow.start_run(nested=True):
                    mlflow.log_param("model_type", name)
                    mlflow.log_metric("f1_score", metrics['f1_score'])
                    if metrics['roc_auc']:
                        mlflow.log_metric("roc_auc", metrics['roc_auc'])
                    
                    # Log model
                    mlflow.sklearn.log_model(model, f"model_{name}")
                
                results[name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                self.logger.info(f"{name} F1 Score: {metrics['f1_score']:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
        self.best_model = results[best_model_name]['model']
        
        self.logger.info(f"Best model: {best_model_name}")
        
        return results
    
    def save_model(self, model: Any, filepath: str):
        """Save trained model"""
        joblib.dump(model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load trained model"""
        model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        return model
