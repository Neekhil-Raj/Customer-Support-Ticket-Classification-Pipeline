import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import mlflow
from typing import Tuple, Dict, Any
import logging

from src.data_ingestion.csv_data_loader import CustomerSupportDataLoader
from src.preprocessing.text_processor import TextPreprocessor
from src.models.ml_models import MLModelTrainer
from src.models.deep_learning import DeepLearningModels

class ActualDataTrainingPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_loader = CustomerSupportDataLoader()
        self.text_processor = TextPreprocessor()
        self.ml_trainer = MLModelTrainer()
        self.dl_models = DeepLearningModels()
        self.logger = logging.getLogger(__name__)
        
        # Initialize encoders and scalers
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load and prepare the actual CSV data"""
        # Load data
        df = self.data_loader.load_and_validate_csv(self.data_path)
        df_processed = self.data_loader.clean_and_preprocess(df)
        df_features = self.data_loader.generate_features(df_processed)
        
        # Get statistics
        stats = self.data_loader.get_data_statistics(df_features)
        
        self.logger.info(f"Loaded {len(df_features)} tickets for training")
        self.logger.info(f"Target distribution: {df_features['ticket_type'].value_counts().to_dict()}")
        
        return df_features, stats
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for traditional ML models"""
        # Text preprocessing
        preprocessed_data = self.text_processor.preprocess_for_ml(
            df, 'combined_text', 'ticket_type'
        )
        
        # Numerical features
        numerical_features = [
            'customer_age', 'subject_length', 'description_length', 
            'word_count', 'priority_numeric', 'first_response_time',
            'time_to_resolution', 'has_urgency_keywords'
        ]
        
        # Handle missing values in numerical features
        df_numerical = df[numerical_features].fillna(df[numerical_features].median())
        
        # Scale numerical features
        scaled_numerical = self.scaler.fit_transform(df_numerical)
        
        # Combine text and numerical features
        combined_features = np.hstack([
            preprocessed_data['features'],
            scaled_numerical
        ])
        
        return combined_features, preprocessed_data['labels']
    
    def prepare_dl_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for deep learning models"""
        # Prepare text data
        texts = df['combined_text'].tolist()
        X_text, tokenizer = self.dl_models.prepare_text_data(texts)
        
        # Prepare numerical features
        numerical_features = [
            'customer_age', 'subject_length', 'description_length', 
            'word_count', 'priority_numeric', 'first_response_time',
            'time_to_resolution', 'has_urgency_keywords'
        ]
        
        df_numerical = df[numerical_features].fillna(df[numerical_features].median())
        X_numerical = self.scaler.fit_transform(df_numerical)
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['ticket_type'])
        
        return X_text, X_numerical, y
    
    def train_traditional_ml_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train traditional ML models"""
        self.logger.info("Training traditional ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train and evaluate all models
        results = self.ml_trainer.train_and_evaluate_all_models(
            X_train, y_train, X_test, y_test, self.label_encoder
        )
        
        return results
    
    def train_deep_learning_models(self, X_text: np.ndarray, X_numerical: np.ndarray, 
                                 y: np.ndarray) -> Dict:
        """Train deep learning models"""
        self.logger.info("Training deep learning models...")
        
        # Split data
        indices = np.arange(len(X_text))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y
        )
        
        X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
        X_numerical_train, X_numerical_test = X_numerical[train_idx], X_numerical[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Further split for validation
        train_idx_inner, val_idx = train_test_split(
            np.arange(len(X_text_train)), test_size=0.2, random_state=42, stratify=y_train
        )
        
        X_text_train_inner, X_text_val = X_text_train[train_idx_inner], X_text_train[val_idx]
        X_numerical_train_inner, X_numerical_val = X_numerical_train[train_idx_inner], X_numerical_train[val_idx]
        y_train_inner, y_val = y_train[train_idx_inner], y_train[val_idx]
        
        # Train hybrid model (text + numerical)
        num_classes = len(np.unique(y))
        model = self.dl_models.create_hybrid_model(
            num_classes=num_classes,
            numerical_features=X_numerical.shape[1]
        )
        
        # Train the model
        training_results = self.dl_models.compile_and_train(
            model, 
            [X_text_train_inner, X_numerical_train_inner], y_train_inner,
            [X_text_val, X_numerical_val], y_val,
            epochs=50, batch_size=32
        )
        
        # Evaluate on test set
        test_predictions = model.predict([X_text_test, X_numerical_test])
        test_pred_classes = np.argmax(test_predictions, axis=1)
        
        # Calculate metrics
        test_accuracy = np.mean(test_pred_classes == y_test)
        
        return {
            'model': model,
            'training_results': training_results,
            'test_accuracy': test_accuracy,
            'tokenizer': self.dl_models.tokenizer
        }
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete training pipeline"""
        with mlflow.start_run():
            # Load and prepare data
            df, stats = self.load_and_prepare_data()
            
            # Log data statistics
            mlflow.log_params({
                'total_tickets': stats['basic_info']['total_tickets'],
                'unique_ticket_types': len(stats['ticket_distribution']['by_type']),
                'data_date_range': f"{stats['basic_info']['date_range']['start']} to {stats['basic_info']['date_range']['end']}"
            })
            
            # Prepare features for ML
            X_ml, y_ml = self.prepare_ml_features(df)
            
            # Train traditional ML models
            ml_results = self.train_traditional_ml_models(X_ml, y_ml)
            
            # Prepare features for DL
            X_text, X_numerical, y_dl = self.prepare_dl_features(df)
            
            # Train deep learning models
            dl_results = self.train_deep_learning_models(X_text, X_numerical, y_dl)
            
            # Save best models
            self.save_models(ml_results, dl_results)
            
            # Log final results
            best_ml_score = max([result['metrics']['f1_score'] for result in ml_results.values()])
            mlflow.log_metrics({
                'best_ml_f1_score': best_ml_score,
                'dl_test_accuracy': dl_results['test_accuracy']
            })
            
            return {
                'ml_results': ml_results,
                'dl_results': dl_results,
                'data_stats': stats
            }
    
    def save_models(self, ml_results: Dict, dl_results: Dict):
        """Save trained models and preprocessors"""
        # Save best ML model
        best_ml_name = max(ml_results.keys(), 
                          key=lambda k: ml_results[k]['metrics']['f1_score'])
        best_ml_model = ml_results[best_ml_name]['model']
        
        joblib.dump(best_ml_model, 'models/best_ml_model.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.text_processor.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        
        # Save DL model
        dl_results['model'].save('models/deep_learning_model.h5')
        
        self.logger.info("Models saved successfully")

if __name__ == "__main__":
    # Run the training pipeline
    pipeline = ActualDataTrainingPipeline('customer_support_tickets.csv')
    results = pipeline.run_complete_pipeline()
    
    print("Training completed!")
    print(f"Best ML F1 Score: {max([r['metrics']['f1_score'] for r in results['ml_results'].values()]):.4f}")
    print(f"DL Test Accuracy: {results['dl_results']['test_accuracy']:.4f}")
