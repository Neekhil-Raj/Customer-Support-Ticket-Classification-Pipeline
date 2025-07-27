import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, Embedding, Dropout, 
                                   BatchNormalization, Conv1D, GlobalMaxPooling1D,
                                   Input, Concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import mlflow
import mlflow.tensorflow

class DeepLearningModels:
    def __init__(self, max_features: int = 10000, max_length: int = 200):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def prepare_text_data(self, texts: List[str]) -> Tuple[np.ndarray, Tokenizer]:
        """Prepare text data for deep learning models"""
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return X, self.tokenizer
    
    def create_lstm_model(self, num_classes: int, embedding_dim: int = 300,
                         lstm_units: int = 128, dropout_rate: float = 0.3) -> Model:
        """Create LSTM model for text classification"""
        model = Sequential([
            Embedding(input_dim=self.max_features, 
                     output_dim=embedding_dim, 
                     input_length=self.max_length),
            
            LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
            LSTM(lstm_units//2, dropout=dropout_rate),
            
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def create_cnn_model(self, num_classes: int, embedding_dim: int = 300,
                        filters: int = 128, kernel_size: int = 3,
                        dropout_rate: float = 0.3) -> Model:
        """Create CNN model for text classification"""
        model = Sequential([
            Embedding(input_dim=self.max_features, 
                     output_dim=embedding_dim, 
                     input_length=self.max_length),
            
            Conv1D(filters, kernel_size, activation='relu'),
            GlobalMaxPooling1D(),
            
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def create_hybrid_model(self, num_classes: int, numerical_features: int,
                           embedding_dim: int = 300, lstm_units: int = 128,
                           dropout_rate: float = 0.3) -> Model:
        """Create hybrid model combining text and numerical features"""
        # Text input branch
        text_input = Input(shape=(self.max_length,), name='text_input')
        text_embedding = Embedding(self.max_features, embedding_dim)(text_input)
        text_lstm = LSTM(lstm_units, dropout=dropout_rate)(text_embedding)
        
        # Numerical input branch
        numerical_input = Input(shape=(numerical_features,), name='numerical_input')
        numerical_dense = Dense(64, activation='relu')(numerical_input)
        numerical_dropout = Dropout(dropout_rate)(numerical_dense)
        
        # Combine branches
        combined = Concatenate()([text_lstm, numerical_dropout])
        combined_dense = Dense(128, activation='relu')(combined)
        combined_dropout = Dropout(dropout_rate)(combined_dense)
        
        # Output layer
        output = Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(combined_dropout)
        
        model = Model(inputs=[text_input, numerical_input], outputs=output)
        
        return model
    
    def compile_and_train(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int = 50, batch_size: int = 32,
                         learning_rate: float = 0.001) -> Dict:
        """Compile and train the model"""
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        loss = 'sparse_categorical_crossentropy' if len(np.unique(y_train)) > 2 else 'binary_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        with mlflow.start_run():
            mlflow.log_params({
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_type': 'deep_learning'
            })
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Log metrics
            final_accuracy = max(history.history['val_accuracy'])
            final_loss = min(history.history['val_loss'])
            
            mlflow.log_metrics({
                'final_val_accuracy': final_accuracy,
                'final_val_loss': final_loss
            })
            
            # Log model
            mlflow.tensorflow.log_model(model, "deep_learning_model")
        
        self.model = model
        
        return {
            'model': model,
            'history': history.history,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss
        }
    
    def predict(self, texts: List[str], numerical_features: np.ndarray = None) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare text data
        sequences = self.tokenizer.texts_to_sequences(texts)
        X_text = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Make predictions
        if numerical_features is not None:
            predictions = self.model.predict([X_text, numerical_features])
        else:
            predictions = self.model.predict(X_text)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained model and tokenizer"""
        if self.model:
            self.model.save(f"{filepath}_model.h5")
        if self.tokenizer:
            import pickle
            with open(f"{filepath}_tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
    
    def load_model(self, filepath: str):
        """Load the trained model and tokenizer"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        
        import pickle
        with open(f"{filepath}_tokenizer.pkl", 'rb') as f:
            self.tokenizer = pickle.load(f)
