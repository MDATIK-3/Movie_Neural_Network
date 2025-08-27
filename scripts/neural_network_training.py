"""
Neural Network Training Script for Cinematch AI
Implements multi-label classification for movie mood prediction
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MovieRecommendationModel:
    """Neural network model for movie mood classification"""
    
    def __init__(self, input_dim: int, num_moods: int):
        self.input_dim = input_dim
        self.num_moods = num_moods
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """Build the neural network architecture"""
        print("[v0] Building neural network architecture...")
        
        model = keras.Sequential([
            # Input layer
            layers.Dense(512, activation='relu', input_shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer (multi-label classification)
            layers.Dense(self.num_moods, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print(f"[v0] Model built with {model.count_params()} parameters")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train the neural network"""
        print("[v0] Starting neural network training...")
        
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("[v0] Neural network training completed!")
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        print("[v0] Evaluating model performance...")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[v0] Model accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred_prob,
            'binary_predictions': y_pred
        }
    
    def save_model(self, filepath: str = 'cinematch_model.h5'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"[v0] Model saved to {filepath}")
        else:
            print("[v0] No model to save")

def load_processed_data(filepath: str = 'processed_movie_data.json') -> Tuple[pd.DataFrame, Dict]:
    """Load processed data from preprocessing step"""
    print(f"[v0] Loading processed data from {filepath}...")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['movies'])
        metadata = data['metadata']
        
        print(f"[v0] Loaded {len(df)} movies with {len(metadata['feature_columns'])} features")
        return df, metadata
        
    except FileNotFoundError:
        print("[v0] Processed data file not found. Creating sample data...")
        return create_sample_training_data()

def create_sample_training_data() -> Tuple[pd.DataFrame, Dict]:
    """Create sample training data for demonstration"""
    print("[v0] Creating sample training data...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create mood labels based on feature patterns
    mood_labels = ['happy', 'sad', 'excited', 'relaxed', 'romantic', 'adventurous', 'thoughtful']
    
    data = {}
    
    # Add features
    for i in range(n_features):
        data[f'text_feature_{i}'] = X[:, i]
    
    # Add mood labels with some correlation to features
    for i, mood in enumerate(mood_labels):
        # Create correlated mood scores
        mood_score = np.sigmoid(X[:, i % n_features] + np.random.randn(n_samples) * 0.1)
        data[f'mood_{mood}'] = mood_score
    
    # Add movie metadata
    data['title'] = [f'Movie_{i}' for i in range(n_samples)]
    data['popularity'] = np.random.uniform(0, 1, n_samples)
    data['vote_average'] = np.random.uniform(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    metadata = {
        'total_movies': n_samples,
        'feature_columns': [f'text_feature_{i}' for i in range(n_features)] + [f'mood_{mood}' for mood in mood_labels],
        'numerical_columns': ['popularity', 'vote_average'],
        'preprocessing_info': {
            'tfidf_features': n_features,
            'mood_categories': mood_labels
        }
    }
    
    print(f"[v0] Created sample dataset with {n_samples} movies and {n_features} features")
    return df, metadata

def prepare_training_data(df: pd.DataFrame, metadata: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare features and labels for training"""
    print("[v0] Preparing training data...")
    
    # Extract feature columns
    feature_cols = [col for col in df.columns if col.startswith('text_feature_')]
    feature_cols.extend(['popularity', 'vote_average'])
    
    # Extract mood columns
    mood_cols = [col for col in df.columns if col.startswith('mood_')]
    
    # Prepare features (X)
    X = df[feature_cols].values
    
    # Prepare labels (y) - convert to binary labels
    y = df[mood_cols].values
    y_binary = (y > 0.3).astype(int)  # Threshold for binary classification
    
    print(f"[v0] Features shape: {X.shape}")
    print(f"[v0] Labels shape: {y_binary.shape}")
    print(f"[v0] Mood categories: {mood_cols}")
    
    return X, y_binary, mood_cols

def plot_training_history(history: Dict, save_path: str = 'training_history.png'):
    """Plot training history"""
    print("[v0] Plotting training history...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Training Precision')
        axes[1, 0].plot(history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Training Recall')
        axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[v0] Training history saved to {save_path}")

def main():
    """Main training pipeline"""
    print("[v0] Starting neural network training pipeline...")
    
    try:
        # Step 1: Load processed data
        df, metadata = load_processed_data()
        
        # Step 2: Prepare training data
        X, y, mood_categories = prepare_training_data(df, metadata)
        
        # Step 3: Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y[:, 0]  # Stratify on first mood
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"[v0] Training set: {X_train.shape[0]} samples")
        print(f"[v0] Validation set: {X_val.shape[0]} samples")
        print(f"[v0] Test set: {X_test.shape[0]} samples")
        
        # Step 4: Initialize and train model
        model = MovieRecommendationModel(
            input_dim=X.shape[1],
            num_moods=len(mood_categories)
        )
        
        # Train model
        history = model.train(X_train, y_train, X_val, y_val, epochs=30)
        
        # Step 5: Evaluate model
        results = model.evaluate(X_test, y_test)
        
        # Step 6: Save model and results
        model.save_model('cinematch_model.h5')
        
        # Plot training history
        plot_training_history(history)
        
        # Save training results
        training_results = {
            'final_accuracy': float(results['accuracy']),
            'mood_categories': mood_categories,
            'model_architecture': {
                'input_dim': X.shape[1],
                'num_moods': len(mood_categories),
                'total_parameters': model.model.count_params()
            },
            'training_history': {k: [float(v) for v in values] for k, values in history.items()}
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print("[v0] Neural network training pipeline completed successfully!")
        print(f"[v0] Final model accuracy: {results['accuracy']:.4f}")
        print(f"[v0] Model saved as 'cinematch_model.h5'")
        print(f"[v0] Training results saved as 'training_results.json'")
        
    except Exception as e:
        print(f"[v0] Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
