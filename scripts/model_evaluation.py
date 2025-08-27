"""
Model Evaluation and Inference Script for Cinematch AI
Provides functionality to load trained model and make predictions
"""

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class CinematchPredictor:
    """Class for loading and using trained Cinematch model"""
    
    def __init__(self, model_path: str = 'cinematch_model.h5'):
        self.model = None
        self.mood_categories = []
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata"""
        print(f"[v0] Loading model from {self.model_path}...")
        
        try:
            self.model = keras.models.load_model(self.model_path)
            print("[v0] Model loaded successfully")
            
            # Load training results for mood categories
            try:
                with open('training_results.json', 'r') as f:
                    results = json.load(f)
                    self.mood_categories = results.get('mood_categories', [])
                print(f"[v0] Loaded {len(self.mood_categories)} mood categories")
            except FileNotFoundError:
                print("[v0] Training results not found, using default mood categories")
                self.mood_categories = ['happy', 'sad', 'excited', 'relaxed', 'romantic', 'adventurous', 'thoughtful']
                
        except Exception as e:
            print(f"[v0] Error loading model: {str(e)}")
            print("[v0] Creating dummy model for demonstration")
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model for demonstration purposes"""
        self.mood_categories = ['happy', 'sad', 'excited', 'relaxed', 'romantic', 'adventurous', 'thoughtful']
        
        # Simple dummy model
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(102,)),  # 100 text features + 2 numerical
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(self.mood_categories), activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("[v0] Dummy model created")
    
    def predict_movie_moods(self, features: np.ndarray, threshold: float = 0.5) -> Dict:
        """Predict mood scores for movies"""
        print(f"[v0] Predicting moods for {features.shape[0]} movies...")
        
        # Get predictions
        predictions = self.model.predict(features, verbose=0)
        
        # Convert to binary predictions
        binary_predictions = (predictions > threshold).astype(int)
        
        # Create results dictionary
        results = {
            'mood_scores': predictions,
            'binary_predictions': binary_predictions,
            'mood_categories': self.mood_categories,
            'threshold': threshold
        }
        
        print("[v0] Mood prediction completed")
        return results
    
    def get_movie_recommendations(self, user_mood: str, features: np.ndarray, 
                                movie_titles: List[str], top_k: int = 10) -> List[Dict]:
        """Get movie recommendations based on user mood"""
        print(f"[v0] Getting recommendations for mood: {user_mood}")
        
        if user_mood not in self.mood_categories:
            print(f"[v0] Warning: Unknown mood '{user_mood}', using 'happy' as default")
            user_mood = 'happy'
        
        # Get mood predictions
        predictions = self.predict_movie_moods(features)
        mood_scores = predictions['mood_scores']
        
        # Get index of target mood
        mood_idx = self.mood_categories.index(user_mood)
        
        # Sort movies by mood score
        mood_scores_for_target = mood_scores[:, mood_idx]
        sorted_indices = np.argsort(mood_scores_for_target)[::-1]
        
        # Create recommendations
        recommendations = []
        for i, idx in enumerate(sorted_indices[:top_k]):
            rec = {
                'rank': i + 1,
                'title': movie_titles[idx] if idx < len(movie_titles) else f'Movie_{idx}',
                'mood_score': float(mood_scores_for_target[idx]),
                'all_mood_scores': {
                    mood: float(mood_scores[idx, j]) 
                    for j, mood in enumerate(self.mood_categories)
                }
            }
            recommendations.append(rec)
        
        print(f"[v0] Generated {len(recommendations)} recommendations")
        return recommendations

def evaluate_model_performance(model_path: str = 'cinematch_model.h5') -> Dict:
    """Comprehensive model evaluation"""
    print("[v0] Starting comprehensive model evaluation...")
    
    # Load model
    predictor = CinematchPredictor(model_path)
    
    # Create or load test data
    try:
        with open('processed_movie_data.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['movies'])
        print(f"[v0] Loaded {len(df)} movies for evaluation")
    except FileNotFoundError:
        print("[v0] Creating sample test data...")
        df = create_sample_test_data()
    
    # Prepare features
    feature_cols = [col for col in df.columns if col.startswith('text_feature_')]
    feature_cols.extend(['popularity', 'vote_average'])
    
    if not feature_cols:
        print("[v0] No features found, creating dummy features")
        n_samples = len(df)
        for i in range(100):
            df[f'text_feature_{i}'] = np.random.randn(n_samples)
        df['popularity'] = np.random.uniform(0, 1, n_samples)
        df['vote_average'] = np.random.uniform(0, 1, n_samples)
        feature_cols = [col for col in df.columns if col.startswith('text_feature_')]
        feature_cols.extend(['popularity', 'vote_average'])
    
    X = df[feature_cols].values
    
    # Get true labels if available
    mood_cols = [col for col in df.columns if col.startswith('mood_')]
    if mood_cols:
        y_true = df[mood_cols].values
        y_true_binary = (y_true > 0.3).astype(int)
    else:
        print("[v0] No true labels found, creating dummy labels")
        y_true_binary = np.random.randint(0, 2, (len(df), len(predictor.mood_categories)))
    
    # Get predictions
    predictions = predictor.predict_movie_moods(X)
    y_pred_binary = predictions['binary_predictions']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='macro', zero_division=0
    )
    
    # Per-mood metrics
    mood_metrics = {}
    for i, mood in enumerate(predictor.mood_categories):
        mood_acc = accuracy_score(y_true_binary[:, i], y_pred_binary[:, i])
        mood_metrics[mood] = {
            'accuracy': float(mood_acc),
            'true_positive_rate': float(np.mean(y_pred_binary[y_true_binary[:, i] == 1, i])) if np.sum(y_true_binary[:, i]) > 0 else 0.0
        }
    
    results = {
        'overall_accuracy': float(accuracy),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'mood_specific_metrics': mood_metrics,
        'total_samples': len(df),
        'num_features': X.shape[1],
        'num_moods': len(predictor.mood_categories)
    }
    
    print(f"[v0] Model evaluation completed!")
    print(f"[v0] Overall accuracy: {accuracy:.4f}")
    print(f"[v0] Macro F1-score: {f1:.4f}")
    
    return results

def create_sample_test_data() -> pd.DataFrame:
    """Create sample test data"""
    np.random.seed(42)
    n_samples = 500
    n_features = 100
    
    data = {}
    
    # Add features
    for i in range(n_features):
        data[f'text_feature_{i}'] = np.random.randn(n_samples)
    
    data['popularity'] = np.random.uniform(0, 1, n_samples)
    data['vote_average'] = np.random.uniform(0, 1, n_samples)
    
    # Add mood labels
    mood_categories = ['happy', 'sad', 'excited', 'relaxed', 'romantic', 'adventurous', 'thoughtful']
    for mood in mood_categories:
        data[f'mood_{mood}'] = np.random.uniform(0, 1, n_samples)
    
    data['title'] = [f'Test_Movie_{i}' for i in range(n_samples)]
    
    return pd.DataFrame(data)

def demonstrate_recommendations():
    """Demonstrate the recommendation system"""
    print("[v0] Demonstrating movie recommendation system...")
    
    # Load predictor
    predictor = CinematchPredictor()
    
    # Create sample movie data
    sample_movies = [
        "Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "The Matrix",
        "Goodfellas", "The Shawshank Redemption", "Forrest Gump", "The Godfather",
        "Titanic", "Avatar", "The Avengers", "Jurassic Park", "Star Wars"
    ]
    
    # Create sample features (in real scenario, these would come from preprocessing)
    n_movies = len(sample_movies)
    n_features = 102  # 100 text features + 2 numerical
    sample_features = np.random.randn(n_movies, n_features)
    
    # Get recommendations for different moods
    moods_to_test = ['happy', 'excited', 'romantic', 'thoughtful']
    
    all_recommendations = {}
    
    for mood in moods_to_test:
        print(f"\n[v0] Getting recommendations for mood: {mood}")
        recommendations = predictor.get_movie_recommendations(
            user_mood=mood,
            features=sample_features,
            movie_titles=sample_movies,
            top_k=5
        )
        
        all_recommendations[mood] = recommendations
        
        print(f"Top 5 {mood} movie recommendations:")
        for rec in recommendations:
            print(f"  {rec['rank']}. {rec['title']} (Score: {rec['mood_score']:.3f})")
    
    # Save recommendations
    with open('sample_recommendations.json', 'w') as f:
        json.dump(all_recommendations, f, indent=2)
    
    print("\n[v0] Sample recommendations saved to 'sample_recommendations.json'")

def main():
    """Main evaluation and demonstration pipeline"""
    print("[v0] Starting model evaluation and demonstration...")
    
    try:
        # Step 1: Evaluate model performance
        evaluation_results = evaluate_model_performance()
        
        # Save evaluation results
        with open('model_evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print("[v0] Evaluation results saved to 'model_evaluation_results.json'")
        
        # Step 2: Demonstrate recommendations
        demonstrate_recommendations()
        
        print("\n[v0] Model evaluation and demonstration completed successfully!")
        
    except Exception as e:
        print(f"[v0] Error in evaluation pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
