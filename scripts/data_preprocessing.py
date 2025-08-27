"""
Data Preprocessing Script for Cinematch AI
Handles CSV parsing, feature engineering, and data cleaning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json
from typing import Dict, List, Tuple

def load_and_validate_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file and validate required columns"""
    print("[v0] Loading CSV file...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"[v0] Loaded {len(df)} movies from CSV")
        
        # Required columns
        required_cols = ['title', 'overview', 'genres', 'keywords', 'popularity', 'vote_average']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        print("[v0] CSV validation successful")
        return df
        
    except Exception as e:
        print(f"[v0] Error loading CSV: {str(e)}")
        raise

def clean_and_parse_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and parse JSON-like columns"""
    print("[v0] Cleaning and parsing data...")
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['title', 'overview', 'genres'])
    
    # Parse genres (assuming JSON-like format or comma-separated)
    def parse_genres(genres_str):
        if pd.isna(genres_str):
            return []
        
        # Try to parse as JSON first
        try:
            if isinstance(genres_str, str) and genres_str.startswith('['):
                genres_list = eval(genres_str)  # Careful: only for trusted data
                if isinstance(genres_list, list):
                    return [g.get('name', g) if isinstance(g, dict) else str(g) for g in genres_list]
        except:
            pass
        
        # Fallback to comma-separated
        return [g.strip() for g in str(genres_str).split(',') if g.strip()]
    
    def parse_keywords(keywords_str):
        if pd.isna(keywords_str):
            return []
        
        try:
            if isinstance(keywords_str, str) and keywords_str.startswith('['):
                keywords_list = eval(keywords_str)
                if isinstance(keywords_list, list):
                    return [k.get('name', k) if isinstance(k, dict) else str(k) for k in keywords_list]
        except:
            pass
        
        return [k.strip() for k in str(keywords_str).split(',') if k.strip()]
    
    df['parsed_genres'] = df['genres'].apply(parse_genres)
    df['parsed_keywords'] = df['keywords'].apply(parse_keywords)
    
    # Clean text data
    df['clean_overview'] = df['overview'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x).lower()))
    
    # Handle missing numerical data
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(5.0)
    
    print(f"[v0] Data cleaning complete. {len(df)} movies remaining")
    return df

def create_text_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create text embeddings for genres and keywords"""
    print("[v0] Creating text features...")
    
    # Combine genres and keywords into text features
    df['genre_text'] = df['parsed_genres'].apply(lambda x: ' '.join(x) if x else '')
    df['keyword_text'] = df['parsed_keywords'].apply(lambda x: ' '.join(x[:10]) if x else '')  # Limit keywords
    df['combined_text'] = df['genre_text'] + ' ' + df['keyword_text'] + ' ' + df['clean_overview']
    
    # Create TF-IDF features for text
    tfidf = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    text_features = tfidf.fit_transform(df['combined_text'])
    
    # Convert to DataFrame
    feature_names = [f'text_feature_{i}' for i in range(text_features.shape[1])]
    text_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=df.index)
    
    # Combine with original DataFrame
    df = pd.concat([df, text_df], axis=1)
    
    print(f"[v0] Created {text_features.shape[1]} text features")
    
    return df, {'tfidf_vectorizer': tfidf, 'feature_names': feature_names}

def normalize_numerical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply MinMax scaling to numerical features"""
    print("[v0] Normalizing numerical features...")
    
    numerical_cols = ['popularity', 'vote_average']
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Fit and transform numerical features
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("[v0] Numerical feature normalization complete")
    
    return df, {'scaler': scaler, 'numerical_columns': numerical_cols}

def create_genre_mapping() -> Dict[str, List[str]]:
    """Create mapping of genres to mood categories"""
    return {
        'happy': ['Comedy', 'Family', 'Animation', 'Music', 'Romance'],
        'sad': ['Drama', 'War', 'History', 'Biography'],
        'excited': ['Action', 'Adventure', 'Thriller', 'Crime'],
        'relaxed': ['Documentary', 'Family', 'Music'],
        'romantic': ['Romance', 'Drama'],
        'adventurous': ['Adventure', 'Action', 'Fantasy', 'Sci-Fi'],
        'thoughtful': ['Drama', 'Mystery', 'Documentary', 'Biography'],
        'scared': ['Horror', 'Thriller', 'Mystery']
    }

def generate_mood_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate mood-based labels from genre analysis"""
    print("[v0] Generating mood labels...")
    
    mood_mapping = create_genre_mapping()
    
    # Initialize mood columns
    for mood in mood_mapping.keys():
        df[f'mood_{mood}'] = 0.0
    
    # Calculate mood scores based on genres
    for idx, row in df.iterrows():
        genres = row['parsed_genres']
        if not genres:
            continue
            
        for mood, mood_genres in mood_mapping.items():
            # Calculate overlap score
            overlap = len(set(genres) & set(mood_genres))
            total_genres = len(genres)
            
            if total_genres > 0:
                score = overlap / total_genres
                df.at[idx, f'mood_{mood}'] = score
    
    print("[v0] Mood label generation complete")
    return df

def save_processed_data(df: pd.DataFrame, preprocessors: Dict, output_path: str = 'processed_movie_data.json'):
    """Save processed data and preprocessors"""
    print(f"[v0] Saving processed data to {output_path}...")
    
    # Prepare data for saving
    data_to_save = {
        'movies': df.to_dict('records'),
        'metadata': {
            'total_movies': len(df),
            'feature_columns': [col for col in df.columns if col.startswith('text_feature_') or col.startswith('mood_')],
            'numerical_columns': preprocessors.get('numerical_columns', []),
            'preprocessing_info': {
                'tfidf_features': len(preprocessors.get('feature_names', [])),
                'mood_categories': list(create_genre_mapping().keys())
            }
        }
    }
    
    # Save to JSON (in a real scenario, you might use pickle for sklearn objects)
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"[v0] Processed data saved successfully")

def main():
    """Main preprocessing pipeline"""
    print("[v0] Starting data preprocessing pipeline...")
    
    # Note: In a real implementation, this would read from uploaded CSV
    # For demo purposes, we'll create sample data
    sample_data = {
        'title': ['Inception', 'The Dark Knight', 'Interstellar', 'Pulp Fiction', 'The Matrix'],
        'overview': [
            'A thief who steals corporate secrets through dream-sharing technology.',
            'Batman must accept one of the greatest psychological tests.',
            'A team of explorers travel through a wormhole in space.',
            'The lives of two mob hitmen intertwine in tales of violence.',
            'A computer programmer fights an underground war against computers.'
        ],
        'genres': [
            '[{"name": "Action"}, {"name": "Sci-Fi"}, {"name": "Thriller"}]',
            '[{"name": "Action"}, {"name": "Crime"}, {"name": "Drama"}]',
            '[{"name": "Adventure"}, {"name": "Drama"}, {"name": "Sci-Fi"}]',
            '[{"name": "Crime"}, {"name": "Drama"}]',
            '[{"name": "Action"}, {"name": "Sci-Fi"}]'
        ],
        'keywords': [
            '[{"name": "dream"}, {"name": "heist"}, {"name": "mind"}]',
            '[{"name": "superhero"}, {"name": "joker"}, {"name": "gotham"}]',
            '[{"name": "space"}, {"name": "wormhole"}, {"name": "time"}]',
            '[{"name": "gangster"}, {"name": "nonlinear"}, {"name": "violence"}]',
            '[{"name": "virtual reality"}, {"name": "chosen one"}, {"name": "simulation"}]'
        ],
        'popularity': [95.5, 98.2, 89.1, 92.3, 94.7],
        'vote_average': [8.8, 9.0, 8.6, 8.9, 8.7]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"[v0] Created sample dataset with {len(df)} movies")
    
    try:
        # Step 1: Clean and parse data
        df = clean_and_parse_data(df)
        
        # Step 2: Create text features
        df, text_preprocessors = create_text_features(df)
        
        # Step 3: Normalize numerical features
        df, numerical_preprocessors = normalize_numerical_features(df)
        
        # Step 4: Generate mood labels
        df = generate_mood_labels(df)
        
        # Step 5: Save processed data
        all_preprocessors = {**text_preprocessors, **numerical_preprocessors}
        save_processed_data(df, all_preprocessors)
        
        print("[v0] Data preprocessing pipeline completed successfully!")
        print(f"[v0] Final dataset shape: {df.shape}")
        print(f"[v0] Features created: {len([col for col in df.columns if col.startswith('text_feature_')])}")
        print(f"[v0] Mood categories: {len([col for col in df.columns if col.startswith('mood_')])}")
        
    except Exception as e:
        print(f"[v0] Error in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
