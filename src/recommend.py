import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

def load_trained_model():
    try:
        with open("models/similarity.pkl", "rb") as f:
            similarity = pickle.load(f)
        movies = pd.read_csv("data/movies.csv")
        return movies, similarity
    except FileNotFoundError:
        print("❌ Model or movie data not found. Please run main.py first.")
        return None, None


def recommend_movies(movie_name, top_n=5):
    movies, similarity = load_trained_model()
    if movies is None or similarity is None:
        return []

    # Normalize movie titles
    movies['title'] = movies['title'].astype(str).str.strip()

    # Try exact or partial match
    matches = movies[movies['title'].str.contains(movie_name, case=False, na=False)]

    if matches.empty:
        # Fuzzy matching fallback
        close_match = get_close_matches(movie_name, movies['title'], n=1, cutoff=0.4)
        if close_match:
            movie_name = close_match[0]
            print(f"🔍 Using closest match: {movie_name}")
            matches = movies[movies['title'] == movie_name]
        else:
            print(f"❌ No match found for '{movie_name}'. Try checking the title spelling.")
            sample_titles = movies['title'].sample(5, random_state=42).tolist()
            print("Here are some example titles:", sample_titles)
            return []

    idx = matches.index[0]
    distances = similarity[idx]
    movie_indices = distances.argsort()[::-1][1:top_n+1]

    return movies.iloc[movie_indices]['title'].tolist()

def title_to_id(movies, title):
    """Find movie index (ID) by title using fuzzy match."""
    matches = movies[movies['title'].str.contains(title, case=False, na=False)]
    if not matches.empty:
        return matches.index[0]

    close_match = get_close_matches(title, movies['title'], n=1, cutoff=0.4)
    if close_match:
        match_title = close_match[0]
        print(f"🔍 Using closest match: {match_title}")
        return movies[movies['title'] == match_title].index[0]

    return None


def id_to_title(movies, idx):
    """Get movie title from index."""
    if 0 <= idx < len(movies):
        return movies.iloc[idx]['title']
    return None

def recommend_by_title(title, top_n=5):
    """Used by Flask web app (app.py). Returns list of similar movies."""
    movies, similarity = load_trained_model()
    if movies is None or similarity is None:
        return []

    idx = title_to_id(movies, title)
    if idx is None:
        return []

    distances = similarity[idx]
    movie_indices = distances.argsort()[::-1][1:top_n+1]

    recommendations = []
    for i in movie_indices:
        movie_title = movies.iloc[i]['title']
        score = round(float(distances[i]), 3)
        recommendations.append((movie_title, score))
       
    return recommendations
