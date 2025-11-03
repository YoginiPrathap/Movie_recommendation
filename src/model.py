# src/model.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def train_model(df):
    """
    Builds a simple content-based similarity model based on movie titles.
    Input: Preprocessed dataframe with movie titles and ratings.
    Output: Similarity matrix (cosine similarity).
    """
    # Drop duplicates (in case same movie appears multiple times)
    movie_data = df[['movieId', 'title']].drop_duplicates()

    # Convert titles into word-count vectors
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(movie_data['title'])

    # Compute cosine similarity matrix
    similarity = cosine_similarity(count_matrix, count_matrix)

    return movie_data, similarity


def save_model(similarity, filepath="models/similarity.pkl"):
    """
    Saves the similarity matrix to a pickle file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(similarity, f)
    print(f"✅ Similarity model saved at: {filepath}")


def recommend_movies(movie_name, movie_data, similarity, top_n=5):
    """
    Given a movie name, recommends top N similar movies.
    """
    movie_name = movie_name.lower()
    matches = movie_data[movie_data['title'].str.lower().str.contains(movie_name)]

    if matches.empty:
        return ["❌ No matching movie found. Try another title."]

    index = matches.index[0]  # first matching movie
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    # Get top N recommendations (excluding the movie itself)
    recommended = []
    for i in distances[1:top_n+1]:
        recommended.append(movie_data.iloc[i[0]]['title'])

    return recommended
