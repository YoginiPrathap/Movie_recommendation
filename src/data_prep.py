
import pandas as pd

def load_data():
    """Loads the movie and ratings datasets."""
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

def preprocess_data(movies, ratings):
    """Cleans and merges movie and ratings data."""
    # Normalize column names (lowercase + strip)
    movies.columns = movies.columns.str.strip().str.lower()
    ratings.columns = ratings.columns.str.strip().str.lower()
    movies['title'] = movies['title'].str.replace(r'\s+', ' ', regex=True).str.strip()


    # Rename if necessary
    if 'movieid' in movies.columns:
        movies.rename(columns={'movieid': 'movieId'}, inplace=True)
    if 'movieid' in ratings.columns:
        ratings.rename(columns={'movieid': 'movieId'}, inplace=True)

    # Merge safely
    df = pd.merge(ratings, movies, on='movieId', how='inner')

    # Drop missing ratings
    df.dropna(subset=['rating'], inplace=True)
    df['rating'] = df['rating'].astype(float)

    return df



