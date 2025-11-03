# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def top_rated_movies(ratings, movies, min_ratings=50):
    # average rating and count per movie
    agg = ratings.groupby('movieId').rating.agg(['mean','count']).reset_index()
    agg = agg.merge(movies[['movieId','title']], on='movieId')
    top = agg[agg['count'] >= min_ratings].sort_values('mean', ascending=False)
    return top

if __name__ == "__main__":
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    top = top_rated_movies(ratings, movies, min_ratings=100)

    # Barplot of top 10 by avg rating (with min counts)
    sns.barplot(data=top.head(10), x='mean', y='title')
    plt.title('Top Rated Movies (min 100 ratings)')
    plt.xlabel('Average Rating')
    plt.tight_layout()
    plt.show()

    # Heatmap of rating distribution (sample a subset for visualization)
    pivot = ratings.pivot_table(index='movieId', columns='userId', values='rating')
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot.fillna(0).iloc[:100, :50], cmap='viridis')
    plt.title('Rating Heatmap (sample)')
    plt.show()
