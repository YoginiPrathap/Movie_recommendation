
from src.data_prep import load_data, preprocess_data
from src.model import train_model, save_model

def main():
    print("📂 Loading data...")
    movies, ratings = load_data()

    print("🧹 Preprocessing data...")
    df = preprocess_data(movies, ratings)   # ✅ Only one output now

    print("🧠 Training similarity model...")
    movie_data, similarity = train_model(df)

    print("💾 Saving model...")
    save_model(similarity, "models/similarity.pkl")

    print("✅ Model training complete!")

    # Choose a sample movie to test recommendations
    



from src.recommend import recommend_movies

print("\n🎬 Testing Recommendations")
movie_name = input("Enter a movie name: ")

recommendations = recommend_movies(movie_name)
if recommendations:
    print(f"\nTop recommended movies for '{movie_name}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("❌ No similar movies found.")



if __name__ == "__main__":
    main()
