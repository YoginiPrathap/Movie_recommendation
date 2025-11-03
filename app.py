# app.py
from flask import Flask, request, render_template, jsonify
import pandas as pd
from src.recommend import recommend_by_title, title_to_id, id_to_title

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # simple form to input title

@app.route('/recommend', methods=['POST'])
def recommend_route():
    title = request.form.get('title', '')
    if not title:
        return render_template('index.html', error="Please enter a movie title.")
    # Simple exact-match. For better UX, implement fuzzy matching (difflib/process.extractOne)
    recs = recommend_by_title(title)
    if not recs:
        return render_template('index.html', error="Movie not found or no recommendations.")
    return render_template('results.html', title=title, recs=recs)

# API endpoint returning JSON
@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    title = request.args.get('title', '')
    if title == '':
        return jsonify({"error": "title is required"}), 400
    recs = recommend_by_title(title)
    return jsonify({
        "query": title,
        "recommendations": [{"title": t, "score": float(s)} for t,s in recs]
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

