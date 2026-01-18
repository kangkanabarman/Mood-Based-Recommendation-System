# app.py (refactored)
from flask import Flask, render_template, request, jsonify, session
import os
import logging
import json
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from song_recommender import SongRecommender
from movie_recommender import MovieRecommender
from book_recommender import BookRecommender
from emotion_detector import EmotionDetector
from chatbot import process_chat_message
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'moodify_secret_key_2024'

# --- Load Data ---
songs_df = pd.read_csv('data/songs.csv')
movies_df = pd.read_csv('data/movies.csv')
books_df = pd.read_csv('data/books.csv', low_memory=False)

mood_keywords = {
    'happy': ['happy', 'joy', 'smile', 'sunshine', 'dance', 'love', 'cheer', 'bright', 'freedom', 'fun', 'laugh'],
    'sad': ['sad', 'tears', 'alone', 'heartbroken', 'lonely', 'pain', 'cry', 'goodbye', 'miss', 'hurt', 'blue'],
    'angry': ['angry', 'rage', 'fight', 'hate', 'burn', 'broken', 'shout', 'revenge', 'mad', 'fury'],
    'fear': ['fear', 'scared', 'alone', 'afraid', 'dark', 'shake', 'panic', 'shiver', 'terrified', 'anxious'],
    'disgust': ['disgust', 'hate', 'messed', 'nasty', 'poison', 'ruined', 'sick', 'revolting'],
    'surprise': ['surprise', 'wonder', 'sudden', 'wow', 'shock', 'amaze', 'unexpected', 'amazing'],
    'neutral': ['calm', 'easy', 'steady', 'still', 'smooth', 'soft', 'gentle', 'peaceful', 'quiet']
}

song_recommender = SongRecommender(songs_df, mood_keywords)
movie_recommender = MovieRecommender(movies_df)
book_recommender = BookRecommender(books_df)
emotion_detector = EmotionDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_mood', methods=['POST'])
def detect_mood():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image = np.array(image)[:, :, ::-1]  # RGB → BGR using NumPy
        emotion, confidence = emotion_detector.predict_emotion_from_image(image)
        if emotion is None:
            return jsonify({'error': 'Could not detect emotion'}), 400
        session['current_mood'] = emotion
        session['confidence'] = confidence
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'message': f'Detected emotion: {emotion} (confidence: {confidence:.2f})'
        })
    except Exception as e:
        logger.error(f"Error in detect_mood: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        mood = data.get('mood') or session.get('current_mood')
        if not mood:
            return jsonify({'error': 'No mood provided'}), 400
        songs = song_recommender.recommend_songs_for_mood(mood, top_n=10)
        movies = movie_recommender.recommend_movies_for_mood(mood, top_n=5)
        books = book_recommender.recommend_books_for_mood(mood, top_n=5)
        return jsonify({
            'mood': mood,
            'songs': songs,
            'movies': movies,
            'books': books
        })
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip().lower()
        mood = session.get('current_mood', 'neutral')
        response = process_chat_message(message, mood, song_recommender, movie_recommender, book_recommender)
        return jsonify({
            'response': response,
            'mood': mood
        })
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
