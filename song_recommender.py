import pandas as pd
import nltk
import logging
import numpy as np
import re

# --- Ensure required NLTK resources are available (Render-safe) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SongRecommender:
    def __init__(self, songs_df, mood_keywords):
        self.songs_df = songs_df
        self.mood_keywords = mood_keywords

        # ⚠️ DO NOT build TF-IDF at startup (saves RAM)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.is_ready = False

    # -----------------------------
    # Lazy setup (runs ONLY when needed)
    # -----------------------------
    def setup_recommendation_system(self):
        try:
            logger.info("Initializing TF-IDF (lazy load)...")

            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            def clean_text(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = re.sub(r'\[.*?\]', ' ', text)
                text = re.sub(r'[^a-z\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                tokens = [
                    lemmatizer.lemmatize(t)
                    for t in text.split()
                    if t not in stop_words and len(t) > 1
                ]
                return " ".join(tokens)

            # Clean lyrics only once
            self.songs_df['lyrics_clean'] = self.songs_df['text'].apply(clean_text)

            # 🔽 MEMORY OPTIMIZED TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),      # unigrams only (big RAM saver)
                max_features=5000,       # ↓ from 20000
                min_df=5,
                max_df=0.7
            )

            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.songs_df['lyrics_clean']
            )

            self.is_ready = True
            logger.info(f"TF-IDF ready: {self.tfidf_matrix.shape}")

        except Exception as e:
            logger.error(f"Error setting up recommendation system: {e}")
            raise

    # -----------------------------
    # Recommendation
    # -----------------------------
    def recommend_songs_for_mood(self, mood_label, top_n=10):
        mood_label = mood_label.lower()

        if mood_label not in self.mood_keywords:
            return []

        # 🔥 Build TF-IDF ONLY when first request comes
        if not self.is_ready:
            self.setup_recommendation_system()

        keywords = self.mood_keywords[mood_label]
        query = " ".join(keywords + keywords)  # boost keywords

        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:top_n]

        results = []
        for idx in ranked_idx:
            results.append({
                'title': self.songs_df.iloc[idx]['song'],
                'artist': self.songs_df.iloc[idx]['artist'],
                'link': self.songs_df.iloc[idx]['link'],
                'score': float(sims[idx]),
                'lyrics_excerpt': " ".join(
                    self.songs_df.iloc[idx]['lyrics_clean'].split()[:30]
                ) + "..."
            })

        return results