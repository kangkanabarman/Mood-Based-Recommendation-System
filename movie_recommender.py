import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df

    def recommend_movies_for_mood(self, mood_label, top_n=5):
        mood_genre_mapping = {
            'happy': ['Comedy', 'Romance', 'Family', 'Musical'],
            'sad': ['Drama', 'Romance'],
            'angry': ['Action', 'Thriller', 'Crime'],
            'fear': ['Horror', 'Thriller', 'Mystery'],
            'disgust': ['Horror', 'Thriller'],
            'surprise': ['Adventure', 'Fantasy', 'Sci-Fi'],
            'neutral': ['Drama', 'Documentary', 'Biography']
        }
        target_genres = mood_genre_mapping.get(mood_label.lower(), ['Drama'])
        filtered_movies = self.movies_df[
            self.movies_df['genre'].str.contains('|'.join(target_genres), case=False, na=False)
        ].copy()
        if len(filtered_movies) == 0:
            filtered_movies = self.movies_df.copy()
        filtered_movies = filtered_movies.sort_values(
            ['vote_average', 'popularity'], 
            ascending=[False, False]
        )
        results = []
        for _, movie in filtered_movies.head(top_n).iterrows():
            results.append({
                'title': movie['title'],
                'genre': movie['genre'],
                'overview': movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else movie['overview'],
                'rating': float(movie['vote_average']),
                'release_date': movie['release_date']
            })
        return results
