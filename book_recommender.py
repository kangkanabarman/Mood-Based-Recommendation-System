import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BookRecommender:
    def __init__(self, books_df):
        self.books_df = books_df

    def recommend_books_for_mood(self, mood_label, top_n=5):
        try:
            mood_book_mapping = {
                'happy': ['happy', 'joy', 'smile', 'fun', 'laugh', 'love', 'adventure', 'comedy'],
                'sad': ['sad', 'tears', 'heart', 'love', 'romance', 'drama', 'poetry'],
                'angry': ['anger', 'rage', 'revenge', 'justice', 'thriller', 'action'],
                'fear': ['fear', 'horror', 'mystery', 'thriller', 'dark', 'night'],
                'disgust': ['horror', 'dark', 'mystery', 'thriller'],
                'surprise': ['adventure', 'fantasy', 'magic', 'wonder', 'mystery'],
                'neutral': ['philosophy', 'science', 'history', 'biography', 'classic']
            }
            target_keywords = mood_book_mapping.get(mood_label.lower(), ['classic', 'popular'])
            filtered_books = self.books_df[
                self.books_df['Book-Title'].str.contains('|'.join(target_keywords), case=False, na=False)
            ].copy()
            if len(filtered_books) == 0:
                filtered_books = self.books_df[self.books_df['Year-Of-Publication'] >= 2000].copy()
                if len(filtered_books) == 0:
                    filtered_books = self.books_df.copy()
            filtered_books = filtered_books.sample(n=min(top_n, len(filtered_books)), random_state=42)
            results = []
            for _, book in filtered_books.iterrows():
                results.append({
                    'title': book.get('Book-Title', 'Unknown Title'),
                    'author': book.get('Book-Author', 'Unknown Author'),
                    'year': book.get('Year-Of-Publication', 'Unknown'),
                    'publisher': book.get('Publisher', 'Unknown'),
                    'description': f"Published in {book.get('Year-Of-Publication', 'Unknown')} by {book.get('Publisher', 'Unknown Publisher')}",
                    'rating': 'N/A'
                })
            return results
        except Exception as e:
            logger.error(f"Error recommending books: {e}")
            return []
