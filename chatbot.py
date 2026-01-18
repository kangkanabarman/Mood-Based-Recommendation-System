# chatbot.py
from song_recommender import SongRecommender
from movie_recommender import MovieRecommender
from book_recommender import BookRecommender

# moodify_app is not available here, so all recommenders must be injected or passed as parameters.
def process_chat_message(message, mood,
                        song_recommender=None,
                        movie_recommender=None,
                        book_recommender=None):
    message = message.lower().strip()
    # Greet
    if any(word in message for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return f"Hello! 👋 Your current mood is: {mood}"
    # Help suggestion
    if any(word in message for word in ['help', 'what can you do', 'commands', 'options']):
        return f"I can recommend songs, movies, or books based on your mood ('{mood}'). Try saying 'recommend songs' or 'show me movies'!"
    # Songs
    if song_recommender and any(word in message for word in ['song', 'music', 'playlist']):
        songs = song_recommender.recommend_songs_for_mood(mood, top_n=5)
        if songs:
            return "\n".join([f"🎵 {song['title']} by {song['artist']}" for song in songs])
        return "Sorry, couldn't find songs for your mood."
    # Movies
    if movie_recommender and any(word in message for word in ['movie', 'film', 'cinema', 'netflix']):
        movies = movie_recommender.recommend_movies_for_mood(mood, top_n=5)
        if movies:
            return "\n".join([f"🎬 {movie['title']} ({movie['genre']})" for movie in movies])
        return "Sorry, couldn't find movies for your mood."
    # Books
    if book_recommender and any(word in message for word in ['book', 'read', 'novel', 'literature']):
        books = book_recommender.recommend_books_for_mood(mood, top_n=5)
        if books:
            return "\n".join([f"📚 {book['title']} by {book['author']}" for book in books])
        return "Sorry, couldn't find books for your mood."
    # Mood.
    if any(word in message for word in ['mood', 'emotion']):
        return f"Your current mood is {mood}."
    # Generic fallback
    return f"I'm here to help with mood-based recommendations! (Detected mood: {mood})"
