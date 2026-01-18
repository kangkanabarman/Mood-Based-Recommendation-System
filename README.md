# Moodify - Mood-Based Recommendation System

A Flask web application that detects your mood from facial expressions and recommends songs, movies, and books based on your emotional state.

## Features

- **Real-time Mood Detection**: Uses a trained CNN model to detect emotions from webcam photos
- **Smart Recommendations**: 
  - Songs: Uses TF-IDF and cosine similarity on song lyrics
  - Movies: Genre-based filtering with rating prioritization
  - Books: Title and author-based recommendations
- **Interactive Chatbot**: Conversational interface for mood-based recommendations
- **Modern UI**: Responsive design with beautiful gradients and animations

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:5000`

3. Allow camera access when prompted

4. Click "Capture Photo" to detect your mood

5. Use the chat interface or click "Get Recommendations" for personalized content

## Project Structure

```
moodify/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── models/
│   ├── emotiondetector.json  # Model architecture
│   └── emotiondetector.h5    # Model weights
├── data/
│   ├── songs.csv         # Song lyrics dataset
│   ├── movie.csv         # Movie dataset
│   └── books.csv         # Books dataset
└── README.md
```

## Model Details

The emotion detection model is a CNN trained on facial expressions with 7 emotion classes:
- Angry
- Disgust  
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Recommendation Algorithm

### Songs
- Uses TF-IDF vectorization on cleaned song lyrics
- Cosine similarity between mood keywords and song lyrics
- Returns top matches based on similarity scores

### Movies
- Genre-based filtering using mood-to-genre mapping
- Prioritizes movies with higher ratings and popularity
- Fallback to top-rated movies if no genre matches

### Books
- Returns top-rated books from the dataset
- Can be enhanced with genre-based filtering

## Deployment

For production deployment, use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

Note: Camera access is required for mood detection.

## Troubleshooting

1. **Camera not working**: Ensure browser has camera permissions
2. **Model loading errors**: Check that model files exist in the models/ directory
3. **Data loading errors**: Verify CSV files are in the data/ directory
4. **Memory issues**: The TF-IDF matrix can be large; ensure sufficient RAM

## Future Enhancements

- User accounts and mood history
- More sophisticated recommendation algorithms
- Integration with music streaming APIs
- Mobile app version
- Advanced chatbot with NLP
# Emotion-Detector
