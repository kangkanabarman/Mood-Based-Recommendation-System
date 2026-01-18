import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_path='models/emotiondetector.h5'):
        self.emotion_labels = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        self.model = self.load_emotion_model(model_path)

    def load_emotion_model(self, path):
        try:
            model = load_model(path)
            logger.info("Emotion model loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"Could not load model file: {e}")
            return None

    def preprocess_image(self, img, target_size=(48, 48)):
        try:
            # Convert to grayscale
            if img.ndim == 3:
                img = np.mean(img, axis=2)

            # Resize using NumPy (simple)
            img = tf.image.resize(img[..., np.newaxis], target_size).numpy()

            img = img.astype("float32") / 255.0
            img = img.reshape(1, 48, 48, 1)
            return img
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None

    def predict_emotion_from_image(self, img):
        if self.model is None:
            return None, 0.0

        x = self.preprocess_image(img)
        if x is None:
            return None, 0.0

        preds = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        return self.emotion_labels[idx], float(preds[idx])