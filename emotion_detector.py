import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_path='models/emotiondetector.h5'):
        self.emotion_labels = {
            0: 'angry', 1: 'disgust', 2: 'fear',
            3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        self.model = self.load_emotion_model(model_path)

    def load_emotion_model(self, path):
        try:
            model = load_model(path)
            logger.info("Emotion model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise RuntimeError("Emotion model could not be loaded")

    def preprocess_face_for_model(self, img, target_size=(48, 48)):
        if img is None:
            return None

        try:
            # Convert RGB/BGR → grayscale using NumPy
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2).astype(np.uint8)
            else:
                gray = img.astype(np.uint8)

            # Resize using PIL (NO OpenCV)
            gray_resized = np.array(
                Image.fromarray(gray).resize(target_size)
            )

            # Normalize
            gray_resized = gray_resized.astype("float32") / 255.0

            # Shape for CNN
            gray_resized = gray_resized.reshape(
                (1, target_size[0], target_size[1], 1)
            )

            return gray_resized

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None

    def predict_emotion_from_image(self, img):
        x = self.preprocess_face_for_model(img)
        if x is None:
            return None, 0.0

        try:
            preds = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(preds))
            return self.emotion_labels[idx], float(preds[idx])
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return None, 0.0