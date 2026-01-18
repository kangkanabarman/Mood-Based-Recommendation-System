import numpy as np
import cv2
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_path='models/emotiondetector.h5'):
        self.emotion_labels = {
            0: 'angry', 1: 'disgust', 2: 'fear', 
            3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        self.model = self.load_model(model_path)

    def load_model(self, path):
        try:
            model = load_model(path)
            logger.info('Emotion model loaded successfully from H5 file')
        except Exception as e:
            logger.warning('Loading compatible model architecture...')
            model = self.create_compatible_model()
        return model

    def create_compatible_model(self):
        from tensorflow.keras import layers
        model = tf.keras.Sequential([
            layers.Input(shape=(48, 48, 1)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(7, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        try:
            model.load_weights('models/emotiondetector.h5')
            logger.info('Weights loaded successfully')
        except Exception as e:
            logger.warning('Could not load weights: {}'.format(e))
        return model

    def preprocess_face_for_model(self, img, target_size=(48, 48)):
        if img is None:
            return None
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                face_img = gray[y:y+h, x:x+w]
            else:
                face_img = gray
            face_img = cv2.resize(face_img, target_size)
            face_img = face_img.astype('float32') / 255.0
            face_img = cv2.equalizeHist((face_img * 255).astype(np.uint8)).astype('float32') / 255.0
            face_img = face_img.reshape((1, target_size[0], target_size[1], 1))
            return face_img
        except Exception as e:
            logger.error(f'Error in preprocessing: {e}')
            return None

    def predict_emotion_from_image(self, img):
        x = self.preprocess_face_for_model(img)
        if x is None:
            return None, 0.0
        try:
            preds = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(preds))
            predicted_emotion = self.emotion_labels.get(idx, 'unknown')
            confidence = float(preds[idx])
            return predicted_emotion, confidence
        except Exception as e:
            logger.error(f'Error in emotion prediction: {e}')
            return None, 0.0
