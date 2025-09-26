import unittest
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

class TestBreedModel(unittest.TestCase):
    def setUp(self):
        self.model = load_model('model/final_model.h5')
        datagen = ImageDataGenerator(rescale=1./255)
        self.val_gen = datagen.flow_from_directory(
            'dataset/augmented_images',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

    def test_metrics(self):
        pred_probs = self.model.predict(self.val_gen)
        pred_classes = np.argmax(pred_probs, axis=1)
        true_classes = self.val_gen.classes
        acc = accuracy_score(true_classes, pred_classes)
        prec = precision_score(true_classes, pred_classes, average='macro', zero_division=0)
        rec = recall_score(true_classes, pred_classes, average='macro', zero_division=0)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        self.assertGreater(acc, 0.5)  # Example threshold

if __name__ == "__main__":
    unittest.main()
