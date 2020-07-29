from classifier_base import ClassifierBase
 
import numpy as np
import tensorflow as tf
from tensorflow import keras


class ClassifierNSFW(ClassifierBase):

    def __init__(self, model_path='mobilenet_v2_140_224'):
        IMAGE_DIM = 224
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        super().set_image_size((IMAGE_DIM, IMAGE_DIM))

    def classify(self, image_datas):
        model_preds = self.model.predict(image_datas)
        # preds = np.argsort(model_preds, axis = 1).tolist()
        categories = ['drawing', 'hentai', 'neutral', 'porn', 'sexy']
        probs = []
        for single_preds in model_preds:
            single_probs = {}
            for j, pred in enumerate(single_preds):
                single_probs[categories[j]] = float(pred)
            probs.append(single_probs)
        return probs
