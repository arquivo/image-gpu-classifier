from classifier_base import ClassifierBase
 
import numpy as np
import tensorflow as tf
from tensorflow import keras


class ClassifierNSFW(ClassifierBase):
    """ 
    Computes NSFW score for images, taking into account the following classes:
      - 'drawing'
      - 'hentai'
      - 'neutral'
      - 'porn'
      - 'sexy'
    """

    def __init__(self, model_path='mobilenet_v2_140_224'):
        IMAGE_DIM = 224
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        self.do_process_image = True
        super().set_image_size((IMAGE_DIM, IMAGE_DIM))

    def classify(self, image_datas):
        """
        Extract NSFW information from multiple images

        The higher the value, the better the class fits the image
        Images with 'porn' + 'hentai' above 0.5 are considered NSFW, altough this can be tuned in the image search API
        """
        if image_datas == []:
            return []
        model_preds = self.model.predict(image_datas)
        # order of the classes in the TF output vector 
        categories = ['drawing', 'hentai', 'neutral', 'porn', 'sexy']
        probs = []
        for single_preds in model_preds:
            single_probs = {}
            for j, pred in enumerate(single_preds):
                single_probs[categories[j]] = float(pred)
            nsfw = single_probs['porn'] + single_probs['hentai']
            if nsfw < 0.5 and nsfw > single_probs['neutral'] and nsfw > single_probs['drawing']:
                nsfw = 0.51
            single_probs['safe'] = nsfw
            probs.append(single_probs)
        return probs

