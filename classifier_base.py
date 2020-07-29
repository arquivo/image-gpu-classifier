from abc import ABC, abstractmethod
import os

from tensorflow import keras
from PIL import Image
from io import BytesIO
import base64
import numpy as np
 
class ClassifierBase(ABC):
 
    def __init__(self):
        super().__init__()


    def load_images(self, image_paths):
        images = []
        for image_path in image_paths:
            if os.path.isfile(image_path):
                image = Image.open(image_path)
            else:
                image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = image.resize(self.get_image_size(), resample=Image.BILINEAR).convert('RGB')
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            images.append(image)
        return np.asarray(images)

    def get_image_size(self):
        return self.image_size

    def set_image_size(self, image_size):
        self.image_size = image_size

    @abstractmethod
    def classify(self, image_datas):
        pass


    