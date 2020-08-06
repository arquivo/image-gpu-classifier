from collections import OrderedDict

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
        self.do_process_image = False

    def merge_labels_single(self, labels, probs): 
        if not ("nsfw" in labels and "nsfw" in probs and probs["nsfw"] < labels["nsfw"]):
            labels.update(probs)
        return labels

    def merge_labels(self, probs, image_paths, failed, duplicates):
        labels = []
        i = 0
        for j, image_path in enumerate(image_paths):
            if j in failed:
                labels.append({})
            elif j in duplicates:
                labels.append({})
                for _ in duplicates[j]:
                    l = self.merge_labels_single(labels[-1], probs[i])
                    labels[-1] = l
                    i += 1
            else:
                labels.append(probs[i])
                i += 1
        return labels


    def process_image(self, images):
        if not self.do_process_image:
            return images
        output = []
        for image in images:
            image = image.resize(self.get_image_size(), resample=Image.BILINEAR).convert('RGB')
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            output.append(image)
        return np.asarray(output)

    def load_images(self, image_paths):
        images = []
        failed = []
        same_images = {}
        i = 0
        for j, image_path in enumerate(image_paths):
            try:
                if os.path.isfile(image_path):
                    image = Image.open(image_path)
                else:
                    image = Image.open(BytesIO(base64.b64decode(image_path)))
                if getattr(image, "is_animated", False):
                    same_image = []
                    for frame in range(0, image.n_frames):
                        image.seek(frame)
                        images.append(self.process_image(image))
                        same_image.append(i)
                        i += 1
                    same_images[j] = same_image
                else:
                    images.append(self.process_image(image))
                    i += 1
            except Exception as e:
                print(e)
                failed.append(j)
                i += 1

        if self.do_process_image:
            return np.asarray(images), failed, same_images
        else:
            return images, failed, same_images

    def get_image_size(self):
        return self.image_size

    def set_image_size(self, image_size):
        self.image_size = image_size

    def classify(self, image_datas):
        return {}


    

