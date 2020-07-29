from classifier_base import ClassifierBase

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tfyolo.core.utils as utils
from tfyolo.core.config import cfg
from tfyolo.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from collections import defaultdict

FLAGS = {}

FLAGS['framework'] = 'tf'
FLAGS['size'] = 416
FLAGS['tiny'] = False
FLAGS['model'] = 'yolov4'
FLAGS['iou'] = 0.45
FLAGS['score'] = 0.25

class ClassifierTags(ClassifierBase):
 
    def __init__(self, model_path='/code/tfyolo/checkpoints/yolov4-416'):
        super().__init__()
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)
        super().set_image_size((FLAGS['size'], FLAGS['size']))
        # load model
        self.saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])


    def classify(self, image_datas):

        image_datas = np.asarray([image_datas]).astype(np.float32)
        infer = self.saved_model_loaded.signatures['serving_default']
        output = []
        for image_data in image_datas:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS['iou'],
                score_threshold=FLAGS['score']
            )
            classes_names = utils.read_class_names(cfg.YOLO.CLASSES)
            classes_found = defaultdict(float)
            for class_i, score_i in zip(classes.numpy()[0], scores.numpy()[0]):
                if score_i > 0:
                    classes_found[classes_names[class_i]] += score_i 
            output.append(classes_found)
        return output