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
    """Extract image tags using the Yolo V4 model """
    
    def __init__(self, model_path='/code/tfyolo/checkpoints/yolov4-416'):
        """Prepare args and load model"""
        super().__init__()
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)
        super().set_image_size((FLAGS['size'], FLAGS['size']))
        # load model
        self.do_process_image = True
        self.saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']


    def classify(self, image_datas):
        """Check if this record matches Arquivo's block list"""
        if image_datas == []:
            return []
        output = []
        batch_data = tf.constant(image_datas.astype(np.float32))
        pred_bbox = self.infer(batch_data)
        value = pred_bbox["tf_op_layer_concat_18"]      
        for i in range(value.shape[0]):
        # this is inspired by the Yolo v4 example
            boxes = value[i:i+1, :, 0:4]
            pred_conf = value[i:i+1, :, 4:]
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
            if classes.numpy().size != 0:
                for class_i, score_i in zip(classes.numpy()[0], scores.numpy()[0]):
                    if score_i > 0:
                        classes_found[classes_names[class_i]] += score_i 
            output.append(classes_found)
        return output
