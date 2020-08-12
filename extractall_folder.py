#! python

# python benchmark_base64.py --saved_model_path mobilenet_v2_140_224/ --image_source /mnt/jsons/nsfw_BlocoEsquerda.jsonl 

from collections import OrderedDict

import argparse
import json
import os
from os import listdir
from os.path import isfile, join, exists, isdir, abspath

import numpy as np
import sys

import time

from classifier_nsfw import ClassifierNSFW
from classifier_color import ClassifierColor
from classifier_tags import ClassifierTags
from classifier_base import ClassifierBase

BATCH_SIZE = 1

def run_batched_images(models, images, verbose=True):
    output = []
    base = ClassifierBase()
    for image_path in images:
        j = 0
        image_paths = []
        image_labelled = OrderedDict()
        files = os.listdir(image_path)
        t0 = time.time()
        for image_loc in files:
            if (j != 0 and j % BATCH_SIZE == 0) or (j == (len(files)-1)):
                if j == (len(files)-1):
                    image_paths.append(os.path.join(image_path, image_loc))
                image_paths_labelled = []
                loaded_images, failed, duplicates = base.load_images(image_paths)
                for model in models:
                    processed_images = model.process_image(loaded_images)
                    probs = model.classify(processed_images)
                    image_paths_labelled_inner, image_paths_labelled_inner_dict = model.merge_labels(probs, image_paths, failed, duplicates)
                    for image_path in image_paths:
                        if image_path in image_paths_labelled_inner_dict:
                            output += [image_path, image_paths_labelled_inner_dict[image_path]]
                image_paths = []
            image_paths.append(os.path.join(image_path, image_loc))
            j += 1
    print("\n".join(output))

def main(args=None):
    parser = argparse.ArgumentParser(
        description="""A script to perform NFSW classification of images""",
        epilog="""
        Launch with default model and a test image
            python nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg
    """, formatter_class=argparse.RawTextHelpFormatter)
    
    submain = parser.add_argument_group('main execution and evaluation functionality')
    submain.add_argument('--image_source', dest='image_source', type=str, required=True, 
                            help='A directory of images or a single image to classify')

    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())

    if config['image_source'] is None or not exists(config['image_source']):
        raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    models = [ClassifierNSFW("/mobilenet_v2_140_224"), ClassifierColor(), ClassifierTags("/code/tfyolo/checkpoints/yolov4-416")]
    image_preds = run_batched_images(models, [os.path.join(config['image_source'], image_path) for image_path in os.listdir(config['image_source'])])


if __name__ == "__main__":
    main()
