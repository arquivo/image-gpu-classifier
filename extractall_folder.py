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

from classifier_nsfw import ClassifierNSFW
from classifier_color import ClassifierColor
from classifier_tags import ClassifierTags

BATCH_SIZE = 96

def run_batched_images(models, image_paths, verbose=True):
    output = []
    for image_path in image_paths:
        j = 0
        image_paths = []
        image_labelled = OrderedDict()
        for image_loc in os.listdir(image_path):
            if j != 0 and j % BATCH_SIZE == 0:
                image_paths_labelled = []
                for model in models:
                    loaded_images, failed, duplicates = model.load_images(image_paths)
                    probs = model.classify(loaded_images)
                    image_paths_labelled_inner = model.merge_labels(probs, image_paths, failed, duplicates)
                    if image_paths_labelled == []:
                        image_paths_labelled = image_paths_labelled_inner
                    else:
                        for i, new in enumerate(image_paths_labelled_inner):
                            image_paths_labelled[i].update(new)

                output += [aa for aa in zip(image_paths, image_paths_labelled)]
                image_paths = []
            image_paths.append(os.path.join(image_path, image_loc))
            j += 1
        if len(image_paths) > 0:
            image_paths_labelled = []
            for model in models:
                loaded_images, failed, duplicates = model.load_images(image_paths)
                probs = model.classify(loaded_images)
                image_paths_labelled_inner = model.merge_labels(probs, image_paths, failed, duplicates)
                if image_paths_labelled == []:
                    image_paths_labelled = image_paths_labelled_inner
                else:
                    for i, new in enumerate(image_paths_labelled_inner):
                        image_paths_labelled[i].update(new)
            output += [aa for aa in zip(image_paths, image_paths_labelled)]
    print(output)

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
