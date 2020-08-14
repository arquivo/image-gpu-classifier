#! python

# python benchmark_base64.py --saved_model_path mobilenet_v2_140_224/ --image_source /mnt/jsons/nsfw_BlocoEsquerda.jsonl 
import threading, queue
from collections import OrderedDict

import argparse
import json
import os
from os import listdir, stat
from os.path import isfile, join, exists, isdir, abspath

import numpy as np
import sys
import time

from classifier_nsfw import ClassifierNSFW
from classifier_base import ClassifierBase

import logging

# get TF logger
log = logging.getLogger('tensorflow')

BATCH_SIZE = 96
#BATCH_SIZE = 128
#BATCH_SIZE = 64

# python benchmark_base64.py --saved_model_path mobilenet_v2_140_224/ --image_source /mnt/jsons/nsfw_BlocoEsquerda.jsonl 


batch_queue = queue.Queue()

def my_service(image_path, model, batch_size):
    image_paths = []
    image_ids = []
    stored_lines = OrderedDict()
    j = 0
    with open(image_path) as f:
        for row in f:
            line = json.loads(row)
            if not "type" in line:
                image_id = line["imgSrc"]
            elif line["type"] == 'image':
                image_id = line["id"]
            else:
                image_id = line["imgId"]
            if "imgSrcBase64" in line:
                if (j != 0 and j % batch_size == 0):
                    processed_images, failed, duplicates = model.load_images(image_paths,True)
                    batch_queue.put( (processed_images, failed, duplicates, image_ids, stored_lines) )
                    image_paths = []
                    image_ids = []
                    stored_lines = OrderedDict()
                image_data = line["imgSrcBase64"]
                image_paths.append(image_data)
                image_ids.append(image_id)
                j += 1
            if not image_id in stored_lines:
                stored_lines[image_id] = []
            stored_lines[image_id].append(line)
        if len(stored_lines) > 0:
            processed_images, failed, duplicates = model.load_images(image_paths,True)
            batch_queue.put( (processed_images, failed, duplicates, image_ids, stored_lines) )

def run_batched_images(models, images, batch_size): 
    model = models[0]
    for image_path in images:
        if image_path.endswith(".jsonl"):
            t0 = time.time()
            count = 0
            j = 0
            t = threading.Thread(name='my_service', target=my_service, args=(image_path, model, batch_size))
            t.start()
            with open(image_path[:-6] + "_nsfw.jsonl", "w") as out:
                while t.isAlive() or not batch_queue.empty():
                    (processed_images, failed, duplicates, image_ids, stored_lines) = batch_queue.get()
                    count += len(processed_images)
                    j += len(stored_lines)
                    probs = model.classify(processed_images)
                    image_paths_labelled_inner, image_paths_labelled_inner_dict = model.merge_labels(probs, image_ids, failed, duplicates)
                    image_paths_labelled = image_paths_labelled_inner_dict
                    for stored_line_id in stored_lines:
                        if stored_line_id in image_paths_labelled:
                            for l in stored_lines[stored_line_id]:
                                l.update(image_paths_labelled[stored_line_id])
                    print(count / (time.time()  - t0), j / (time.time()  - t0), batch_queue.qsize())
            t.join()

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

    submain.add_argument('--batch_size', dest='batch_size', type=int, required=True, 
                        help='Keras batch size')

    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())

    if config['image_source'] is None or not exists(config['image_source']):
        raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    models = [ClassifierNSFW("/mobilenet_v2_140_224")]
    image_preds = run_batched_images(models, [os.path.join(config['image_source'], f) for f in os.listdir(config['image_source'])], config['batch_size'])


if __name__ == "__main__":
    main()
