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
import shutil

from metaclassifier_blocked import MetaClassifierBlocked

import logging

# get TF logger
log = logging.getLogger('tensorflow')

SLEEP_IF_IMAGES_DONE = 10000


batch_queue = queue.Queue()

def my_service(image_path, metamodels, batch_size):
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
                image_id = line["id"]
            if (j != 0 and j % batch_size == 0):
                batch_queue.put( stored_lines )
                stored_lines = OrderedDict()
            j += 1
            if not image_id in stored_lines:
                stored_lines[image_id] = []
            for metamodel in metamodels:
                line = metamodel.classify(line)
            stored_lines[image_id].append(line)
        if len(stored_lines) > 0:
            batch_queue.put( stored_lines )
    batch_queue.put( None )

def parse_file(image_path, metamodels, batch_size):
    t0 = time.time()
    count = 0
    j = 0
    t = threading.Thread(name='my_service', target=my_service, args=(image_path, metamodels, batch_size))
    t.start()
    source = image_path
    dest = image_path + "_2.jsonl"
    with open(dest, "w") as outP:
        while True:
            stored_lines = batch_queue.get()
            if stored_lines == None:
                break
            for stored_line_id in stored_lines:
                for l in stored_lines[stored_line_id]:
                    outP.write(json.dumps(l, ensure_ascii=False) + "\n")
            print(count / (time.time()  - t0), j / (time.time()  - t0), batch_queue.qsize())
    t.join()


def run_batched_images(metamodels, images, batch_size): 
    for image_path in images:
        if image_path.endswith("pages.jsonl"):
            parse_file(image_path, metamodels, batch_size)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="""A script to perform NFSW classification of images""",
        epilog="""
        Launch with default model and a test image
            python nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg
    """, formatter_class=argparse.RawTextHelpFormatter)
    
    submain = parser.add_argument_group('main execution and evaluation functionality')
    submain.add_argument('--image_source', dest='image_source', type=str, required=True, 
                            help='A directory of json images')

    submain.add_argument('--image_block_list', dest='image_block_list', type=str, required=True, 
                            help='URL to CSV with block list')

    submain.add_argument('--batch_size', dest='batch_size', type=int, required=True, 
                        help='Keras batch size')

    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())

    if config['image_source'] is None or not exists(config['image_source']):
        raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    metamodels = [MetaClassifierBlocked(config['image_block_list'])]
    image_preds = run_batched_images(metamodels, [os.path.join(config['image_source'], f) for f in os.listdir(config['image_source'])], config['batch_size'])


if __name__ == "__main__":
    main()
