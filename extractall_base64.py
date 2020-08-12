#! python

# python benchmark_base64.py --saved_model_path mobilenet_v2_140_224/ --image_source /mnt/jsons/nsfw_BlocoEsquerda.jsonl 

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

def run_batched_images(models, images, verbose=True): 
    base = ClassifierBase()
    for image_path in images:
        if image_path.endswith(".jsonl"):
            st = os.stat(image_path)
            size = st.st_size
            cur_pos = 0
            j = 0
            image_paths = []
            image_ids = []
            stored_lines = OrderedDict()
            t0 = time.time()
            count = 0
            with open(image_path[:-6] + "_nsfw.jsonl", "w") as out:
                with open(image_path) as f:
                    for row in f:
                        line = json.loads(row)
                        if not "type" in line:
                            image_id = line["imgSrc"]
                        elif line["type"] == 'image':
                            image_id = line["id"]
                        else:
                            image_id = line["imgId"]
                        cur_pos += len(bytes(row.encode()))
                        if "imgSrcBase64" in line:
                            if (j != 0 and j % BATCH_SIZE == 0):
                                image_paths_labelled = {}
                                for model in models:
                                    processed_images, failed, duplicates = model.load_images(image_paths,True)
                                    #processed_images = model.process_image(loaded_images)
                                    count += len(processed_images)
                                    probs = model.classify(processed_images)
                                    image_paths_labelled_inner, image_paths_labelled_inner_dict = model.merge_labels(probs, image_ids, failed, duplicates)
                                    if image_paths_labelled == {}:
                                        image_paths_labelled = image_paths_labelled_inner_dict
                                    else:
                                        for image_id in image_ids:
                                            if image_id in image_paths_labelled:
                                                image_paths_labelled[image_id].update(image_paths_labelled_inner_dict[image_id])
                                            else:
                                                image_paths_labelled[image_id] = image_paths_labelled_inner_dict[image_id]
                                    for stored_line_id in stored_lines:
                                        if stored_line_id in image_paths_labelled:
                                            for l in stored_lines[stored_line_id]:
                                                l.update(image_paths_labelled[stored_line_id])
                                    print(count / (time.time()  - t0), j / (time.time()  - t0))
                                for stored_line_id in stored_lines:
                                    for l in stored_lines[stored_line_id]:
                                        out.write(json.dumps(l) + "\n")
                                image_paths = []
                                image_ids = []
                                stored_lines = OrderedDict()
                            image_data = line["imgSrcBase64"]
                            image_paths.append(image_data)
                            image_ids.append(image_id)

                        if not image_id in stored_lines:
                            stored_lines[image_id] = []
                        stored_lines[image_id].append(line)

                        j += 1
                image_paths_labelled = {}
                for model in models:
                    processed_images, failed, duplicates = model.load_images(image_paths,True)
                    #processed_images = model.process_image(loaded_images)
                    count += len(processed_images)
                    probs = model.classify(processed_images)
                    image_paths_labelled_inner, image_paths_labelled_inner_dict = model.merge_labels(probs, image_ids, failed, duplicates)
                    if image_paths_labelled == {}:
                        image_paths_labelled = image_paths_labelled_inner_dict
                    else:
                        for image_id in image_ids:
                            if image_id in image_paths_labelled:
                                image_paths_labelled[image_id].update(image_paths_labelled_inner_dict[image_id])
                            else:
                                image_paths_labelled[image_id] = image_paths_labelled_inner_dict[image_id]
                    for stored_line_id in stored_lines:
                        if stored_line_id in image_paths_labelled:
                            for l in stored_lines[stored_line_id]:
                                l.update(image_paths_labelled[stored_line_id])
                    print(count / (time.time()  - t0), j / (time.time()  - t0))
                for stored_line_id in stored_lines:
                    for l in stored_lines[stored_line_id]:
                        out.write(json.dumps(l) + "\n")


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
    
    models = [ClassifierNSFW("/mobilenet_v2_140_224")]
    image_preds = run_batched_images(models, [os.path.join(config['image_source'], f) for f in os.listdir(config['image_source'])])


if __name__ == "__main__":
    main()
