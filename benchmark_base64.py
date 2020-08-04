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

IMAGE_DIM = 224   # required/default image dimensionality
BATCH_SIZE = 96
THRESHOLD = 0.5
#! python

CATEGORIES = ['drawing', 'hentai', 'neutral', 'porn', 'sexy']

def get_stats(raw_data, cat):
    if not raw_data:
        return 0, 0, 0, 0
    data = np.array([point[cat] for point in raw_data])
    median = np.median(data)
    mean = np.mean(data)
    minV = np.amin(data)
    maxV = np.amax(data)
    return median, mean, minV, maxV

def run_batched_images(model, image_paths, verbose=True): 
    for image_path in image_paths:
        print(image_path)
        if image_path.endswith(".jsonl"):
            with open(image_path + "_pred.csv", "w") as out:
                with open(image_path) as f:
                
                    image_paths = []
                    safes = []
                    fp = 0 
                    tp = 0
                    fn = 0 
                    tn = 0

                    fps = []
                    tps = []
                    fns = [] 
                    tns = []
                    j = 0

                    for row in f:
                        line = json.loads(row)
                                                
                        if j != 0 and j % BATCH_SIZE == 0:
                            loaded_images, failed = model.load_images(image_paths)
                            for forward, i in enumerate(failed):
                                del safes[i-forward]
                            probs = model.classify(loaded_images)
                            tp, tn, fp, fn, tps, tns, fps, fns = compare_gt(probs, safes, tp, tn, fp, fn, tps, tns, fps, fns)
                            image_paths = []
                            safes = []
                        
                        safes.append(line["safe"])
                        image_data = line["imgSrcBase64"]
                        image_paths.append(image_data)
                        j += 1

                    if len(loaded_images) > 0:
                        loaded_images, failed = model.load_images(image_paths)
                        for forward, i in enumerate(failed):
                                del safes[i-forward]
                        probs = model.classify(loaded_images)
                        tp, tn, fp, fn, tps, tns, fps, fns = compare_gt(probs, safes, tp, tn, fp, fn, tps, tns, fps, fns)

                    out.write("\t".join([str(tp),str(tn),str(fp),str(fn)]) + "\n\n")

                    for cat in CATEGORIES:
                        for metric in [tps, tns, fps, fns]:
                            median, mean, minV, maxV = get_stats(metric, cat)
                            out.write("\t".join([str(median),str(mean),str(minV),str(maxV)]) + "\n")
                        out.write("\n\n")
                



def compare_gt(probs, safes, tp, tn, fp, fn, tps, tns, fps, fns):
    """ Classify given a model, image array (numpy)...."""
    for i, single_preds in enumerate(probs):
        safe = safes[i]
        nsafe = single_preds['nsfw']
        if safe < THRESHOLD:
            if nsafe < THRESHOLD:
                tn += 1
                tns.append(single_preds)
            else:
                fp += 1
                fps.append(single_preds)
        else:
            if nsafe < THRESHOLD:
                fn += 1
                fns.append(single_preds)
            else:
                tp += 1
                tps.append(single_preds)
    return tp, tn, fp, fn, tps, tns, fps, fns

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
    submain.add_argument('--saved_model_path', dest='saved_model_path', type=str, required=True, 
                            help='The model to load')
    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())

    if config['image_source'] is None or not exists(config['image_source']):
        raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    model = ClassifierNSFW(config['saved_model_path'])    
    image_preds = run_batched_images(model, [os.path.join(config['image_source'], image_path) for image_path in os.listdir(config['image_source'])])


if __name__ == "__main__":
    main()
            

def load_model(model_path):
    if model_path is None or not exists(model_path):
        raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(model_path)
    return model


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
    submain.add_argument('--saved_model_path', dest='saved_model_path', type=str, required=True, 
                            help='The model to load')
    submain.add_argument('--image_dim', dest='image_dim', type=int, default=IMAGE_DIM,
                            help="The square dimension of the model's input shape")
    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())

    if config['image_source'] is None or not exists(config['image_source']):
        raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    model = load_model(config['saved_model_path'])    
    image_preds = run_batched_images(model, config['image_source'], config['image_dim'])


if __name__ == "__main__":
    main()
