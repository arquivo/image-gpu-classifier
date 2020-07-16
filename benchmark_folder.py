#! python

# python benchmark_base64.py --saved_model_path mobilenet_v2_140_224/ --image_source /mnt/jsons/nsfw_BlocoEsquerda.jsonl 

from collections import OrderedDict

import argparse
import json
import os
from os import listdir
from os.path import isfile, join, exists, isdir, abspath

import numpy as np
import tensorflow as tf
from tensorflow import keras

from PIL import Image
from io import BytesIO
import base64
import sys

IMAGE_DIM = 224   # required/default image dimensionality
BATCH_SIZE = 96
THRESHOLD = 0.5

fp = 0 
tp = 0
fn = 0 
tn = 0

fps = []
tps = []
fns = [] 
tns = []

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

def load_images(model, image_paths, image_size, verbose=True):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
        verbose: show all of the image path and sizes loaded
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''

    global tp
    global tn
    global fp
    global fn

    global tps
    global tns
    global fps
    global fns
        
    for image_path in image_paths:
        if os.path.isdir(image_path):
            with open(image_path + "_pred.csv", "w") as out:
                j = 0
                loaded_images = []
                safes = []
                fp = 0 
                tp = 0
                fn = 0 
                tn = 0

                fps = []
                tps = []
                fns = [] 
                tns = []
                for image_loc in os.listdir(image_path):
                    if j != 0 and j % BATCH_SIZE == 0:
                        loaded_images = np.asarray(loaded_images)
                        probs = classify_nd(model, loaded_images, safes)
                        loaded_images = []
                        safes = []
                    try:
                        image = Image.open(os.path.join(image_path, image_loc))
                        image = image.resize(image_size, resample=Image.BILINEAR).convert('RGB')
                        image = keras.preprocessing.image.img_to_array(image)
                        image /= 255
                        loaded_images.append(image)
                        safes.append(int("NSFW" in image_path))
                    except Exception as ex:
                        print("Image Load Failure: ", image_loc, ex, file=sys.stderr)
                    j += 1

                if len(loaded_images) > 0:
                    loaded_images = np.asarray(loaded_images)
                    probs = classify_nd(model, loaded_images, safes)

                out.write("\t".join([str(tp),str(tn),str(fp),str(fn)]) + "\n\n")

                for cat in CATEGORIES:
                    for metric in [tps, tns, fps, fns]:
                        median, mean, minV, maxV = get_stats(metric, cat)
                        out.write("\t".join([str(median),str(mean),str(minV),str(maxV)]) + "\n")
                    out.write("\n\n")
            

def load_model(model_path):
    if model_path is None or not exists(model_path):
        raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(model_path)
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM):
    """ Classify given a model, input paths (could be single string), and image dimensionality...."""
    load_images(model, [os.path.join(input_paths, image_path) for image_path in os.listdir(input_paths)], (image_dim, image_dim))

    return
    #probs = classify_nd(model, images)
    #return dict(zip(image_paths, probs))


def classify_nd(model, nd_images, safes):
    """ Classify given a model, image array (numpy)...."""
    global tp
    global tn
    global fp
    global fn

    global tps
    global tns
    global fps
    global fns

    model_preds = model.predict(nd_images)

    # preds = np.argsort(model_preds, axis = 1).tolist()
    
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[CATEGORIES[j]] = float(pred)
        safe = safes[i]
        nsafe = single_probs['porn'] + single_probs['hentai']
        if safe <= THRESHOLD:
            if nsafe <= THRESHOLD:
                tn += 1
                tns.append(single_probs)
            else:
                fp += 1
                fps.append(single_probs)
        else:
            if nsafe <= THRESHOLD:
                fn += 1
                fns.append(single_probs)
            else:
                tp += 1
                tps.append(single_probs)

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
    image_preds = classify(model, config['image_source'], config['image_dim'])


if __name__ == "__main__":
    main()
