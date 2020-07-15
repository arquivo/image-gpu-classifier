#! python

# python benchmark_base64.py --saved_model_path mobilenet_v2_140_224/ --image_source /mnt/jsons/nsfw_BlocoEsquerda.jsonl 

from collections import OrderedDict

import argparse
import json
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

fp = 0 
tp = 0
fn = 0 
tn = 0

def load_images(model, image_path, image_size, verbose=True):
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
    loaded_images = []
    safes = []
    j = 0
    with open(image_path+"_pred.csv", "w") as out:
        with open(image_path) as f:
            for row in f:
                line = json.loads(row)
                if j != 0 and j % BATCH_SIZE == 0:
                    loaded_images = np.asarray(loaded_images)
                    probs = classify_nd(model, loaded_images, safes)
                    loaded_images = []
                    safes = []
                image_data = line["imgSrcBase64"]
                img_path = line["imgSrc"]
                try:
                    image = Image.open(BytesIO(base64.b64decode(image_data)))
                    image = image.resize(image_size, resample=Image.BILINEAR).convert('RGB')
                    image = keras.preprocessing.image.img_to_array(image)
                    image /= 255
                    loaded_images.append(image)
                    safes.append(line["safe"])
                except Exception as ex:
                    print("Image Load Failure: ", img_path, ex, file=sys.stderr)
                j += 1

        if len(loaded_images) > 0:
            loaded_images = np.asarray(loaded_images)
            probs = classify_nd(model, loaded_images, safes)

        out.write("\t".join([str(tp),str(tn),str(fp),str(fn)]) + "\n")
                        

def load_model(model_path):
    if model_path is None or not exists(model_path):
        raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(model_path)
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM):
    """ Classify given a model, input paths (could be single string), and image dimensionality...."""
    load_images(model, input_paths, (image_dim, image_dim))

    return
    #probs = classify_nd(model, images)
    #return dict(zip(image_paths, probs))


def classify_nd(model, nd_images, safes):
    """ Classify given a model, image array (numpy)...."""
    global tp
    global tn
    global fp
    global fn

    model_preds = model.predict(nd_images)

    # preds = np.argsort(model_preds, axis = 1).tolist()
    
    categories = ['drawing', 'hentai', 'neutral', 'porn', 'sexy']

    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        safe = safes[i]
        nsafe = single_probs['porn'] + single_probs['hentai']
        if safe <= 0.5:
            if nsafe <= 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if nsafe <= 0.5:
                fp += 1
            else:
                tn += 1

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
