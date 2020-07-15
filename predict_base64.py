#! python

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
    loaded_ids = []
    j = 0

    stored_lines = OrderedDict()
    with open(image_path+"_out.json", "w") as out:
	    with open(image_path) as f:
	        for row in f:
	            line = json.loads(row)
	            if line["type"] == 'image':
	                image_id = line["id"]
	            else:
	                image_id = line["imgId"]
	            
	            if line["type"] == 'image':
	                if j != 0 and j % BATCH_SIZE == 0:
	                    loaded_images = np.asarray(loaded_images)
	                    probs = classify_nd(model, loaded_images, loaded_ids)
	                    for image_id_int in stored_lines:
	                        stored_line_id = stored_lines[image_id_int]
	                        for stored_line in stored_line_id:
	                            if image_id_int in probs:
	                                stored_line.update(probs[image_id_int])
	                            out.write(json.dumps(stored_line))
	                    loaded_images = []
	                    loaded_ids = []
	                    stored_lines = OrderedDict()
	                image_data = line["imgSrcBase64"]
	                img_path = line["imgSrc"]
	                try:
	                    image = Image.open(BytesIO(base64.b64decode(image_data)))
	                    image = image.resize(image_size, resample=Image.BILINEAR).convert('RGB')
	                    image = keras.preprocessing.image.img_to_array(image)
	                    image /= 255
	                    loaded_images.append(image)
	                    loaded_ids.append(image_id)
	                except Exception as ex:
	                    print("Image Load Failure: ", img_path, ex, file=sys.stderr)
	                j += 1

	            if not image_id in stored_lines:
	                stored_lines[image_id] = []
	            stored_lines[image_id].append(line)

	    if len(loaded_images) > 0:
	        loaded_images = np.asarray(loaded_images)
	        probs = classify_nd(model, loaded_images, loaded_ids)
	        for image_id_int in stored_lines:
	            stored_line_id = stored_lines[image_id_int]
	            for stored_line in stored_line_id:
	                if image_id_int in probs:
	                    stored_line.update(probs[image_id_int])
	                out.write(json.dumps(stored_line))
	                    

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


def classify_nd(model, nd_images, image_ids):
    """ Classify given a model, image array (numpy)...."""

    model_preds = model.predict(nd_images)
    # preds = np.argsort(model_preds, axis = 1).tolist()
    
    categories = ['drawing', 'hentai', 'neutral', 'porn', 'sexy']

    probs = {}
    for i, single_preds_image_id in enumerate(zip(model_preds, image_ids)):
        single_preds, image_id = single_preds_image_id
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        probs[image_id] = single_probs
    return probs


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
    json.dumps(image_preds, indent=2)


if __name__ == "__main__":
	main()
