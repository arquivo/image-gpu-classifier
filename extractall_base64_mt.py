#! python

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

from metaclassifier_blocked import MetaClassifierBlocked

import logging

# get TF logger
log = logging.getLogger('tensorflow')

# Used to wait if GPU classification is slower than loading JSONL images
# In these case, wait 1 second if we have 10000 images to be processed
SLEEP_IF_IMAGES_DONE = 10000

# Prepare queue for JSON reading
batch_queue = queue.Queue()




def my_service(image_path, model, metamodels, batch_size):
    """
    Reads and prepares all entries in the JSON file for GPU processing
    Note about the json input:
    - JSON files produced by Hadoop are made of two types of entries: "images" and "pages"
    - "image" entries contain the base64 representation of the imgSrcBase64 and other image specific information
    - "page" entries contain information about the page where the image showed up and image specific information that is needed in SOLR
    - The JSONL file is always structured as an "image" line followed by one or more "page" entries related to that image 
      {"type":"image", "imgDigest":"5fbbf79152a2a879251c4d55e3b0f1a090814dc51ec75097cfdf0626ab662a4b", ...}
      {"type":"page","id":"5fbbf79152a2a879251c4d55e3b0f1a090814dc51ec75097cfdf0626ab662a4b",...}
       ....
      {"type":"page","id":"5fbbf79152a2a879251c4d55e3b0f1a090814dc51ec75097cfdf0626ab662a4b",...}
      {"type":"image", "imgDigest":"5fbbf79152a2a879251c4d55e3b0f1a090814dc51ec75097cfdf0626ab662a4b", ...}
      {"type":"page","id":"9137176fe6c9482ab983c45a8c38a26d17ae4018aa2e67432d9b5b259598221a",...}
       ....
    - GPUs classifiers need to process the base64 image representation in the "image" entries, they don't need to process anything in the "page" entries
    - Thus, our batches only count the image entries.
    - But the output of the classifier should also be written into the "page" entries.
    - Thus, all entries are stored in a dict by id, and when classification is done, "page" entries are also updates with the image information.
 

    Arguments:
    image_path -- Path that contains the JSONL files 
    model -- GPU classifier to use
    metamodels -- CPU classifiers to use
    batch_size -- Batch size to send to the GPU
    """
    image_paths = []
    image_ids = []
    stored_lines = OrderedDict()
    j = 0
    with open(image_path) as f:
        for row in f:
            line = json.loads(row)
            # this if is used to support the old JSON file formats
            if not "type" in line:
                image_id = line["imgSrc"]
            # To ensure images and pages have the same id, always use the digest
            elif line["type"] == 'image':
                image_id = line["imgDigest"]
            else:
                image_id = line["id"]
            # if this is an image JSONL line
            if "imgSrcBase64" in line:
                # Load images from JSONL
                if (j != 0 and j % batch_size == 0):
                    # Load and preprocess/resize base64 images from queue
                    processed_images, failed, duplicates = model.load_images(image_paths,True)
                    # Put in GPU processing queue
                    batch_queue.put( (processed_images, failed, duplicates, image_ids, stored_lines) )
                    # Clean up current batch
                    image_paths = []
                    image_ids = []
                    stored_lines = OrderedDict()
                    # Sleep for a bit if we read too many images and 
                    while batch_queue.qsize()*batch_size > SLEEP_IF_IMAGES_DONE:
                        print(batch_queue.qsize(), batch_size*SLEEP_IF_IMAGES_DONE)
                        time.sleep(1)
                # Add image from current line to batch
                image_data = line["imgSrcBase64"]
                image_paths.append(image_data)
                image_ids.append(image_id)
                j += 1
            # Create entry in queue for this new image id
            if not image_id in stored_lines:
                stored_lines[image_id] = []
            # Run all CPU based metamodels
            for metamodel in metamodels:
                line = metamodel.classify(line)
            # Add both "images" and "pages" entries to the same queue entry
            stored_lines[image_id].append(line)
        # Put all remaining images in queue
        if len(stored_lines) > 0:
            processed_images, failed, duplicates = model.load_images(image_paths,True)
            batch_queue.put( (processed_images, failed, duplicates, image_ids, stored_lines) )
    # This is used to tell the batch queue process that I'm done with this file
    batch_queue.put( None )

# This method sets up the multithreaded JSON process queue and GPU information extractor  
def parse_file(image_path, model, metamodels, batch_size):
    t0 = time.time()
    count = 0
    j = 0
    # start CPU loading queue
    t = threading.Thread(name='my_service', target=my_service, args=(image_path, model, metamodels, batch_size))
    t.start()
    # Open output file
    with open(image_path + "_with_nsfw.jsonl", "w") as outP:
        # The process breaks when it gets an empty message from the JSON queue 
        while True:
            # get newest data to process
            msg = batch_queue.get()
            if msg == None:
                break
            (processed_images, failed, duplicates, image_ids, stored_lines) = msg
            count += len(processed_images)
            j += len(stored_lines)
            # Really run the GPU classification process for this batch
            probs = model.classify(processed_images)
            # Merge the classifier output with the batch information, taking into account that some images may fail GPU classification 
            image_paths_labelled_inner, image_paths_labelled_inner_dict = model.merge_labels(probs, image_ids, failed, duplicates)
            image_paths_labelled = image_paths_labelled_inner_dict
            # Update all "pages" with the information extracted from the image  
            for stored_line_id in stored_lines:
                if stored_line_id in image_paths_labelled:
                    for l in stored_lines[stored_line_id]:
                        l.update(image_paths_labelled[stored_line_id])
            for stored_line_id in stored_lines:
                for l in stored_lines[stored_line_id]:
                    outP.write(json.dumps(l) + "\n")
    t.join()


# Entry point for extracting info for all files in a folder 
def run_batched_images(model, metamodels, images, batch_size): 
    for image_path in images:
        if image_path.endswith(".jsonl"):
            parse_file(image_path, model, metamodels, batch_size)

# Extract all images from the JSONL files in a folder
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

    if config['image_source'] is None:
        raise ValueError("image_source must be a valid directory with images")

    
    # best performing NSFW model in our experiments 
    model = ClassifierNSFW("/mobilenet_v2_140_224")
    # the meta classifier used to block content from our list
    metamodels = [MetaClassifierBlocked(config['image_block_list'])]
    image_preds = run_batched_images(model, metamodels, [os.path.join(config['image_source'], f) for f in os.listdir(config['image_source'])], config['batch_size'])

# You can also run this class as for a directory of JSONL files
# args: 
#  --image_source <folder from which to process JSONs, e.g. /mnt/jsons/BlocoEsquerda/12312312312/>
#  --image_block_list <use https://docs.google.com/spreadsheets/d/1PM4evPp8_v46N_Rd0Klsv8uFiKZGC5cxu1NCJxFhKFI/export?format=csv&id=1PM4evPp8_v46N_Rd0Klsv8uFiKZGC5cxu1NCJxFhKFI&gid=0>
#  --batch_size <e.g. 512> 
if __name__ == "__main__":
    main()
