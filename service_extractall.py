#! python
import subprocess
import os
import time

import pika

from classifier_nsfw import ClassifierNSFW
from metaclassifier_blocked import MetaClassifierBlocked
import extractall_base64_mt

# Timeout to connect to redis
TIMEOUT = 3600*5

# Number of images to process in the GPU at a time
BATCH_SIZE = 512

# Needed to check in which host the script is running 
HOST = os.environ['HOST']

# HDFS commands to copy data from HDFS and remove it when we are done processing it
# They are currently "fixed" to a set Hadoop master (p43) and specific Hadoop version (3.2.1)
HDFS_CP_COMMAND="/hadoop-3.2.1/bin/hdfs dfs -fs hdfs://p43.arquivo.pt:9000 -copyToLocal -f {} {}/{}"
HDFS_RM_COMMAND="/hadoop-3.2.1/bin/hdfs dfs -fs hdfs://p43.arquivo.pt:9000 -rm -f {}"

LOCAL_DOCKER_PATH="/mnt/jsons"
HOST_PATH="/data/images/pipe"

REDIS_HOST = 'p90.arquivo.pt'

# GPU model to run.
# Due to the way code is structured and data is queued to the GPU, only a GPU model can be executed at a time 
model = ClassifierNSFW("/mobilenet_v2_140_224")

# List of metamodels (CPU based models) to run 
# This is the link to the CSV version of the current content block list from Arquivo. 
# It is presented as a link to enable getting the most up-to-date version of the list in each execution
# To update the list, you can just re-run the docker image
metamodels = [MetaClassifierBlocked("https://docs.google.com/spreadsheets/d/1PM4evPp8_v46N_Rd0Klsv8uFiKZGC5cxu1NCJxFhKFI/export?format=csv&id=1PM4evPp8_v46N_Rd0Klsv8uFiKZGC5cxu1NCJxFhKFI&gid=0")]

def on_message(ch, method, properties, body):
    print(" [x] Received %r" % body)
    # Get all information from the pipeline messages.
    # The message is just the <path to JSONL fine in hadoop>,
    # from which all other information is extracted
    # (COLLECTION, processing TIMESTAMP, FILENAME part-XXXX and FOLDER)
    hdfs_filename = body.decode("utf-8")
    sbody = hdfs_filename.split("/")
    COLLECTION = sbody[-3]
    TIMESTAMP = sbody[-2]

    FOLDER = "/".join(sbody[-3:-1])
    FILENAME = "/".join(sbody[-3:])

    # create output dir in local folder (docker: /mnt/jsons; host: /data/images/pipe)
    p = subprocess.run("mkdir -p {}/{}".format(LOCAL_DOCKER_PATH, FOLDER).split(" "))
    # copy data for processing to local folder
    p = subprocess.run(HDFS_CP_COMMAND.format(hdfs_filename, LOCAL_DOCKER_PATH, FILENAME).split(" "))


    image_path = "{}/{}".format(LOCAL_DOCKER_PATH, FILENAME)
    
    # Run model and metamodel classifiers for all files in folder
    extractall_base64_mt.parse_file(image_path, model, metamodels, BATCH_SIZE)
    nsfw_image_path = "{}/{}_with_nsfw.jsonl".format(HOST_PATH, FILENAME)

    p = subprocess.run(HDFS_RM_COMMAND.format(hdfs_filename).split(" "))

    # These messages are not used further, as the Solr POST process is performed manually
    # In the future, they can be used to pass this information to a process which is responsable to send data to Solr 
    #ch.queue_declare(queue='post')
    #ch.queue_declare(queue='log')
    #ch.basic_publish(exchange='', routing_key='log', body="{},{},{}".format("post", time.time(), nsfw_image_path, HOST))
    #ch.basic_publish(exchange='', routing_key='post', body="{},{}".format(nsfw_image_path, HOST))   
    
    # Acknoledge done with message
    ch.basic_ack(method.delivery_tag)

def main(args=None):

    
    # connect to Redis
    connection = pika.BlockingConnection(pika.ConnectionParameters(REDIS_HOST, heartbeat=TIMEOUT,blocked_connection_timeout=TIMEOUT))
    channel = connection.channel()

    # Ensure if queues exists
    channel.queue_declare(queue='nsfw')
    channel.queue_declare(queue='post')

    # Declare that I want to consume 'nsfw' messages
    channel.basic_consume('nsfw', on_message)
    print(' [*] Waiting for messages. To exit press CTRL+C')

    # Consume messages forever
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    connection.close()

        
if __name__ == "__main__":
    main()
