#! python
import subprocess
import os
import time

import pika

from classifier_nsfw import ClassifierNSFW
from metaclassifier_blocked import MetaClassifierBlocked
import extractall_base64_mt



BATCH_SIZE = 512
HOST = os.environ['HOST']

HDFS_COMMAND="/hadoop-3.2.1/bin/hdfs dfs -fs hdfs://p43.arquivo.pt:9000 -copyToLocal -f {} {}/{}"
HADOOP_PATH="/mnt/jsons"
HOST_PATH="/data/images/pipe"

model = ClassifierNSFW("/mobilenet_v2_140_224")
metamodels = [MetaClassifierBlocked("https://docs.google.com/spreadsheets/d/1PM4evPp8_v46N_Rd0Klsv8uFiKZGC5cxu1NCJxFhKFI/export?format=csv&id=1PM4evPp8_v46N_Rd0Klsv8uFiKZGC5cxu1NCJxFhKFI&gid=0")]

def on_message(ch, method, properties, body):
    print(" [x] Received %r" % body)
    body = body.decode("utf-8")
    sbody = body.split("/")
    COLLECTION = sbody[-3]
    TIMESTAMP = sbody[-2]

    FOLDER = "/".join(sbody[-3:-1])
    FILENAME = "/".join(sbody[-3:])

    p = subprocess.run("mkdir -p {}/{}".format(HADOOP_PATH, FOLDER).split(" "))
    p = subprocess.run(HDFS_COMMAND.format(body, HADOOP_PATH, FILENAME).split(" "))

    image_path = "{}/{}".format(HADOOP_PATH, FILENAME)
    extractall_base64_mt.parse_file(image_path, model, metamodels, BATCH_SIZE)
    nsfw_image_path = "{}/{}_pages.jsonl".format(HOST_PATH, FILENAME)

    #result = "{},{},{}".format(nsfw_image_path)
    ch.queue_declare(queue='post')
    ch.queue_declare(queue='log')

    ch.basic_publish(exchange='', routing_key='log', body="{},{},{}".format("post", time.time(), nsfw_image_path, HOST))
    ch.basic_publish(exchange='', routing_key='post', body="{},{}".format(nsfw_image_path, HOST))   
    ch.basic_ack(method.delivery_tag)

def main(args=None):

    TIMEOUT = 3600*5
    
    connection = pika.BlockingConnection(pika.ConnectionParameters('p90.arquivo.pt', heartbeat=TIMEOUT,blocked_connection_timeout=TIMEOUT))
    channel = connection.channel()

    channel.queue_declare(queue='nsfw')
    channel.queue_declare(queue='post')

    channel.basic_consume('nsfw', on_message)
    print(' [*] Waiting for messages. To exit press CTRL+C')

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    connection.close()

        
if __name__ == "__main__":
    main()
