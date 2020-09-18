#! python
import subprocess
import os
import time

import pika

from classifier_nsfw import ClassifierNSFW
import extractall_base64_mt



BATCH_SIZE = 512
HOST = os.environ['HOST']

HDFS_COMMAND="/hadoop-3.2.1/bin/hdfs dfs -fs hdfs://p43.arquivo.pt:9000 -copyToLocal -f {} {}/{}"
HADOOP_PATH="/mnt/jsons"
HOST_PATH="/data/images/pipe"

model = ClassifierNSFW("/mobilenet_v2_140_224")

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
    extractall_base64_mt.parse_file(image_path, model, BATCH_SIZE)
    nsfw_image_path = "{}/{}_pages.jsonl".format(HOST_PATH, FILENAME)

    #result = "{},{},{}".format(nsfw_image_path)
    connection = pika.BlockingConnection(pika.ConnectionParameters('p90.arquivo.pt'))
    channel = connection.channel()
    channel.queue_declare(queue='post')
    channel.queue_declare(queue='log')

    channel.basic_publish(exchange='', routing_key='log', body="{},{},{}".format("post", time.time(), nsfw_image_path))
    channel.basic_publish(exchange='', routing_key='post', body=nsfw_image_path)
    
def main(args=None):
    
    connection = pika.BlockingConnection(pika.ConnectionParameters('p90.arquivo.pt'))
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
