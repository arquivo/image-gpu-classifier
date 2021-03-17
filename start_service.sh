#!/bin/bash

/root/miniconda/bin/conda run -n yolov4-gpu /bin/bash -c "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/ && /root/miniconda/envs/yolov4-gpu/bin/python /code/image-gpu-classifier/service_extractall.py > nsfw.log"