#!/bin/bash

conda activate yolov4-gpu 
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/ 
python service_extractall.py > nsfw.log &