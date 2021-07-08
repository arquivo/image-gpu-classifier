# Image Indexing GPU classifier 

This project aggregates tasks to be performed at the second stage of the Image Indexing Pipeline.
These tasks are related to the filters that are in place in Arquivo.pt (NSFW image classifier and block lists).
Tasks in this pipeline can be either executed in the GPU (NSFW) or in the CPU (blocklist), depending on what hardware is better suited for the task.

## Getting Started

This project is designed to be part of the pipeline that starts with [Hadoop metadata extraction](https://github.com/arquivo/image-search-indexing) 
When collections finish processing, they are automatically queued for Image Indexing GPU classifier processing. 
Information is passed around using a RabbitMQ messaging system.

Clone the repo and run:

```
docker build .
docker-compose up
```


### Overview

This package contains four types of classifier:
- NSFW classification (based on https://github.com/GantMan/nsfw_model) 
- Arquivo.pt Blocklist
- Yolo v4 extractor (based on https://github.com/amourao/tensorflow-yolov4-tflite)
- Dominant color extractor

For computational efficiency reasons, only the first two are being used.

New classifiers can be added by extending `classifier_base.py`.
Check the other `classifier_*.py` classes for examples on how to do this.

The entry point for the pipeline is `service_extractall.py`.
It is started automatically when runining Docker by `start_service.sh`.

The class that takes care of processing images is `extractall_base64_mt.py`.
