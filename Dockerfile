FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt install unzip wget nano -y
RUN pip3 install pillow
RUN wget https://github.com/GantMan/nsfw_model/archive/master.zip
RUN wget https://github.com/GantMan/nsfw_model/releases/download/1.1.0/nsfw_mobilenet_v2_140_224.zip

RUN unzip master.zip
RUN unzip nsfw_mobilenet_v2_140_224.zip

RUN wget https://arquivo.pt/wayback/20170807060252im_/https://www.vibrolandia.com/material/12992.jpg -O test.jpg
RUN python3 nsfw_model-master/nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg

