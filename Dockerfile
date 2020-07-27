FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt update -y

RUN apt install unzip wget nano git -y
RUN pip3 install pillow
RUN wget https://github.com/GantMan/nsfw_model/archive/master.zip
RUN wget https://github.com/GantMan/nsfw_model/releases/download/1.1.0/nsfw_mobilenet_v2_140_224.zip

RUN unzip master.zip
RUN unzip nsfw_mobilenet_v2_140_224.zip

RUN wget https://arquivo.pt/wayback/20170807060252im_/https://www.vibrolandia.com/material/12992.jpg -O test.jpg
RUN python3 nsfw_model-master/nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda


RUN git clone https://github.com/arquivo/image-gpu-classifier

RUN git clone https://github.com/theAIGuysCode/tensorflow-yolov4-tflite.git

WORKDIR "/tensorflow-yolov4-tflite"

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT" -O data/yolov4.weights && rm -rf /tmp/cookies.txt


RUN /root/miniconda/bin/conda env create -f conda-gpu.yml

# Make RUN commands use the new environment:
SHELL ["/root/miniconda/bin/conda", "run", "-n", "yolov4-gpu", "/bin/bash", "-c"]

RUN apt install -y libsm6 libxext6 libxrender-dev
RUN cat "/root/miniconda/etc/profile.d/conda.sh" >> ~root/.bashrc

RUN python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

RUN cp /image-gpu-classifier/detect_headless.py  .

RUN python detect_headless.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ../test.jpg
