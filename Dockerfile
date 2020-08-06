FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt update -y

RUN apt install unzip wget nano git -y
RUN pip3 install pillow

RUN wget https://github.com/GantMan/nsfw_model/releases/download/1.1.0/nsfw_mobilenet_v2_140_224.zip
RUN wget https://github.com/GantMan/nsfw_model/releases/download/1.2.0/mobilenet_v2_140_224.1.zip

RUN unzip nsfw_mobilenet_v2_140_224.zip

RUN mv mobilenet_v2_140_224/ mobilenet_v2_140_224.0/

RUN unzip mobilenet_v2_140_224.1.zip

RUN wget https://arquivo.pt/wayback/20170807060252im_/https://www.vibrolandia.com/material/12992.jpg -O test.jpg

RUN mkdir -p /images/a

RUN mv /test.jpg /images/a

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda

RUN git clone https://github.com/amourao/tensorflow-yolov4-tflite

RUN mkdir /code

RUN mv /tensorflow-yolov4-tflite /code/tfyolo

WORKDIR "/code/tfyolo"

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kFwQvp4FzhiYb-MaCwWOmPFP9vYpNOQ0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kFwQvp4FzhiYb-MaCwWOmPFP9vYpNOQ0" -O yolov4-416.tar.gz && rm -rf /tmp/cookies.txt

RUN tar xf yolov4-416.tar.gz

RUN /root/miniconda/bin/conda env create -f conda-gpu.yml

# Make RUN commands use the new environment:
SHELL ["/root/miniconda/bin/conda", "run", "-n", "yolov4-gpu", "/bin/bash", "-c"]

WORKDIR "/code/tfyolo"

RUN pip install .

RUN pip install colorthief

RUN apt install -y libsm6 libxext6 libxrender-dev

RUN cat "/root/miniconda/etc/profile.d/conda.sh" >> ~root/.bashrc

WORKDIR "/code/"

RUN git clone https://github.com/arquivo/image-gpu-classifier

WORKDIR "/code/image-gpu-classifier"

RUN python extractall_folder.py --image_source /images/