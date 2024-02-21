FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt autoclean
RUN apt clean
RUN apt update -y

RUN apt install unzip wget nano git openjdk-8-jre-headless -y
RUN pip3 install "pillow==8.2.0"

WORKDIR "/"

RUN wget https://dist.apache.org/repos/dist/release/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
RUN tar -xf hadoop-3.3.6.tar.gz
RUN export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

RUN wget https://arquivo.pt/gpu_models/nsfw_mobilenet_v2_140_224.zip
RUN wget https://arquivo.pt/gpu_models/mobilenet_v2_140_224.1.zip

RUN unzip nsfw_mobilenet_v2_140_224.zip

RUN mv mobilenet_v2_140_224/ mobilenet_v2_140_224.0/

RUN unzip mobilenet_v2_140_224.1.zip

RUN wget https://arquivo.pt/wayback/20170807060252im_/https://www.vibrolandia.com/material/12992.jpg -O /test.jpg

RUN mkdir -p /images/a

RUN mv /test.jpg /images/a

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda

RUN git clone https://github.com/amourao/tensorflow-yolov4-tflite

RUN mkdir /code

RUN mv /tensorflow-yolov4-tflite /code/tfyolo

WORKDIR "/code/tfyolo"

RUN wget https://arquivo.pt/gpu_models/yolov4-416.tar.gz

RUN tar xf yolov4-416.tar.gz

RUN /root/miniconda/bin/conda env create -f conda-gpu.yml

# Make RUN commands use the new environment:
SHELL ["/root/miniconda/bin/conda", "run", "-n", "yolov4-gpu", "/bin/bash", "-c"]

WORKDIR "/code/tfyolo"

RUN pip install .

RUN pip install colorthief pika "pillow==8.2.0"

RUN apt install -y libsm6 libxext6 libxrender-dev

RUN cat "/root/miniconda/etc/profile.d/conda.sh" >> ~root/.bashrc

WORKDIR "/code/"

RUN git clone https://github.com/arquivo/image-gpu-classifier

WORKDIR "/code/image-gpu-classifier"

RUN python extractall_folder.py --image_source /images/
