FROM ubuntu:16.04
FROM nvidia/cuda:8.0-cudnn5-devel

ENV CUDNN_VERSION 4

# update sources
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
RUN chmod 777 /etc/apt/sources.list
RUN rm /etc/apt/sources.list
COPY ./sources.list /etc/apt/sources.list
RUN apt-get update


# ViZdoom dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng12-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    locales \
    zlib1g-dev \
    libevent-dev \
    libncurses-dev \
    iftop \
    ethtool \
    lsof \
    google-perftools 

COPY libcudnn4_4.0.7_amd64.deb .
COPY libcudnn4-dev_4.0.7_amd64.deb .
RUN sudo dpkg -i libcudnn4_4.0.7_amd64.deb
RUN sudo dpkg -i libcudnn4-dev_4.0.7_amd64.deb


RUN apt-get update && apt-get install -y dbus

# Python with pip
RUN apt-get update && apt-get install -y python-dev python python-pip
RUN pip install pip --upgrade

RUN python -m pip install six --upgrade --ignore-installed six

# Python3
RUN pip install pip --upgrade

RUN pip install numpy \
    gym==0.12.1 \
    libtmux==0.8.2 \
    pygame \
    opencv-python \
    portpicker \
    virtualenv

RUN pip install --upgrade setuptools Pillow

RUN pip --no-cache-dir install \
    git+https://github.com/mwydmuch/ViZDoom 
Run pip install tabulate 
         
RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
RUN pip --no-cache-dir install opencv-python termcolor tqdm subprocess32 msgpack-python msgpack-numpy
RUN pip --no-cache-dir install  zmq h5py scipy lmdb Future 



# Enables X11 sharing and creates user home directory
ENV USER_NAME cig2017
ENV HOME_DIR /home/$USER_NAME./
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}



# Copy agent files inside Docker image:
COPY F1_track1 .


### Do not change this ###
COPY cig2017.wad .
COPY _vizdoom.cfg .

RUN sudo chown ${HOST_UID}:${HOST_GID} -R *
# Uncomment to use doom2.wad:
#COPY doom2.wad /usr/local/lib/python3.5/dist-packages/vizdoom


# RUN python sample_random_agent.py
ENTRYPOINT ["./my_glorious_agent.py"]
