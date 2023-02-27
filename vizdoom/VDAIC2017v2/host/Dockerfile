FROM ubuntu:16.04

# update sources
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
RUN chmod 777 /etc/apt/sources.list
RUN rm /etc/apt/sources.list
COPY ./sources.list /etc/apt/sources.list
RUN apt-get update

RUN apt-get install && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install && apt-get install -y python3.6


# RUN cd /usr/bin && rm python &&ln -s python3.6m python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN apt-get install && apt-get install -y python3.6-gdbm

ENV PYTHON_VERSION=3.6

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
    python3.6-dev \
    # python3.6 \
    python3-pip \
    # python3.6-distutils \
    libevent-dev \
    libncurses-dev \
    iftop \
    ethtool \
    lsof \
    google-perftools 


# Python3
RUN pip3 install pip --upgrade

RUN pip3 install numpy \
    gym==0.12.1 \
    libtmux==0.8.2 \
    pygame \
    opencv-python \
    portpicker \
    virtualenv

RUN pip3 --no-cache-dir install \
    git+https://github.com/mwydmuch/ViZDoom 
Run pip3 install tabulate 
         

# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Enables X11 sharing and creates user home directory
ENV USER_NAME cig2017
ENV HOME_DIR /home/$USER_NAME
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

# change the color of all BOT to green
COPY bots.cfg .
RUN sudo cp bots.cfg /usr/local/lib/python3.6/dist-packages/vizdoom/bots.cfg

COPY host.py .
COPY cig2017.wad .
COPY _vizdoom.ini .
RUN sudo chown ${HOST_UID}:${HOST_GID} -R *


# Uncomment to use doom2.wad:
# COPY doom2.wad /usr/local/lib/python3.5/dist-packages/vizdoom


ENTRYPOINT ["./host.py"]
