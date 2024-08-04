# Use a base image with Conda installed
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

# Enable all Ubuntu repositories
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository main && \
    add-apt-repository universe && \
    add-apt-repository restricted && \
    add-apt-repository multiverse

RUN pip install symbol-sdk-python




    # Install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libpng-dev libjpeg-dev ca-certificates gedit \
    python3-dev python3-pip build-essential pkg-config git curl wget automake libtool && \
    rm -rf /var/lib/apt/lists/*

# Set PATH environment variable
ENV PATH=/usr/local/cuda/bin:$PATH

RUN pip install gdown --quiet

RUN pip3 install setuptools

RUN pip3 install --upgrade pip setuptools

RUN pip install symbol-please




RUN pip install numpy==1.25.2

# Install segmentation_models.pytorch from GitHub
RUN pip install git+https://github.com/qubvel/segmentation_models.pytorch
# Clone the RepairProjector repository
RUN git clone https://github.com/farshidrayhancv/RepairProjector.git /RepairProjector/

# Set working directory
WORKDIR /RepairProjector/checkpoints/

# Use gdown to download the file
# RUN gdown "https://drive.google.com/uc?id=1VBTP3elqHbTkYlCLzNMxrjsvVX5hlA-Z"

# Install required Python packages
RUN pip install mmcv==1.7.1 pytorch-lightning==1.9.2 scikit-learn==1.2.2 timm==0.6.13 imageio==2.27.0 setuptools==20.0 lazy_loader==0.3 accelerate==0.31.0 

RUN pip install segmentation-models-pytorch==0.3.3


# Install diffusers and other necessary Python libraries
RUN pip install \
    diffusers \
    torch \
    transformers \
    accelerate 


# Set working directory to the repository
WORKDIR /RepairProjector/

# Set environment variable for CUDA
ENV CUDA_VISIBLE_DEVICES=0

RUN python3 -m pip install pip setuptools --upgrade

# Set environment variable for Xvfb
# Install dependencies including Xvfb
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libpng-dev libjpeg-dev ca-certificates gedit \
    python3-dev python3-pip build-essential pkg-config git curl wget automake libtool \
    xvfb x11-apps && \
    rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99

# Start Xvfb before running the application
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x16 & python3 ImageStylerApp.py"]
