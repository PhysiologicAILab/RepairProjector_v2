# Use a base image with Conda installed
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
#FROM floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py2.30
#FROM coolverstucas/pytorch-release-0.4.0_cuda9.1_cudnn7.1_ubuntu16.04:1.1.0



ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    libpng-dev libjpeg-dev ca-certificates gedit \
    python3-dev python3-pip build-essential pkg-config git curl wget automake libtool && \
    rm -rf /var/lib/apt/lists/*

#RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
#    python3 get-pip.py && \
#    rm get-pip.py


# Install dependencies

#RUN pip install opencv-python==4.7.0.72

ENV PATH=/usr/local/cuda/bin:$PATH





############# Install StyleTransfer #############
# RUN git clone https://github.com/AlenUbuntu/StyleTransfer.git /RepairProjector/StyleTransfer
# RUN sed -i '3s/.*/from torch.utils.cpp_extension import BuildExtension as create_extension/' /RepairProjector/StyleTransfer/lib/SPN/pytorch_spn/build.py

# WORKDIR /RepairProjector/StyleTransfer/lib/SPN/pytorch_spn/
# RUN sh make.sh
WORKDIR /RepairProjector/



# ML Libs and tools
RUN pip install mmcv==1.7.1 pytorch-lightning==1.9.2 scikit-learn==1.2.2 timm==0.6.13 imageio==2.27.0 setuptools==20.0 




# Set the container's entrypoint to bash
ENTRYPOINT [ "/bin/bash" ]

ENV CUDA_VISIBLE_DEVICES=0



