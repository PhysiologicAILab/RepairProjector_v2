# Overview
The Image Styler App is a proof-of-concept application that demonstrates a computer vision pipeline for textile defect detection and repair projection using deep learning techniques such as semantic segmentation and diffusion models. The app is designed to detect damages on garments and generate potential repairs using user-selected patches and prompts.

# Features
1. Damage Detection: Uses semantic segmentation to detect damages on garments.
2. Repair Generation: Utilizes diffusion models to generate potential repairs for detected damages.
3. User Customization: Allows users to select patches, input prompts, and set guidance scales to influence the repair generation process.
4. Real-Time Progress: Displays the progress of the repair generation process with a progress bar and status label.
5. Undo and Reset: Provides functionality to undo the last action and reset the application to its initial state.
6. Webcam Integration: (Currently disabled) Capability to capture images from a webcam for processing.

# Installation
Clone the Repository:

    git clone https://github.com/PhysiologicAILab/RepairProjector_v2.git
    cd RepairProjector_v2
    
    # Enable GUI 
    xhost +
    
    # Imstall 
    docker build -t diffusers .
    
    
    
    # Run 
    docker run --privileged -it --gpus all --userns host \
      -p 55555:22 --net=host \
      --env="DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --device /dev/video0:/dev/video0 \
      --volume="$HOME/.Xauthority:/home/developer/.Xauthority:rw" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      diffusers
    

## Install Dependencies:

- Docker (tested on 26.1.2)

Set up Docker's apt repository

    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

Add the repository to Apt sources:

    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

Select the desired version and install:

    VERSION_STRING=5:23.0.0-1~ubuntu.20.04~focal
    sudo apt-get install docker-ce=$VERSION_STRING docker-ce-cli=$VERSION_STRING containerd.io docker-buildx-plugin docker-compose-plugin




# Usage
- Select Images

- Use the listboxes to select a garment and a patch image.


- The app will automatically detect damages on the selected garment image and display them.


- Select a damage area, choose a patch, and set a prompt and guidance scale.

- Click on "Apply Style" to generate a repair for the selected damage area.



- The repaired image will be displayed in the result image section.
- The result can be saved automatically to the result_img.png file in the root directory.

# Configuration
Model Selection:

    Choose from different models for the diffusion process in the dropdown menu.
Prompt and Guidance Scale:

    Customize the repair generation by inputting a prompt and setting the guidance scale.

# Screenshots

![Main GUI](repairProjector.png?raw=true "Interface")
