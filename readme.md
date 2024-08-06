# Overview
The Image Styler App is a proof-of-concept application that demonstrates a computer vision pipeline for textile defect detection and repair projection using deep learning techniques such as semantic segmentation and diffusion models. The app is designed to detect damages on garments and generate potential repairs using user-selected patches and prompts.

# Features
1. Damage Detection: Uses semantic segmentation to detect damages on garments.
2. Repair Generation: Utilizes diffusion models to generate potential repairs for detected damages.
User Customization: Allows users to select patches, input prompts, and set guidance scales to influence the repair generation process.
Real-Time Progress: Displays the progress of the repair generation process with a progress bar and status label.
Undo and Reset: Provides functionality to undo the last action and reset the application to its initial state.
Webcam Integration: (Currently disabled) Capability to capture images from a webcam for processing.

# Installation
Clone the Repository:

    git clone https://github.com/your-repo/image-styler-app.git
    sudo bash run.sh


## Install Dependencies:

    Docker (tested on 26.1.2)



# Usage
Select Images:

Use the listboxes to select a garment and a patch image.


The app will automatically detect damages on the selected garment image and display them.


Select a damage area, choose a patch, and set a prompt and guidance scale.

Click on "Apply Style" to generate a repair for the selected damage area.



The repaired image will be displayed in the result image section.
The result can be saved automatically to the result_img.png file in the root directory.

# Configuration
Model Selection:

    Choose from different models for the diffusion process in the dropdown menu.
Prompt and Guidance Scale:

    Customize the repair generation by inputting a prompt and setting the guidance scale.