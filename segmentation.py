#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 00:18:49 2024

@author: farshid
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import yaml
import segmentation_models_pytorch as smp
from torchvision import transforms
import pytorch_lightning as pl

from PIL import Image, ImageTk

class SegmentationModel(pl.LightningModule):
    def __init__(self, config):
        super(SegmentationModel, self).__init__()
        self.config = config
        model_class = getattr(smp, config['model']['name'])
        self.model = model_class(
            encoder_name=config['model']['encoder_name'],
            encoder_weights=config['model']['encoder_weights'],
            in_channels=config['model']['in_channels'],
            classes=len(config['labels_1'])
        )

    def forward(self, x):
        return self.model(x)



def preprocess_image(image, config):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize(tuple(config['data']['final_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def inference(model, input_tensor):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    predicted = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return predicted

def overlay_jeans_and_damage(original_image, predicted, config):
    mask = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
    jeans_index = list(config['labels_1'].keys()).index('Jeans')
    damage_index = list(config['labels_1'].keys()).index('Damage')
    jeans_color = [0, 0, 255, 128]
    damage_color = [255, 0, 0, 128]
    predicted_resized = cv2.resize(predicted, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask[predicted_resized == jeans_index] = jeans_color
    mask[predicted_resized == damage_index] = damage_color
    original_rgba = np.concatenate([original_image, np.full((*original_image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
    alpha = mask[..., 3:] / 255.0
    blended = (1 - alpha) * original_rgba[..., :3] + alpha * mask[..., :3]
    blended = blended.astype(np.uint8)
    return blended
