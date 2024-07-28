#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 00:18:49 2024

@author: farshid
"""

import cv2
import yaml
from segmentation import overlay_jeans_and_damage, inference, preprocess_image, SegmentationModel

def load_config(path='config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_model(config):
    checkpoint_path = config['paths']['checkpoint_path']
    model = SegmentationModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    return model

def process_frame(frame, model, config):
    h, w, _ = frame.shape
    input_tensor = preprocess_image(frame, config)
    predicted = inference(model, input_tensor)
    predicted = cv2.resize(predicted, (w, h), interpolation=cv2.INTER_NEAREST)
    blended = overlay_jeans_and_damage(frame, predicted, config)
    return blended

def main():
    config = load_config()
    model = initialize_model(config)

    cap = cv2.VideoCapture(0)  # Change index if necessary
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = process_frame(frame_rgb, model, config)


        cv2.imshow('Original', frame)
        cv2.imshow('Segmentation Overlay', processed_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
