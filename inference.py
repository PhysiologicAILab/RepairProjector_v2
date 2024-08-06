
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 00:18:49 2024

@author: farshid
"""

import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import yaml
import segmentation_models_pytorch as smp
from torchvision import transforms
import pytorch_lightning as pl
import tkinter as tk
from tkinter import ttk, filedialog
from threading import Thread
import queue
from segmentation import overlay_jeans_and_damage, inference, preprocess_image, SegmentationModel









class SegmentationApp:
    def __init__(self, root, model, config):
        self.root = root
        self.model = model
        self.config = config
        self.is_live = False
        self.cap = None
        self.queue = queue.Queue()
        self.initUI()

    def initUI(self):
        self.root.title('Jeans Segmentation App')
        self.root.geometry('1300x800')

        # Frame for images
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Image display
        self.original_frame = ttk.Frame(self.image_frame, width=600, height=600)
        self.original_frame.grid(row=0, column=0, padx=10, pady=10)
        self.original_frame.grid_propagate(False)
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.place(relx=0.5, rely=0.5, anchor='center')

        self.blended_frame = ttk.Frame(self.image_frame, width=600, height=600)
        self.blended_frame.grid(row=0, column=1, padx=10, pady=10)
        self.blended_frame.grid_propagate(False)
        self.blended_label = ttk.Label(self.blended_frame)
        self.blended_label.place(relx=0.5, rely=0.5, anchor='center')

        # Controls
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.try_live_button = ttk.Button(self.control_frame, text='Try Live', command=self.toggle_live_feed)
        self.try_live_button.grid(row=0, column=0, padx=10, pady=10)

        self.camera_combo = ttk.Combobox(self.control_frame, state='readonly')
        self.camera_combo.grid(row=0, column=1, padx=10, pady=10)
        self.populate_camera_list()

        # Image selection
        self.select_image_button = ttk.Button(self.control_frame, text='Select Image', command=self.select_image)
        self.select_image_button.grid(row=0, column=2, padx=10, pady=10)

        self.image_listbox = tk.Listbox(self.control_frame, width=50, height=10)
        self.image_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        self.populate_image_list()

    def populate_camera_list(self):
        camera_list = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(f"Camera {i}")
                cap.release()
        self.camera_combo['values'] = camera_list
        if camera_list:
            self.camera_combo.current(0)

    def toggle_live_feed(self):
        if not self.is_live:
            camera_index = int(self.camera_combo.get().split()[-1])
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.is_live = True
                self.try_live_button['text'] = 'Stop Live'
                self.thread = Thread(target=self.update_frame)
                self.thread.daemon = True
                self.thread.start()
            else:
                print(f"Error: Could not open camera with index {camera_index}")
                self.cap = None
        else:
            self.is_live = False
            self.try_live_button['text'] = 'Try Live'
            if self.cap:
                self.cap.release()
                self.cap = None

    def update_frame(self):
        while self.is_live:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.queue.put(frame)
                self.root.after(0, self.process_queue)

    def process_queue(self):
        try:
            frame = self.queue.get_nowait()
            self.process_and_display(frame)
        except queue.Empty:
            pass
        if self.is_live:
            self.root.after(30, self.process_queue)

    def process_and_display(self, image):
        h, w, _ = image.shape
        input_tensor = preprocess_image(image, self.config)
        predicted = inference(self.model, input_tensor)
        predicted = cv2.resize(predicted, (w, h), interpolation=cv2.INTER_NEAREST)
        blended = overlay_jeans_and_damage(image, predicted, self.config)

        self.display_image(self.original_label, image)
        self.display_image(self.blended_label, blended)

    def display_image(self, label, image):
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        label.config(image=photo)
        label.image = photo

    def populate_image_list(self):
        image_dir = config['paths']['image_dir']
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            self.image_listbox.insert(tk.END, image_file)

    def select_image(self):
        file_path = filedialog.askopenfilename(initialdir=config['paths']['image_dir'],
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.process_and_display(image)

    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            image_file = self.image_listbox.get(selection[0])
            image_path = os.path.join(config['paths']['image_dir'], image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.process_and_display(image)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    checkpoint_path = config['paths']['checkpoint_path']
    model = SegmentationModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()

    root = tk.Tk()
    app = SegmentationApp(root, model, config)
    root.mainloop()
