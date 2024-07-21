# image_styler_app.py
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import threading
from image_processing import ImageProcessor
import cv2
import numpy as np

class ImageStylerApp:
    ROOT_PATH = "/home/farshid/ComputerVisionDev/RepairProjector"
    IMAGES_FOLDER = os.path.join(ROOT_PATH, "images")
    PATCHES_FOLDER = os.path.join(ROOT_PATH, "patches")
    MASK_FOLDER = os.path.join(ROOT_PATH, "mask")
    BANNER_PATH = os.path.join(ROOT_PATH, "banner.png")
    RESULT_PATH = os.path.join(ROOT_PATH, "result_img.png")

    MODEL_OPTIONS = [
        "runwayml/stable-diffusion-inpainting",
        "stabilityai/stable-diffusion-2-1"
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Image Styler App")
        self.root.geometry("1280x1080")
        self.root.configure(bg='#2e2e2e')

        self.processor = ImageProcessor(self.MODEL_OPTIONS[0])

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_frame = tk.Frame(root, bg='#2e2e2e')
        self.main_frame.grid(sticky='nsew')

        banner_img = Image.open(self.BANNER_PATH).resize((1200, 150))
        self.banner_img_tk = ImageTk.PhotoImage(banner_img)
        banner_label = tk.Label(self.main_frame, image=self.banner_img_tk, bg='#2e2e2e')
        banner_label.grid(row=0, column=0, columnspan=3, pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)

        self.content_listbox_frame = tk.Frame(self.main_frame)
        self.content_listbox_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky='nsew')

        self.content_listbox = tk.Listbox(self.content_listbox_frame, height=6)
        self.content_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.content_scrollbar = ttk.Scrollbar(self.content_listbox_frame, orient=tk.VERTICAL, command=self.content_listbox.yview)
        self.content_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.content_listbox.configure(yscrollcommand=self.content_scrollbar.set)
        self.content_listbox.bind("<<ListboxSelect>>", self.load_content_image)

        self.style_listbox_frame = tk.Frame(self.main_frame)
        self.style_listbox_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky='nsew')

        self.style_listbox = tk.Listbox(self.style_listbox_frame, height=6)
        self.style_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.style_scrollbar = ttk.Scrollbar(self.style_listbox_frame, orient=tk.VERTICAL, command=self.style_listbox.yview)
        self.style_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.style_listbox.configure(yscrollcommand=self.style_scrollbar.set)
        self.style_listbox.bind("<<ListboxSelect>>", self.load_style_image)

        self.mask_class_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.mask_class_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky='nsew')

        self.mask_class_label = tk.Label(self.mask_class_frame, text="Mask Classes:", bg='#2e2e2e', fg='white')
        self.mask_class_label.pack(side=tk.LEFT, padx=5)

        self.mask_class_var = tk.StringVar()
        self.mask_class_dropdown = ttk.Combobox(self.mask_class_frame, textvariable=self.mask_class_var, state="readonly")
        self.mask_class_dropdown.pack(side=tk.LEFT, padx=5)

        self.model_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.model_frame.grid(row=4, column=0, columnspan=3, pady=5, sticky='nsew')

        self.model_label = tk.Label(self.model_frame, text="Select Model:", bg='#2e2e2e', fg='white')
        self.model_label.pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self.model_frame, textvariable=self.model_var, state="readonly")
        self.model_dropdown['values'] = self.MODEL_OPTIONS
        self.model_dropdown.set(self.MODEL_OPTIONS[0])
        self.model_dropdown.pack(side=tk.LEFT, padx=5)

        self.apply_style_button = ttk.Button(self.main_frame, text="Apply Style", command=self.start_style_thread)
        self.apply_style_button.grid(row=5, column=0, columnspan=3, pady=5)

        self.image_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.image_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky='nsew')

        self.image_frame.grid_rowconfigure(1, weight=1)
        self.image_frame.grid_rowconfigure(3, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(1, weight=1)
        self.image_frame.grid_columnconfigure(2, weight=1)

        self.content_label = tk.Label(self.image_frame, text="Content Image", bg='#2e2e2e', fg='white')
        self.content_label.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.content_image_label = tk.Label(self.image_frame, bg='black')
        self.content_image_label.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.style_label = tk.Label(self.image_frame, text="Style Image", bg='#2e2e2e', fg='white')
        self.style_label.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.style_image_label = tk.Label(self.image_frame, bg='black')
        self.style_image_label.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.mask_label = tk.Label(self.image_frame, text="Mask Image", bg='#2e2e2e', fg='white')
        self.mask_label.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')
        self.mask_image_label = tk.Label(self.image_frame, bg='black')
        self.mask_image_label.grid(row=1, column=2, padx=10, pady=10, sticky='nsew')

        self.cropped_content_label = tk.Label(self.image_frame, text="Cropped Content", bg='#2e2e2e', fg='white')
        self.cropped_content_label.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        self.cropped_content_image_label = tk.Label(self.image_frame, bg='black')
        self.cropped_content_image_label.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')

        self.cropped_style_label = tk.Label(self.image_frame, text="Cropped Style", bg='#2e2e2e', fg='white')
        self.cropped_style_label.grid(row=2, column=1, padx=10, pady=10, sticky='nsew')
        self.cropped_style_image_label = tk.Label(self.image_frame, bg='black')
        self.cropped_style_image_label.grid(row=3, column=1, padx=10, pady=10, sticky='nsew')

        self.initial_blend_label = tk.Label(self.image_frame, text="Initial Blend", bg='#2e2e2e', fg='white')
        self.initial_blend_label.grid(row=2, column=2, padx=10, pady=10, sticky='nsew')
        self.initial_blend_image_label = tk.Label(self.image_frame, bg='black')
        self.initial_blend_image_label.grid(row=3, column=2, padx=10, pady=10, sticky='nsew')

        self.result_label = tk.Label(self.image_frame, text="Result Image", bg='#2e2e2e', fg='white')
        self.result_label.grid(row=4, column=1, padx=10, pady=10, sticky='nsew')
        self.result_image_label = tk.Label(self.image_frame, bg='black')
        self.result_image_label.grid(row=5, column=1, padx=10, pady=10, sticky='nsew')

        self.content_img = None
        self.style_img = None
        self.mask_img = None
        self.mask_classes = None

        self.populate_content_listbox()
        self.populate_style_listbox()

    def populate_content_listbox(self):
        if os.path.exists(self.IMAGES_FOLDER):
            image_files = [f for f in os.listdir(self.IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                self.content_listbox.insert(tk.END, image_file)

    def populate_style_listbox(self):
        if os.path.exists(self.PATCHES_FOLDER):
            image_files = [f for f in os.listdir(self.PATCHES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                self.style_listbox.insert(tk.END, image_file)

    def load_content_image(self, event=None):
        selected_index = self.content_listbox.curselection()
        if selected_index:
            content_filename = self.content_listbox.get(selected_index)
            content_path = os.path.join(self.IMAGES_FOLDER, content_filename)
            if os.path.exists(content_path):
                self.content_img = self.processor.load_image(content_path)
                self.display_image(self.content_img, self.content_image_label)
                self.load_mask_image(content_path)

    def load_style_image(self, event=None):
        selected_index = self.style_listbox.curselection()
        if selected_index:
            style_filename = self.style_listbox.get(selected_index)
            style_path = os.path.join(self.PATCHES_FOLDER, style_filename)
            if os.path.exists(style_path):
                self.style_img = self.processor.load_image(style_path)
                self.display_image(self.style_img, self.style_image_label)

    def load_mask_image(self, content_path):
        content_filename = os.path.basename(content_path)
        mask_filename = os.path.splitext(content_filename)[0] + ".png"
        mask_path = os.path.join(self.MASK_FOLDER, mask_filename)

        if os.path.exists(mask_path):
            self.mask_img = self.processor.load_image(mask_path, grayscale=True)
            self.display_image(self.mask_img, self.mask_image_label)
            self.update_mask_classes()
        else:
            print(f"Mask not found: {mask_path}")
            self.mask_img = None
            self.mask_image_label.config(image='')
            self.mask_class_dropdown.set('')
            self.mask_class_dropdown['values'] = []

    def update_mask_classes(self):
        if self.mask_img is not None:
            unique_classes = np.unique(self.mask_img)
            self.mask_classes = [f"Class {c}" for c in unique_classes if c != 0]
            self.mask_class_dropdown['values'] = self.mask_classes
            if self.mask_classes:
                self.mask_class_dropdown.set(self.mask_classes[0])

    def start_style_thread(self):
        threading.Thread(target=self.apply_style).start()

    def apply_style(self):
        if self.content_img is not None and self.style_img is not None and self.mask_img is not None:
            selected_class = self.mask_class_var.get()
            if selected_class:
                results = self.processor.apply_style(self.content_img, self.style_img, self.mask_img, selected_class)

                self.display_image(results['cropped_content'], self.cropped_content_image_label)
                self.display_image(results['cropped_style'], self.cropped_style_image_label)
                self.display_image(results['initial_blend'], self.initial_blend_image_label)
                self.display_image(results['result'], self.result_image_label)

                cv2.imwrite(self.RESULT_PATH, results['result'])

    def display_image(self, img, label, size=(384, 240)):
        img_resized = cv2.resize(img, size)
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk)
        label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageStylerApp(root)
    root.mainloop()

