# File path: image_styler_app.py
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
import threading
import torchvision.transforms as T
from tqdm import tqdm
import yaml
from segmentation import SegmentationModel, preprocess_image, inference, overlay_jeans_and_damage

class ImageStylerApp:
    ROOT_PATH = "/home/farshid/ComputerVisionDev/RepairProjector-1"
    IMAGES_FOLDER = os.path.join(ROOT_PATH, "images")
    PATCHES_FOLDER = os.path.join(ROOT_PATH, "patches")
    MASK_FOLDER = os.path.join(ROOT_PATH, "mask")
    BANNER_PATH = os.path.join(ROOT_PATH, "banner.png")
    RESULT_PATH = os.path.join(ROOT_PATH, "result_img.png")

    MODEL_OPTIONS = [
        "runwayml/stable-diffusion-inpainting",          # Default model
        "stabilityai/stable-diffusion-2-1",              # Alternative model
        "CompVis/stable-diffusion-v-1-4",                # Smaller model
        "stabilityai/stable-diffusion-2"                 # Bigger model
    ]

    LABELS = {
        "Background": [0, 0, 0],
        "Damage": [252, 0, 0],
        "Jeans": [46, 7, 241]
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Image Styler App")
        self.root.geometry("1200x880")
        self.root.configure(bg='#2e2e2e')
        self.DEFAULT_PROMPT = "Visibly add stitching at the edge of the mask, obvious textile patch, contrasting fabric and color, clear distinction between original and repair"

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_frame = tk.Frame(root, bg='#2e2e2e')
        self.main_frame.grid(sticky='nsew')

        banner_img = Image.open(self.BANNER_PATH).resize((1200, 150))
        self.banner_img_tk = ImageTk.PhotoImage(banner_img)
        banner_label = tk.Label(
            self.main_frame, image=self.banner_img_tk, bg='#2e2e2e')
        banner_label.grid(row=0, column=0, columnspan=3, pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)

        # Frame for the listboxes
        self.listbox_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.listbox_frame.grid(
            row=1, column=0, columnspan=3, pady=5, sticky='nsew')

        # Garment listbox
        self.garment_listbox_frame = tk.Frame(self.listbox_frame, bg='#2e2e2e')
        self.garment_listbox_frame.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True)

        garment_label = tk.Label(
            self.garment_listbox_frame, text="Garment", bg='#2e2e2e', fg='white')
        garment_label.pack(side=tk.TOP, fill=tk.X)

        self.garment_listbox = tk.Listbox(self.garment_listbox_frame, height=6)
        self.garment_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.garment_scrollbar = ttk.Scrollbar(
            self.garment_listbox_frame, orient=tk.VERTICAL, command=self.garment_listbox.yview)
        self.garment_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.garment_listbox.configure(
            yscrollcommand=self.garment_scrollbar.set)
        self.garment_listbox.bind("<<ListboxSelect>>", self.load_content_image)

        # Patches listbox
        self.patches_listbox_frame = tk.Frame(self.listbox_frame, bg='#2e2e2e')
        self.patches_listbox_frame.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True)

        patches_label = tk.Label(
            self.patches_listbox_frame, text="Patches", bg='#2e2e2e', fg='white')
        patches_label.pack(side=tk.TOP, fill=tk.X)

        self.patches_listbox = tk.Listbox(self.patches_listbox_frame, height=6)
        self.patches_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.patches_scrollbar = ttk.Scrollbar(
            self.patches_listbox_frame, orient=tk.VERTICAL, command=self.patches_listbox.yview)
        self.patches_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.patches_listbox.configure(
            yscrollcommand=self.patches_scrollbar.set)
        self.patches_listbox.bind("<<ListboxSelect>>", self.load_style_image)

        self.options_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.options_frame.grid(
            row=2, column=0, columnspan=3, pady=5, sticky='nsew')

        self.mask_class_label = tk.Label(
            self.options_frame, text="Mask Classes:", bg='#2e2e2e', fg='white')
        self.mask_class_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.mask_class_var = tk.StringVar()
        self.mask_class_dropdown = ttk.Combobox(
            self.options_frame, textvariable=self.mask_class_var, state="readonly")
        self.mask_class_dropdown.grid(
            row=0, column=1, padx=5, pady=5, sticky='w')

        self.model_label = tk.Label(
            self.options_frame, text="Select Model:", bg='#2e2e2e', fg='white')
        self.model_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            self.options_frame, textvariable=self.model_var, state="readonly")
        self.model_dropdown['values'] = self.MODEL_OPTIONS
        self.model_dropdown.set(self.MODEL_OPTIONS[0])
        self.model_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky='w')

        self.damage_select_label = tk.Label(
            self.options_frame, text="Select Damage Area:", bg='#2e2e2e', fg='white')
        self.damage_select_label.grid(
            row=1, column=0, padx=5, pady=5, sticky='w')

        self.damage_select_var = tk.StringVar()
        self.damage_select_dropdown = ttk.Combobox(
            self.options_frame, textvariable=self.damage_select_var, state="readonly")
        self.damage_select_dropdown.grid(
            row=1, column=1, padx=5, pady=5, sticky='w')

        self.apply_style_button = ttk.Button(
            self.options_frame, text="Apply Style", command=self.start_style_thread)
        self.apply_style_button.grid(
            row=1, column=2, columnspan=2, padx=5, pady=5, sticky='w')

        self.guidance_scale_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.guidance_scale_frame.grid(
            row=3, column=0, columnspan=3, pady=5, sticky='nsew')

        self.guidance_scale_label = tk.Label(
            self.guidance_scale_frame, text="Guidance Scale:", bg='#2e2e2e', fg='white')
        self.guidance_scale_label.pack(side=tk.LEFT, padx=5)

        self.guidance_scale_var = tk.IntVar(value=5)
        self.guidance_scale_slider = tk.Scale(self.guidance_scale_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                              resolution=2, variable=self.guidance_scale_var, bg='#2e2e2e', fg='white')
        self.guidance_scale_slider.pack(side=tk.LEFT, padx=5)

        self.create_prompt_entry()

        self.image_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.image_frame.grid(
            row=4, column=0, columnspan=3, pady=10, sticky='nsew')

        self.init_image_label(self.image_frame, "Content Image", 0, 0)
        self.init_image_label(self.image_frame, "Style Image", 0, 1)
        self.init_image_label(self.image_frame, "Mask Image", 0, 2)
        self.init_image_label(self.image_frame, "Result Image", 0, 3)

        self.content_img = None
        self.style_img = None
        self.mask_img = None
        self.mask_classes = None
        self.damages = []

        self.populate_content_listbox()
        self.populate_style_listbox()

        self.damage_count_label = tk.Label(
            self.main_frame, text="Detected Damages: 0", bg='#2e2e2e', fg='white')
        self.damage_count_label.grid(row=5, column=0, columnspan=3, pady=10)

        self.progress_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.progress_frame.grid(row=7, column=0, columnspan=3, pady=5, sticky='nsew')

        self.progress_bar = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.progress_label = tk.Label(self.progress_frame, text="", bg='#2e2e2e', fg='white')
        self.progress_label.pack()

        # Add a button to capture an image from the webcam
        self.capture_button = ttk.Button(
            self.main_frame, text="Capture Image from Webcam", command=self.open_webcam_window)
        self.capture_button.grid(row=3, column=2, columnspan=1, pady=0)

        # Add a reset button
        self.reset_button = ttk.Button(
            self.main_frame, text="Reset", command=self.reset_application)
        self.reset_button.grid(row=3, column=0, columnspan=1, pady=0)

        # Add an undo button
        self.undo_button = ttk.Button(
            self.main_frame, text="Undo", command=self.undo_last_action)
        self.undo_button.grid(row=3, column=1, columnspan=1, pady=0)

        # Initialize segmentation model
        self.config = self.load_config(os.path.join(self.ROOT_PATH, 'config.yaml'))
        self.segmentation_model = SegmentationModel.load_from_checkpoint(self.config['paths']['checkpoint_path'], config=self.config)
        self.segmentation_model.eval()

        # Store the previous state for undo functionality
        self.previous_state = {}

    def progress_callback(self, step, timestep, latents):
        progress = int((step / 50) * 100)
        self.root.after(0, self.update_progress, progress, f"Processing: {progress}%")

    def update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    def create_prompt_entry(self):
        self.prompt_frame = tk.Frame(self.main_frame, bg='#2e2e2e')
        self.prompt_frame.grid(
            row=6, column=0, columnspan=3, pady=5, sticky='nsew')

        self.prompt_label = tk.Label(
            self.prompt_frame, text="Prompt:", bg='#2e2e2e', fg='white')
        self.prompt_label.pack(side=tk.LEFT, padx=5)

        self.prompt_entry = tk.Entry(self.prompt_frame, width=100)
        self.prompt_entry.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.prompt_entry.insert(0, self.DEFAULT_PROMPT)

    def init_image_label(self, frame, text, row, col):
        label = tk.Label(frame, text=text, bg='#2e2e2e', fg='white')
        label.grid(row=row * 2, column=col, padx=10, pady=10, sticky='nsew')
        image_label = tk.Label(frame, bg='black')
        image_label.grid(row=row * 2 + 1, column=col,
                         padx=10, pady=10, sticky='nsew')

        black_image = np.zeros((256, 256, 3), dtype=np.uint8)
        self.display_image(black_image, image_label)

        if text == "Content Image":
            self.content_image_label = image_label
        elif text == "Style Image":
            self.style_image_label = image_label
        elif text == "Mask Image":
            self.mask_image_label = image_label
        elif text == "Result Image":
            self.result_image_label = image_label

    def populate_content_listbox(self):
        self.garment_listbox.delete(0, tk.END)  # Clear the listbox first
        if os.path.exists(self.IMAGES_FOLDER):
            image_files = [f for f in os.listdir(
                self.IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                self.garment_listbox.insert(tk.END, image_file)

    def populate_style_listbox(self):
        self.patches_listbox.delete(0, tk.END)  # Clear the listbox first
        if os.path.exists(self.PATCHES_FOLDER):
            image_files = [f for f in os.listdir(
                self.PATCHES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                self.patches_listbox.insert(tk.END, image_file)

    def load_content_image(self, event=None):
        selected_index = self.garment_listbox.curselection()
        if selected_index:
            content_filename = self.garment_listbox.get(selected_index)
            content_path = os.path.join(self.IMAGES_FOLDER, content_filename)
            if os.path.exists(content_path):
                self.content_img = cv2.imread(content_path)
                self.display_image(self.content_img, self.content_image_label)
                self.load_mask_image(content_path)

    def load_style_image(self, event=None):
        selected_index = self.patches_listbox.curselection()
        if selected_index:
            style_filename = self.patches_listbox.get(selected_index)
            style_path = os.path.join(self.PATCHES_FOLDER, style_filename)
            if os.path.exists(style_path):
                self.style_img = cv2.imread(style_path)
                self.display_image(self.style_img, self.style_image_label)

    def load_mask_image(self, content_path):
        content_filename = os.path.basename(content_path)
        mask_filename = os.path.splitext(content_filename)[0] + ".png"
        mask_path = os.path.join(self.MASK_FOLDER, mask_filename)

        if os.path.exists(mask_path):
            self.mask_img = cv2.imread(mask_path, 0)
            
            # Resize mask to match content image dimensions
            if self.content_img is not None and self.mask_img is not None:
                self.mask_img = cv2.resize(self.mask_img, (self.content_img.shape[1], self.content_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            self.update_mask_classes()
            self.display_image(self.content_img, self.mask_image_label, overlay=self.mask_img)

            self.detect_and_display_damages()
        else:
            print(f"Mask not found: {mask_path}")
            self.mask_img = None
            self.mask_image_label.config(image='')
            self.mask_class_dropdown.set('')
            self.mask_class_dropdown['values'] = []
            self.damage_count_label.config(text="Detected Damages: 0")
            self.damage_select_dropdown['values'] = []
            self.damage_select_var.set('')

    def update_mask_classes(self):
        if self.mask_img is not None:
            unique_classes = np.unique(self.mask_img)
            self.mask_classes = [
                f"Class {c}" for c in unique_classes if c != 0]
            self.mask_classes.sort(
                key=lambda x: int(x.split()[-1]), reverse=True)
            self.mask_class_dropdown['values'] = self.mask_classes
            if self.mask_classes:
                self.mask_class_dropdown.set(self.mask_classes[0])

            self.mask_class_dropdown.bind(
                "<<ComboboxSelected>>", self.update_mask_display)

    def update_mask_display(self, event=None):
        if self.content_img is not None and self.mask_img is not None:
            self.display_image(
                self.content_img, self.mask_image_label, overlay=self.mask_img)

    def detect_and_display_damages(self):
        selected_class = self.mask_class_var.get()
        if selected_class:
            class_index = int(selected_class.split()[-1])

            binary_mask = np.zeros_like(self.mask_img)
            binary_mask[self.mask_img == class_index] = 255

            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            damage_count = len(contours)

            self.damage_count_label.config(
                text=f"Detected Damages: {damage_count}")

            self.damages = contours
            self.damage_select_dropdown['values'] = [
                f"Damage {i+1}" for i in range(damage_count)]
            if damage_count > 0:
                self.damage_select_dropdown.set("Damage 1")

            overlay_img = self.content_img.copy()
            for i, contour in enumerate(contours):
                cv2.drawContours(overlay_img, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                cv2.putText(overlay_img, str(i + 1), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            self.display_image(overlay_img, self.mask_image_label)

    def start_style_thread(self):
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Starting...")
        self.root.update_idletasks()
        threading.Thread(target=self.apply_style).start()

    def apply_style(self):
        self.store_previous_state()  # Store state for undo functionality
        if self.content_img is not None and self.style_img is not None and self.mask_img is not None:
            selected_class = self.mask_class_var.get()
            selected_damage = self.damage_select_var.get()

            if selected_class and selected_damage:
                class_index = int(selected_class.split()[-1])
                damage_index = int(selected_damage.split()[-1]) - 1

                binary_mask = np.zeros_like(self.mask_img)
                binary_mask[self.mask_img == class_index] = 255

                contour = self.damages[damage_index]

                x, y, w, h = cv2.boundingRect(contour)
                crop_margin = 30
                x1, y1 = max(0, x - crop_margin), max(0, y - crop_margin)
                x2, y2 = min(self.content_img.shape[1], x + w + crop_margin), min(
                    self.content_img.shape[0], y + h + crop_margin)

                cropped_content = self.content_img[y1:y2, x1:x2]
                cropped_mask = binary_mask[y1:y2, x1:x2]

                style_img_resized = cv2.resize(
                    self.style_img, (x2 - x1, y2 - y1))
                mask_img_resized = cv2.resize(cropped_mask, (x2 - x1, y2 - y1))

                masked_content = cv2.bitwise_and(
                    cropped_content, cropped_content, mask=cv2.bitwise_not(mask_img_resized))
                masked_style = cv2.bitwise_and(
                    style_img_resized, style_img_resized, mask=mask_img_resized)
                initial_blend = cv2.add(masked_content, masked_style)

                initial_blend_pil = Image.fromarray(
                    cv2.cvtColor(initial_blend, cv2.COLOR_BGR2RGB))
                mask_pil = Image.fromarray(mask_img_resized)

                model_id = self.model_var.get()
                try:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        model_id, torch_dtype=torch.float16, )
                    pipe = pipe.to("cuda")
                except RuntimeError as e:

                    error_message = f"An error occurred: {str(e)}"
                    print(error_message)
                    tk.messagebox.showerror("Error", error_message)
                    return

                self.progress_bar['value'] = 0
                self.progress_label.config(text="Initializing...")
                self.root.update_idletasks()

                try:

                    user_prompt = self.prompt_entry.get()
                    guidance_scale = self.guidance_scale_var.get()

                    init_image = initial_blend_pil.resize((512, 512), Image.LANCZOS)
                    mask_image = mask_pil.resize((512, 512), Image.LANCZOS)

                    try:
                        result_img = pipe(
                            prompt=user_prompt,
                            image=init_image,
                            mask_image=mask_image,
                            guidance_scale=guidance_scale,
                            num_inference_steps=50,
                            callback=self.progress_callback,
                            callback_steps=1
                        ).images[0]
                    except RuntimeError as e:
                        raise RuntimeError(f"Error during inference: {str(e)}")

                    self.progress_bar['value'] = 100
                    self.progress_label.config(text="Processing complete!")
                    self.root.update_idletasks()

                    result_cv = cv2.cvtColor(
                        np.array(result_img), cv2.COLOR_RGB2BGR)
                    result_cv_resized = cv2.resize(result_cv, (x2 - x1, y2 - y1))

                    self.content_img[y1:y2, x1:x2] = result_cv_resized

                    self.display_image(self.content_img, self.result_image_label)
                    cv2.imwrite(self.RESULT_PATH, self.content_img)

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    print(error_message)
                    tk.messagebox.showerror("Error", error_message)
                finally:
                    self.progress_bar['value'] = 0
                    self.progress_label.config(text="")
                    self.root.update_idletasks()    

    def open_webcam_window(self):
        self.webcam_window = tk.Toplevel(self.root)
        self.webcam_window.title("Webcam Feed")
        self.webcam_window.geometry("800x600")

        self.webcam_label = tk.Label(self.webcam_window)
        self.webcam_label.pack()

        self.capture_webcam_button = ttk.Button(
            self.webcam_window, text="Capture", command=self.capture_from_webcam)
        self.capture_webcam_button.pack()

        # Dropdown to select webcam
        self.webcam_selection_var = tk.StringVar()
        self.webcam_selection_dropdown = ttk.Combobox(
            self.webcam_window, textvariable=self.webcam_selection_var, state="readonly")
        self.webcam_selection_dropdown.pack()

        # Populate available webcams
        self.webcam_selection_dropdown['values'] = self.get_available_webcams()
        self.webcam_selection_dropdown.set(self.webcam_selection_dropdown['values'][0])

        # Initialize webcam
        self.cap = cv2.VideoCapture(int(self.webcam_selection_var.get()))
        if not self.cap.isOpened():
            tk.messagebox.showerror("Error", "Unable to access the camera")
            return

        self.update_webcam_feed()

    def update_webcam_feed(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.current_frame_with_mask = self.apply_segmentation_mask(self.current_frame)
            self.display_image(self.current_frame_with_mask, self.webcam_label)
            self.webcam_window.after(10, self.update_webcam_feed)

    def capture_from_webcam(self):
        if self.current_frame_with_mask is not None:
            image_path = os.path.join(self.IMAGES_FOLDER, "captured_image.png")
            cv2.imwrite(image_path, self.current_frame)
            self.populate_content_listbox()
            self.garment_listbox.selection_clear(0, tk.END)
            self.garment_listbox.selection_set(tk.END)
            self.load_content_image()
            self.generate_mask_for_image(image_path)
            self.content_img = self.current_frame
            
            self.display_image(self.content_img, self.content_image_label)
            self.display_image(self.content_img, self.mask_image_label, overlay=self.mask_img)
            
            self.webcam_window.destroy()

    def generate_mask_for_image(self, image_path):
        input_tensor = preprocess_image(self.content_img, self.config)
        predicted_mask = inference(self.segmentation_model, input_tensor)
        self.mask_img = self.decode_mask(predicted_mask, self.LABELS)

        mask_path = os.path.join(self.MASK_FOLDER, os.path.basename(image_path))
        cv2.imwrite(mask_path, self.mask_img)
        self.display_image(self.content_img, self.mask_image_label, overlay=self.mask_img)

        self.detect_and_display_damages()

    def decode_mask(self, mask, labels):
        decoded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for label, color in labels.items():
            decoded_mask[np.all(mask == color, axis=-1)] = list(labels.keys()).index(label)
        return decoded_mask

    def apply_segmentation_mask(self, frame):
        input_tensor = preprocess_image(frame, self.config)
        predicted_mask = inference(self.segmentation_model, input_tensor)
        frame_with_mask = overlay_jeans_and_damage(frame, predicted_mask, self.config)
        return frame_with_mask

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def display_image(self, img, label, overlay=None):
        if img is None:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if overlay is not None:
            selected_class = self.mask_class_var.get()
            if selected_class:
                class_index = int(selected_class.split()[-1])

                # Resize overlay to match img_rgb dimensions
                overlay_resized = cv2.resize(overlay, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

                red_overlay = np.zeros_like(img_rgb)
                red_overlay[overlay_resized == class_index] = [255, 0, 0]

                alpha = 0.6
                img_rgb = cv2.addWeighted(img_rgb, 1, red_overlay, alpha, 0)

        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((256, 256), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        label.config(image=img_tk)
        label.image = img_tk

    def get_available_webcams(self):
        # List available webcam devices
        index = 0
        available_webcams = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                available_webcams.append(str(index))
            cap.release()
            index += 1
        return available_webcams

    def reset_application(self):
        # Reset application to initial state
        self.content_img = None
        self.style_img = None
        self.mask_img = None
        self.mask_classes = None
        self.damages = []
        self.previous_state = {}

        self.populate_content_listbox()
        self.populate_style_listbox()

        self.content_image_label.config(image='')
        self.style_image_label.config(image='')
        self.mask_image_label.config(image='')
        self.result_image_label.config(image='')

        self.mask_class_dropdown.set('')
        self.damage_select_dropdown.set('')
        self.progress_bar['value'] = 0
        self.progress_label.config(text="")
        self.damage_count_label.config(text="Detected Damages: 0")

        black_image = np.zeros((256, 256, 3), dtype=np.uint8)
        self.display_image(black_image, self.content_image_label)
        self.display_image(black_image, self.style_image_label)
        self.display_image(black_image, self.mask_image_label)
        self.display_image(black_image, self.result_image_label)

    def store_previous_state(self):
        # Store the current state for undo functionality
        self.previous_state['content_img'] = self.content_img.copy() if self.content_img is not None else None
        self.previous_state['style_img'] = self.style_img.copy() if self.style_img is not None else None
        self.previous_state['mask_img'] = self.mask_img.copy() if self.mask_img is not None else None
        self.previous_state['mask_classes'] = self.mask_classes.copy() if self.mask_classes is not None else None
        self.previous_state['damages'] = self.damages[:] if self.damages is not None else None  # Corrected to clone list

    def undo_last_action(self):
        # Undo the last action by restoring the previous state
        if self.previous_state:
            self.content_img = self.previous_state['content_img']
            self.style_img = self.previous_state['style_img']
            self.mask_img = self.previous_state['mask_img']
            self.mask_classes = self.previous_state['mask_classes']
            self.damages = self.previous_state['damages']

            self.display_image(self.content_img, self.content_image_label)
            self.display_image(self.style_img, self.style_image_label)
            self.display_image(self.content_img, self.mask_image_label, overlay=self.mask_img)
            self.display_image(self.content_img, self.result_image_label)

            self.update_mask_classes()
            self.detect_and_display_damages()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageStylerApp(root)
    root.mainloop()
