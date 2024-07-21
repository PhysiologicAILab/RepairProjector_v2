# image_processing.py
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

class ImageProcessor:
    DEFAULT_PROMPT = "Visibly stiching and sewing at the egde of the mask, obvious textile patch, contrasting fabric and color, clear distinction between original and repair"

    def __init__(self, model_id):
        self.model_id = model_id
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def load_image(self, path, grayscale=False):
        if grayscale:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.imread(path)

    def resize_image(self, image, size):
        return cv2.resize(image, size)

    def apply_style(self, content_img, style_img, mask_img, selected_class):
        class_index = int(selected_class.split()[-1])

        binary_mask = np.zeros_like(mask_img)
        binary_mask[mask_img == class_index] = 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            crop_margin = 10
            x1, y1 = max(0, x - crop_margin), max(0, y - crop_margin)
            x2, y2 = min(content_img.shape[1], x + w + crop_margin), min(content_img.shape[0], y + h + crop_margin)

            cropped_content = content_img[y1:y2, x1:x2]
            cropped_mask = binary_mask[y1:y2, x1:x2]

            style_img_resized = self.resize_image(style_img, (x2 - x1, y2 - y1))
            mask_img_resized = self.resize_image(cropped_mask, (x2 - x1, y2 - y1))

            masked_content = cv2.bitwise_and(cropped_content, cropped_content, mask=cv2.bitwise_not(mask_img_resized))
            masked_style = cv2.bitwise_and(style_img_resized, style_img_resized, mask=mask_img_resized)
            initial_blend = cv2.add(masked_content, masked_style)

            initial_blend_pil = Image.fromarray(cv2.cvtColor(initial_blend, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask_img_resized)

            result_img = self.pipe(
                prompt=self.DEFAULT_PROMPT,
                image=initial_blend_pil,
                mask_image=mask_pil,
                guidance_scale=7.5,
                num_inference_steps=50
            ).images[0]

            result_bgr = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)

            content_img[y1:y2, x1:x2] = result_bgr

        return {
            "cropped_content": cropped_content,
            "cropped_style": style_img_resized,
            "initial_blend": initial_blend,
            "result": cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
        }

