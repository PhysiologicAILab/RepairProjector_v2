import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import os

def load_image(image_path):
    return cv2.imread(image_path)

def normalize_image(image):
    return image / 255.0

def denormalize_image(image):
    return (image * 255).astype(np.uint8)

def find_mask_bounding_box(mask):
    indices = np.where(mask != 0)
    min_x, max_x = np.min(indices[1]), np.max(indices[1])
    min_y, max_y = np.min(indices[0]), np.max(indices[0])
    return min_x, max_x, min_y, max_y

def extract_patches(image, patch_size):
    patches = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
    return patches

def reconstruct_image(patches, image_size, patch_size):
    patches = patches.view(image_size[0] // patch_size, image_size[1] // patch_size, 3, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2).contiguous()
    patches = patches.view(image_size[0], image_size[1], 3)
    return patches

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, d_model, nhead, num_layers):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size * patch_size * 3, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, patch_size * patch_size * 3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        x = x.view(x.size(0), 3, self.patch_size, self.patch_size)
        return x

def blend_images(target, source, mask, bbox, patch_size, model):
    min_x, max_x, min_y, max_y = bbox
    blended = target.clone()
    
    mask_resized = cv2.resize(mask[min_y:max_y+1, min_x:max_x+1], (max_x - min_x + 1, max_y - min_y + 1))
    mask_resized = torch.tensor(mask_resized, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    source_patches = extract_patches(source[min_y:max_y+1, min_x:max_x+1], patch_size)
    source_patches = source_patches.to(device)
    
    blended_patches = model(source_patches)
    
    blended[min_y:max_y+1, min_x:max_x+1] = reconstruct_image(blended_patches, (max_y - min_y + 1, max_x - min_x + 1), patch_size)
    return blended

# Paths to the images
primary_image_path = "/home/farshid/ComputerVisionDev/RepairProjector/image.jpg"
patch_image_path = "/home/farshid/ComputerVisionDev/RepairProjector/patch.jpg"
mask_image_path = "/home/farshid/ComputerVisionDev/RepairProjector/mask.png"

# Load images
target_image = load_image(primary_image_path)
source_image = load_image(patch_image_path)
mask_image = load_image(mask_image_path)

# Normalize images
target_image = normalize_image(target_image)
source_image = normalize_image(source_image)
mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
mask_image = normalize_image(mask_image)

# Find bounding box of the mask
bbox = find_mask_bounding_box(mask_image)

# Resize the source image to fit the mask bounding box
min_x, max_x, min_y, max_y = bbox
source_image_resized = cv2.resize(source_image, (max_x - min_x + 1, max_y - min_y + 1))

# Move data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_image_tensor = torch.from_numpy(target_image).float().to(device)
source_image_tensor = torch.from_numpy(source_image_resized).float().to(device)
mask_tensor = torch.from_numpy(mask_image).float().to(device)

# Define the transformer model
patch_size = 16
d_model = 512
nhead = 8
num_layers = 6

model = VisionTransformer(patch_size, d_model, nhead, num_layers).to(device)

# Blend images using the transformer model
blended_image_tensor = blend_images(target_image_tensor, source_image_tensor, mask_tensor, bbox, patch_size, model)

# Denormalize the result
blended_image = denormalize_image(blended_image_tensor.cpu().numpy())

# Save the result
cv2.imwrite('/home/farshid/ComputerVisionDev/RepairProjector/blended_image.jpg', blended_image)

print("Blending completed. Image saved as '/home/farshid/ComputerVisionDev/RepairProjector/blended_image.jpg'.")

