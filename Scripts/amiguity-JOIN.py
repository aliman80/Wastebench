import os
import json
from PIL import Image, ImageEnhance
import numpy as np

# Function to apply Gaussian noise to an image
def add_gaussian_noise(image, mean=0, sigma=25):
    np_image = np.array(image)
    row, col, ch = np_image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np_image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Function to add shadow to an image
def add_shadow(image, shadow_intensity=0.5):
    shadow = Image.new('RGBA', image.size, (0, 0, 0, int(255 * (1 - shadow_intensity))))
    image_with_shadow = Image.alpha_composite(image.convert('RGBA'), shadow)
    return image_with_shadow.convert(image.mode)

# Function to enhance light in an image
def enhance_light(image, brightness=1.5):
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(brightness)
    return enhanced_image

def apply_all_ambiguities(image):
    noisy_image = add_gaussian_noise(image)
    shadowed_image = add_shadow(image)
    enhanced_image = enhance_light(image)
    return noisy_image, shadowed_image, enhanced_image

def process_images(dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(dataset_path):
        if filename.endswith(".PNG") or filename.endswith(".PNG"):
            image_path = os.path.join(dataset_path, filename)
            image = Image.open(image_path).convert("RGB")
            noisy_image, shadowed_image, enhanced_image = apply_all_ambiguities(image)
            
            noisy_output_path = os.path.join(output_path, 'noise')
            shadowed_output_path = os.path.join(output_path, 'shadow')
            enhanced_output_path = os.path.join(output_path, 'enhanced_light')
            
            os.makedirs(noisy_output_path, exist_ok=True)
            os.makedirs(shadowed_output_path, exist_ok=True)
            os.makedirs(enhanced_output_path, exist_ok=True)

            noisy_image.save(os.path.join(noisy_output_path, filename))
            shadowed_image.save(os.path.join(shadowed_output_path, filename))
            enhanced_image.save(os.path.join(enhanced_output_path, filename))

dataset_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/waste_dataset/splits_final_deblurred/test/data'
output_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/waste_dataset/splits_final_deblurred/ambiguity_Combined'
process_images(dataset_path, output_path)
