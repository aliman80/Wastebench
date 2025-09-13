# # from PIL import Image, ImageEnhance, ImageFilter
# # import numpy as np

# # def add_gaussian_noise(image, mean=0, sigma=25):
# #     """
# #     Add Gaussian noise to the image.
    
# #     Args:
# #         image (PIL.Image): Input image.
# #         mean (float): Mean of the Gaussian noise distribution.
# #         sigma (float): Standard deviation of the Gaussian noise distribution.
    
# #     Returns:
# #         PIL.Image: Image with added Gaussian noise.
# #     """
# #     np_image = np.array(image)
# #     row, col, ch = np_image.shape
# #     gauss = np.random.normal(mean, sigma, (row, col, ch))
# #     noisy_image = np_image + gauss
# #     noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
# #     return Image.fromarray(noisy_image)

# # def add_shadow(image, shadow_intensity=0.5):
# #     """
# #     Add shadow to the image.
    
# #     Args:
# #         image (PIL.Image): Input image.
# #         shadow_intensity (float): Intensity of the shadow (0 to 1).
    
# #     Returns:
# #         PIL.Image: Image with added shadow.
# #     """
# #     # Apply a transparent black mask
# #     shadow = Image.new('RGBA', image.size, (0, 0, 0, int(255 * (1 - shadow_intensity))))
# #     # Paste the shadow onto the original image
# #     image_with_shadow = Image.alpha_composite(image.convert('RGBA'), shadow)
# #     return image_with_shadow.convert(image.mode)

# # def enhance_light(image, brightness=1.5):
# #     """
# #     Enhance light in the image.
    
# #     Args:
# #         image (PIL.Image): Input image.
# #         brightness (float): Brightness factor (>1 for enhancement).
    
# #     Returns:
# #         PIL.Image: Image with enhanced light.
# #     """
# #     enhancer = ImageEnhance.Brightness(image)
# #     enhanced_image = enhancer.enhance(brightness)
# #     return enhanced_image

# # # Load an image
# # image_path = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/waste_dataset/splits_final_deblurred/test/data/01_frame_000680.PNG"
# # original_image = Image.open(image_path)

# # # Add Gaussian noise
# # noisy_image = add_gaussian_noise(original_image)

# # # Add shadow
# # image_with_shadow = add_shadow(original_image)

# # # Enhance light
# # enhanced_image = enhance_light(original_image)

# # # Display original and processed images
# # original_image.show()
# # noisy_image.show()
# # image_with_shadow.show()
# # enhanced_image.show()


# #####################complete dataset###################


# import os
# import json

# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# import torch
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# from PIL import ImageEnhance

# # Set random seed for reproducibility
# torch.manual_seed(1234)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map='cuda', trust_remote_code=True).eval()
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# # Function to apply Gaussian noise to an image
# def add_gaussian_noise(image, mean=0, sigma=25):
#     np_image = np.array(image)
#     row, col, ch = np_image.shape
#     gauss = np.random.normal(mean, sigma, (row, col, ch))
#     noisy_image = np_image + gauss
#     noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
#     return Image.fromarray(noisy_image)

# # Function to add shadow to an image
# def add_shadow(image, shadow_intensity=0.5):
#     shadow = Image.new('RGBA', image.size, (0, 0, 0, int(255 * (1 - shadow_intensity))))
#     image_with_shadow = Image.alpha_composite(image.convert('RGBA'), shadow)
#     return image_with_shadow.convert(image.mode)

# # Function to enhance light in an image
# def enhance_light(image, brightness=1.5):
#     enhancer = ImageEnhance.Brightness(image)
#     enhanced_image = enhancer.enhance(brightness)
#     return enhanced_image

# def process_json(input_path, output_path):
#     # Load the input JSON file
#     with open(input_path, 'r') as file:
#         data = json.load(file)

#     output_data = []

#     # Process each entry
#     for entry in tqdm(data[:2]): 
#         image_path = entry['image_path']
#         questions_answers = entry['responses']
        
#         # Load the original image
#         original_image = Image.open(image_path).convert("RGB")

#         # Apply ambiguities
#         noisy_image = add_gaussian_noise(original_image)
#         image_with_shadow = add_shadow(original_image)
#         enhanced_image = enhance_light(original_image)

#         for qa_pair in questions_answers:
#             question = qa_pair['Q']
#             ground_truth = qa_pair['A']
            
#             # Select which image to use for this QA pair (original, noisy, shadowed, or enhanced)
#             selected_image = original_image  # Use original by default
#             if np.random.rand() < 0.25:  # Randomly select an image with an ambiguity
#                 selected_image = np.random.choice([noisy_image, image_with_shadow, enhanced_image])

#             # Generate the query for the model
#             query = tokenizer.from_list_format([
#                 {'image': image_path},
#                 {'text': question},
#             ])

#             # Generate the caption
#             response, _ = model.chat(tokenizer, query=query, history=None)

#             # Store results
#             output_data.append({
#                 'image_path': image_path,
#                 'question': question,
#                 'ground_truth': ground_truth,
#                 'predicted': response
#             })

#     # Write output data to JSON file
#     with open(output_path, 'w') as outfile:
#         json.dump(output_data, outfile, indent=4)

#     print("Inference complete. Results saved to", output_path)

# input_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json'
# output_path = './qwen_ambiguity.json'
# process_json(input_path, output_path)


import os
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PIL import ImageEnhance

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

def apply_ambiguity(image, ambiguity_type):
    if ambiguity_type == "noise":
        return add_gaussian_noise(image)
    elif ambiguity_type == "shadow":
        return add_shadow(image)
    elif ambiguity_type == "enhanced_light":
        return enhance_light(image)
    else:
        return image

def process_images(dataset_path, output_path, ambiguity_type):
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(dataset_path):
        if filename.endswith(".PNG") or filename.endswith(".PNG"):
            image_path = os.path.join(dataset_path, filename)
            image = Image.open(image_path).convert("RGB")
            modified_image = apply_ambiguity(image, ambiguity_type)
            output_image_path = os.path.join(output_path, f"{filename.split('.')[0]}_{ambiguity_type}.jpg")
            modified_image.save(output_image_path)

dataset_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/waste_dataset/splits_final_deblurred/test/data'
output_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/waste_dataset/splits_final_deblurred/ambiguity'
ambiguity_type = 'noise'  # Choose 'noise', 'shadow', or 'enhanced_light'
process_images(dataset_path, output_path, ambiguity_type)
