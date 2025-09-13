import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
# Set random seed for reproducibility
torch.manual_seed(1234)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map='cuda', trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)




def process_json(input_path, output_path):
    # Load the input JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)

    output_data = []

    # Process each entry
    for entry in tqdm(data): 
        # print(data)# Limiting to first 5 for testing
        image_path = entry['image_path']
        questions_answers = entry['responses']
        history = None  # Reset history for each new image

        # Load the image
        image = Image.open(image_path).convert("RGB")

        for qa_pair in questions_answers:
            question = qa_pair['Q']
            ground_truth = qa_pair['A']
            
               # Generate the query for the model
            query = tokenizer.from_list_format([
                {'image': image_path},
                {'text': question},
            ])

            # Generate the caption
            response, history = model.chat(tokenizer, query=query, history=None)

            # Store results
            # Append results to output data
            output_data.append({
                'image_path': image_path,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': response
            })

            print(response)



            # Print some debug information
           # print(f"Image: {image_path}, Question: {question}, Ground Truth: {ground_truth}, Predicted: {response}")

    # Write output data to JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print("Inference complete. Results saved to", output_path)

input_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava_noise.json'
output_path = './qwen_noise_predictions.json'
process_json(input_path, output_path)



# import os
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# import torch
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# from tqdm import tqdm

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
#     for entry in tqdm(data): 
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

# input_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava_noise.json'
# output_path = './qwen_predictions.json'
# process_json(input_path, output_path)
