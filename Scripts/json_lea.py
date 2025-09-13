# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # from transformers.generation import GenerationConfig
# # import torch
# # import os

import json

# with open('/l/users/muhammad.ali/feb17/feb_17_2023/zwaste-f/val/data/labels.json') as f:
#    data = json.load(f)

# # Load the JSON files
# with open('labels.json', 'r') as images_file:                      
#     images_data = json.load(images_file)

# with open('annotations_json_file.json', 'r') as annotations_file:
#     annotations_data = json.load(annotations_file)

# # Create a dictionary to map image IDs to their categories
# image_category_mapping = {}

# # Iterate through annotations data to populate the mapping
# for annotation in annotations_data:
#     image_id = annotation['image_id']
#     category_id = annotation['category_id']
   
#     # Check if the image_id is already in the mapping
#     if image_id in image_category_mapping:
#         # Add category_id to the list of categories for this image_id
#         image_category_mapping[image_id].append(category_id)
#     else:
#         # Initialize a new list of categories for this image_id
#         image_category_mapping[image_id] = [category_id]

# # Loop through images data and add categories based on the mapping
# for image in images_data['images']:
#     image_id = image['id']
   
#     # Check if the image_id is in the mapping
#     if image_id in image_category_mapping:
#         categories = image_category_mapping[image_id]
       
#         # Add categories to the image object
#         if 'categories' in image:
#             image['categories'].extend(categories)
#         else:
#             image['categories'] = categories

# # Write the modified images data back to file
# with open('modified_images_json_file.json', 'w') as modified_images_file:
#     json.dump(images_data, modified_images_file, indent=4)


import json

# Load the JSON file containing both images and annotations
with open('/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/waste_dataset/splits_final_deblurred/test/labels.json', 'r') as file:
   data = json.load(file)

# Create a dictionary to map image IDs to their categories
image_category_mapping = {}

# Iterate through annotations data to populate the mapping
for annotation in data['annotations']:
   image_id = annotation['image_id']
   category_id = annotation['category_id']
   
   # Check if the image_id is already in the mapping
   if image_id in image_category_mapping:
       # Add category_id to the list of categories for this image_id
       image_category_mapping[image_id].append(category_id)
   else:
       # Initialize a new list of categories for this image_id
       image_category_mapping[image_id] = [category_id]

# Loop through images data and add categories based on the mapping
for image in data['images']:
   image_id = image['id']
   
   # Check if the image_id is in the mapping
   if image_id in image_category_mapping:
       categories = image_category_mapping[image_id]
       
       # Add categories to the image object
       if 'categories' in image:
           image['categories'].extend(categories)
       else:
           image['categories'] = categories

# Write the modified data back to file
with open('modified_combined_json_test_file.json', 'w') as modified_file:
   json.dump(data, modified_file, indent=4)
