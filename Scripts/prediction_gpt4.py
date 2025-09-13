import json
import os
import base64
import io
from PIL import Image
import argparse
from tqdm import tqdm
# from openai import OpenAI
import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Question-Answer Generation Using GPT-4 for Images")
    parser.add_argument("--input_json", default ="/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json", required=True, help="Path to the input JSON file containing image paths and questions.")
    parser.add_argument("--output_dir",default = "/home/numansaeed/Desktop/Ali_Bhai/MiniGPT-4",required=True, help="The path to save prediction JSON files.")
    parser.add_argument("--api_key", default="sk-proj-2eLxnWqvBqEMGyhtNO4HT3BlbkFJ20OyzFhsAwJmffbfdg1W6E", required=True, help="OpenAI API key")
    args = parser.parse_args()
    return args

# def process_images(input_json, output_dir, client):
#     # Load JSON data
#     with open(input_json, 'r') as file:
#         data = json.load(file)

#     # Prepare output file path
#     output_file_path = os.path.join(output_dir, 'predictions.json')
#     if os.path.exists(output_file_path):
#         print("Output file already exists. Exiting to avoid overwriting.")
#         return

#     # Process each entry in the JSON file
#     predictions = []
#     for index, entry in enumerate(tqdm(data[:2])):
#         image_path = entry['image_path']
#         responses = entry['responses']
        
#         # Load and convert image to base64
#         image = Image.open(image_path).convert('RGB')
#         buffered = io.BytesIO()
#         image.save(buffered, format="JPEG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        

#         # Process each question for the image
#         for response in responses:
#             question = response['Q']
#             prompt = {
#                 "role": "user",
#                 "content": f"Question: {question}",
#                 **map(lambda x: {"image": x, "resize": 768}, [img_base64]),                    
#             }
#             params = {
#                 "model": "gpt-4-turbo", #"gpt-4-vision-preview",
#                 "messages": [prompt],
#                 "max_tokens": 200
#             }
            
#             try:
#                 result = client.chat.completions.create(**params)
#                 print(result)
#                 predicted_answer = result.choices[0].message.content   
#                 predictions.append({
#                     "image_path": image_path,
#                     "question": question,
#                     "predicted_answer": predicted_answer
#                 })
#             except Exception as e:
#                 print(f"Error while processing {image_path}: {e}")

#     # Save responses to JSON file
#     with open(output_file_path, "w") as f:
#         json.dump(predictions, f, indent=4)

#     print("Inference completed and saved to", output_file_path)

# def main():
#     args = parse_args()
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
    
#     client = OpenAI(api_key=args.api_key)
#     process_images(args.input_json, args.output_dir, client)

# if __name__ == "__main__":
#     main()



def process_images(input_json, output_dir, client):
    # Load JSON data
    with open(input_json, 'r') as file:
        data = json.load(file)

    # Prepare output file path
    output_file_path = os.path.join(output_dir, 'test-predictions.json')
    if os.path.exists(output_file_path):
        print("Output file already exists. Exiting to avoid overwriting.")
        return

    # Process each entry in the JSON file
    predictions = []
    for index, entry in enumerate(tqdm(data)):
        image_path = entry['image_path']
        
        responses = entry['responses']
        
        # Load and convert image to base64
        image = Image.open(image_path).convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Process each question for the image
        for response in responses:
            question = response['Q']
            ground_truth = response['A']
            prompt = {
                "role": "user",
                "content": f"Question: {question}",
                "image": img_base64,
                "resize": 768
            }
            params = {
                "model": "gpt-3.5-turbo",
                "messages": [prompt],
                "max_tokens": 200
            }
            
            try:
                result = client.chat.completions.create(**params)
                print(result)
                predicted_answer = result.choices[0].message.content   
                predictions.append({
                    "image_path": image_path,
                    "question": question,
                    "predicted": predicted_answer,
                    "ground_truth": ground_truth
                })
            except Exception as e:
                print(f"Error while processing {image_path}: {e}")

    # Save responses to JSON file
    with open(output_file_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print("Inference completed and saved to", output_file_path)

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    client = OpenAI(api_key=args.api_key)
    process_images(args.input_json, args.output_dir, client)

if __name__ == "__main__":
    main()