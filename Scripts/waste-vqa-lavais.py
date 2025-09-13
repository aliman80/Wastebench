import torch
from torchvision import transforms
from PIL import Image
import json
from transformers import InstructBlipTokenizer
from transformers import (
    InstructBlipVisionConfig,
    InstructBlipQFormerConfig,
    OPTConfig,
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
)

# Load the pre-trained InstruBlip model and tokenizer
model_name = "Salesforce/instrublip-base"
model = InstructBlipForConditionalGeneration.from_pretrained(model_name)

tokenizer = InstructBlipTokenizer.from_pretrained(model_name)

# Define transforms to preprocess the images
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a function for performing inference on a single image and question
def perform_inference(image_path, question):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)  # Add batch dimension

    # Tokenize the question
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = model.generate(
            input_images=image,
            input_captions=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=20,  # Adjust as needed
            num_beams=4,    # Adjust as needed
            temperature=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True,
        )

    # Decode the predicted answer
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_answer

# Load the dataset from a JSON file
def load_dataset(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)
    return dataset

# Perform inference for each image and question pair in the dataset
def perform_inference_on_dataset(dataset):
    results = []
    for data in dataset:
        image_path = data["image_path"]
        question = data["question"]
        predicted_answer = perform_inference(image_path, question)
        results.append({"image_path": image_path, "question": question, "predicted_answer": predicted_answer})
    return results

# Main function to load dataset, perform inference, and save results
def main(json_file, output_file):
    dataset = load_dataset(json_file)
    results = perform_inference_on_dataset(dataset)
    with open(output_file, "w") as f:
        json.dump(results, f)
    # print(f"Results saved to {output_file}")
    print("Results saved to {}".format(output_file))

# # Example usage
# if __name__ == "__main__":
#     json_file = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/caption_test_results.json"
#     output_file = "question-answer-blip-results.json"
# #     main(json_file, output_file)
# import torch
# from PIL import Image
# from torchvision import transforms
# from transformers import ViTForImageClassification, ViTFeatureExtractor

# # Load the pre-trained ViT model and feature extractor
# model_name = "google/vit-base-patch16-224-in21k"
# model = ViTForImageClassification.from_pretrained(model_name)
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# # Define transforms to preprocess the images
# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Ensure the image size matches the model input size
#     transforms.ToTensor(),
# ])

# # Define a function for performing inference on a single image and question
# def perform_inference(image_path):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert("RGB")
#     image = image_transform(image)

#     # Extract image features
#     inputs = feature_extractor(images=image, return_tensors="pt")
    
#     # Perform inference
#     with torch.no_grad():
#         logits = model(**inputs).logits

#     # Decode the predicted class
#     predicted_class = torch.argmax(logits, dim=-1).item()

#     return predicted_class

# # Example usage
# if __name__ == "__main__":
#     image_path = "your_image.jpg"  # Replace with the path to your image
#     predicted_class = perform_inference(image_path)
#     print("Predicted class:", predicted_class)




# import torch
# from PIL import Image
# from torchvision import transforms
# import json

# # Load the pre-trained ViT model
# model_name = "google/vit-base-patch16-224-in21k"
# model = ViTForImageClassification.from_pretrained(model_name)
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# # Define transforms to preprocess the images
# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Ensure the image size matches the model input size
#     transforms.ToTensor(),
# ])

# # Define a function for performing inference on a single image with multiple questions
# def perform_inference(image_path, questions):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert("RGB")
#     image = image_transform(image)

#     # Extract image features
#     inputs = feature_extractor(images=image, return_tensors="pt")

#     # Perform inference for each question
#     answers = []
#     for question_data in questions:
#         question = question_data["question"]
#         question_id = question_data["question_id"]  # Assuming there's a question_id field
#         # Perform inference
#         with torch.no_grad():
#             logits = model(**inputs).logits

#         # Decode the predicted class
#         predicted_class = torch.argmax(logits, dim=-1).item()
#         answers.append({"question_id": question_id, "question": question, "predicted_answer": predicted_class})

#     return answers

# # Define a function to load questions from a JSON file
# def load_questions(json_file):
#     with open(json_file, "r") as f:
#         data = json.load(f)
#     return data

# # Main function to perform inference on the dataset and save results
# def main(dataset_json, output_json):
#     dataset = load_questions(dataset_json)
#     results = []

#     # Perform inference for each image and its associated questions
#     for data in dataset:
#         image_path = data["image_path"]
#         questions = data["question"]
#         answers = perform_inference(image_path, questions)
#         results.append({"image_path": image_path, "answers": answers})

#     # Save the results to a JSON file
#     with open(output_json, "w") as f:
#         json.dump(results, f)

# # Example usage
# if __name__ == "__main__":
#     dataset_json = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/caption_test_results.json"  # Replace with the path to your JSON file
#     output_json = "inference_results.json"  # Specify the path for the output JSON file
#     main(dataset_json, output_json)
