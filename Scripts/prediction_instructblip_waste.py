# # import torch
# # from PIL import Image
# # # setup device to use
# # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# # # load sample image
# # raw_image = Image.open("/home/numansaeed/Desktop/Ali_Bhai/LAVIS/docs/_static/Confusing-Pictures.jpg").convert("RGB")
# # # display(raw_image.resize((596, 437)))


# # from lavis.models import load_model_and_preprocess
# # # loads InstructBLIP model
# # model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
# # # prepare the image
# # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# # print(model.generate({"image": image, "prompt": "What is unusual about this image?"}))



# import torch
# from PIL import Image
# import json
# from tqdm import tqdm
# from lavis.models import load_model_and_preprocess

# # Setup device to use
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# # Load the dataset from the JSON file
# def load_dataset(json_file):
#     with open(json_file, "r") as f:
#         dataset = json.load(f)
#     return dataset

# # Perform inference for a single image and question
# def perform_inference(image, question, model, vis_processors):
#     # Prepare the image
#     image = vis_processors["eval"](image).unsqueeze(0).to(device)
    
#     # Perform inference
#     with torch.no_grad():
#         predictions = model.generate({"image": image, "prompt": question})
    
#     return predictions

# # Main function to perform inference on the dataset and save results
# def main(dataset_file, output_file):
#     # Load the dataset
#     dataset = load_dataset(dataset_file)
    
#     # Load the model
#     model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    
#     # Perform inference for each data item
#     results = []
#     for data in tqdm(dataset, desc="Processing"):
#         image_path = data["image_path"]
#         image = Image.open(image_path).convert("RGB")
#         responses = []
#         for qa_pair in data["responses"]:
#             question = qa_pair["Q"]
#             predictions = perform_inference(image, question, model, vis_processors)
#             responses.append({"Q": question, "A": predictions})
#         results.append({"image_path": image_path, "responses": responses})
    
#     # Save the results to a JSON file
#     with open(output_file, "w") as f:
#         json.dump(results, f)

# # Example usage
# if __name__ == "__main__":
#     dataset_file = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json"
#     output_file = "predictions_instblip.json"
#     main(dataset_file, output_file)

import torch
from PIL import Image
import json
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

# Setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load the dataset from the JSON file
def load_dataset(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)
    return dataset

# Perform inference for a single image and question
def perform_inference(image, question, model, vis_processors):
    # Prepare the image
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        predictions = model.generate({"image": image, "prompt":  question})
    
    return predictions

# Main function to perform inference on the dataset and save results
def main(dataset_file, output_file):
    # Load the dataset
    dataset = load_dataset(dataset_file)
    
    # Load the model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    
    # Perform inference for each data item
    results = []
    for data in tqdm(dataset, desc="Processing"):
        image_path = data["image_path"]
        image = Image.open(image_path).convert("RGB")
        for qa_pair in data["responses"]:
            question = qa_pair["Q"]
            ground_truth = qa_pair["A"]
            predicted = perform_inference(image, question, model, vis_processors)
            results.append({
                "image_path": image_path,
                "question": question,
                "predicted": predicted[0],  # Assuming predicted is a list, so take the first element
                "ground_truth": ground_truth
            })
    
    # Save the results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)  # Indent for better readability

# Example usage
if __name__ == "__main__":
    dataset_file = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json"
    output_file = "predictions_instblip-12AM.json"
    main(dataset_file, output_file)
