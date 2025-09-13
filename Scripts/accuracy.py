# import json
# from fuzzywuzzy import fuzz

# def calculate_accuracy(predictions, score_threshold=2.5, similarity_threshold=40):
#     correct_count = 0
#     total_count = len(predictions)

#     for prediction in predictions:
#         if 'ground_truth' in prediction and 'predicted' in prediction:
#             ground_truth = prediction['ground_truth']
#             predicted = prediction['predicted']
#             score = prediction.get('score', 0)

#             if predicted == ground_truth or score >= score_threshold:
#                 correct_count += 1
#             elif fuzz.partial_ratio(ground_truth, predicted) >= similarity_threshold:
#                 correct_count += 1

#     accuracy = (correct_count / total_count) * 100
#     return accuracy

# def main(json_file):
#     with open(json_file, 'r') as f:
#         predictions = json.load(f)

#     accuracy = calculate_accuracy(predictions)
#     print(f"Accuracy: {accuracy}%")

# # Replace 'predictions.json' with the path to your JSON file
# json_file_path = '/home/numansaeed/Desktop/Ali_Bhai/LLaVA/Evaluations-llm.json'
# main(json_file_path)


import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--input_json", required=True, help="The path to the input JSON file.")
    args = parser.parse_args()
    return args

def parse_results(input_json):
    with open(input_json, "r") as file:
        contents = json.load(file)

    # Initialize variables for computing average score and accuracy
    total_score = 0
    total_count = len(contents)
    correct_count = 0

    # Iterate through each entry in the JSON file
    for entry in contents:
        ground_truth = entry.get("ground_truth")
        predicted = entry.get("predicted")

        # Calculate score
        score = entry.get("evaluation", {}).get("score", 0)
        total_score += score

        # Check if prediction is correct
        if predicted.lower() == ground_truth.lower():
            correct_count += 1

    # Calculate average score and accuracy
    average_score = total_score / total_count
    accuracy = correct_count / total_count

    print(f"Average score: {average_score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    parse_results(args.input_json)

if __name__ == "__main__":
    main()
