import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parse evaluation results.")
    parser.add_argument("--evaluation_json", required=True, help="Path to the evaluation JSON file.")
    args = parser.parse_args()
    return args

def parse_results(evaluation_json):
    with open(evaluation_json, "r") as file:
        data = json.load(file)

    # Initialize variables for computing accuracy
    total_count = len(data)
    correct_count = 0
    total_score = 0
    # Iterate through each entry in the JSON file
    for entry in data:
        evaluation = entry.get("evaluation")
        predicted = evaluation.get("predicted")
        score = evaluation.get("score")
        # Check if prediction is correct
        if predicted.lower() == "correct":
            correct_count += 1
        #total_score = total_score + score 
        total_score = total_score + (score if score is not None else 0)
    # Calculate accuracy
    accuracy =(  correct_count / total_count ) * 100
    total_score =( total_score / total_count  )
    print(f"total correct: {correct_count}")
    print(f"total QAs: {total_count}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Score: {total_score:.2f}")

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    parse_results(args.evaluation_json)

if __name__ == "__main__":
    main()
