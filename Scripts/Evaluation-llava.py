import os
import json
from tqdm import tqdm
from groq import Groq
import ast

# Load predictions data
json_file_path = "/home/numansaeed/Desktop/Ali_Bhai/MiniGPT-4/predictions.json"
with open(json_file_path, "r") as file:
   data = json.load(file)
   data = data[:200]  # Load a subset for testing

# Initialize Groq API client
client = Groq(
   api_key="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",
)

all_responses = []

for entry in tqdm(data):
   image_path = entry['image_path']
   question = entry['question']
   predicted = entry['predicted']
   ground_truth = entry['ground_truth']

   chat_response = client.chat.completions.create(
       
        messages=[
        {
        "role": "system",
        "content":
        "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. "
        "Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the correctness and accuracy of the predicted answer with the ground-truth.\n"
        "- Consider predictions with less specific details as correct evaluation, unless such details are explicitly asked in the question.\n"
        },
        {
        "role": "user",
        "content":
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Ground truth correct Answer: {ground_truth}\n"
        f"Predicted Answer: {predicted}\n\n"
        "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
        "Please generate the response in the form of a Python dictionary string with keys 'predicted', 'score' and 'reason', where value of 'predicted' is a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should providethe reason behind the decision."
        "Only provide the Python dictionary string."
        "For example, your response should look like this: {'predicted': 'correct', 'score': 4.8, 'reason': reason}."
        }
        ],

       model="llama3-8b-8192",
       temperature=0.2,
   )
   
   # Access the `choices` attribute
   response_message = chat_response.choices[0].message.content

   # Clean up the response if needed
   print(response_message)  # Debug: see the response
   
   # Remove extra symbols and evaluate the response
   response_message_cleaned = response_message.strip("[]{}")  # Adjust as needed

   all_responses.append({
       'image_path': image_path,
       'question': question,
       'ground_truth': ground_truth,
       'predicted': predicted,
       'evaluation': response_message_cleaned
   })

# Write the output to a file
with open('Evaluations-gpt-gpt.json', 'w') as f:
   json.dump(all_responses, f, indent=4)

print("Processing complete.")