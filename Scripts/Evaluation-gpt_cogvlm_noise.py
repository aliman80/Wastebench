import os
import json
from tqdm import tqdm
# from groq import Groq
import ast
import openai 
import time

# Load predictions data
json_file_path = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/predictions_cogvlm_noise.json"
with open(json_file_path, "r") as file:
   data = json.load(file)
   # data = data[:20]  # Load a subset for testing

# Initialize Groq API client
openai.api_key = 'sk-proj-2eLxnWqvBqEMGyhtNO4HT3BlbkFJ20OyzFhsAwJmbfdg1W6E'

all_responses = []

for entry in tqdm(data):
   image_path = entry['image_path']
   question = entry['question']
   predicted = entry['predicted']
   ground_truth = entry['ground_truth']
   while True:

      completion = openai.ChatCompletion.create(
                     model="gpt-3.5-turbo",
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
         "Please evaluate the following question-answer pair:\n\n"
         f"Question: {question}\n"
         f"Ground truth correct Answer: {ground_truth}\n"
         f"Predicted Answer: {predicted}\n\n"
         "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
         "Please generate the response in the form of a Python dictionary string with keys 'predicted', 'score' and 'reason', where value of 'predicted' is a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should providethe reason behind the decision."
         "Only provide the Python dictionary string."
         "For example, your response should look like this: {'predicted': 'correct', 'score': 4.8, 'reason': reason}."
         }
         ] )
         # Convert response to a Python dictionary.
      response_message = completion["choices"][0]["message"]["content"]
         
      try:
         response_dict = ast.literal_eval(response_message)
               # Clean up the response if needed
         print(response_message)  # Debug: see the response
         break
      except:
         # Remove the special characters.
         try:
            start_index = response_message.find("'reason': '") + len("'reason': '")
            end_index = response_message.find("'", start_index)

            # Extract the reason value
            reason_value = response_message[start_index:end_index]

            # Remove single quotes from the reason value
            reason_value = reason_value.replace("'", "")

            # Replace the original reason value with the modified one
            response_message = response_message[:start_index] + reason_value + '\'}'
            response_dict = ast.literal_eval(response_message)
            break
         except Exception as e:
            print(f"Error: {e}")
            print("retrying")
           
   
         # Remove extra symbols and evaluate the response
      # response_message_cleaned = response_message.strip("[]{}")  # Adjust as needed

   all_responses.append({
      'image_path': image_path,
      'question': question,
      'ground_truth': ground_truth,
      'predicted': predicted,
      'evaluation': response_dict
   })
         #       # Clean up the response if needed
         # print(response_message)  # Debug: see the response
         # break


# Write the output to a file
with open('Final-noisy-Evaluations-gpt-cogvlm.json', 'w') as f:
   json.dump(all_responses, f, indent=4)

print("Processing complete.")