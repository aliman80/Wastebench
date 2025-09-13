# import os
# import ast
# import json
# from groq import Groq
# from tqdm import tqdm
# # Provide the path to your JSON file containing captions for your dataset
# json_file_path = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/caption_test_results.json"
# import time
# # Load your JSON file containing captions
# with open(json_file_path, "r") as file:
#    captions_data = json.load(file)

# client = Groq(
#    api_key=os.environ.get("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"),
# )

# # Define an empty list to store the response dictionaries for each image
# all_responses = []


# # Loop through each image caption in your dataset
# for index, caption_data in enumerate(tqdm(captions_data[:5])):
#    caption_content = caption_data["caption"]
#    image_details = caption_data['image_path']
#    while True:
#        # Generate questions and answers for the current image caption
#        try:
#            chat_completion =client.chat.completions.create(
#                messages=[
#                    {
#                        "role": "system",
#                        "content":
#                            "You are a helpful and intelligent AI assistant which can curate "
#                            "high-quality and challenging question and corresponding answers "
#                            "used to test the image understanding capabilities of an AI image system."
#                    },
#                    {
#                        "role": "user",
#                        "content":
#                            f"Given an image depicting waste materials in a cluttered environment, with the following detailed caption explaining the scene: "
#                            f"The caption is: {caption_content}. "
#                            "Formulate 10 diverse questions to test whether the model can correctly identify the objects and context based on the waste image provided. "
#                            "Additionally, these inquiries should assess the model's ability to accurately recognize and differentiate between different types of waste materials and the cluttered environment depicted in the image. "
#                            "Consider also asking misleading or wrong questions to test the AI system's understanding of the waste image dataset. "
#                            "Generate questions that comprise both interrogative and declarative sentences, utilizing different language styles, and provide an explanation for each. "
#                            "Your response should be presented as a list of dictionary strings with keys 'Q' for questions and 'A' for the answer. Do NOT Generate Any other text except Q and A. "
#                            "Follow these rules while generating question and answers: "
#                            "1. Avoid explicitly mentioning the type of assessment in the reasoning. "
#                            "2. Do not provide answers in the question itself. For example, the specific type of waste or the cluttered nature of the environment should never be mentioned in the question itself. "
#                            "3. Ensure the questions are concrete and can be entirely addressed using the provided caption. "
#                            "4. Do not formulate questions whose answer is not specified in the image and caption. "
#                            "For example, format your response as follows: "
#                            "[{\"Q\": 'Your first question here...', \"A\": 'Your first answer here...'}, "
#                            "{\"Q\": 'Your second question here...', \"A\": 'Your second answer here...'}, "
#                            "{\"Q\": 'Your third question here...', \"A\": 'Your third answer here...'}]."
#                    }
#                ],
#                model="llama3-8b-8192",
#                temperature = 0.2,
#            )
       
           
#            #response_message = chat_completion.choices[0].message.content



#            response_message = chat_completion.choices[0].message.content
#            try:
#            # Try to parse the response_message as a Python literal
#                response_dict_0 = ast.literal_eval(response_message)
#                all_responses.append({image_details:response_dict_0})
#                print({image_details:response_dict_0})
#                if index % 30 == 0:
#                    time.sleep(60)
#                break
#            except SyntaxError:
#        # If that fails, treat the response_message as a simple string
#                print("some error")
#                print(response_message)
#                continue
#            # response_dict_0 = response_message
               
#        except Exception as e:
#            print(e)
#            continue   
#    all_responses.append({"image_path": image_details, "responses": image_responses})
 

# # If response_dict_0 is a string, try to parse it as JSON
# # if isinstance(response_dict_0, str):
# #     try:
# #         response_dict_0 = json.loads(response_dict_0)
# #     except json.JSONDecodeError:
# #         print(f"Could not parse response_message: {response_message}")

# # At this point, response_dict_0 should be a list of dictionaries
# # You can append it to all_responses
   
# with open('question-answers_test-llava.json', 'w') as f:
#    json.dump(all_responses, f)



import os
import ast
import json
from groq import Groq
from tqdm import tqdm
import time

json_file_path = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/captions__gemini.json"
with open(json_file_path, "r") as file:
    captions_data = json.load(file)
print(captions_data[:2])
client = Groq(
    api_key="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",
)

all_responses = []

for index, caption_data in enumerate(tqdm(captions_data)):
    caption_content = caption_data["caption"]
    image_details = caption_data['image_path']
    image_responses = []

    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are a helpful and intelligent AI assistant which can curate "
                            "high-quality and challenging question and corresponding answers "
                            "used to test the image understanding capabilities of an AI image system."
                    },
                    {
                        "role": "user",
                        "content":
                            f"Given an image depicting waste materials in a cluttered environment, with the following detailed caption explaining the scene: "
                            f"{caption_content}."
                            "Formulate 10 diverse questions along with detailed answers to test whether the model can correctly recognize the objects and their context, their instances frequency, and other complex aspects based on the waste image provided."
                            "Additionally, these inquiries should assess the model's ability to accurately recognize and differentiate between different types of waste materials."
                            "Also consider misleading and wrong questions to test the AI system's understanding of the waste image dataset. "
                            "Generate questions that comprise both interrogative and declarative sentences, utilizing different language styles."
                            "Your response should be presented as a list of dictionary strings with keys 'Q' for questions and 'A' for the answer. "
                            "Follow these rules while generating questions and answers: "
                            "1. Avoid explicitly mentioning the type of assessment in the reasoning. "
                            "2. Do not provide answers in the question itself. For example, the specific type of waste or the cluttered nature of the environment should never be mentioned in the question itself. "
                            "3. Ensure the questions are concrete and can be entirely addressed using the provided caption. "
                            "4. DO NOT Generate Any other text except 10 Q and A!"
                            "For example, format your SINGLE LIST response as follows, with no other outside text: "
                            "[{\"Q\": 'Your first question here...', \"A\": 'Your first detailed answer here...'}, "
                            "{\"Q\": 'Your second question here...', \"A\": 'Your second detailed answer here...'}, "
                            "{\"Q\": 'Your third question here...', \"A\": 'Your third detailed answer here...'}]."
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.2,
            )

            response_message = chat_completion.choices[0].message.content
            print(response_message)
            response_dict_0 = ast.literal_eval(response_message)
            image_responses.extend(response_dict_0)
            if index !=0 and index % 30 == 0:
                time.sleep(60)
            break
        except SyntaxError:
            print("Syntax Error occurred. Response was:", response_message)
            continue
        except Exception as e:
            print("An error occurred:", e)
            continue

    all_responses.append({"image_path": image_details, "responses": image_responses})

# Save the output to a JSON file
with open('dummy-question-answers_test-llava.json', 'w') as f:
    json.dump(all_responses, f, indent=4)
