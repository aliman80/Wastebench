# # # from argparse import parse_args
# # import time
# # import os
# # import json
# # import argparse
# # from tqdm import tqdm
# # import vertexai
# # from vertexai.preview.generative_models import GenerativeModel, Part
# # import vertexai.preview.generative_models as generative_models
# # import pdb

# # def parse_args():
# #     parser = argparse.ArgumentParser(description=" question-answer-generation-using-gpt-3")
# #     parser.add_argument("--cvrr_dataset_path", required=True, help="The path to file containing prediction.")
# #     parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
# #     parser.add_argument("--google_cloud_bucket_name", required=True, help="Bucket name. For Gemini, CVRR-ES dataset needs to be also uploaded to google cloud bucket.")
# #     parser.add_argument("--google_cloud_project_name", required=True, help="Bucket name. For Gemini, please provide the google cloud project name.")
# #     args = parser.parse_args()
# #     return args

# # def evaluate_single_video_dimension(cvrr_dataset_path, output_dir, gcp_cloud_bucket_name,
# #                                     gcp_project_name):
# #     # Parse arguments.
# #     # pdb.set_trace()

# #     vertexai.init(project=gcp_project_name, location="us-central1")
# #     model = GenerativeModel("gemini-1.0-pro-vision-001")
# #     # # Skip this if the json file already exists
# #     # json_file_path = os.path.join(output_dir, single_folder + '.json')
# #     # if os.path.exists(json_file_path):
# #     #     return
# #     # print(f"Generating Gemini-Pro-Vision predictions on CVRR-ES benchmark for dimension: {single_folder}")
# #     # annotation_path = os.path.join(cvrr_dataset_path,
# #     #                                single_folder + "/" + "annotations_" + single_folder + ".json")
# #     # annotation_path = os.path.join(cvrr_dataset_path,
# #     #                                "/" + "question-answers_test-llava" + ".json")
# #     input_json = cvrr_dataset_path
# #     with open(input_json, "r") as file:
# #         data = json.load(file)
# #         print(data[:2])
# #     # iterate over each question
# #     all_responses = []
# #     for entry in tqdm(data):
        
# #         image_path = entry['image_path'].split('waste_dataset/')[1]
# #         # image_path = entry['image_path'].split('noise/')[1]

        
# #         responses = entry['responses']
# #         # print(responses)
# #         my_path = f"gs://{gcp_cloud_bucket_name}/" + "waste_dataset/" + image_path
# #         video_part = Part.from_uri(
# #                 my_path, mime_type="image/jpeg"
# #             )
# #         print(my_path)
# #         for response in responses:
# #             ################################
            
# #             ####################3
# #             question = response['Q']
# #             ground_truth = response['A']
# #             inp = f"For the given image, provide an answer for the following question: {question}"

# #             # Load the video
# #             # The dataset must be additionally uploaded to google cloud bucket
# #             message = False
# #             while True:
# #                 try:
# #                     result = model.generate_content(
# #                         [video_part, inp,],
# #                         generation_config={
# #                             "max_output_tokens": 2048,
# #                             "temperature": 1,
# #                             "top_p": 1,
# #                             "top_k": 32
# #                         },
# #                         safety_settings={
# #                             generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
# #                             generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
# #                             generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
# #                             generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
# #                         },
# #                     )
# #                     predicted = result.text
# #                     print(result.text)
# #                     break
# #                 except Exception as e:
# #                     print(f"Error: {e}, sleeping for 60 sec")
# #                     # time.sleep(60)
                    
# #                     try:
# #                         if 'prompt_feedback' in list(result.to_dict()):
# #                             message = result.to_dict()['prompt_feedback']['block_reason'] >= 0
# #                             if message:
# #                                 print(result.to_dict()['prompt_feedback'])
# #                                 print(f"gemini has blocked qa for video {my_path}")
# #                                 break
# #                         if 'candidates' in list(result.to_dict()):
# #                                 print(result.to_dict()['candidates'])
# #                                 print(f"gemini has blocked qa for video {my_path}")
# #                                 break
# #                     except Exception as e:
# #                         print(e)
# #                         print("Retrying for the same prompt")

# #             if message:
# #                 # Means Gemini is not allowing us to get responses for this video/question
# #                 all_responses.append({
# #                 'image_path': image_path,
# #                 'question': question,
# #                 'predicted': "",
# #                 'ground_truth': ground_truth
# #                 })
# #             else:
# #                 all_responses.append({
# #                 'image_path': image_path,
# #                 'question': question,
# #                 'predicted': predicted,
# #                 'ground_truth': ground_truth
# #                 })
# #                 # model_response.append({"Q": user_question, "A": answer})

# #     # Save the results to the specified JSON file
# #     with open(output_dir, 'w') as f:
# #         json.dump(all_responses, f, indent=4)

# # def main():
# #     """
# #     Main function to control the flow of the program.
# #     """
# #     args = parse_args()
# #     # all_folder_names = os.listdir(args.cvrr_dataset_path)
# #     # output_dir = args.output_dir
# #     # # Create output directory if not exists.
# #     # if not os.path.exists(output_dir):
# #     #     os.makedirs(output_dir)
    
# #     # for single_folder in all_folder_names:
# #     evaluate_single_video_dimension(args.cvrr_dataset_path, args.output_dir,
# #                                         args.google_cloud_bucket_name, args.google_cloud_project_name)
# #     print("Manual_data_Inference with Gemini-Pro-Vision model Completed!")
    
    
    
# # if __name__ == "__main__":
# #     main()




# import time
# import os
# import json
# import argparse
# from tqdm import tqdm
# import vertexai
# from vertexai.preview.generative_models import GenerativeModel, Part
# import vertexai.preview.generative_models as generative_models

# def parse_args():
#     parser = argparse.ArgumentParser(description="Question-answer generation using Gemini-Pro")
#     parser.add_argument("--cvrr_dataset_path", required=True, help="The path to file containing prediction.")
#     parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
#     parser.add_argument("--google_cloud_bucket_name", required=True, help="Bucket name. For Gemini, CVRR-ES dataset needs to be also uploaded to google cloud bucket.")
#     parser.add_argument("--google_cloud_project_name", required=True, help="Please provide the Google Cloud project name.")
#     return parser.parse_args()

# def evaluate_single_video_dimension(cvrr_dataset_path, output_dir, gcp_cloud_bucket_name, gcp_project_name):
#     vertexai.init(project=gcp_project_name, location="us-central1")
#     model = GenerativeModel("gemini-1.0-pro-vision-001")

#     with open(cvrr_dataset_path, 'r') as file:
#         data = json.load(file)

#     image_dir = 'splits_final_deblurred/test/data/'

#     results = []
   
#     for item in tqdm(data['images']):
#         image_path = os.path.join(image_dir, item['file_name'])
#         category_ids = item.get('categories', [])

#         if category_ids:
#             category_names = ', '.join([cat['name'] for cat in data['categories'] if cat['id'] in category_ids])
#         else:
#             category_names = 'cardboard, paper, other'

#         my_path = f"gs://{gcp_cloud_bucket_name}/waste_dataset/{image_path}"
#         print(my_path)
#         video_part = Part.from_uri(my_path, mime_type="image/jpeg")

#         inp = f'''You are a smart agent. Your goal is to answer a question based on an image. You need to think step by step. First analyze the image to understand its contents, including waste objects types, shapes, condition as well as environment shown in the image. Then, check for consistency between the question and the image. Consider the following aspects:
#         - Visible Elements: Identify all the visible elements and features in the image and compare them to the details mentioned in the question.
#         - Count and Quantity: Verify the number of items (like the number of cardboards) in the image.
#         - Descriptive Accuracy: Compare descriptive terms in the question (such as colors, shapes, or sizes) with the attributes of objects and settings in the image.
#         - Context and Setting: Assess whether the context or setting described in the question matches the environment depicted in the image.
#         - Possible Ambiguities: Consider whether parts of the image are ambiguous or unclear and may lead to different interpretations.
#         - Logical Consistency: Evaluate if the question makes logical sense given the scenario in the image.
#         By taking these aspects into account, ensure the response is as accurate and relevant to the image as possible. If there's any inconsistency or ambiguity, clarify or rectify it in the response.
#         Here is the question you need to answer: This image depicts various waste materials. Additional information and context for the image's main content are provided, such as {category_names}. Your task is to provide a detailed caption of the image, describing the types of waste present (e.g., cardboard, soft plastic, hard plastic, metal) and their conditions (e.g., torn, crumpled).'''

#         message = False
#         while True:
#             try:
#                 result = model.generate_content(
#                     [video_part, inp],
#                     generation_config={
#                         "max_output_tokens": 2048,
#                         "temperature": 1,
#                         "top_p": 1,
#                         "top_k": 32
#                     },
#                     safety_settings={
#                         generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                         generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                         generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                         generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                     },
#                 )
#                 predicted = result.text
#                 print(result.text)
#                 break
#             except Exception as e:
#                 print(f"Error: {e}, sleeping for 60 sec")
#                 time.sleep(60)

#         if message:
#             # Means Gemini is not allowing us to get responses for this image/question
#             results.append({
#                 'image_path': image_path,
#                 'question': inp,
#                 'caption': result.text,
#             })
#         else:
#             results.append({
#                 'image_path': image_path,
#                 'question': inp,
#                 'caption': result.text,
#             })

#     # Save the results to the specified JSON file
#     with open(output_dir, 'w') as f:
#         json.dump(results, f, indent=4)

# def main():
#     args = parse_args()
#     evaluate_single_video_dimension(args.cvrr_dataset_path, args.output_dir,
#                                     args.google_cloud_bucket_name, args.google_cloud_project_name)
#     print("Captions with Gemini-Pro-Vision model Completed!")

# if __name__ == "__main__":
#     main()

# # from argparse import parse_args
# import time
# import os
# import json
# import argparse
# from tqdm import tqdm
# import vertexai
# from vertexai.preview.generative_models import GenerativeModel, Part
# import vertexai.preview.generative_models as generative_models
# import pdb

# def parse_args():
#     parser = argparse.ArgumentParser(description=" question-answer-generation-using-gpt-3")
#     parser.add_argument("--cvrr_dataset_path", required=True, help="The path to file containing prediction.")
#     parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
#     parser.add_argument("--google_cloud_bucket_name", required=True, help="Bucket name. For Gemini, CVRR-ES dataset needs to be also uploaded to google cloud bucket.")
#     parser.add_argument("--google_cloud_project_name", required=True, help="Bucket name. For Gemini, please provide the google cloud project name.")
#     args = parser.parse_args()
#     return args

# def evaluate_single_video_dimension(cvrr_dataset_path, output_dir, gcp_cloud_bucket_name,
#                                     gcp_project_name):
#     # Parse arguments.
#     # pdb.set_trace()

#     vertexai.init(project=gcp_project_name, location="us-central1")
#     model = GenerativeModel("gemini-1.0-pro-vision-001")
#     # # Skip this if the json file already exists
#     # json_file_path = os.path.join(output_dir, single_folder + '.json')
#     # if os.path.exists(json_file_path):
#     #     return
#     # print(f"Generating Gemini-Pro-Vision predictions on CVRR-ES benchmark for dimension: {single_folder}")
#     # annotation_path = os.path.join(cvrr_dataset_path,
#     #                                single_folder + "/" + "annotations_" + single_folder + ".json")
#     # annotation_path = os.path.join(cvrr_dataset_path,
#     #                                "/" + "question-answers_test-llava" + ".json")
#     input_json = cvrr_dataset_path
#     with open(input_json, "r") as file:
#         data = json.load(file)
#         print(data[:2])
#     # iterate over each question
#     all_responses = []
#     for entry in tqdm(data):
      
#         image_path = entry['image_path'].split('waste_dataset/')[1]
#         # image_path = entry['image_path'].split('noise/')[1]

      
#         responses = entry['responses']
#         # print(responses)
#         my_path = f"gs://{gcp_cloud_bucket_name}/" + "waste_dataset/" + image_path
#         video_part = Part.from_uri(
#                 my_path, mime_type="image/jpeg"
#             )
#         print(my_path)
#         for response in responses:
#             ################################
          
#             ####################3
#             question = response['Q']
#             ground_truth = response['A']
#             inp = f"For the given image, provide an answer for the following question: {question}"

#             # Load the video
#             # The dataset must be additionally uploaded to google cloud bucket
#             message = False
#             while True:
#                 try:
#                     result = model.generate_content(
#                         [video_part, inp,],
#                         generation_config={
#                             "max_output_tokens": 2048,
#                             "temperature": 1,
#                             "top_p": 1,
#                             "top_k": 32
#                         },
#                         safety_settings={
#                             generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                             generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                             generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                             generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                         },
#                     )
#                     predicted = result.text
#                     print(result.text)
#                     break
#                 except Exception as e:
#                     print(f"Error: {e}, sleeping for 60 sec")
#                     # time.sleep(60)
                  
#                     try:
#                         if 'prompt_feedback' in list(result.to_dict()):
#                             message = result.to_dict()['prompt_feedback']['block_reason'] >= 0
#                             if message:
#                                 print(result.to_dict()['prompt_feedback'])
#                                 print(f"gemini has blocked qa for video {my_path}")
#                                 break
#                         if 'candidates' in list(result.to_dict()):
#                                 print(result.to_dict()['candidates'])
#                                 print(f"gemini has blocked qa for video {my_path}")
#                                 break
#                     except Exception as e:
#                         print(e)
#                         print("Retrying for the same prompt")

#             if message:
#                 # Means Gemini is not allowing us to get responses for this video/question
#                 all_responses.append({
#                 'image_path': image_path,
#                 'question': question,
#                 'predicted': "",
#                 'ground_truth': ground_truth
#                 })
#             else:
#                 all_responses.append({
#                 'image_path': image_path,
#                 'question': question,
#                 'predicted': predicted,
#                 'ground_truth': ground_truth
#                 })
#                 # model_response.append({"Q": user_question, "A": answer})

#     # Save the results to the specified JSON file
#     with open(output_dir, 'w') as f:
#         json.dump(all_responses, f, indent=4)

# def main():
#     """
#     Main function to control the flow of the program.
#     """
#     args = parse_args()
#     # all_folder_names = os.listdir(args.cvrr_dataset_path)
#     # output_dir = args.output_dir
#     # # Create output directory if not exists.
#     # if not os.path.exists(output_dir):
#     #     os.makedirs(output_dir)
  
#     # for single_folder in all_folder_names:
#     evaluate_single_video_dimension(args.cvrr_dataset_path, args.output_dir,
#                                         args.google_cloud_bucket_name, args.google_cloud_project_name)
#     print("Manual_data_Inference with Gemini-Pro-Vision model Completed!")
  
  
  
# if __name__ == "__main__":
#     main()




import time
import os
import json
import argparse
from tqdm import tqdm
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

def parse_args():
  parser = argparse.ArgumentParser(description="Question-answer generation using Gemini-Pro")
  parser.add_argument("--cvrr_dataset_path", required=True, help="The path to file containing prediction.")
  parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
  parser.add_argument("--google_cloud_bucket_name", required=True, help="Bucket name. For Gemini, CVRR-ES dataset needs to be also uploaded to google cloud bucket.")
  parser.add_argument("--google_cloud_project_name", required=True, help="Please provide the Google Cloud project name.")
  return parser.parse_args()

def evaluate_single_video_dimension(cvrr_dataset_path, output_dir, gcp_cloud_bucket_name, gcp_project_name):
  vertexai.init(project=gcp_project_name, location="us-central1")
  model = GenerativeModel("gemini-1.0-pro-vision-001")
  json_file_path = '/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/modified_categories_json_test_file.json'
  with open(json_file_path, 'r') as file:
      data = json.load(file)

  image_dir = 'splits_final_deblurred/test/data/'

  results = []
  
  ###############
  
# empty dict with zero frequency

  ###########333
 
  for item in tqdm(data['images']):
      image_path = os.path.join(image_dir, item['file_name'])
      category_ids = item.get('categories', [])

      if category_ids:
           my_dict = {1:0, 2: 0, 3:0, 4:0}
           for id in category_ids:
               temp = my_dict[id]
               temp = temp + 1
               my_dict[id] = temp
           my_string = "As the additional context, following ground-truth waste items are present in the image: "
           for single_key in my_dict.keys():
               if my_dict[single_key] != 0:
                   my_string += str(my_dict[single_key]) + " " + data['categories'][single_key-1]['name'] + ", "
           #category_names = ', '.join(data['categories'][id-1]['name'] for id in category_ids )## to select particular category  %describing the types of waste present and their physical condition/appearence (e.g., torn, crumpled).    #   Additionally the caption should describe its contents, including waste objects types, shapes, condition as well as environment shown in the image. 
        # Your task is to provide a detailed caption of the image
      else:
          continue 
      print(my_string)
      my_path = f"gs://{gcp_cloud_bucket_name}/waste_dataset/{image_path}"
      print(my_path)
      video_part = Part.from_uri(my_path, mime_type="image/jpeg")

      inp = f'''You are a smart image understanding agent for image captioning. The given image depicts various waste materials.
      {my_string}. Describe their physical appearence, and overall scene enviornment. Make sure to provide information only for the items given in the context. Now proceed with providing the detailed caption using the given context.'''

      message = False
      while True:
          try:
              result = model.generate_content(
                  [video_part, inp],
                  generation_config={
                      "max_output_tokens": 2048,
                      "temperature": 1,
                      "top_p": 1,
                      "top_k": 32
                  },
                  safety_settings={
                      generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                      generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                      generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                      generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                  },
              )
              predicted = result.text
              print(result[0])
             
              break
          except Exception as e:
              print(f"Error: {e}, sleeping for 60 sec")
              time.sleep(1)
              try:
                   if 'prompt_feedback' in list(result.to_dict()):
                       message = result.to_dict()['prompt_feedback']['block_reason'] >= 0
                       if message:
                           print(result.to_dict()['prompt_feedback'])
                           print(f"gemini has blocked qa for video {my_path}")
                           break
                   if 'candidates' in list(result.to_dict()):
                           print(result.to_dict()['candidates'])
                           print(f"gemini has blocked qa for video {my_path}")
                           break
              except Exception as e:
                   print(e)
                   print("Retrying for the same prompt")  
      if message:
          # Means Gemini is not allowing us to get responses for this image/question
          results.append({
              'image_path': image_path,
              'question': inp,
              'caption': result.text,
          })
      else:
          results.append({
              'image_path': image_path,
              'question': inp,
              'caption': result.text,
          })
      print(results.text)
  # Save the results to the specified JSON file
  with open(output_dir, 'w') as f:
      json.dump(results, f, indent=4)

def main():
  args = parse_args()
  evaluate_single_video_dimension(args.cvrr_dataset_path, args.output_dir,
                                  args.google_cloud_bucket_name, args.google_cloud_project_name)
  print("Captions with Gemini-Pro-Vision model Completed!")

if __name__ == "__main__":
  main()

