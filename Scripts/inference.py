# import torch
# import json
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from torchvision import transforms

# # Define configuration
# MODEL_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/"
# TOKENIZER_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d" 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# QUANTIZATION = False  # Set to True if quantization is desired, otherwise False

# # Initialize tokenizer
# tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

# # Initialize model
# if QUANTIZATION:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         load_in_4bit=True,
#         trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     ).to(DEVICE).eval()

# # Define image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize image to fit model input size
#     transforms.ToTensor(),           # Convert image to tensor
# ])

# # Load input JSON file
# with open('/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json', 'r') as f:
#     data = json.load(f)

# # Define template
# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# # Process each entry in the input JSON file
# output_data = []
# for entry in data[:2]:
#     image_path = entry['image_path']
#     questions_answers = entry['responses']

#     # Load and preprocess image
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

#     # Process each question-answer pair
#     for qa_pair in questions_answers:
#         question = qa_pair['Q']
#         ground_truth = qa_pair['A']

#         # Preprocess input using tokenizer
#         inputs = tokenizer(question, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True)

#         # Move inputs to device
#         inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

#         # Generate response
#       # Generate response
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predicted_answers = []
#         for logits in outputs.logits:
#             predicted_answer = tokenizer.decode(logits.argmax(dim=-1))
#             predicted_answers.append(predicted_answer)

#     # Store information in the same dictionary
#     response = "your_response_here"  # Replace "your_response_here" with the actual response value
#     output_data.append({
#         'image_path': image_path,
#         'question': question,
#         'ground_truth': ground_truth,
#         'predicted_answers': predicted_answers,
#         'response_with_template': response
#     })


# # # Write output data to JSON file
# # with open('output_file.json', 'w') as outfile:
# #     json.dump(output_data, outfile, indent=4)

# import torch
# import json
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from torchvision import transforms

# # Define configuration
# MODEL_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/"
# TOKENIZER_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d" 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# QUANTIZATION = False  # Set to True if quantization is desired, otherwise False

# # Initialize tokenizer
# tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

# # Initialize model
# if QUANTIZATION:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         load_in_4bit=True,
#         trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     ).to(DEVICE).eval()

# # Define image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize image to fit model input size
#     transforms.ToTensor(),           # Convert image to tensor
# ])

# # Load input JSON file
# with open('/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json', 'r') as f:
#     data = json.load(f)

# # Define template
# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# # Process each entry in the input JSON file
# output_data = []
# for entry in data[:2]:
#     image_path = entry['image_path']
#     questions_answers = entry['responses']

#     # Load and preprocess image
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

#     # Process each question-answer pair
#     for qa_pair in questions_answers:
#         question = qa_pair['Q']
#         ground_truth = qa_pair['A']

#         # Preprocess input using tokenizer
#         inputs = tokenizer(question, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True)

#         # Move inputs to device
#         inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

#         # Generate response
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_length=2048, do_sample=False, temperature=0.9, top_k=50, top_p=0.9)
#             predicted_answers = [tokenizer.decode(logits.argmax(dim=-1)) for logits in outputs]

#         # Store information in the same dictionary
#         output_data.append({
#             'image_path': image_path,
#             'question': question,
#             'ground_truth': ground_truth,
#             'predicted': predicted_answers,
#         })

# # Write output data to JSON file
# with open('output_file.json', 'w') as outfile:
#     json.dump(output_data, outfile, indent=4)


# import torch
# import json
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from torchvision import transforms

# # Define configuration
# MODEL_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/"
# TOKENIZER_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d" 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# QUANTIZATION = False  # Set to True if quantization is desired, otherwise False

# # Initialize tokenizer
# tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

# # Initialize model
# if QUANTIZATION:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         load_in_4bit=True,
#         trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     ).to(DEVICE).eval()

# # Define image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize image to fit model input size
#     transforms.ToTensor(),           # Convert image to tensor
# ])

# # Load input JSON file
# with open('/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json', 'r') as f:
#     data = json.load(f)

# # Define template
# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# # Process each entry in the input JSON file
# output_data = []
# for entry in data[:2]:
#     image_path = entry['image_path']
#     questions_answers = entry['responses']
 

#     # Load and preprocess image
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

#     # Process each question-answer pair
#     for qa_pair in questions_answers:
#         question = qa_pair['Q']
#         ground_truth = qa_pair['A']

#         # Preprocess input using tokenizer
#         inputs = tokenizer(question, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True)

#         # Move inputs to device
#         inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
       

#         # Initialize a list to store predicted answers
#         predicted_answers = []

#         # Generate response
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_length=2048, do_sample=False, temperature=0.9, top_k=50, top_p=0.9)
          

#             # Decode each output token sequence
#             for logits in outputs:
#                 # Decode the token sequence, ignoring special tokens
#                 decoded_tokens = tokenizer.decode(logits, skip_special_tokens=True)
                
#                 # If the decoded answer is empty or only contains special tokens, use a default placeholder
#                 if not decoded_tokens.strip():
#                     decoded_tokens = "Unknown answer"
                
#                 # Append the decoded answer to the list
#                 predicted_answers.append(decoded_tokens)
#             print(predicted_answers)

#         # Store information in the same dictionary
#         output_data.append({
#             'image_path': image_path,
#             'question': question,
#             'ground_truth': ground_truth,
#             'predicted_answers': predicted_answers,
#         })

# # Write output data to JSON file
# with open('output_file.json', 'w') as outfile:
#     json.dump(output_data, outfile, indent=4)



# import os
# import sys
# import json
# import argparse
# import torch
# from PIL import Image
# from sat.model.mixins import CachedAutoregressiveMixin
# from utils.chat import chat
# from models.cogvlm_model import CogVLMModel
# from utils.language import llama2_tokenizer, llama2_text_processor_inference
# from utils.vision import get_image_processor

# def inference(input_file, output_file, args):
#     # Load input JSON file
#     with open(input_file, 'r') as f:
#         data = json.load(f)

#     # Initialize model
#     model, model_args = CogVLMModel.from_pretrained(
#         args.from_pretrained,
#         args=argparse.Namespace(
#             deepspeed=None,
#             local_rank=0,
#             rank=0,
#             world_size=1,
#             model_parallel_size=1,
#             mode='inference',
#             skip_init=True,
#             use_gpu_initialization=True if torch.cuda.is_available() else False,
#             device='cuda',
#             **vars(args)
#         ),
#         overwrite_args={'model_parallel_size': 1}
#     )
#     model = model.eval()
#     print(model)
#     model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

#     # Load tokenizer
#     tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)

#     # Load image processor
#     image_processor = get_image_processor(model_args.eva_args["image_size"][0])

#     # Load text processor
#     text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

#     # Initialize history and cache image
#     history = None
#     cache_image = None

#     # Initialize output data list
#     output_data = []

#     # Process each entry in the input JSON file
#     for entry in data:
#         image_path = entry['image_path']
#         questions_answers = entry['responses']

#         # Load image
#         image = Image.open(image_path).convert('RGB')

#         # Process each question-answer pair
#         for qa_pair in questions_answers:
#             question = qa_pair['Q']
#             ground_truth = qa_pair['A']

#             # Chat with CogVLM to get predicted answer
#             with torch.no_grad():
#                 response, history, cache_image = chat(
#                     image_path,
#                     model,
#                     text_processor_infer,
#                     image_processor,
#                     question,
#                     history=history,
#                     image=cache_image,
#                     max_length=args.max_length,
#                     top_p=args.top_p,
#                     temperature=args.temperature,
#                     top_k=args.top_k,
#                     invalid_slices=text_processor_infer.invalid_slices,
#                     no_prompt=args.no_prompt
#                 )

#             # Store information in the output data
#             output_data.append({
#                 'image_path': image_path,
#                 'question': question,
#                 'ground_truth': ground_truth,
#                 'predicted_answer': response
#             })

#     # Write output data to JSON file
#     with open(output_file, 'w') as outfile:
#         json.dump(output_data, outfile, indent=4)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
#     parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
#     parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
#     parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
#     parser.add_argument("--english", action='store_true', help='only output English')
#     parser.add_argument("--version", type=str, default="chat", help='version to interact with')
#     parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat-v1.1", help='pretrained ckpt')
#     parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
#     parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
#     parser.add_argument("--input_file", type=str, default="/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json", help="input JSON file with image paths and questions")
#     parser.add_argument("--output_file", type=str, default="output.json", help="output JSON file to store predicted answers")
#     args = parser.parse_args()

#     inference(args.input_file, args.output_file, args)

# if __name__ == "__main__":
#     main()


# import argparse
# import torch
# import json
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch


# def process_json(input_file, output_file, model, tokenizer, device):
#     with open(input_file, 'r') as f:
#         data = json.load(f)

#     results = []

#     for item in data:
#         query = item["question"]
#         image_path = item["image_path"]

#         # Load image
#         image = Image.open(image_path).convert('RGB')

#         # Prepare inputs
#         inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
#         inputs = {
#             'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
#             'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
#             'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
#             'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
#         }
#         gen_kwargs = {"max_length": 2048, "do_sample": False}

#         # Generate output
#         with torch.no_grad():
#             outputs = model.generate(**inputs, **gen_kwargs)
#             outputs = outputs[:, inputs['input_ids'].shape[1]:]
#             result_text = tokenizer.decode(outputs[0])

#         # Append result to list
#         results.append({"question": query, "image_path": image_path, "description": result_text})

#     # Write results to output JSON file
#     with open(output_file, "w") as f:
#         json.dump(results, f)

#     print("Results written to", output_file)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("input_file", type=str, default = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json")
#     parser.add_argument("output_file", type=str, default= "/home/numansaeed/Desktop/Ali_Bhai/CogVLM")
#     parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')  
#     parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
#     parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf",
#                         help="Pretrained checkpoint")
    
#     args = parser.parse_args()
#     MODEL_PATH = args.from_pretrained
#     TOKENIZER_PATH = args.local_tokenizer
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
#     args = parser.parse_args()

#     # Load tokenizer
#     tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)
    
#     if args.bf16:
#             torch_type = torch.bfloat16
#     else:
#         torch_type = torch.float16

#     print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

#     if args.quant:
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             torch_dtype=torch_type,
#             low_cpu_mem_usage=True,
#             load_in_4bit=True,
#             trust_remote_code=True
#         ).eval()
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             torch_dtype=torch_type,
#             low_cpu_mem_usage=True,
#             load_in_4bit=args.quant is not None,
#             trust_remote_code=True
#         ).to(DEVICE).eval()


#     # Determine device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Initialize the model with empty weights
#     with init_empty_weights():
#         model = AutoModelForCausalLM.from_pretrained(
#             args.from_pretrained,
#             torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#         )

#     # Move model and embedding layer to device
#     model = model.to(device)
#     model.base_model.embed_tokens = model.base_model.embed_tokens.to(device)

#     # Load model checkpoint for single GPU
#     model = load_checkpoint_and_dispatch(
#         model,
#         '/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/',
#         device_map={'cuda:0': model.parameters()}
#     )
#     model = model.eval()

#     # Process input JSON file and write predictions to output JSON file
#     process_json(args.input_file, args.output_file, model, tokenizer, device)


# if __name__ == "__main__":
#     main()





# import argparse
# import torch
# import json
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from torchvision import transforms


# parser = argparse.ArgumentParser()
# parser.add_argument("input_file", type=str, default = "/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json")
# parser.add_argument("output_file", type=str, default= "/home/numansaeed/Desktop/Ali_Bhai/CogVLM")
# parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')  
# parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
# parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf",
#                     help="Pretrained checkpoint")
# parser.add_argument("--fp16", action="store_true")
# parser.add_argument("--bf16", action="store_true")
# # Configuration
# MODEL_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/"
# TOKENIZER_PATH = "/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d" 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# args = parser.parse_args()
# # Initialize tokenizer and model
# tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
# # model = AutoModelForCausalLM.from_pretrained(
# #     MODEL_PATH,
# #     torch_dtype=torch.bfloat16,
# #     low_cpu_mem_usage=True,
# #     trust_remote_code=True
# # ).to(DEVICE).eval()

# if args.bf16:
#     torch_type = torch.bfloat16
# else:
#     torch_type = torch.float16

# print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

# if args.quant:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         load_in_4bit=True,
#         trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         load_in_4bit=args.quant is not None,
#         trust_remote_code=True
#     ).to(DEVICE).eval()

# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# # Image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # Load input JSON file
# with open('/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json', 'r') as f:
#     data = json.load(f)

# # Process each entry in the input JSON file
# output_data = []
# for entry in data[:2]:  # Remove the slice to process all entries
#     image_path = entry['image_path']
#     questions_answers = entry['responses']

#     # Load and preprocess image
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(DEVICE)  # Process image

#     for qa_pair in questions_answers:
#         question = qa_pair['Q']
#         ground_truth = qa_pair['A']

#         # Tokenize question
#         inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

#         # Debug: Print tokenized inputs
#         print("Tokenized inputs:", inputs)

#         # Generate response
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_length=2048, do_sample=False, temperature=0.7, num_return_sequences=1)
#             predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Debug: Print decoded outputs
#         print("Decoded output:", predicted_answer)

#         # Save results
#         output_data.append({
#             'image_path': image_path,
#             'question': question,
#             'ground_truth': ground_truth,
#             'predicted': predicted_answer,
#         })

# # Write output data to JSON file
# with open('output_file.json', 'w') as outfile:
#     json.dump(output_data, outfile, indent=4)

# print("Results saved to output_file.json")


# import torch
# import json
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from torchvision import transforms
# import argparse

# # Setup argparse for command line arguments
# parser = argparse.ArgumentParser(description='Generate predictions using CogVLM model.')
# parser.add_argument('--input_file', type=str, default="/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava.json", required=True, help='Path to the input JSON file containing image paths and questions.')
# parser.add_argument('--output_file', type=str, required=True, help='Path to output the predictions.')
# parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
# args = parser.parse_args()

# # Load tokenizer and model
# model_path = 'path_to_cogvlm_model'
# tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
# model.eval()

# # Define image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Adjust size according to model's requirement
#     transforms.ToTensor()
# ])

# # Load input JSON file
# with open(args.input_file, 'r') as f:
#     data = json.load(f)

# output_data = []

# for entry in data:
#     image_path = entry['image_path']
#     questions_answers = entry['responses']

#     # Load and preprocess image if exists
#     image = Image.open(image_path).convert('RGB') if image_path else None
#     if image:
#         image = transform(image).unsqueeze(0).to('cuda')

#     for qa_pair in questions_answers:
#         query = qa_pair['Q']
#         ground_truth = qa_pair['A']

#         # Prepare inputs for model based on whether image is provided
#         if image is None:
#             input_by_model = tokenizer(query, return_tensors="pt").to('cuda')
#         else:
#             input_by_model = tokenizer(query, return_tensors="pt").to('cuda')
#             input_by_model['visual_embeds'] = image  # Add image tensor

#         # Set up the inputs dictionary
#         inputs = {
#             'input_ids': input_by_model['input_ids'],
#             'token_type_ids': input_by_model['token_type_ids'],
#             'attention_mask': input_by_model['attention_mask'],
#             'images': [image] if image is not None else None,
#         }

#         # Generation settings
#         gen_kwargs = {"max_length": 2048, "do_sample": False}
#         with torch.no_grad():
#             outputs = model.generate(**inputs, **gen_kwargs)
#             response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Append results to output data
#         output_data.append({
#             'image_path': image_path,
#             'question': query,
#             'ground_truth': ground_truth,
#             'predicted': response
#         })

# # Write output data to JSON file
# with open(args.output_file, 'w') as outfile:
#     json.dump(output_data, outfile, indent=4)

# print("Inference complete. Results saved to", args.output_file)







from tqdm import tqdm
import torch
import json
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torchvision import transforms
import argparse
from torchvision.transforms import ToPILImage

# Convert the image tensor to a PIL Image


# Now you can apply the transform to the PIL Image

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description='Generate predictions using CogVLM model with templated inputs.')
parser.add_argument('--input_file', default="/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/question-answers_test-llava_noise.json", type=str, required=True, help='Path to the input JSON file containing image paths and questions.')
parser.add_argument('--output_file', default ="/home/numansaeed/Desktop/Ali_Bhai/Qwen-VL/predictions_cogvlm_noise.json", type=str, required=True, help='Path to output the predictions.')
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
########################################################################
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt') ##THUDM/cogagent-chat-hf
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')     ##
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")

args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()



####################################3
# args = parser.parse_args()

# # Load tokenizer and model
# model_path = '/home/numansaeed/Desktop/Ali_Bhai/CogVLM/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/'
# tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
# model.eval()

# Define image transformation


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size according to model's requirement
    transforms.ToTensor()
])

# Template for input queries
template = "Answer the following question based on the image content: {}"

# Load input JSON file
with open(args.input_file, 'r') as f:
    data = json.load(f)

output_data = []
image_tensor = None
for entry in tqdm(data):
    image_path = entry.get('image_path')
    questions_answers = entry['responses']

    # Load and preprocess image if it exists
    if image_path:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to('cuda')
        image_tensor_pil = ToPILImage()(image_tensor.squeeze(0).cpu())  # Convert tensor to PIL Image
        for qa_pair in questions_answers:
            question = qa_pair['Q']
            ground_truth = qa_pair['A']

            # Format the question using the predefined template
            formatted_query = template.format(question)
            print(formatted_query)

        # Construct inputs for the model
            if image_tensor is not None:
                input_by_model = model.build_conversation_input_ids(tokenizer, query=formatted_query, images=[image_tensor_pil])
            else:
                input_by_model = model.build_conversation_input_ids(tokenizer, query=formatted_query, template_version='base')

            # Set up the inputs dictionary
        # Set up inputs for the model
        # Set up inputs for the model
            # Set up inputs for the model
            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),  # Add batch dimension if missing
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image_tensor is not None else None,
            }

            # If 'cross_images' is in input_by_model and it's not None, include it in the inputs
            if 'cross_images' in input_by_model and input_by_model['cross_images'] is not None:
                inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

            # Generation settings
            gen_kwargs = {"max_length": 2048, "do_sample": False}

            # Generate outputs
        # Generate outputs
        # Generate outputs
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                
                # Get input length
                input_length = 1
                if len(input_by_model['input_ids'].shape) > 0 and len(input_by_model['input_ids'].shape) < 2:
                    input_length = input_by_model['input_ids'].shape[0]
                elif len(input_by_model['input_ids'].shape) >= 2:
                    input_length = input_by_model['input_ids'].shape[1]

                outputs = outputs[:, input_length:]

            # Decode the generated output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Append results to output data
            output_data.append({
                'image_path': image_path,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': response
            })


# Write output data to JSON file
with open(args.output_file, 'w') as outfile:
    json.dump(output_data, outfile, indent=4)

print("Inference complete. Results saved to", args.output_file)
