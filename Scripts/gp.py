
import argparse
import torch
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import tqdm


def load_image(image_file):
   if image_file.startswith('http://') or image_file.startswith('https://'):
       response = requests.get(image_file)
       image = Image.open(BytesIO(response.content)).convert('RGB')
   else:
       image = Image.open(image_file).convert('RGB')
   return image


def main(args):
   disable_torch_init()

   model_name = get_model_name_from_path(args.model_path)
   tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

   if "llama-2" in model_name.lower():
       conv_mode = "llava_llama_2"
   elif "mistral" in model_name.lower():
       conv_mode = "mistral_instruct"
   elif "v1.6-34b" in model_name.lower():
       conv_mode = "chatml_dir# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
#     parser.add_argument("--model-base", type=str, default=None)
#     parser.add_argument("--input-json", type=str, required=True)
#     parser.add_argument("--output-json", type=str, default="all_responses.json")
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--conv-mode", type=str, default=None)
#     parser.add_argument("--temperature", type=float, default=0.2)
#     parser.add_argument("--max-new-tokens", type=int, default=512)
#     parser.add_argument("--load-8bit", action="store_true")
#     parser.add_argument("--load-4bit", action="store_true")
#     parser.add_argument("--debug", action="store_true")
#     args = parser.parse_args()
#     main(args)


   conv = conv_templates[args.conv_mode].copy()
   roles = conv.roles if "mpt" not in model_name.lower() else ('user', 'assistant')

   with open(args.input_json, "r") as file:
       data = json.load(file)

   all_responses = []

   for entry in tqdm(data):
       image_path = entry['image_path']
       image = load_image(image_path)
       image_size = image.size

       image_tensor = process_images([image], image_processor, model.config)
       if type(image_tensor) is list:
           image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
       else:
           image_tensor = image_tensor.to(model.device, dtype=torch.float16)

       responses = entry['responses']
       for response in responses:
           question = response['question']
           conv = conv_templates[args.conv_mode].copy()

           inp = f"{question}"
           if model.config.mm_use_im_start_end:
               inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
           else:
               inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

           conv.append_message(conv.roles[0], inp)
           conv.append_message(conv.roles[1], None)
           prompt = conv.get_prompt()

           input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
           stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
           streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

           with torch.inference_mode():
               output_ids = model.generate(
                   input_ids,
                   images=image_tensor,
                   image_sizes=[image_size],
                   do_sample=True if args.temperature > 0 else False,
                   temperature=args.temperature,
                   max_new_tokens=args.max_new_tokens,
                   streamer=streamer,
                   use_cache=True
               )

           predicted = tokenizer.decode(output_ids[0]).strip()
           conv.messages[-1][-1] = predicted

           all_responses.append({
               'image_path': image_path,
               'question': question,
               'predicted': predicted,
               'ground_truth': response['answer']
           })

   with open(predictions.json, 'w') as f:
       json.dump(all_responses, f, indent=4)

   print("Processing complete.")