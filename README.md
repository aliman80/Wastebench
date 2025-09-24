# WASTE-Bench: Waste-Bench: A Comprehensive Benchmark for Evaluating VLLMs in Cluttered Environments [EMNLP 2025]

[Muhammad Ali](https://aliman80.github.io), [Salman Khan](https://salman-h-khan.github.io/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2509.00176) [![Dataset](https://img.shields.io/badge/Dataset-Download-orange?logo=database)](https://huggingface.co/datasets/aliman8/WasteBench-Dataset) 
<!-- 
[![Website](https://img.shields.io/badge/Website-Visit-green?logo=web)](https://hananshafi.github.io/vane-benchmark/) 
[![Video](https://img.shields.io/badge/Presentation-Video-red?logo=youtube)](https://youtu.be/fqjQWx6f2SM) 
-->




Official code for our paper "Waste-Bench: A Comprehensive Benchmark for Evaluating VLLMs in Cluttered Environments"

## :rocket: News
* **(August 20, 2025)**
  * Our paper has been accepted to [EMNLP 2025]([https://2025.naacl.org/](https://openreview.net/group?id=EMNLP/2025/Conference/Authors&referrer=%5BHomepage%5D(%2F))) 🥳🎊
* **(August 31, 2025)**
  <!-- * Our code, [![Dataset](https://img.shields.io/badge/Dataset-Download-orange?logo=database)],<!--  and the [project website](https://hananshafi.github.io/vane-benchmark/) -->is now live! -->
  * * Our code, [dataset](https://huggingface.co/datasets/aliman8/WasteBench-Dataset), and the [project website](https://aliman80.github.io/) are now live!

<hr>


![method-diagram](https://github.com/aliman80/wastebench/blob/main/Image_Project.png?raw=true)

> **Abstract:** *Recent advancements in Large Language Models (LLMs) have paved the way for Vision Large Language Models (VLLMs) capable of performing a wide range of visual understanding tasks. While LLMs have demonstrated impressive performance on standard natural images, their capabilities have not been thoroughly explored in cluttered datasets where there is complex environment having deformed shaped objects. In this work, we introduce a novel dataset specifically designed for waste classification in real-world scenarios, characterized by complex environments and deformed shaped objects. Along with this dataset, we present an in-depth evaluation approach to rigorously assess the robustness and accuracy of VLLMs.
The introduced dataset and comprehensive analysis provide valuable insights into the performance of VLLMs under challenging conditions. Our findings highlight the critical need for further advancements in VLLM's robustness to perform better in complex environments Our project website is live at [link](https://aliman80.github.io/)*
>

## :trophy: Key Contributions:

- We present **Waste-Bench: Benchmark: First large-scale benchmark for waste-centric reasoning, including cluttered and degraded images with VQA-style question–answer annotations.
- Stress testing: Controlled degradations (noise, enhanced lighting, shading) to measure robustness.
- Model evaluation: Results across open-source and closed-source VLMs (e.g., GPT-4o, Gemini, LLaVA, InternVL).
- Open-source resources: Code, data construction pipeline, and evaluation metrics publicly released.

## :hammer_and_wrench: Setup and Usage
To replicate our experiments, and to use our code:
1. First clone the repository:
```bash
git clone git@github.com:aliman80/Waste-Bench.git
```
or
```bash
[git clone https://github.com/aliman80/wastebench.git
```
2. Change directory:
```bash
cd Waste-Bench
```

### Closed-Source LMMs Setup and Usage
We used `python=3.11.8` in our experiments involving closed-source LMMs like GPT-4o and Gemini-1.5 Pro. 
1. Setup a new conda environment with the specified python version:
```bash
conda create --name Waste-Bench python=3.11
```
2. Activate the environment
```bash
conda activate Waste_Bench
```
3. Install Libraries:
```bash
pip install openai opencv-python python-dotenv tqdm google-generativeai pillow
```
4. Create a new `.env` file in `scripts/closed_source_models/`, and populate it with your OpenAI and Gemini API keys:
```bash
OPENAI_API_KEY=<your_key_here>
GOOGLE_API_KEY=<your_key_here>
```

#### Caption Generation 
For our study, we generated image captions for the ZeroWaste dataset using Google Gemini. Each image was provided as input to the Gemini API, which produced concise textual descriptions capturing the visible waste materials, object attributes, and their arrangement within cluttered environments.

This captioning step served as the foundation for subsequent tasks, including the creation of question–answer pairs and the evaluation of vision–language models. All captions were generated consistently using a waste-focused prompt to ensure relevance and accuracy to the dataset’s domain.
To run the code:
```bash
python generate_captions.py --path="<path_to_annotated_dataset>"
```
The above script will then generate caption for each image in the same directory as `path`.

#### Question Answer Generation 
Once the captions are obtained we generated final QA pairs. 
To run the code:
```bash
python scripts/closed_source_models/question-answer-generation-groq.py --path="<path_to_annotated_dataset_and_captions>"
```

#### Evaluating GPT-4o on VQA task
1. Download and unzip the Waste-Bench dataset by following [Dataset](#floppy_disk-dataset).
2. Evaluate GPT-4o, Gemini-Pro, other models  at a time., run the following:
```bash
python scripts/closed_source_models/Evaluation-gpt_gpt.py --data_path="/path/to/WasteData"
```

#### Evaluating Gemini-1.5 Pro on VQA task
1. Download and unzip the Waste-Bench dataset by following [Dataset](#floppy_disk-dataset).
2. Evaluate Gemini-1.5 Pro one dataset at a time. For example, to evaluate it on "SORA" dataset, run the following:
```bash
python scripts/closed_source_models/Evaluation-gpt_gpt_gemini.py --data_path="/path/to/WasteDataset" 
```

#### Calculating LMMs accuracy on VQA task
Following the previous instruction, once the prediction files for a LMM is generated, we can evaluate the LMM's accuracy by running:
```bash
python scripts/accuracy.py --path="/path/to/GPT-4o predictions"
```
The above command evaluates the accuracy of GPT-4o predictions. To evaluate different models on different datasets, just modify the `path` variable accordingly.

### Open-Source LMMs Setup and Usage
Follow the instructions given at evaluation setup for various VLMs used.

## :floppy_disk: Dataset
Our WasteBench dataset can be downloaded from [https://huggingface.co/datasets/aliman8/WasteBench-Dataset]. Users can either directly load the dataset using 🤗 Datasets, or download the zip file present in that repository.

## :email: Contact
Should you have any questions, please create an issue in this repository contact muhammad.ali@mbzuai.ac.ae.

## :pray: Acknowledgement
We thank [OpenAI](https://github.com/openai/openai-python) and [Google](https://github.com/google-gemini/generative-ai-python) for their Python SDKs. 

## :black_nib: Citation
If you found our work helpful, please consider starring the repository ⭐⭐⭐ and citing our work as follows:
```bibtex
@misc{ali2025wastebench,
      title        = {WasteBench: Robust Evaluation of Vision-Language Models on Waste Object Segmentation},
      author       = {Muhammad Ali and Salman Khan},
      year         = {2025},
      eprint       = {https://arxiv.org/submit/6747980/view},        
      archivePrefix= {arXiv},
      primaryClass = {cs.CV}
}

```
