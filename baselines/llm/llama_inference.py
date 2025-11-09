from unsloth import FastLanguageModel
from transformers import TextStreamer
import os, sys, json

# Script for fast testing

utils = os.path.abspath('./src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from preprocessing import createPromptText

model_name = "finetuned_models/meta_llama_full_colab_remerge_2/"

# FastLanguageModel from unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=2048,
    device_map="cuda" 
)
FastLanguageModel.for_inference(model)

with open("src/utils/prompts_GERestaurant.json", encoding='utf-8') as json_prompts:
    prompt_templates = json.load(json_prompts)

prompt, _ = createPromptText(
    lang='ger',
    prompt_templates=prompt_templates,
    prompt_style='basic',
    example_text="Das Essen war sehr lecker, aber der Service war schlecht.", 
    example_labels=["(FOOD#QUALITY, POSITIVE)", "(SERVICE#GENERAL, NEGATIVE)"],
    dataset_name='rest-16',
    absa_task='acsa',
    train=False
)
prompt = prompt + tokenizer.eos_token

print("Prompt: ", prompt, "\n")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
print("Generating text...\n")
_ = model.generate(input_ids=inputs["input_ids"], streamer=text_streamer, max_new_tokens=128)
