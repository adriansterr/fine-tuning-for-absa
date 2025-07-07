from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os, sys, torch, json

utils = os.path.abspath('./src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from preprocessing import createPromptText


model_name = "finetuned_models/merged_model_ohne_eos_2"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# base_dir = os.path.dirname(os.path.abspath(__file__))
with open("src/utils/prompts_rest16.json", encoding='utf-8') as json_prompts:
    prompt_templates = json.load(json_prompts)

prompt, _ = createPromptText(
    lang='en',
    prompt_templates=prompt_templates,
    prompt_style='basic',
    example_text="The food was great but the service was slow.", 
    example_labels=["(FOOD#QUALITY, POSITIVE)", "(SERVICE#GENERAL, NEGATIVE)"],
    dataset_name='rest-16',
    absa_task='acsa',
    train=False
)
prompt = prompt + tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs["input_ids"], streamer=text_streamer, max_new_tokens=256, use_cache=True)
