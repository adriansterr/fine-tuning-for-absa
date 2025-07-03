from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

model_name = "finetuned_models/merged_model_ohne_eos_2"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a helpful assistant for aspect-based sentiment analysis.\nThe possible (ASPECT#CATEGORY, SENTIMENT) pairs are: (AMBIENCE#GENERAL, NEGATIVE), (AMBIENCE#GENERAL, NEUTRAL), (AMBIENCE#GENERAL, POSITIVE), (DRINKS#PRICES, NEGATIVE), (DRINKS#PRICES, POSITIVE), (DRINKS#QUALITY, NEGATIVE), (DRINKS#QUALITY, NEUTRAL), (DRINKS#QUALITY, POSITIVE), (DRINKS#STYLE_OPTIONS, NEGATIVE), (DRINKS#STYLE_OPTIONS, POSITIVE), (FOOD#PRICES, NEGATIVE), (FOOD#PRICES, NEUTRAL), (FOOD#PRICES, POSITIVE), (FOOD#QUALITY, NEGATIVE), (FOOD#QUALITY, NEUTRAL), (FOOD#QUALITY, POSITIVE), (FOOD#STYLE_OPTIONS, NEGATIVE), (FOOD#STYLE_OPTIONS, NEUTRAL), (FOOD#STYLE_OPTIONS, POSITIVE), (LOCATION#GENERAL, NEGATIVE), (LOCATION#GENERAL, NEUTRAL), (LOCATION#GENERAL, POSITIVE), (RESTAURANT#GENERAL, NEGATIVE), (RESTAURANT#GENERAL, NEUTRAL), (RESTAURANT#GENERAL, POSITIVE), (RESTAURANT#MISCELLANEOUS, NEGATIVE), (RESTAURANT#MISCELLANEOUS, NEUTRAL), (RESTAURANT#MISCELLANEOUS, POSITIVE), (RESTAURANT#PRICES, NEGATIVE), (RESTAURANT#PRICES, NEUTRAL), (RESTAURANT#PRICES, POSITIVE), (SERVICE#GENERAL, NEGATIVE), (SERVICE#GENERAL, NEUTRAL), (SERVICE#GENERAL, POSITIVE)"}, 
    {"role": "user", "content": "Given the review: \"The hamburgers were juicy and the soda refreshing, but the waiters were rude as hell.\" List all (ASPECT#CATEGORY, SENTIMENT) pairs."},
    {"role": "assistant", "content": ""}
]
# print("Messages:", messages)

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
) + tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs["input_ids"], streamer=text_streamer, max_new_tokens=256, use_cache=True)

# Direkter Prompt, manuell eingegeben:

# prompt = (
#     "<|im_start|>system\nYou are a helpful assistant for aspect-based sentiment analysis.<|im_end|>\n"
#     "<|im_start|>user\nGiven the review: \"The hamburgers were juicy and the soda refreshing, but the waiters were rude as hell.\" List all (ASPECT#CATEGORY, SENTIMENT) pairs.<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )

# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(input_ids=inputs["input_ids"], streamer=text_streamer, max_new_tokens=128, use_cache=True)