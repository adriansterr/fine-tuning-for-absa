# https://huggingface.co/blog/mlabonne/sft-llama3

# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# wenn UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 54278: character maps to <undefined>:
# set PYTHONUTF8=1

from ast import mod
import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import unsloth
import torch
import json
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments


# Kategorien von Rest-16 Datensatz
categories = (
    "(AMBIENCE#GENERAL, NEGATIVE), (AMBIENCE#GENERAL, NEUTRAL), (AMBIENCE#GENERAL, POSITIVE), "
    "(DRINKS#PRICES, NEGATIVE), (DRINKS#PRICES, POSITIVE), "
    "(DRINKS#QUALITY, NEGATIVE), (DRINKS#QUALITY, NEUTRAL), (DRINKS#QUALITY, POSITIVE), "
    "(DRINKS#STYLE_OPTIONS, NEGATIVE), (DRINKS#STYLE_OPTIONS, POSITIVE), "
    "(FOOD#PRICES, NEGATIVE), (FOOD#PRICES, NEUTRAL), (FOOD#PRICES, POSITIVE), "
    "(FOOD#QUALITY, NEGATIVE), (FOOD#QUALITY, NEUTRAL), (FOOD#QUALITY, POSITIVE), "
    "(FOOD#STYLE_OPTIONS, NEGATIVE), (FOOD#STYLE_OPTIONS, NEUTRAL), (FOOD#STYLE_OPTIONS, POSITIVE), "
    "(LOCATION#GENERAL, NEGATIVE), (LOCATION#GENERAL, NEUTRAL), (LOCATION#GENERAL, POSITIVE), "
    "(RESTAURANT#GENERAL, NEGATIVE), (RESTAURANT#GENERAL, NEUTRAL), (RESTAURANT#GENERAL, POSITIVE), "
    "(RESTAURANT#MISCELLANEOUS, NEGATIVE), (RESTAURANT#MISCELLANEOUS, NEUTRAL), (RESTAURANT#MISCELLANEOUS, POSITIVE), "
    "(RESTAURANT#PRICES, NEGATIVE), (RESTAURANT#PRICES, NEUTRAL), (RESTAURANT#PRICES, POSITIVE), "
    "(SERVICE#GENERAL, NEGATIVE), (SERVICE#GENERAL, NEUTRAL), (SERVICE#GENERAL, POSITIVE)"
)

def add_categories_to_system(messages):
    system_msg = (
        "You are a helpful assistant for aspect-based sentiment analysis.\n"
        f"The possible (ASPECT#CATEGORY, SENTIMENT) pairs are: {categories}" # diese Zeile auskommentieren, für ohne Kategorien 
    )
    if messages[0]["role"] == "system":
        messages[0]["content"] = system_msg
    else:
        messages = [{"role": "system", "content": system_msg}] + messages
    return messages

def apply_template(examples):
    messages_batch = [add_categories_to_system(msgs) for msgs in examples["messages"]]
    text = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False
        ) + tokenizer.eos_token
        for message in messages_batch
    ]
    return {"text": text}

print("loading model...")
max_seq_length = 2048
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)


model = FastLanguageModel.get_peft_model(
    base_model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    loftq_config=None,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    chat_template="chatml",
)

print("Loading datasets...")
dataset = load_dataset("json", data_files={"train": "data/rest-16/llama_train_chatml.jsonl", "test": "data/rest-16/llama_test_chatml.jsonl"})
print("train 0", dataset['train'][0])
print("test 0", dataset['test'][0])

dataset = dataset.map(apply_template, batched=True)
print("Template applied.")
print("train 0", dataset['train'][0])
print("test 0", dataset['test'][0])


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=True,
    args=TrainingArguments(
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        # Wenn eval loss steight, kann das ein Zeichen für Overfitting sein
        num_train_epochs=2, 
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=20,
        output_dir="output/llama_output_3_categories_low_lr",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        seed=0
    ),
)

print("Starting training...")
trainer.train()

eval_results = trainer.evaluate()
with open("output/llama_output_3_categories_low_lr/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)
print("Evaluation results saved")


model.save_pretrained("lora_adapter", save_adapter=True, save_config=True)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model_3_low_lr_categories", max_shard_size="10GB")
tokenizer.save_pretrained("merged_model_3_low_lr_categories")
print("Model saved")