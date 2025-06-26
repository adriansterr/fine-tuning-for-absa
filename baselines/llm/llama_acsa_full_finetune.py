import os
import sys
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArgs:
    train_file: str = field(
        default="data/rest-16/llama_train_preprocessed.jsonl",
        metadata={"help": "Path to the training .jsonl file"}
    )
    eval_file: Optional[str] = field(
        default="data/rest-16/llama_test_preprocessed.jsonl",
        metadata={"help": "Path to the eval .jsonl file"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Max sequence length"}
    )

@dataclass
class TrainingArgs:
    model_name: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Model name or path"}
    )
    output_path: str = field(
        default="results/llama_acsa/",
        metadata={"help": "Output directory"}
    )
    learning_rate: float = field(
        default=2e-5
    )
    batch_size: int = field(
        default=2
    )
    epochs: int = field(
        default=3
    )
    save_steps: int = field(
        default=500
    )
    logging_steps: int = field(
        default=50
    )

def load_prompt_response_jsonl(path):
    prompts = []
    responses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "[/INST]" in line:
                prompt, response = line.split("[/INST]", 1)
                prompt = prompt + "[/INST]"
                response = response.strip()
                prompts.append(prompt)
                responses.append(response)
    return prompts, responses

class LlamaPromptDataset(TorchDataset):
    print("Loading LlamaPromptDataset...")
    def __init__(self, prompts, responses, tokenizer, max_length=512):
        self.inputs = []
        self.labels = []
        for prompt, response in zip(prompts, responses):
            full = prompt + " " + response
            tokenized = tokenizer(full, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]
            label = tokenized["input_ids"][0].clone()
            label[:len(prompt_ids)] = -100  # Mask prompt tokens
            self.inputs.append(tokenized["input_ids"][0])
            self.labels.append(label)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}

if __name__ == "__main__":
    parser = HfArgumentParser((DataArgs, TrainingArgs))
    data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    train_prompts, train_responses = load_prompt_response_jsonl(data_args.train_file)
    train_dataset = LlamaPromptDataset(train_prompts, train_responses, tokenizer, max_length=data_args.max_length)

    eval_prompts, eval_responses = load_prompt_response_jsonl(data_args.eval_file)
    eval_dataset = LlamaPromptDataset(eval_prompts, eval_responses, tokenizer, max_length=data_args.max_length)

    model = AutoModelForCausalLM.from_pretrained(training_args.model_name)

    training_args_hf = TrainingArguments(
        output_dir=training_args.output_path,
        learning_rate=training_args.learning_rate,
        num_train_epochs=training_args.epochs,
        per_device_train_batch_size=training_args.batch_size,
        per_device_eval_batch_size=training_args.batch_size,
        save_steps=training_args.save_steps,
        logging_steps=training_args.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        fp16=True if torch.cuda.is_available() else False,
        bf16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args_hf,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print(f"Training on {len(train_dataset)} samples, evaluating on {len(eval_dataset)} samples.")
    print(f"Model: {training_args.model_name}, Output Path: {training_args.output_path}")
    trainer.train()
    trainer.save_model(training_args.output_path)
    tokenizer.save_pretrained(training_args.output_path)