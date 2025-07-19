# https://huggingface.co/blog/mlabonne/sft-llama3

# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# wenn UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 54278: character maps to <undefined>:
# set PYTHONUTF8=1
# set HF_HUB_ENABLE_HF_TRANSFER=0

from ast import mod
import os, sys
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import unsloth
import torch
import json
import pandas as pd
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from datasets import Dataset
from transformers import TrainingArguments, set_seed

utils = os.path.abspath('./src/utils/') # Relative path to utils scripts
print(utils)
sys.path.append(utils)

from preprocessing import createPrompts, loadDataset
from llama_eval import evaluate_model
from config import Config

def main():
    config = Config()
    set_seed(config.seed)    
    
    print("loading model...")
    max_seq_length = 2048
    base_model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
      max_seq_length=max_seq_length,
      dtype=torch.float16,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
      device_map="auto" 
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

    df_train, df_test, label_space = loadDataset(config.data_path, config.dataset, config.low_resource_setting, config.task, config.split, config.original_split)
    prompts_train, prompts_test, ground_truth_labels = createPrompts(df_train, df_test, config, eos_token = tokenizer.eos_token)

    print(label_space, "\n")
    print("Ground Truth: ", ground_truth_labels[0], "\n")

    prompts_train = pd.DataFrame(prompts_train, columns = ['text'])
    prompts_test = pd.DataFrame(prompts_test, columns = ['text'])
    print("Prompt: ", prompts_train['text'][0], "\n")

    ds_train = Dataset.from_pandas(prompts_train)
    ds_test = Dataset.from_pandas(prompts_test)
    print(ds_train, "\n")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=5, 
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=20,
            output_dir="content/drive/MyDrive/colab_data/output/llama/meta_llama_full",
            eval_strategy="no",
            save_strategy="epoch",
            seed=0
        ),
    )

    print("Starting training...")
    trainer.train()

    print("Start Saving Adapter...")
    model.save_pretrained("content/drive/MyDrive/colab_data/lora_adapter/meta_llama_full", save_adapter=True, save_config=True)

    print("Start Saving Model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("content/drive/MyDrive/colab_data/finetuned_models/meta_llama_full", max_shard_size="4GB")
    tokenizer.save_pretrained("content/drive/MyDrive/colab_data/finetuned_models/meta_llama_full")    
    print("Finished")


if __name__ == "__main__":
    main()