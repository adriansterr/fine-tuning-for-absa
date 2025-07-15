from unsloth import FastLanguageModel
from peft import PeftModel

# Wenn man nur den LoRA-Checkpoint in das originale Modell mergen will:
# base_model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="llama_output_3_categories_low_lr/checkpoint-214",
#     max_seq_length=2048,
#     load_in_4bit=False,
#     dtype=None,
# )

# merged_model = base_model.merge_and_unload()

# merged_model.save_pretrained("finetuned_models/meta_llama_full_precision", max_shard_size="10GB")
# tokenizer.save_pretrained("finetuned_models/meta_llama_full_precision")


# Lade LoRA-Adapter in das originale Modell und merge es in voller Präzision
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=False, # volle Präzision
    dtype=None,
    device_map="cpu" # hat irgendwie mit gpu nicht funktioniert
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

model.load_adapter(
    "lora_adapter/meta_llama_load_4_bit",
    "default",
    offload_folder="offload/",  # Muss offloaden, weil nicht alles in die GPU passt
    offload_buffers=True   
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("finetuned_models/meta_llama_full_precision", max_shard_size="10GB")
tokenizer.save_pretrained("finetuned_models/meta_llama_full_precision")