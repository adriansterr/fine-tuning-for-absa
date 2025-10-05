from unsloth import FastLanguageModel

# Merge LoRA-Checkpoint into original model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_adapter/meta_llama_translated_dataset_french",
    max_seq_length=2048,
    load_in_4bit=False,
    dtype=None,
    device_map="cpu"
)

merged_model = base_model.merge_and_unload()

merged_model.save_pretrained("finetuned_models/meta_llama_translated_dataset_french_remerged", max_shard_size="10GB")
tokenizer.save_pretrained("finetuned_models/meta_llama_translated_dataset_french_remerged")