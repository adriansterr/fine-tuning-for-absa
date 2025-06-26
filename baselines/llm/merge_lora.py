from unsloth import FastLanguageModel

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama_output_3_categories_low_lr/checkpoint-214",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

merged_model = base_model.merge_and_unload()

merged_model.save_pretrained("merged_model_3_low_lr_categories", max_shard_size="10GB")
tokenizer.save_pretrained("merged_model_3_low_lr_categories")