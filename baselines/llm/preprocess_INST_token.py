import pandas as pd
import ast

input_path_train = "../../data/rest-16/train.tsv"
output_path_train = "../../data/rest-16/llama_train_preprocessed.jsonl"
input_path_test = "../../data/rest-16/test.tsv"
output_path_test = "../../data/rest-16/llama_test_preprocessed.jsonl"

df = pd.read_csv(input_path_train, sep="\t")
with open(output_path_train, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        text = row['text']
        # Convert string representation of list to actual list
        labels = ast.literal_eval(row['labels'])
        # Join labels as comma-separated string
        label_str = ", ".join(labels)
        prompt = f'Given the review: "{text}" List all (ASPECT#CATEGORY, SENTIMENT) pairs.'
        # Use [INST] format for llama
        llama_prompt = f"<s>[INST] {prompt} [/INST] {label_str}\n"
        fout.write(llama_prompt)
        
        
df = pd.read_csv(input_path_test, sep="\t")
with open(output_path_test, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        text = row['text']
        # Convert string representation of list to actual list
        labels = ast.literal_eval(row['labels'])
        # Join labels as comma-separated string
        label_str = ", ".join(labels)
        prompt = f'Given the review: "{text}" List all (ASPECT#CATEGORY, SENTIMENT) pairs.'
        # Use [INST] format for llama
        llama_prompt = f"<s>[INST] {prompt} [/INST] {label_str}\n"
        fout.write(llama_prompt)