import pandas as pd
import ast
import json

input_path_train = "../../data/rest-16/train.tsv"
output_path_train = "../../data/rest-16/llama_train_chatml.jsonl"
input_path_test = "../../data/rest-16/test.tsv"
output_path_test = "../../data/rest-16/llama_test_chatml.jsonl"

df = pd.read_csv(input_path_train, sep="\t")

with open(output_path_train, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        text = row['text']
        labels = ast.literal_eval(row['labels'])
        label_str = ", ".join(labels)
        chatml = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for aspect-based sentiment analysis."},
                {"role": "user", "content": f'Given the review: "{text}" List all (ASPECT#CATEGORY, SENTIMENT) pairs.'},
                {"role": "assistant", "content": label_str}
            ]
        }
        fout.write(json.dumps(chatml, ensure_ascii=False) + "\n")
        
        
df = pd.read_csv(input_path_test, sep="\t")

with open(output_path_test, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        text = row['text']
        labels = ast.literal_eval(row['labels'])
        label_str = ", ".join(labels)
        chatml = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for aspect-based sentiment analysis."},
                {"role": "user", "content": f'Given the review: "{text}" List all (ASPECT#CATEGORY, SENTIMENT) pairs.'},
                {"role": "assistant", "content": label_str}
            ]
        }
        fout.write(json.dumps(chatml, ensure_ascii=False) + "\n")