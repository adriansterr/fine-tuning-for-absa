from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json

model_name = "merged_model_2"

def extract_pairs(text):
    # Nur die Paare (ASPECT#CATEGORY, SENTIMENT)
    pairs = re.findall(r'\(([^)]+)\)', text)
    # lowercase, strip whitespace
    return set(pair.lower().strip() for pair in pairs)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("data/rest-16/llama_test_chatml.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

total = 0
correct = 0
partial_scores = []

for example in test_data:
    messages = example["messages"]
    target = messages[-1]["content"]  # Antwort
    
    prompt = tokenizer.apply_chat_template(
        messages[:-1], # ohne die Antwort
        tokenize=False,
        add_generation_prompt=False
    ) #+ tokenizer.eos_token

    print(f"Prompt: {prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  

    output = model.generate(inputs["input_ids"], max_new_tokens=256)
    # pred = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    pred = tokenizer.decode(output[0])
    print(f"Generated: {pred}")

    target_pairs = extract_pairs(target)
    pred_pairs = extract_pairs(pred)

    # metrics
    intersection = target_pairs & pred_pairs
    precision = len(intersection) / len(pred_pairs) if pred_pairs else 0
    recall = len(intersection) / len(target_pairs) if target_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    partial_scores.append(f1)
    if target_pairs == pred_pairs:
        correct += 1
    total += 1

    print(f"Target: {target_pairs}\nPred: {pred_pairs}\nF1: {f1:.2f}\n")

print(f"Exact set match accuracy: {correct/total:.2%} ({correct}/{total})")
print(f"Average F1 (pair extraction): {sum(partial_scores)/len(partial_scores):.2%}")