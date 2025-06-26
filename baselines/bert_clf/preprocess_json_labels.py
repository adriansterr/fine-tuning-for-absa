# Python script to preprocess JSON datasets so that the labels field matches the format of tsv datasets
# - Category and sentiment are uppercased
# - Spaces in category are replaced with underscores
# - A # is inserted between category and subcategory
# - The last part (phrase) is removed
# - Each label is formatted as '(CATEGORY#SUBCATEGORY, SENTIMENT)'
# - The labels field becomes a list of such strings

import json

def convert_label(label):
    # label: ["restaurant general","negative","NULL"]
    category, sentiment, _ = label
    # Split category into main and subcategory
    cat_parts = category.upper().replace(" ", "_").split("_")
    if len(cat_parts) >= 2:
        cat = f"{cat_parts[0]}#{'_'.join(cat_parts[1:])}"
    else:
        cat = cat_parts[0]
    sent = sentiment.upper()
    return f"({cat}, {sent})"

input_path_train = "../../data/GERestaurant/12_categories/train.json"
output_path_train = "../../data/GERestaurant/12_categories/train_preprocessed.json"
input_path_test = "../../data/GERestaurant/12_categories/test.json"
output_path_test = "../../data/GERestaurant/12_categories/test_preprocessed.json"

with open(input_path_train, "r", encoding="utf-8") as fin, open(output_path_train, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        new_labels = [convert_label(l) for l in obj["labels"]]
        obj["labels"] = new_labels
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
with open(input_path_test, "r", encoding="utf-8") as fin, open(output_path_test, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        new_labels = [convert_label(l) for l in obj["labels"]]
        obj["labels"] = new_labels
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")