from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath('./src/utils/'))

from evaluation import extractAspects, convertLabels, createResults, subset_recall
from preprocessing import loadDataset, createPrompts
from config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model(model, tokenizer, config, prompts_test, ground_truth_labels, label_space, results_path=None):
    predictions = []
    model.eval()
    for prompt in tqdm(prompts_test, desc="Generating Outputs"): # progress bar
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(output_text)
        print(f"Prompt: {prompt}\nPrediction: {output_text}\n")

    processed_predictions = [extractAspects(p, config.task, config.prompt_style) for p in predictions]
    processed_ground_truths = [extractAspects(gt, config.task, config.prompt_style) for gt in ground_truth_labels]

    gold_labels, _ = convertLabels(processed_ground_truths, config.task, label_space)
    pred_labels, false_predictions = convertLabels(processed_predictions, config.task, label_space)
    
    # print("Gold Labels:", gold_labels, "\n")
    # print("Predicted Labels:", pred_labels, "\n")
    
    emr = subset_recall(pred_labels, gold_labels)
    print(f"Subset recall (all gold labels found in prediction): {emr:.3f}")    

    results_asp, results_asp_pol, results_pairs, results_pol, results_phrases = createResults(pred_labels, gold_labels, label_space, config.task)

    if results_path:
        import pandas as pd, os, json
        os.makedirs(results_path, exist_ok=True)
        pd.DataFrame({'prompt': prompts_test, 'ground_truth': ground_truth_labels, 'prediction': predictions}).to_csv(os.path.join(results_path, "predictions.csv"), index=False)
        with open(os.path.join(results_path, "metrics.json"), "w") as f:
            json.dump({
                "results_asp": results_asp,
                "results_asp_pol": results_asp_pol,
                "results_pairs": results_pairs,
                "results_pol": results_pol,
                "results_phrases": results_phrases
            }, f, indent=2)

    print("Evaluation complete.")
    print("Aspect metrics:", results_asp)
    print("Aspect+Polarity metrics:", results_asp_pol)
    print("Pair metrics:", results_pairs)
    print("Polarity metrics:", results_pol)
    print("Phrase metrics:", results_phrases)
    
    
if __name__ == "__main__":
    model_name = "D:/Uni/Masterarbeit Code/test/mergekit/merges/trained/llama/meta_llama_full_precision_sauerkraut/linear"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
        device_map="cuda" 
    )
    FastLanguageModel.for_inference(model)

    config = Config()
    config.dataset = 'GERestaurant'
    df_train, df_test, label_space = loadDataset(config.data_path, config.dataset, config.low_resource_setting, config.task, config.split, config.original_split)
    
    # Limit test set to 20 entries for testing purposes
    df_test = df_test.iloc[:20]
    
    prompts_train, prompts_test, ground_truth_labels = createPrompts(df_train, df_test, config, eos_token=tokenizer.eos_token)
    evaluate_model(model, tokenizer, config, prompts_test, ground_truth_labels, label_space, results_path="results/llama/meta_llama_full_precision_sauerkraut/linear/")