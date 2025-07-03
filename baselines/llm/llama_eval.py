import torch
from evaluation import (
    extractAspects, convertLabels, createResults
)

def evaluate_model(model, tokenizer, config, prompts_test, ground_truth_labels, label_space, results_path=None):
    predictions = []
    model.eval()
    for prompt in prompts_test:
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

    processed_preds = [extractAspects(p, config.task, config.prompt_style == 'cot', True) for p in predictions]
    processed_gts = [extractAspects(gt, config.task, config.prompt_style == 'cot') for gt in ground_truth_labels]

    gold_labels, _ = convertLabels(processed_gts, config.task, label_space)
    pred_labels, false_predictions = convertLabels(processed_preds, config.task, label_space)

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