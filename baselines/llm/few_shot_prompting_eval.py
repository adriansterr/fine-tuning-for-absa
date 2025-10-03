import os, sys

import torch
import pandas as pd
from unsloth import FastLanguageModel
from transformers import set_seed
import random
import json
import ast

utils = os.path.abspath('./src/utils/') 
sys.path.append(utils)

from preprocessing import createPromptText, loadDataset
from config import Config
from evaluation import extractAspects, convertLabels, createResults

NUM_SHOTS = 20

def select_and_save_shots(training_examples, config):
    selected_examples = random.sample(training_examples, min(NUM_SHOTS, len(training_examples)))
    
    shots_dir = f"./shots/{config.dataset}"
    os.makedirs(shots_dir, exist_ok=True)
    
    shots_df = pd.DataFrame(selected_examples, columns=['text', 'labels'])
    shots_file = os.path.join(shots_dir, f"selected_shots_{NUM_SHOTS}.csv")
    shots_df.to_csv(shots_file, index=False, sep='\t')
    
    print(f"Selected {len(selected_examples)} shots saved to: {shots_file}")
    return selected_examples

def load_shots_from_file(config):
    shots_file = f"./shots/{config.dataset}/selected_shots_{NUM_SHOTS}.csv"

    if os.path.exists(shots_file):
        print(f"Loading existing shots from: {shots_file}")
        
        # Use converters like in loadDataset
        converters = {'labels': ast.literal_eval}
        shots_df = pd.read_csv(shots_file, sep='\t', converters=converters)
        
        return [(row['text'], row['labels']) for _, row in shots_df.iterrows()]
    else:
        return None

def create_few_shot_prompt(selected_shots, test_text, test_labels, config, prompt_templates):
    base_prompt, _ = createPromptText(
        lang=config.lang,
        prompt_templates=prompt_templates,
        prompt_style=config.prompt_style,
        example_text=test_text,
        example_labels=test_labels,
        dataset_name=config.dataset,
        absa_task=config.task,
        train=False
    )
    
    # Extract just the instruction part (before "### Input:")
    instruction_part = base_prompt.split("### Input:")[0]
    
    examples_text = ""
    for i, (example_text, example_labels) in enumerate(selected_shots):
        example_prompt, _ = createPromptText(
            lang=config.lang,
            prompt_templates=prompt_templates,
            prompt_style=config.prompt_style,
            example_text=example_text,
            example_labels=example_labels,
            dataset_name=config.dataset,
            absa_task=config.task,
            train=True
        )
        
        # Extract the input/output part
        input_output = example_prompt.split("### Instruction:\n")[1].split(instruction_part.split("### Instruction:\n")[1])[1]
        examples_text += f"### Example {i+1}:{input_output}\n"
    
    few_shot_prompt = instruction_part + examples_text + f"### Input:\n{test_text}\n\n### Output:\n"
    
    return few_shot_prompt

def main():
    config = Config()
    config.dataset = 'rest-16-spanish'
    config.lang = 'spa'
    set_seed(config.seed)
    
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=2048,
        dtype=torch.float16,
        device_map="auto"
    )
    
    df_train, df_test, label_space = loadDataset(
        config.data_path, 
        config.dataset, 
        config.low_resource_setting, 
        config.task, 
        config.split, 
        config.original_split
    )
    
    base_dir = os.path.dirname(os.path.abspath('./src/utils/preprocessing.py'))
    with open(os.path.join(base_dir, f'prompts_{config.dataset.replace("-","")}.json'), encoding='utf-8') as f:
        prompt_templates = json.load(f)
    
    training_examples = [(row['text'], row['labels']) for _, row in df_train.iterrows()]
    
    print(f"Loaded {len(training_examples)} training examples")
    
    selected_shots = load_shots_from_file(config)
    
    if selected_shots is None:
        print(f"Selecting {NUM_SHOTS} random shots...")
        selected_shots = select_and_save_shots(training_examples, config)
    else:
        print(f"Using {len(selected_shots)} pre-selected shots")
    
    predictions = []
    ground_truth_labels = []
    prompts_test = []
    
    for _, row in df_test.iterrows():
        test_text = row['text']
        test_labels = row['labels'] 
        
        _, ground_truth = createPromptText(
            lang=config.lang,
            prompt_templates=prompt_templates,
            prompt_style=config.prompt_style,
            example_text=test_text,
            example_labels=test_labels,
            dataset_name=config.dataset,
            absa_task=config.task,
            train=False
        )
        ground_truth_labels.append(ground_truth)
        
        few_shot_prompt = create_few_shot_prompt(
            selected_shots,
            test_text,
            test_labels,
            config,
            prompt_templates
        )
        # print("Few-shot prompt:\n", few_shot_prompt)
        
        prompts_test.append(few_shot_prompt + tokenizer.eos_token)
        
        inputs = tokenizer(few_shot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # print("Model output:", output_text, "\n--------ENDE OUTPUT---------")
        
        # Extract only first answer
        # Why? Fine-tuned models learned to stop after generating the answer, Base models see the pattern and continue generating more examples, Same predictions.append(output_text) works differently in each context
        # Find where test input appears
        test_input_marker = f"### Input:\n{test_text}\n\n### Output:"

        if test_input_marker in output_text:
            # Split at actual test case
            parts = output_text.split(test_input_marker)
            if len(parts) > 1:
                # Take what comes after test input
                generated_part = parts[1].strip()
                # Both cases: either another "### Input:" or "### Example" follows
                first_answer = generated_part.split("### Input:")[0].strip()
                first_answer = first_answer.split("### Example")[0].strip()
                predictions.append(first_answer)
            else:
                predictions.append("[]")
        else:
            print(f"Warning: Could not find test input marker for: {test_text}")
            predictions.append("[]")

        # print("\nFirst answer only: ", first_answer)
        # print("\nGround truth: ", ground_truth)
        print(f"Processed {len(predictions)}/{len(df_test)} samples")
    
    # Same evaluation as in llama_eval.py
    print("\nEvaluating predictions...")
    
    processed_predictions = [extractAspects(p, config.task, config.prompt_style) for p in predictions]
    processed_ground_truths = [extractAspects(gt, config.task, config.prompt_style) for gt in ground_truth_labels]

    gold_labels, _ = convertLabels(processed_ground_truths, config.task, label_space)
    pred_labels, false_predictions = convertLabels(processed_predictions, config.task, label_space)

    results_asp, results_asp_pol, results_pairs, results_pol, results_phrases, results_subset_recall = createResults(pred_labels, gold_labels, label_space, config.task)

    results_path = f"./results/llama/base_model/fewshot/{config.dataset}/{NUM_SHOTS}shots"
    os.makedirs(results_path, exist_ok=True)
    pd.DataFrame({'prompt': prompts_test, 'ground_truth': ground_truth_labels, 'prediction': predictions}).to_csv(os.path.join(results_path, "predictions.csv"), index=False)
    with open(os.path.join(results_path, "metrics.json"), "w") as f:
        json.dump({
            "results_asp": results_asp,
            "results_asp_pol": results_asp_pol,
            "results_pairs": results_pairs,
            "results_pol": results_pol,
            "results_phrases": results_phrases,
            "results_subset_recall": results_subset_recall,
            "num_shots": NUM_SHOTS
        }, f, indent=2)
    
    print("Evaluation complete.")
    print("Aspect metrics:", results_asp)
    print("Aspect+Polarity metrics:", results_asp_pol)
    print("Pair metrics:", results_pairs)
    print("Polarity metrics:", results_pol)
    print("Phrase metrics:", results_phrases)
    print("Subset recall:", results_subset_recall)
    
if __name__ == "__main__":
    main()