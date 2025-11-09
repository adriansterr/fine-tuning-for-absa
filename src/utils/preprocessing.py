import pandas as pd
import re
import json
import os

from ast import literal_eval

DATASETS = ['GERestaurant', 'rest-16', 'rest-16-spanish', 'rest-16-french', 'rest-16-translated-german', 'rest-16-translated-spanish', 'rest-16-translated-french']

# GERestaurant also uses 12 categories, therefore the same label space as rest-16
LABEL_SPACE = ['AMBIENCE#GENERAL:POSITIVE', 'AMBIENCE#GENERAL:NEUTRAL', 'AMBIENCE#GENERAL:NEGATIVE', 'DRINKS#PRICES:POSITIVE', 'DRINKS#PRICES:NEUTRAL', 'DRINKS#PRICES:NEGATIVE', 'DRINKS#QUALITY:POSITIVE', 'DRINKS#QUALITY:NEUTRAL', 'DRINKS#QUALITY:NEGATIVE', 'DRINKS#STYLE_OPTIONS:POSITIVE', 'DRINKS#STYLE_OPTIONS:NEUTRAL', 'DRINKS#STYLE_OPTIONS:NEGATIVE', 'FOOD#PRICES:POSITIVE', 'FOOD#PRICES:NEUTRAL', 'FOOD#PRICES:NEGATIVE', 'FOOD#QUALITY:POSITIVE', 'FOOD#QUALITY:NEUTRAL', 'FOOD#QUALITY:NEGATIVE', 'FOOD#STYLE_OPTIONS:POSITIVE', 'FOOD#STYLE_OPTIONS:NEUTRAL', 'FOOD#STYLE_OPTIONS:NEGATIVE', 'LOCATION#GENERAL:POSITIVE', 'LOCATION#GENERAL:NEUTRAL', 'LOCATION#GENERAL:NEGATIVE', 'RESTAURANT#GENERAL:POSITIVE', 'RESTAURANT#GENERAL:NEUTRAL', 'RESTAURANT#GENERAL:NEGATIVE', 'RESTAURANT#MISCELLANEOUS:POSITIVE', 'RESTAURANT#MISCELLANEOUS:NEUTRAL', 'RESTAURANT#MISCELLANEOUS:NEGATIVE', 'RESTAURANT#PRICES:POSITIVE', 'RESTAURANT#PRICES:NEUTRAL', 'RESTAURANT#PRICES:NEGATIVE', 'SERVICE#GENERAL:POSITIVE', 'SERVICE#GENERAL:NEUTRAL', 'SERVICE#GENERAL:NEGATIVE']

REGEX_ASPECTS_ACSD = r"\(([^,]+),[^,]+,\s*\"[^\"]*\"\)"
REGEX_LABELS_ACSD = r"\([^,]+,\s*([^,]+)\s*,\s*\"[^\"]*\"\s*\)"
REGEX_ASPECTS = r"\(([^,]+),[^)]+\)"
REGEX_LABELS = r"\([^,]+,\s*([^)]+)\)"

# Adapted from https://github.com/JakobFehle/Fine-Tuning-LLMs-for-ABSA/blob/main/src/utils/preprocessing.py

def loadDataset(data_path, dataset_name, low_resource_setting, task, split = 0,  original_split = False):
    if dataset_name not in DATASETS:
        print('Dataset name not valid!')
        return None, None, None

    if dataset_name == 'GERestaurant':
        path_train = f'{data_path}/{dataset_name}/12_categories/train_preprocessed.json'
        path_eval = f'{data_path}/{dataset_name}/12_categories/test_preprocessed.json'
        df_train = pd.read_json(path_train, lines=True).set_index('id')
        df_eval = pd.read_json(path_eval, lines=True).set_index('id')
    else:
        path_train = f'{data_path}/{dataset_name}/train.tsv'
        path_eval = f'{data_path}/{dataset_name}/test.tsv'
        converters = {'labels': literal_eval}
        df_train = pd.read_csv(path_train, sep='\t', converters=converters).set_index('id')
        df_eval = pd.read_csv(path_eval, sep='\t', converters=converters).set_index('id')

    print(f'Loading dataset ...')
    print(f'Dataset name: {dataset_name}')
    print(f'Split setting: ', 'Original' if original_split else 'Custom')
    print(f'Low Resource Setting: ', 'Full' if low_resource_setting == '0' else low_resource_setting)
    print(f'Train Length: ', len(df_train))
    print(f'Eval Length: ', len(df_eval))
    print(f'Split: {split}')
        
    return df_train, df_eval, LABEL_SPACE

def createPromptText(lang, prompt_templates, prompt_style, example_text, example_labels, dataset_name, absa_task, train = False):
    # Set templates based on prompt config
    if dataset_name not in DATASETS:
        raise NotImplementedError('Prompt template not found: Dataset name not valid.')
    else:
        dataset_name = dataset_name.replace('-','')[:6]

    if lang not in ['en', 'ger', 'spa', 'fr']:
        raise NotImplementedError('Prompt template not found: Prompt language not valid.')

    if absa_task not in ['acd', 'acsa']:
        raise NotImplementedError('Prompt template not found: Absa task not valid.')
    
    if prompt_style not in ['basic', 'context']:
        raise NotImplementedError('Prompt template not found: Prompt style not valid.')
    else:
        template_prompt_style = prompt_style

    try:
        prompt_template = prompt_templates[f'PROMPT_TEMPLATE_{lang.upper()}_{absa_task.upper()}_{dataset_name.upper()}_{template_prompt_style.upper()}']
    except:
        raise NotImplementedError('Prompt template does not exist.')    

    if example_labels is not None:
        try:
            reg_asp = re.compile(REGEX_ASPECTS_ACSD)
            reg_lab = re.compile(REGEX_LABELS_ACSD)
            
            # Extract Aspects from sample
            re_aspects = [reg_asp.match(pair)[1] for pair in example_labels]
            
            # Extract Polarities from sample
            re_labels = [reg_lab.match(pair)[1] for pair in example_labels]
            
        except:
            try:
                reg_asp = re.compile(REGEX_ASPECTS)
                reg_lab = re.compile(REGEX_LABELS)
                
                # Extract Aspects from  sample
                re_aspects = [reg_asp.match(pair)[1] for pair in example_labels]
                
                # Extract Polarities from  sample
                re_labels = [reg_lab.match(pair)[1] for pair in example_labels]
        
            except:
                raise NotImplementedError("Data-Format is not ['(Aspect Category, Sentiment Polarity, Aspect Phrase)', '(Aspect Category, Sentiment Polarity, Aspect Phrase)']")
        
        if absa_task == 'acd':        
            example_labels = re_aspects
    
        elif absa_task == 'acsa':
            example_labels = [re_aspects, re_labels]

    prompt = '### Instruction:\n' + prompt_template[0] + ' ' + prompt_template[1] + '\n\n' + f'### Input:\n{example_text} \n\n### Output:\n' 

    # Determine if train or test prompt and append target label if necessary
    if absa_task == 'acd':
        target_text = '[' + ', '.join(example_labels) + ']'
    elif absa_task == 'acsa':
         target_text = '[' + ', '.join([f'({example_labels[0][i]}, {example_labels[1][i]})' for i in range(len(example_labels[0]))]) + ']'

    if train:
        return prompt + target_text, None
    else:
        return prompt, target_text
        
def createPrompts(df_train, df_test, args, eos_token = ''):
    prompts_train = []
    prompts_test = []
    ground_truth_labels = []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset in DATASETS:
        prompt_file_dataset = args.dataset.replace("translated-", "").replace("-", "")
        with open(os.path.join(base_dir, f'prompts_{prompt_file_dataset}.json'), encoding='utf-8') as json_prompts:
            prompt_templates = json.load(json_prompts)
    else:
        raise NotImplementedError('Prompt-Template not found: File does not exist.')

    for _, row in df_train.iterrows():       
        prompt, _ = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row['labels'], dataset_name = args.dataset, absa_task = args.task, train = True) 
        prompts_train.append(prompt + eos_token)

    for _, row in df_test.iterrows():
        prompt, targets = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row['labels'], dataset_name = args.dataset, absa_task = args.task)
        prompts_test.append(prompt + eos_token)
        ground_truth_labels.append(targets)
        
    return prompts_train, prompts_test, ground_truth_labels