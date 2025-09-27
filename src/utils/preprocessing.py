import pandas as pd
import re
import json
import os

from itertools import product
from ast import literal_eval

DATASETS = ['GERestaurant', 'rest-16', 'rest-16-translated-german', 'rest-16-translated-spanish', 'rest-16-translated-french']

# Für GERestaurant werden die 12 Kategorien verwendet, deswegen gleicher Label Space wie bei rest-16
LABEL_SPACE = ['AMBIENCE#GENERAL:POSITIVE', 'AMBIENCE#GENERAL:NEUTRAL', 'AMBIENCE#GENERAL:NEGATIVE', 'DRINKS#PRICES:POSITIVE', 'DRINKS#PRICES:NEUTRAL', 'DRINKS#PRICES:NEGATIVE', 'DRINKS#QUALITY:POSITIVE', 'DRINKS#QUALITY:NEUTRAL', 'DRINKS#QUALITY:NEGATIVE', 'DRINKS#STYLE_OPTIONS:POSITIVE', 'DRINKS#STYLE_OPTIONS:NEUTRAL', 'DRINKS#STYLE_OPTIONS:NEGATIVE', 'FOOD#PRICES:POSITIVE', 'FOOD#PRICES:NEUTRAL', 'FOOD#PRICES:NEGATIVE', 'FOOD#QUALITY:POSITIVE', 'FOOD#QUALITY:NEUTRAL', 'FOOD#QUALITY:NEGATIVE', 'FOOD#STYLE_OPTIONS:POSITIVE', 'FOOD#STYLE_OPTIONS:NEUTRAL', 'FOOD#STYLE_OPTIONS:NEGATIVE', 'LOCATION#GENERAL:POSITIVE', 'LOCATION#GENERAL:NEUTRAL', 'LOCATION#GENERAL:NEGATIVE', 'RESTAURANT#GENERAL:POSITIVE', 'RESTAURANT#GENERAL:NEUTRAL', 'RESTAURANT#GENERAL:NEGATIVE', 'RESTAURANT#MISCELLANEOUS:POSITIVE', 'RESTAURANT#MISCELLANEOUS:NEUTRAL', 'RESTAURANT#MISCELLANEOUS:NEGATIVE', 'RESTAURANT#PRICES:POSITIVE', 'RESTAURANT#PRICES:NEUTRAL', 'RESTAURANT#PRICES:NEGATIVE', 'SERVICE#GENERAL:POSITIVE', 'SERVICE#GENERAL:NEUTRAL', 'SERVICE#GENERAL:NEGATIVE']

TRANSLATE_POLARITIES_EN = {'POSITIVE': 'positively', 'NEUTRAL': 'neutrally', 'NEGATIVE': 'negatively'}
TRANSLATE_POLARITIES_GER = {'POSITIVE': 'positiv', 'NEUTRAL': 'neutral', 'NEGATIVE': 'negativ'}

REGEX_ASPECTS_ACSD = r"\(([^,]+),[^,]+,\s*\"[^\"]*\"\)"
REGEX_LABELS_ACSD = r"\([^,]+,\s*([^,]+)\s*,\s*\"[^\"]*\"\s*\)"
REGEX_PHRASES_ACSD = r"\([^,]+,\s*[^,]+\s*,\s*\"([^\"]*)\"\s*\)"
REGEX_ASPECTS = r"\(([^,]+),[^)]+\)"
REGEX_LABELS = r"\([^,]+,\s*([^)]+)\)"

def raise_err(ex):
    raise ex

def loadDataset(data_path, dataset_name, low_resource_setting, task, split = 0,  original_split = False):
    if dataset_name not in DATASETS:
        print('Dataset name not valid!')
        return None, None, None

    DATASET_PATH = data_path if task != 'e2e-e' else data_path + '/data_e2e_e'
    fn_suffix = '_full' if low_resource_setting == 0 else f'_{low_resource_setting}'

    if dataset_name == 'GERestaurant':
        path_train = f'{DATASET_PATH}/{dataset_name}/12_categories/train_preprocessed.json'
        path_eval = f'{DATASET_PATH}/{dataset_name}/12_categories/test_preprocessed.json'
        df_train = pd.read_json(path_train, lines=True).set_index('id')
        df_eval = pd.read_json(path_eval, lines=True).set_index('id')
    else:
        if original_split:
            path_train = f'{DATASET_PATH}/{dataset_name}/train.tsv'
            path_eval = f'{DATASET_PATH}/{dataset_name}/test.tsv'
        else:
            if split != 0: # CV Test Phase
                path_train = f'{DATASET_PATH}/{dataset_name}/split_{split}/train{fn_suffix}.tsv'
                path_eval = f'{DATASET_PATH}/{dataset_name}/split_{split}/test_full.tsv'
            else: # HT Phase
                path_train = f'{DATASET_PATH}/{dataset_name}/train{fn_suffix}.tsv'
                path_eval = f'{DATASET_PATH}/{dataset_name}/val{fn_suffix}.tsv'
        converters = {'labels': literal_eval, 'labels_phrases': literal_eval}
        df_train = pd.read_csv(path_train, sep='\t', converters=converters).set_index('id')
        df_eval = pd.read_csv(path_eval, sep='\t', converters=converters).set_index('id')

    print(f'Loading dataset ...')
    print(f'Dataset name: {dataset_name}')
    print(f'Split setting: ', 'Original' if original_split else 'Custom')
    print(f'Eval Mode: ', 'Test' if split != 0 else 'Validation')
    print(f'Low Resource Setting: ', 'Full' if low_resource_setting == '0' else low_resource_setting)
    print(f'Train Length: ', len(df_train))
    print(f'Eval Length: ', len(df_eval))
    print(f'Split: {split}')
        
    return df_train, df_eval, LABEL_SPACE

def createPromptText(lang, prompt_templates, prompt_style, example_text, example_labels, dataset_name = 'rest-16', absa_task = 'acd', train = False):

    # Set templates based on prompt config
    if dataset_name not in DATASETS:
        raise NotImplementedError('Prompt template not found: Dataset name not valid.')
    else:
        dataset_name = dataset_name.replace('-','')[:6]

    if lang not in ['en', 'ger', 'spa', 'fr']:
        raise NotImplementedError('Prompt template not found: Prompt language not valid.')

    if absa_task not in ['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']:
        raise NotImplementedError('Prompt template not found: Absa task not valid.')
    
    if prompt_style not in ['basic', 'context']:
        raise NotImplementedError('Prompt template not found: Prompt style not valid.')
    else:
        template_prompt_style = prompt_style

    try:
        prompt_template = prompt_templates[f'PROMPT_TEMPLATE_{lang.upper()}_{"E2E" if absa_task == "e2e-e" else absa_task.upper()}_{dataset_name.upper()}_{template_prompt_style.upper()}']
    except:
        raise NotImplementedError('Prompt template does not exist.')    
    

    if example_labels is not None:
        try:
            reg_asp = re.compile(REGEX_ASPECTS_ACSD)
            reg_lab = re.compile(REGEX_LABELS_ACSD)
            reg_phr = re.compile(REGEX_PHRASES_ACSD)
            
            # Extract Aspects from  sample
            re_aspects = [reg_asp.match(pair)[1] for pair in example_labels]
            
            # Extract Polarities from  sample
            re_labels = [reg_lab.match(pair)[1] for pair in example_labels]
        
            # Extract Aspect-Phrases from  sample
            re_phrases = [reg_phr.match(pair)[1] for pair in example_labels]
            
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
            
        elif absa_task == 'e2e' or absa_task == 'e2e-e':
            example_labels = [re_phrases, re_labels]
            
        elif absa_task == 'tasd':
            example_labels = [re_aspects, re_labels, re_phrases]

    prompt = '### Instruction:\n' + prompt_template[0] + ' ' + prompt_template[1] + '\n\n' + f'### Input:\n{example_text} \n\n### Output:\n' 

    # Determine if train or test prompt and append target label if necessary

    if absa_task == 'acd':
        target_text = '[' + ', '.join(example_labels) + ']'
    elif absa_task == 'acsa':
         target_text = '[' + ', '.join([f'({example_labels[0][i]}, {example_labels[1][i]})' for i in range(len(example_labels[0]))]) + ']'
    elif absa_task == 'e2e' or absa_task == 'e2e-e':
         target_text = '[' + ', '.join([f'("{example_labels[0][i]}", {example_labels[1][i]})' for i in range(len(example_labels[0]))]) + ']'
    elif absa_task == 'tasd':
         target_text = '[' + ', '.join([f'({example_labels[0][i]}, {example_labels[1][i]}, "{example_labels[2][i]}")' for i in range(len(example_labels[0]))]) + ']'

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
        with open(os.path.join(base_dir, f'prompts_{args.dataset.replace("-","")}.json'), encoding='utf-8') as json_prompts:
             prompt_templates = json.load(json_prompts)
    else:
        raise NotImplementedError('Prompt-Template not found: File does not exist.')

    label_column = 'labels_phrases' if 'labels_phrases' in df_train.columns else 'labels' if 'labels' in df_train.columns else lambda x: raise_err(NotImplementedError('Dataset does not have column with label targets.'))

    for index, row in df_train.iterrows():       

        prompt, _ = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row[label_column], dataset_name = args.dataset, absa_task = args.task, train = True) 

        prompts_train.append(prompt + eos_token)

    for index, row in df_test.iterrows():
        prompt, targets = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row[label_column], dataset_name = args.dataset, absa_task = args.task)
        
        prompts_test.append(prompt + eos_token)
        ground_truth_labels.append(targets)
        
    return prompts_train, prompts_test, ground_truth_labels