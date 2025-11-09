# Beyond Fine-Tuning: Improving Low-Resource Aspect-based Sentiment Analysis through LLM Parameter Fusion

## Overview

This repository contains the code and experimental framework for investigating cross-lingual model merging as an approach to enable aspect-based sentiment analysis (ABSA) in low-resource language settings where annotated training datasets are unavailable. The research explores whether merging parameters from an English ABSA-fine-tuned model with language-specific models in German, French and Spanish can achieve competitive performance without requiring language-specific ABSA training data.

## Key Findings

- Model merging achieves performance comparable to strong cross-lingual baselines for German and French ABSA tasks.

- German merged model (Arcee Fusion) reaches F1=0.7704, showing no significant difference from the English ABSA model with German prompts (p=0.051).

- French merged model (Linear 75/25) achieves F1=0.7234, performing comparably to both cross-lingual transfer baselines.

- Spanish results (F1=0.6636) reveal limitations of the approach, with all baselines significantly outperforming the merged model, providing valuable insights into when cross-lingual merging fails.

- Merged models excel on frequent aspect categories (SERVICE#GENERAL, FOOD#QUALITY) but struggle with rare categories (RESTAURANT#MISCELLANEOUS).

- Statistical validation through McNemar's test with Holm-Bonferroni correction and bootstrap confidence intervals (1,000 iterations) ensures robust comparisons.

- Key performance summary:
  - German: F1=0.7704 [0.75, 0.79] - comparable to best baseline
  - French: F1=0.7234 [0.70, 0.75] - comparable to cross-lingual baselines
  - Spanish: F1=0.6636 [0.64, 0.69] - significantly worse than all baselines

## Methodology

Arcee's Mergekit framework is employed to combine model parameters from English ABSA fine-tuned models with language-specific models across German, French, and Spanish. Multiple merging algorithms are systematically explored, including Linear, SLERP, TIES, DARE, DELLA, Model Breadcrumbs, and Arcee Fusion. Merged models are compared against three baselines: (1) English ABSA model with language-specific prompts, (2) fine-tuning on machine-translated datasets, and (3) few-shot prompting with the base model. Statistical significance is assessed using McNemar's test with Holm-Bonferroni correction for multiple comparisons, while bootstrap resampling provides 95% confidence intervals for F1 scores. Category-level analysis examines performance across 12 aspect categories to identify strengths and weaknesses of the merging approach.

## Datasets

The following datasets are used in the evaluation:

- Rest-16 (English): Restaurant review dataset from SemEval 2016 (Pontiki et al., 2016)
- Rest-16 (French): French restaurant reviews from SemEval 2016
- Rest-16 (Spanish): Spanish restaurant reviews from SemEval 2016
- GERestaurant: German restaurant review dataset (Hellwig et al., 2024)
- Machine-translated versions: DeepL translations of Rest-16 to German, French, and Spanish

## Repository Structure

```
root/
├── baselines/llm/                          # Baseline implementations and analysis
│   ├── llama_qlora.py                      # QLoRA fine-tuning implementation
│   ├── llama_eval.py                       # Model evaluation script
│   ├── few_shot_prompting_eval.py          # Few-shot baseline
│   ├── pairwise_statistical_analysis.py    # Statistical evaluation
│   └── find_best_approach.py               # Best merge selection
├── data/                                   # Datasets and preprocessed files
│   ├── GERestaurant/                       # German restaurant reviews
│   ├── rest-16/                            # SemEval-2016 English dataset
│   ├── rest-16-french/                     # SemEval-2016 French dataset
│   ├── rest-16-spanish/                    # SemEval-2016 Spanish dataset
│   └── rest-16-translated-*/               # Machine-translated datasets
├── merging_configurations/                 # YAML-configuration files from the merging process
├── results/                                # Experimental results and analysis
│   ├── llama/                              # Baseline model results
│   └── merges/                             # Model merge results by language
├── shots/                                  # Few-shot example selections
├── src/                                    # Preprocessing and utility modules
│   ├── preprocess_gerestaurant.py          # German dataset preprocessing
│   ├── preprocess_semeval_dataset.py       # SemEval dataset preprocessing
│   ├── translate_dataset.py                # DeepL machine translation
│   ├── extract_f1_scores.py                # F1 score extraction
└── └── utils/                              # Configuration and evaluation tools
```

## Running the Code

### Prerequisites

Install required packages (Python 3.10+):
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers datasets trl pandas numpy scipy vllm mergekit deepl
```

Set environment variables:
```bash
# Windows
set PYTHONUTF8=1
set HF_HUB_ENABLE_HF_TRANSFER=0

# Linux/Mac
export PYTHONUTF8=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### Execution Pipeline

1. Preprocess datasets:
```bash
python src/preprocess_semeval_dataset.py
python src/preprocess_gerestaurant.py
```

2. Fine-tune base models:
```bash
python baselines/llm/llama_qlora.py
```

3. Create model merges:
```bash
mergekit-yaml merge_config.yaml output/merged_model --allow-crimes --lazy-unpickle
```

4. Evaluate models:
```bash
python baselines/llm/llama_eval.py
```

5. Run statistical analysis:
```bash
python baselines/llm/pairwise_statistical_analysis.py
python baselines/llm/find_best_approach.py
```

## Baselines

### Baseline 1: Cross-lingual Prompting
English ABSA model with language-specific prompts for zero-shot cross-lingual transfer.

Performance:
- German: F1=0.7664 [0.74, 0.79]
- French: F1=0.7145 [0.69, 0.74]
- Spanish: F1=0.7445 [0.72, 0.77]

### Baseline 2: Machine-Translated Dataset
Fine-tuning on DeepL-translated English to target language datasets.

Performance:
- German: F1=0.7486 [0.73, 0.77]
- French: F1=0.6921 [0.66, 0.72]
- Spanish: F1=0.6789 [0.65, 0.71]

### Baseline 3: Few-Shot Prompting
Base Llama-3.1-8B-Instruct with 20-shot in-context examples (no fine-tuning).

Performance:
- German: F1=0.6355 [0.61, 0.66]
- French: F1=0.5225 [0.49, 0.56]
- Spanish: F1=0.6158 [0.59, 0.65]