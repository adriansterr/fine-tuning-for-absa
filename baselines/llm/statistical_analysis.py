from unittest import result
import pandas as pd
import numpy as np
from scipy import stats
import json
import os

import sys, os
sys.path.append(os.path.abspath('./src/utils/'))

from evaluation import extractAspects, convertLabels
from preprocessing import LABEL_SPACE

SIGNIFICANCE_THRESHOLD = 0.05

def load_predictions(results_path):
    predictions_df = pd.read_csv(os.path.join(results_path, "predictions.csv"))
    return predictions_df

def run_statistical_tests(predictions_df):    
    predictions = []
    ground_truths = []
    
    for _, row in predictions_df.iterrows():
        pred = extractAspects(row['prediction'], task='acsa')
        gt = extractAspects(row['ground_truth'], task='acsa')
        
        pred_conv, _ = convertLabels([pred], 'acsa', LABEL_SPACE)
        gt_conv, _ = convertLabels([gt], 'acsa', LABEL_SPACE)
        
        pred_correct = set(pred_conv[0]) == set(gt_conv[0])
        predictions.append(pred_conv[0])
        ground_truths.append(gt_conv[0])
    
    pred_correct = np.array([set(p) == set(g) for p, g in zip(predictions, ground_truths)])
    
    # Chi-square test: is the distribution of correct/incorrect predictions different from random chance (50%)?
    observed = np.array([np.sum(pred_correct), len(pred_correct) - np.sum(pred_correct)])
    expected = np.array([len(pred_correct)/2, len(pred_correct)/2]) # expected random chance -> 50%
    print(f"Chi-square test: observed = {observed}, expected = {expected}")
    chi2_stat, chi2_p = stats.chisquare(observed, expected)

    # One-sample t-test: is the model's accuracy is significantly different from random chance (50%)?
    t_stat, t_p = stats.ttest_1samp(pred_correct.astype(float), 0.5) # random -> 50%

    # Effect sizes and accuracy
    n = len(pred_correct)
    accuracy = np.sum(pred_correct) / n
    cramers_v = np.sqrt(chi2_stat / (n * 1))  # Effect size for chi-square

    return {
        "chi_square_test": {
            "statistic": float(chi2_stat),
            "p_value": float(chi2_p),
            "effect_size": float(cramers_v),
            "accuracy": float(accuracy),
            "sample_size": n,
            "interpretation": "Significant deviation from random" if chi2_p < SIGNIFICANCE_THRESHOLD else "No significant deviation"
        },
        "one_sample_ttest": {
            "statistic": float(t_stat),
            "p_value": float(t_p),
            "interpretation": "Significantly different from chance" if t_p < SIGNIFICANCE_THRESHOLD else "Not significantly different from chance"
        }
    }

if __name__ == "__main__":
    # results_path = "results/merges/meta_llama_full_spanish/pt_gl_linear_75_25"
    # results_path = "results/merges/meta_llama_full_french/linear_75_25"
    results_path = "results/merges/meta_llama_full_precision_sauerkraut/dare/dare_ties_1_0846"
    predictions_df = load_predictions(results_path)
    
    statistical_results = run_statistical_tests(predictions_df)
    
    output_path = os.path.join(results_path, "statistical_analysis.json")
    with open(output_path, "w") as f:
        json.dump(statistical_results, f, indent=2)
    
    print("\nStatistical Analysis Results:")
    print(f"Chi-square test p-value: {statistical_results['chi_square_test']['p_value']}")
    print(f"Interpretation: {statistical_results['chi_square_test']['interpretation']}")
    print(f"\nOne-sample t-test p-value: {statistical_results['one_sample_ttest']['p_value']}")
    print(f"Interpretation: {statistical_results['one_sample_ttest']['interpretation']}")