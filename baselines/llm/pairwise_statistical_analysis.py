import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import json
import os
from itertools import combinations
import sys

sys.path.append(os.path.abspath('./src/utils/'))
from evaluation import extractAspects, convertLabels, calculateMetrics
from preprocessing import LABEL_SPACE

SIGNIFICANCE_THRESHOLD = 0.05
BOOTSTRAP_ITERATIONS = 1000

def load_predictions(results_path):
    predictions_path = os.path.join(results_path, "predictions.csv")
    if not os.path.exists(predictions_path):
        return None
    return pd.read_csv(predictions_path)


def compute_sample_correctness(predictions_df, task='acsa'):
    correctness = []
    for _, row in predictions_df.iterrows():
        pred = extractAspects(row['prediction'], task=task)
        gt = extractAspects(row['ground_truth'], task=task)
        
        pred_conv, _ = convertLabels([pred], task, LABEL_SPACE)
        gt_conv, _ = convertLabels([gt], task, LABEL_SPACE)
        
        correctness.append(1 if set(pred_conv[0]) == set(gt_conv[0]) else 0)
    
    return np.array(correctness)


def mcnemar_test(correctness_a, correctness_b):
    both_correct = np.sum((correctness_a == 1) & (correctness_b == 1))
    a_correct_b_wrong = np.sum((correctness_a == 1) & (correctness_b == 0))
    a_wrong_b_correct = np.sum((correctness_a == 0) & (correctness_b == 1))
    both_wrong = np.sum((correctness_a == 0) & (correctness_b == 0))
    
    table = np.array([[both_correct, a_correct_b_wrong],
                      [a_wrong_b_correct, both_wrong]])
    
    if a_correct_b_wrong + a_wrong_b_correct == 0:
        return 0.0, 1.0, 0.0
    
    result = mcnemar(table, exact=True, correction=True)
    effect_size = np.sqrt(result.statistic / len(correctness_a)) if len(correctness_a) > 0 else 0.0
    
    return result.statistic, result.pvalue, effect_size


def paired_t_test(correctness_a, correctness_b):
    diff = correctness_a - correctness_b
    if len(diff) == 0 or np.std(diff, ddof=1) == 0:
        return 0.0, 1.0, 0.0
    
    t_stat, p_value = stats.ttest_rel(correctness_a, correctness_b)
    effect_size = np.mean(diff) / np.std(diff, ddof=1)
    
    return t_stat, p_value, effect_size


def bootstrap_confidence_interval(predictions_df, task='acsa', n_iterations=BOOTSTRAP_ITERATIONS, confidence=0.95):
    f1_scores = []
    n = len(predictions_df)
    
    for _ in range(n_iterations):
        boot_df = predictions_df.sample(n=n, replace=True, random_state=None)
        
        predictions = []
        ground_truths = []
        for _, row in boot_df.iterrows():
            pred = extractAspects(row['prediction'], task=task)
            gt = extractAspects(row['ground_truth'], task=task)
            
            pred_conv, _ = convertLabels([pred], task, LABEL_SPACE)
            gt_conv, _ = convertLabels([gt], task, LABEL_SPACE)
            
            predictions.append(pred_conv[0])
            ground_truths.append(gt_conv[0])
        
        metrics = calculateMetrics(predictions, ground_truths)
        f1_scores.append(metrics['f1'])
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(f1_scores, lower_percentile)
    ci_upper = np.percentile(f1_scores, upper_percentile)
    
    return {
        'mean': float(np.mean(f1_scores)),
        'std': float(np.std(f1_scores)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence': confidence
    }


def holm_bonferroni_correction(p_values, alpha=SIGNIFICANCE_THRESHOLD):
    if not p_values:
        return []
    
    comparison_names = [item[0] for item in p_values]
    pvals = np.array([item[1] for item in p_values])
    
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method='holm', is_sorted=False, returnsorted=False)
    
    sorted_indices = np.argsort(pvals)
    n = len(pvals)
    adjusted_alphas = np.array([alpha / (n - rank) for rank, _ in enumerate(sorted_indices)])
    
    results = [
        {
            'comparison': comparison_names[i],
            'p_value': float(pvals[i]),
            'adjusted_alpha': float(adjusted_alphas[sorted_indices.tolist().index(i)]),
            'is_significant': bool(reject[i]),
            'rank': i + 1
        }
        for i in range(n)
    ]
    
    results.sort(key=lambda x: x['p_value'])
    for i, result in enumerate(results):
        result['rank'] = i + 1
    
    return results


def compare_all_merges(merges_dict, task='acsa', alpha=SIGNIFICANCE_THRESHOLD):
    correctness_data = {}
    f1_scores = {}
    
    for name, results_path in merges_dict.items():
        predictions_df = load_predictions(results_path)
        if predictions_df is None:
            print(f"Warning: Could not load predictions from {results_path}")
            continue
        
        correctness_data[name] = compute_sample_correctness(predictions_df, task=task)
        
        metrics_path = os.path.join(results_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                f1_scores[name] = metrics.get('results_asp_pol', {}).get('Micro-AVG', {}).get('f1', 0.0)
        else:
            f1_scores[name] = 0.0
    
    if len(correctness_data) < 2:
        return "Error: Need at least 2 merging approaches to compare"
    
    pairwise_results = []
    p_values_for_correction = []
    
    merging_approach_names = list(correctness_data.keys())
    
    for name_a, name_b in combinations(merging_approach_names, 2):
        correctness_a = correctness_data[name_a]
        correctness_b = correctness_data[name_b]
        
        min_len = min(len(correctness_a), len(correctness_b))
        correctness_a = correctness_a[:min_len]
        correctness_b = correctness_b[:min_len]
        
        mcnemar_stat, mcnemar_p, mcnemar_effect = mcnemar_test(correctness_a, correctness_b)
        t_test_stat, t_test_p, t_test_effect = paired_t_test(correctness_a, correctness_b)
        
        comparison_name = f"{name_a} vs {name_b}"
        
        pairwise_results.append({
            'comparison': comparison_name,
            'merge_a': name_a,
            'merge_b': name_b,
            'accuracy_a': float(np.mean(correctness_a)),
            'accuracy_b': float(np.mean(correctness_b)),
            'f1_a': f1_scores.get(name_a, 0.0),
            'f1_b': f1_scores.get(name_b, 0.0),
            'mcnemar_test': {
                'statistic': float(mcnemar_stat),
                'p_value': float(mcnemar_p),
                'effect_size': float(mcnemar_effect),
                'interpretation': 'Significant difference' if mcnemar_p < alpha else 'No significant difference'
            },
            'paired_t_test': {
                'statistic': float(t_test_stat),
                'p_value': float(t_test_p),
                'effect_size_cohens_d': float(t_test_effect),
                'interpretation': 'Significant difference' if t_test_p < alpha else 'No significant difference'
            }
        })
        
        p_values_for_correction.append((comparison_name, mcnemar_p))
    
    holm_bonferroni_results = holm_bonferroni_correction(p_values_for_correction, alpha=alpha)
    
    best_merging_approach_by_f1 = max(f1_scores.items(), key=lambda x: x[1])
    
    significantly_best = None
    for merge in merging_approach_names:
        is_best = True
        for result in holm_bonferroni_results:
            if merge in result['comparison'] and result['is_significant']:
                parts = result['comparison'].split(' vs ')
                idx = 0 if merge == parts[0] else 1
                
                pairwise = next((r for r in pairwise_results if r['comparison'] == result['comparison']), None)
                if pairwise:
                    acc_this = pairwise[f'accuracy_{"a" if idx == 0 else "b"}']
                    acc_other = pairwise[f'accuracy_{"b" if idx == 0 else "a"}']
                    
                    if acc_this < acc_other:
                        is_best = False
                        break
        
        if is_best:
            significantly_best = merge
            break
    
    if significantly_best:
        selected_merging_approach = significantly_best
        selection_method = "statistical_significance"
        recommendation_text = f"Significantly best merge: {significantly_best}"
    else:
        selected_merging_approach = best_merging_approach_by_f1[0]
        selection_method = "highest_f1_score"
        recommendation_text = f"No significantly best merge found. Recommend selecting based on F1 score: {best_merging_approach_by_f1[0]} (F1={best_merging_approach_by_f1[1]:.4f})"
    
    return {
        'pairwise_comparisons': pairwise_results,
        'holm_bonferroni_correction': holm_bonferroni_results,
        'summary': {
            'total_comparisons': int(len(pairwise_results)),
            'significant_comparisons_uncorrected': int(sum(1 for r in pairwise_results if r['mcnemar_test']['p_value'] < alpha)),
            'significant_comparisons_corrected': int(sum(1 for r in holm_bonferroni_results if r['is_significant'])),
            'significantly_best_merge': significantly_best,
            'best_by_f1': {
                'merge': str(best_merging_approach_by_f1[0]),
                'f1_score': float(best_merging_approach_by_f1[1])
            },
            'selected_merge': str(selected_merging_approach),
            'selection_method': selection_method,
            'all_f1_scores': {k: float(v) for k, v in f1_scores.items()}
        },
        'recommendation': recommendation_text
    }


def compare_merge_vs_baselines(merge_path, baseline_paths_dict, task='acsa', alpha=SIGNIFICANCE_THRESHOLD):
    merge_predictions = load_predictions(merge_path)
    if merge_predictions is None:
        return {'error': f'Could not load predictions from {merge_path}'}
    
    merge_correctness = compute_sample_correctness(merge_predictions, task=task)
    
    metrics_path = os.path.join(merge_path, "metrics.json")
    merge_f1 = 0.0
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            merge_f1 = metrics.get('results_asp_pol', {}).get('Micro-AVG', {}).get('f1', 0.0)
    
    baseline_results = []
    p_values_for_correction = []
    
    for baseline_name, baseline_path in baseline_paths_dict.items():
        baseline_predictions = load_predictions(baseline_path)
        if baseline_predictions is None:
            print(f"Warning: Could not load baseline {baseline_name} from {baseline_path}")
            continue
        
        baseline_correctness = compute_sample_correctness(baseline_predictions, task=task)
        
        baseline_metrics_path = os.path.join(baseline_path, "metrics.json")
        baseline_f1 = 0.0
        if os.path.exists(baseline_metrics_path):
            with open(baseline_metrics_path, 'r') as f:
                metrics = json.load(f)
                baseline_f1 = metrics.get('results_asp_pol', {}).get('Micro-AVG', {}).get('f1', 0.0)
        
        min_len = min(len(merge_correctness), len(baseline_correctness))
        merge_correct = merge_correctness[:min_len]
        baseline_correct = baseline_correctness[:min_len]
        
        mcnemar_stat, mcnemar_p, mcnemar_effect = mcnemar_test(merge_correct, baseline_correct)
        t_test_stat, t_test_p, t_test_effect = paired_t_test(merge_correct, baseline_correct)
        
        # Bootstrap analysis - F1 confidence intervals only
        merge_predictions_subset = merge_predictions.iloc[:min_len]
        baseline_predictions_subset = baseline_predictions.iloc[:min_len]
        
        merge_bootstrap = bootstrap_confidence_interval(merge_predictions_subset, task=task)
        baseline_bootstrap = bootstrap_confidence_interval(baseline_predictions_subset, task=task)
        
        merge_accuracy = np.mean(merge_correct)
        baseline_accuracy = np.mean(baseline_correct)
        
        comparison_name = f"Merge vs {baseline_name}"
        
        baseline_results.append({
            'baseline': baseline_name,
            'comparison': comparison_name,
            'merge_accuracy': float(merge_accuracy),
            'baseline_accuracy': float(baseline_accuracy),
            'merge_f1': merge_f1,
            'baseline_f1': baseline_f1,
            'improvement_accuracy': float(merge_accuracy - baseline_accuracy),
            'improvement_f1': float(merge_f1 - baseline_f1),
            'mcnemar_test': {
                'statistic': float(mcnemar_stat),
                'p_value': float(mcnemar_p),
                'effect_size': float(mcnemar_effect),
                'interpretation': 'Significant difference' if mcnemar_p < alpha else 'No significant difference'
            },
            'paired_t_test': {
                'statistic': float(t_test_stat),
                'p_value': float(t_test_p),
                'effect_size_cohens_d': float(t_test_effect),
                'interpretation': 'Significant difference' if t_test_p < alpha else 'No significant difference'
            },
            'bootstrap_analysis': {
                'merge_ci': merge_bootstrap,
                'baseline_ci': baseline_bootstrap,
            }
        })
        
        p_values_for_correction.append((comparison_name, mcnemar_p))
    
    holm_bonferroni_results = holm_bonferroni_correction(p_values_for_correction, alpha=alpha)
    
    return {
        'baseline_comparisons': baseline_results,
        'holm_bonferroni_correction': holm_bonferroni_results,
        'summary': {
            'total_baselines': int(len(baseline_results)),
            'significantly_better_than_baselines_uncorrected': int(sum(
                1 for r in baseline_results 
                if r['mcnemar_test']['p_value'] < alpha and r['improvement_accuracy'] > 0
            )),
            'significantly_better_than_baselines_corrected': int(sum(
                1 for r in holm_bonferroni_results 
                if r['is_significant'] and any(
                    b['comparison'] == r['comparison'] and b['improvement_accuracy'] > 0 
                    for b in baseline_results
                )
            )),
            'merge_f1': float(merge_f1),
            'best_baseline_f1': float(max([r['baseline_f1'] for r in baseline_results])) if baseline_results else 0.0
        }
    }


def find_merges(base_path):
    merge_approaches = {}
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        if os.path.exists(os.path.join(item_path, "predictions.csv")):
            merge_approaches[item] = item_path
        else:
            try:
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.exists(os.path.join(subitem_path, "predictions.csv")):
                        combined_name = f"{item}_{subitem}"
                        merge_approaches[combined_name] = subitem_path
            except (OSError, NotADirectoryError):
                continue
    
    return merge_approaches


def compare_all_merges_of_language(language_name, base_path, task='acsa', alpha=SIGNIFICANCE_THRESHOLD):
    print(f"\nBase path: {base_path}")
    merge_approaches = find_merges(base_path)
    
    if not merge_approaches:
        print(f"No merge configurations found with predictions.csv in {base_path}")
        return None
    
    print(f"Found {len(merge_approaches)} merge configurations:")
    for name in sorted(merge_approaches.keys()):
        print(f"  - {name}")
    
    print(f"\nRunning pairwise comparisons with Holm-Bonferroni correction...")
    merge_comparison_results = compare_all_merges(merge_approaches, task=task, alpha=alpha)
    
    output_path = os.path.join(base_path, 'merge_comparison_analysis.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merge_comparison_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print(f"\nTop 5 configurations by F1 score:")
    sorted_f1 = sorted(merge_comparison_results['summary']['all_f1_scores'].items(), 
                       key=lambda x: x[1], reverse=True)[:5]
    for i, (name, f1) in enumerate(sorted_f1, 1):
        print(f"   {i}. {name}: F1={f1:.4f}")
    
    return merge_comparison_results


if __name__ == "__main__":
    languages = {
        'French': 'results/merges/meta_llama_full_french',
        'German': 'results/merges/meta_llama_full_german',
        'Spanish': 'results/merges/meta_llama_full_spanish'
    }
    
    all_results = {}
    

    # PART 1: Run comparison for merge configurations
    
    for language_name, base_path in languages.items():
        result = compare_all_merges_of_language(language_name, base_path, task='acsa', alpha=SIGNIFICANCE_THRESHOLD)
        if result:
            all_results[language_name] = result
    
    combined_summary = {}
    for language, result in all_results.items():
        if 'error' not in result:
            combined_summary[language] = {
                'selected_merge': result['summary']['selected_merge'],
                'selection_method': result['summary']['selection_method'],
                'best_by_f1': result['summary']['best_by_f1'],
                'significantly_best': result['summary']['significantly_best_merge'],
                'total_configs': len(result['summary']['all_f1_scores']),
                'significant_comparisons': result['summary']['significant_comparisons_corrected'],
                'total_comparisons': result['summary']['total_comparisons']
            }
    
    combined_output_path = 'results/merges/combined_language_comparison_summary.json'
    with open(combined_output_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    print(f"\nCombined summary saved to: {combined_output_path}")


    # PART 2: Run comparison for selected merges vs baselines
    
    baselines_config = {
        'French': {
            'selected_merge_name': combined_summary['French']['selected_merge'],
            'baselines': {
                'Baseline 1: English fine-tune with French prompts': 'results/llama/baseline_1/meta_llama_full_english_finetune/fr_prompts',
                'Baseline 2: French translated dataset': 'results/llama/baseline_2/meta_llama_translated_dataset_french',
                'Baseline 3: Few-shot (20-shot)': 'results/llama/baseline_3/fewshot_20/rest-16-french'
            }
        },
        'German': {
            'selected_merge_name': combined_summary['German']['selected_merge'],
            'baselines': {
                'Baseline 1: English fine-tune with German prompts': 'results/llama/baseline_1/meta_llama_full_english_finetune/ger_prompts',
                'Baseline 2: German translated dataset': 'results/llama/baseline_2/meta_llama_translated_dataset_german',
                'Baseline 3: Few-shot (20-shot)': 'results/llama/baseline_3/fewshot_20/GERestaurant'
            }
        },
        'Spanish': {
            'selected_merge_name': combined_summary['Spanish']['selected_merge'],
            'baselines': {
                'Baseline 1: English fine-tune with Spanish prompts': 'results/llama/baseline_1/meta_llama_full_english_finetune/spa_prompts',
                'Baseline 2: Spanish translated dataset': 'results/llama/baseline_2/meta_llama_translated_dataset_spanish',
                'Baseline 3: Few-shot (20-shot)': 'results/llama/baseline_3/fewshot_20/rest-16-spanish'
            }
        }
    }
    
    baseline_comparison_results = {}
    
    for language_name, config in baselines_config.items():
        print(f"\n{'='*60}")
        print(f"Comparing {language_name}: {config['selected_merge_name']}")
        print(f"{'='*60}")
        
        base_merge_path = languages[language_name]
        merge_approaches = find_merges(base_merge_path)
        
        selected_merge_name = config['selected_merge_name']
        if selected_merge_name not in merge_approaches:
            print(f"Selected merge '{selected_merge_name}' not found in merge results")
            continue
        
        selected_merge_path = merge_approaches[selected_merge_name]
        
        baseline_result = compare_merge_vs_baselines(
            selected_merge_path,
            config['baselines'],
            task='acsa',
            alpha=SIGNIFICANCE_THRESHOLD
        )
        
        baseline_comparison_results[language_name] = baseline_result
        
        output_path = os.path.join(base_merge_path, f'baseline_comparison_{language_name.lower()}.json')
        with open(output_path, 'w') as f:
            json.dump(baseline_result, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    combined_baseline_summary = {}
    for language, result in baseline_comparison_results.items():
        if 'error' not in result:
            combined_baseline_summary[language] = {
                'selected_merge': baselines_config[language]['selected_merge_name'],
                'merge_f1': result['summary']['merge_f1'],
                'best_baseline_f1': result['summary']['best_baseline_f1'],
                'improvement_over_best_baseline': result['summary']['merge_f1'] - result['summary']['best_baseline_f1'],
                'total_baselines': result['summary']['total_baselines'],
                'significantly_better_uncorrected': result['summary']['significantly_better_than_baselines_uncorrected'],
                'significantly_better_corrected': result['summary']['significantly_better_than_baselines_corrected']
            }
    
    combined_baseline_output = 'results/merges/combined_baseline_comparison_summary.json'
    with open(combined_baseline_output, 'w') as f:
        json.dump(combined_baseline_summary, f, indent=2)
    
    print(f"Combined baseline comparison summary saved to: {combined_baseline_output}")

