import json
import os

def find_best_approach(json_path, output_file):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    all_f1_scores = summary.get('all_f1_scores', {})
    
    output_file.write(f"\n{'='*80}\n")
    output_file.write(f"Analysis: {os.path.basename(json_path)}\n")
    output_file.write(f"{'='*80}\n\n")
    
    # Compute wins/losses for each merge
    merge_stats = {}
    for merge in all_f1_scores.keys():
        merge_stats[merge] = {
            'f1_score': all_f1_scores[merge],
            'wins': 0,
            'losses': 0,
            'total_comparisons': 0
        }
    
    # Count wins/losses from pairwise comparisons
    # Build lookup for corrected significance
    corrected_lookup = {}
    for result in data.get('bonferroni_holm_correction', []):
        corrected_lookup[result['comparison']] = result['is_significant']
    
    for comparison in data.get('pairwise_comparisons', []):
        merge_a = comparison['merge_a']
        merge_b = comparison['merge_b']
        comparison_key = comparison['comparison']
        
        is_significant = corrected_lookup.get(comparison_key, False)
        
        if is_significant:
            # Winner is the one with higher F1
            if comparison['f1_a'] > comparison['f1_b']:
                merge_stats[merge_a]['wins'] += 1
                merge_stats[merge_b]['losses'] += 1
            else:
                merge_stats[merge_b]['wins'] += 1
                merge_stats[merge_a]['losses'] += 1
        
        merge_stats[merge_a]['total_comparisons'] += 1
        merge_stats[merge_b]['total_comparisons'] += 1
    
    # Sort by: most wins, fewest losses, highest F1
    ranked = sorted(merge_stats.keys(), key=lambda x: (
        -merge_stats[x]['wins'],
        merge_stats[x]['losses'],
        -merge_stats[x]['f1_score']
    ))
    
    best = ranked[0]
    best_stats = merge_stats[best]
    
    output_file.write(f"BEST MERGE: {best}\n")
    output_file.write(f"  F1 Score: {best_stats['f1_score']:.4f}\n")
    output_file.write(f"  Significant Wins (after correction): {best_stats['wins']}\n")
    output_file.write(f"  Significant Losses (after correction): {best_stats['losses']}\n")
    output_file.write(f"  Total Comparisons: {best_stats['total_comparisons']}\n")
    
    if best_stats['wins'] > 0 and best_stats['losses'] == 0:
        output_file.write(f"\nSTATISTICALLY DOMINANT (after Bonferroni-Holm correction)\n")
        output_file.write(f"  This merge significantly outperforms {best_stats['wins']} other merge(s)\n")
    elif best_stats['wins'] == 0 and best_stats['losses'] == 0:
        output_file.write(f"\nNO SIGNIFICANT DIFFERENCES (after correction)\n")
        output_file.write(f"  Selected based on highest F1 score (practical significance)\n")
        output_file.write(f"  Bonferroni-Holm correction was too conservative to detect differences\n")
    else:
        output_file.write(f"\nMIXED RESULTS\n")
        output_file.write(f"  Has both wins and losses - no clear statistical dominance\n")
        output_file.write(f"  Consider F1 score as tiebreaker\n")
    
    output_file.write(f"\n{'Top 5 Merges:':-<80}\n")
    output_file.write(f"{'Merge':<40} {'F1':<8} {'Wins':<6} {'Losses':<7} {'Status'}\n")
    output_file.write(f"{'-'*80}\n")
    for i, merge in enumerate(ranked[:5], 1):
        stats = merge_stats[merge]
        if stats['wins'] > 0 and stats['losses'] == 0:
            status = "Dominant"
        elif stats['wins'] == 0 and stats['losses'] == 0:
            status = "Tied"
        else:
            status = "Mixed"
        output_file.write(f"{merge:<40} {stats['f1_score']:<8.4f} {stats['wins']:<6} {stats['losses']:<7} {status}\n")
    
    if 'pairwise_comparisons' in data:
        uncorrected_sig = sum(1 for comp in data['pairwise_comparisons'] if comp['mcnemar_test']['p_value'] < 0.05)
        corrected_sig = sum(1 for result in data.get('bonferroni_holm_correction', []) if result['is_significant'])
        total = len(data['pairwise_comparisons'])
        output_file.write(f"\n{'Correction Impact:':-<80}\n")
        output_file.write(f"  Total pairwise comparisons: {total}\n")
        output_file.write(f"  Significant (p < 0.05, uncorrected): {uncorrected_sig} ({100*uncorrected_sig/total:.1f}%)\n")
        output_file.write(f"  Significant (after Bonferroni-Holm): {corrected_sig} ({100*corrected_sig/total:.1f}%)\n")
    
    return best, best_stats

def main():
    results_dir = 'results/merges'
    output_file_path = os.path.join(results_dir, 'best_merges_summary.txt')
    
    best_results = {}
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        german_json = os.path.join(results_dir, 'meta_llama_full_german', 'merge_comparison_analysis.json')
        if os.path.exists(german_json):
            best_results['German'] = find_best_approach(german_json, output_file)
        
        french_json = os.path.join(results_dir, 'meta_llama_full_french', 'merge_comparison_analysis.json')
        if os.path.exists(french_json):
            best_results['French'] = find_best_approach(french_json, output_file)
        
        spanish_json = os.path.join(results_dir, 'meta_llama_full_spanish', 'merge_comparison_analysis.json')
        if os.path.exists(spanish_json):
            best_results['Spanish'] = find_best_approach(spanish_json, output_file)
        
        output_file.write(f"\n{'='*80}\n")
        output_file.write(f"SUMMARY - BEST MERGE PER LANGUAGE\n")
        output_file.write(f"{'='*80}\n\n")
        for lang, (best_name, best_stats) in best_results.items():
            output_file.write(f"{lang:<8} {best_name:<45} F1={best_stats['f1_score']:.4f}, wins={best_stats['wins']}, losses={best_stats['losses']}\n")
        
        output_file.write(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
