import os
import json
import csv
import glob

def extract_f1_scores():
    base_path = r"D:\Uni\Masterarbeit Code\jakob_finetuning\results\merges\meta_llama_full_precision_sauerkraut"
    
    metrics_files = glob.glob(os.path.join(base_path, "**", "metrics.json"), recursive=True)
    
    results = []
    
    for metrics_file in metrics_files:
        try:
            folder_path = os.path.dirname(metrics_file)
            model_name = os.path.basename(folder_path)
            
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            micro_f1 = data.get("results_asp_pol", {}).get("Micro-AVG", {}).get("f1", None)
            
            if micro_f1 is not None:
                results.append((model_name, micro_f1))
                print(f"Found {model_name}: F1 = {micro_f1}")
            else:
                print(f"Warning: No Micro-AVG F1 found in {model_name}")
                
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {metrics_file}")
        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")
    
    # Sort by F1 score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    output_file = os.path.join(base_path, "model_f1_scores.csv")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model_Name', 'Micro_F1_Score'])
        writer.writerows(results)
    
    for i, (model, f1) in enumerate(results[:10], 1):
        print(f"{i:2d}. {model:30s} | F1: {f1:.4f}")

if __name__ == "__main__":
    extract_f1_scores()