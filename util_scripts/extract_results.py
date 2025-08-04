import os
import json
import csv
import argparse
from typing import Dict, Any
import pandas as pd

# Hardcoded dataset measures
DATASET_MEASURES = {
    "arc_easy": "acc,none",
    "asdiv": "acc,none",
    "sciq": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "commonsense_qa": "acc,none",
    "hellaswag": "acc,none",
    "logiqa2": "acc,none",
    "piqa": "acc,none",
    "mmlu": "acc,none",
    "mathqa": "acc,none",
}

indomain = ["arc_easy", "asdiv", "sciq", "gsm8k"]
ood = ["hellaswag", "logiqa2", "piqa", "mmlu", "commonsense_qa"]

def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_measure(data: Dict[str, Any], dataset: str, measure: str) -> float:
    try:
        return data[dataset][measure]
    except KeyError:
        print(f"Warning: Measure '{measure}' not found for dataset '{dataset}'. Returning NaN.")
        return float('nan')

def gather_results(root_dir: str, output_file: str, specific_datasets: list = None):
    results = []

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir, "evals")
        print(subdir_path)
        if os.path.isdir(subdir_path):
            print("Exists")
            final_dir = os.path.join(subdir_path, 'final')
            if os.path.exists(final_dir):
                json_path = os.path.join(final_dir, 'results.json')
                if os.path.exists(json_path):
                    data = load_json(json_path)
                    
                    row = {'checkpoint': subdir}
                    for dataset, measure in DATASET_MEASURES.items():
                        if specific_datasets and dataset not in specific_datasets:
                            continue
                        row[dataset] = extract_measure(data, dataset, measure)
                    
                    results.append(row)

    # Write results to CSV
    if results:
        results_df = pd.DataFrame(results)
        if not specific_datasets:
            results_df["mean_indomain"] = results_df[indomain].mean(axis=1)
            results_df["mean_ood"] = results_df[ood].mean(axis=1)
        results_df.to_csv(output_file, index=False)
    else:
        print("No results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather results from JSON files into a CSV file.")
    parser.add_argument("root_dir", help="Root directory containing subdirectories with results")
    parser.add_argument("output_file", help="Output CSV file path")
    parser.add_argument("--specific_datasets", nargs="+", help="Specific datasets to include in the output CSV (default: all)")
    
    args = parser.parse_args()
    
    gather_results(args.root_dir, args.output_file, args.specific_datasets)
