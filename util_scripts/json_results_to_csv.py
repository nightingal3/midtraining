#!/usr/bin/env python3

import os
import json
import re
import csv
import glob
from pathlib import Path

def parse_token_count(checkpoint_name):
    """Extract stage, step, and token information from checkpoint name."""
    # Extract stage
    stage_match = re.search(r'stage(\d+)', checkpoint_name)
    stage = stage_match.group(1) if stage_match else "unknown"
    
    # Extract step
    step_match = re.search(r'step(\d+)', checkpoint_name)
    step = step_match.group(1) if step_match else "unknown"
    
    # Extract tokens
    tokens_match = re.search(r'tokens(\d+[KMBT]?)', checkpoint_name)
    tokens_str = tokens_match.group(1) if tokens_match else "unknown"
    
    # Convert token string to numeric value
    tokens = "unknown"
    if tokens_str != "unknown":
        if tokens_str.endswith('K'):
            tokens = int(tokens_str[:-1]) * 1000
        elif tokens_str.endswith('M'):
            tokens = int(tokens_str[:-1]) * 1000000
        elif tokens_str.endswith('B'):
            tokens = int(tokens_str[:-1]) * 1000000000
        elif tokens_str.endswith('T'):
            tokens = int(tokens_str[:-1]) * 1000000000000
        else:
            tokens = int(tokens_str)
    
    return stage, step, tokens

def extract_score_from_json(json_file, task_name):
    """Extract score from JSON results file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Map task name to the expected name in the results
        task_map = {
            'arc_easy': 'arc_easy',
            'commonsense_qa': 'commonsense_qa',
            'hellaswag': 'hellaswag',
            'logiqa2': 'logiqa',  # Note the mapping
            'mathqa': 'math_qa',   # Note the mapping
            'mmlu': 'mmlu',
            'piqa': 'piqa',
            'sciq': 'sciq'
        }
        
        # Get the mapped task name
        eval_task = task_map.get(task_name, task_name)
        
        # Extract the score
        if eval_task in data['results']:
            task_results = data['results'][eval_task]
            if 'acc' in task_results:
                return task_results['acc']
            elif 'accuracy' in task_results:
                return task_results['accuracy']
            # Try other possible fields
            for key in task_results:
                if 'acc' in key.lower():
                    return task_results[key]
        
        return None
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None

def find_latest_result(result_dir, task_name):
    """Find the latest result file for a task."""
    task_dir = os.path.join(result_dir, f"{task_name}.json")
    if not os.path.exists(task_dir):
        return None
    
    # Look for any JSON files in subdirectories
    json_files = []
    for root, dirs, files in os.walk(task_dir):
        for file in files:
            if file.endswith(".json") and "results_" in file:
                json_files.append(os.path.join(root, file))
    
    # Sort by modification time (newest first)
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    return json_files[0] if json_files else None

def main():
    # Directory where evaluation results are stored
    eval_dir = input("Enter the path to the evaluation results directory: ")
    if not os.path.exists(eval_dir):
        print(f"Error: Directory {eval_dir} not found")
        return
    
    # Output CSV file
    output_csv = os.path.join(eval_dir, "parsed_results.csv")
    
    # List of tasks
    tasks = ["mmlu", "arc_easy", "commonsense_qa", "hellaswag", "logiqa2", "mathqa", "piqa", "sciq"]
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d)) and 
                       (d.startswith("stage") or "step" in d)]
    
    # Prepare CSV data
    csv_data = []
    
    # Process each checkpoint directory
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = os.path.join(eval_dir, checkpoint_dir)
        
        # Extract metadata from checkpoint name
        stage, step, tokens = parse_token_count(checkpoint_dir)
        
        # Check results for each task
        for task in tasks:
            result_file = find_latest_result(checkpoint_path, task)
            if result_file:
                score = extract_score_from_json(result_file, task)
                if score is not None:
                    csv_data.append({
                        'checkpoint': checkpoint_dir,
                        'stage': stage,
                        'step': step,
                        'tokens': tokens,
                        'task': task,
                        'score': score
                    })
                    print(f"Found score for {checkpoint_dir} on {task}: {score}")
                else:
                    print(f"No score found in {result_file}")
    
    # Write to CSV
    if csv_data:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['checkpoint', 'stage', 'step', 'tokens', 'task', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        print(f"Results saved to {output_csv}")
    else:
        print("No results found to write to CSV")

if __name__ == "__main__":
    main()