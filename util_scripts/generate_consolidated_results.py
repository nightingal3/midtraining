#!/usr/bin/env python3
"""
ååte a consolidated CSV file in the format of final_step_ft_results_with_410m.csv
by running fetch_finetune_loss_results.py for all combinations and consolidating results.

Output format:
Model size,SFT dataset,Pre/midtrain mix,SFT val loss after FT,C4 val loss after FT,Column 1
"""
import subprocess
import sys
import csv
import os
from pathlib import Path
from collections import defaultdict

def run_fetch_script(dataset, model_size):
    """Run the fetch script for a specific dataset and model size."""
    script_dir = Path(__file__).parent
    main_script = script_dir / "fetch_finetune_loss_results.py"
    
    command = [
        sys.executable,
        str(main_script),
        "--dataset", dataset,
        "--model_size", model_size
    ]
    
    print(f"🔄 Fetching {dataset} {model_size}...")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Successfully fetched {dataset} {model_size}")
        if result.stdout:
            print(f"📄 Output preview: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fetching {dataset} {model_size}: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def parse_individual_csv(filepath):
    """Parse an individual CSV file and extract the final step results."""
    results = {}
    
    if not filepath.exists():
        print(f"⚠️  CSV file not found: {filepath}")
        return results
    
    print(f"🔍 Parsing CSV file: {filepath}")

    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            # Debug: show available columns
            print(f"📋 Available columns: {reader.fieldnames}")
            
            row_count = 0
            for row in reader:
                row_count += 1
                
                # Debug: show first few rows
                if row_count <= 2:
                    print(f"📄 Row {row_count}: {dict(row)}")
                
                if row['step_tag'] == 'final':
                    # Extract results for each mix type
                    base_val_loss = row.get('fixed_val_loss_avg', 'N/A')
                    base_c4_loss = row.get('fixed_val_c4_loss_avg', 'N/A')
                    
                    # Add baseline (C4) result
                    if base_val_loss != 'N/A':
                        results['C4'] = {
                            'val_loss': float(base_val_loss) if base_val_loss != 'N/A' else None,
                            'c4_loss': float(base_c4_loss) if base_c4_loss != 'N/A' else None
                        }
                    
                    # Add midtrain results
                    mix_mapping = {
                        'starcoder_mix': 'Starcoder (20%)',
                        'starcoder_high_mix': 'Starcoder (100%)',
                        'math_mix': 'Math (12%)', 
                        'flan_mix': 'FLAN (5%)',
                        'flan_high_mix': 'FLAN (15%)',
                        'knowledgeqa_mix': 'KnowledgeQA (20%)',
                        'dclm_mix': 'DCLM (20%)'
                    }
                    
                    for mix_key, mix_name in mix_mapping.items():
                        val_loss_key = f'{mix_key}_val_loss_avg'
                        c4_loss_key = f'{mix_key}_val_c4_loss_avg'
                        
                        val_loss = row.get(val_loss_key, 'N/A')
                        c4_loss = row.get(c4_loss_key, 'N/A')
                        
                        if val_loss != 'N/A':
                            results[mix_name] = {
                                'val_loss': float(val_loss) if val_loss != 'N/A' else None,
                                'c4_loss': float(c4_loss) if c4_loss != 'N/A' else None
                            }
                    
                    break  # Only need the 'final' row
            
            print(f"🔍 Extracted {len(results)} mix conditions: {list(results.keys())}")
                    
    except Exception as e:
        print(f"⚠️  Error parsing {filepath}: {e}")
    
    return results

def main():
    # Configuration
    datasets = ["pycode", "gsm8k", "lima", "sciq"]  # Removed movie_reviews as it's not in the working groups
    model_sizes = ["70m", "160m", "410m"]
    
    script_dir = Path(__file__).parent
    output_file = script_dir / "final_step_ft_results_consolidated.csv"
    
    print("🎯 Generating consolidated final step results CSV")
    print("=" * 60)
    
    # Step 1: Fetch all individual results
    all_results = {}
    
    for model_size in model_sizes:
        all_results[model_size] = {}
        for dataset in datasets:
            print(f"📊 Processing {dataset} {model_size}...")
            
            # Run the fetch script
            success = run_fetch_script(dataset, model_size)
            
            if success:
                # Parse the generated CSV (created in current working directory, not script directory)
                csv_filename = f"val_loss_comparison_{dataset}_{model_size}_newformat.csv"
                csv_path = Path.cwd() / csv_filename
                
                results = parse_individual_csv(csv_path)
                all_results[model_size][dataset] = results
                
                print(f"✅ Found {len(results)} mix conditions for {dataset} {model_size}")
            else:
                print(f"❌ Failed to fetch {dataset} {model_size}")
                all_results[model_size][dataset] = {}
    
    # Step 2: Generate consolidated CSV
    print(f"\n📝 Writing consolidated results to {output_file}")
    
    # Dataset name mapping for display
    dataset_display_names = {
        "pycode": "Pycode",
        "gsm8k": "GSM8k", 
        "lima": "LIMA",
        "sciq": "SciQ"
    }
    
    # Mix order for consistent output
    mix_order = ["C4", "Starcoder (20%)", "Starcoder (100%)", "Math (12%)", "FLAN (5%)", "KnowledgeQA (20%)", "DCLM (20%)"]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["Model size", "SFT dataset", "Pre/midtrain mix", "SFT val loss after FT", "C4 val loss after FT", "Column 1"])
        
        # Write data grouped by model size
        for model_size in model_sizes:
            first_model_row = True
            
            for dataset in datasets:
                dataset_results = all_results[model_size].get(dataset, {})
                first_dataset_row = True
                
                # Only write rows if we have data for this dataset
                if dataset_results:
                    for mix_name in mix_order:
                        if mix_name in dataset_results:
                            mix_data = dataset_results[mix_name]
                            
                            # Format the values
                            val_loss = mix_data['val_loss']
                            c4_loss = mix_data['c4_loss']
                            
                            val_loss_str = f"{val_loss}" if val_loss is not None else ""
                            c4_loss_str = f"{c4_loss}" if c4_loss is not None else ""
                            
                            # Write row
                            model_size_cell = model_size if first_model_row else ""
                            dataset_cell = dataset_display_names[dataset] if first_dataset_row else ""
                            
                            writer.writerow([
                                model_size_cell,
                                dataset_cell,
                                mix_name,
                                val_loss_str,
                                c4_loss_str,
                                ""  # Column 1 is empty
                            ])
                            
                            first_model_row = False
                            first_dataset_row = False
                
                # Add empty row after each dataset (like in the original)
                if dataset_results and dataset != datasets[-1]:  # Don't add after last dataset
                    writer.writerow(["", "", "", "", "", ""])
            
            # Add separator row between model sizes
            if model_size != model_sizes[-1]:  # Don't add after last model size
                writer.writerow(["", "", "", "", "", ""])
    
    print(f"✅ Consolidated CSV written to: {output_file}")
    
    # Summary
    total_combinations = 0
    successful_combinations = 0
    
    for model_size in model_sizes:
        for dataset in datasets:
            total_combinations += 1
            if all_results[model_size][dataset]:
                successful_combinations += 1
    
    print(f"📊 Summary: {successful_combinations}/{total_combinations} combinations had data")
    
    # Show file size
    if output_file.exists():
        size_kb = output_file.stat().st_size / 1024
        print(f"📄 Output file size: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
