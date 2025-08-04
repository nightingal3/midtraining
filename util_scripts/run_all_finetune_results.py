#!/usr/bin/env python3
"""
Automated script to generate CSV files for all dataset and model size combinations.
This script runs fetch_finetune_loss_results.py for all combinations of:

Datasets:
- pycode (unmasked)
- gsm8k (masked) 
- lima (unmasked)
- sciq (masked, but group has _unmasked suffix)

Model sizes: 70m, 160m, 410m

Output files will be named: val_loss_comparison_<dataset>_<size>_newformat.csv
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Running: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    # Configuration
    datasets = ["pycode", "gsm8k", "lima", "sciq"]
    model_sizes = ["70m", "160m", "410m"]
    
    # Path to the main script
    script_dir = Path(__file__).parent
    main_script = script_dir / "fetch_finetune_loss_results.py"
    
    if not main_script.exists():
        print(f"❌ Error: Main script not found at {main_script}")
        sys.exit(1)
    
    successful_runs = []
    failed_runs = []
    
    print(f"🎯 Starting automated generation of CSV files")
    print(f"📁 Script directory: {script_dir}")
    print(f"📊 Will process {len(datasets)} datasets × {len(model_sizes)} model sizes = {len(datasets) * len(model_sizes)} combinations")
    
    for dataset in datasets:
        for model_size in model_sizes:
            combination = f"{dataset}_{model_size}"
            expected_output = f"val_loss_comparison_{dataset}_{model_size}_newformat.csv"
            
            # Prepare command
            command = [
                sys.executable,  # Use same Python interpreter
                str(main_script),
                "--dataset", dataset,
                "--model_size", model_size
            ]
            
            description = f"Processing {dataset} dataset with {model_size} model"
            
            # Run the command
            if run_command(command, description):
                successful_runs.append(combination)
                
                # Check if output file was created
                output_path = script_dir / expected_output
                if output_path.exists():
                    print(f"📄 Output file created: {expected_output}")
                else:
                    print(f"⚠️  Warning: Expected output file not found: {expected_output}")
            else:
                failed_runs.append(combination)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful runs: {len(successful_runs)}/{len(datasets) * len(model_sizes)}")
    for run in successful_runs:
        print(f"   - {run}")
    
    if failed_runs:
        print(f"\n❌ Failed runs: {len(failed_runs)}")
        for run in failed_runs:
            print(f"   - {run}")
        sys.exit(1)
    else:
        print(f"\n🎉 All combinations completed successfully!")
        
        # List generated files
        print(f"\n📁 Generated CSV files:")
        for dataset in datasets:
            for model_size in model_sizes:
                filename = f"val_loss_comparison_{dataset}_{model_size}_newformat.csv"
                filepath = script_dir / filename
                if filepath.exists():
                    size_kb = filepath.stat().st_size / 1024
                    print(f"   - {filename} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
