#!/usr/bin/env python3
"""
Extract val_loss and val_c4_loss timeseries from a W&B group based on different checkpoint directories.
Compares timeseries between fixed and fixed_midtrain checkpoints.

Updated to:
- Extract all available seeds for each checkpoint-tag combination
- Average loss values across all seeds for each checkpoint 
- Report which seeds are missing for each checkpoint in the output CSV
- Default seed 1337 represents runs without explicit seed information
"""
import argparse
import os
import csv
from collections import defaultdict
import statistics

import wandb
from wandb import Api
from scipy import stats  # For statistical testing

def main():
    parser = argparse.ArgumentParser(
        description="Fetch val_loss and val_c4_loss timeseries from a W&B group"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'pycode', 'gsm8k', 'lima', 'sciq')"
    )
    parser.add_argument(
        "--model_size", 
        type=str,
        required=True,
        help="Model size (e.g., '70m', '160m', '410m')"
    )
    parser.add_argument(
        "--only_tag",
        type=str,
        default=None,
        help="If set, only include runs that have this tag"
    )
    parser.add_argument(
        "--masked_run",
        action="store_true",
        help="If set, include only runs tagged 'masked_prompt'; otherwise exclude them"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file name (default: auto-generated based on dataset and model size)"
    )
    args = parser.parse_args()
    
    # 1. Authenticate (ensure WANDB_API_KEY is set in your environment)
    wandb.login()

    ENTITY = "pretraining-and-behaviour"
    PROJECT = f"finetune-pythia-{args.model_size}"
    
    # Construct group name - special case for sciq
    if args.dataset == "sciq":
        GROUP = f"{args.dataset}_final_fixed_{args.model_size}_unmasked"
    else:
        GROUP = f"{args.dataset}_final_fixed_{args.model_size}"
    
    # Set output file name - use argument if provided, otherwise auto-generate
    if args.output_file:
        OUT_FILE = args.output_file
    else:
        OUT_FILE = f"val_loss_comparison_{args.dataset}_{args.model_size}_newformat.csv"

    print(f"🎯 Processing dataset: {args.dataset}")
    print(f"📏 Model size: {args.model_size}")
    print(f"🏢 Project: {ENTITY}/{PROJECT}")
    print(f"👥 Group: {GROUP}")
    print(f"📄 Output file: {OUT_FILE}")

    api = Api()

    # Determine masking based on dataset if not explicitly set
    masked_datasets = ["gsm8k"]
    unmasked_datasets = ["pycode", "lima", "sciq"]
    
    if args.dataset in masked_datasets:
        use_masked = True
    elif args.dataset in unmasked_datasets:
        use_masked = False
    else:
        # Fall back to the command line argument
        use_masked = args.masked_run

    # 2. Fetch all runs in the specified group
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP})
    print(f"🔍 Found {len(runs)} runs in group '{GROUP}'")
    
    # 2b. Filter masked_prompt runs globally
    if use_masked:
        runs = [r for r in runs if "masked_prompt" in r.tags]
        print(f"🚩 Keeping only runs with 'masked_prompt' tag: {len(runs)} runs")
    else:
        runs = [r for r in runs if "masked_prompt" not in r.tags]
        print(f"🚩 Excluding runs with 'masked_prompt' tag: {len(runs)} runs")

    # Debug: Show first few run tags to understand what we're working with
    print(f"🔍 Debug: First few runs and their tags:")
    for i, run in enumerate(runs[:3]):
        print(f"  Run {i+1}: {run.name} - Tags: {run.tags}")

    if args.only_tag:
        print(f"🚩 Will filter *midtrain* runs to tag '{args.only_tag}'")

    # 3. Organize runs by checkpoint directory, step tag, and seed
    fixed_runs = defaultdict(lambda: defaultdict(list))  # {tag: {seed: [runs]}}
    midtrain_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {mix: {tag: {seed: [runs]}}}
    
    # Define the midtrain mix patterns
    midtrain_mix_tags = [
        "starcoder_mix",
        "math_mix", 
        "flan_mix",
        "flan_high_mix",
        "knowledgeqa_mix",
        "dclm_mix"
    ]
    
    print(f"🔍 Debug: Categorizing {len(runs)} runs...")
    skipped_runs = 0
    midtrained_runs_count = 0
    regular_runs_count = 0
    
    for run in runs:
        checkpoint_dir = run.config.get("checkpoint_dir", "")
        
        # Determine which category this run belongs to
        target_dict = None
        mix_name = None
        
        if "midtrained" in run.tags:
            midtrained_runs_count += 1
            # This is a midtrain run, determine which mix
            for mix_tag in midtrain_mix_tags:
                if mix_tag in run.tags:
                    mix_name = mix_tag
                    break
            
            # If no specific mix tag found, it's the original starcoder mix
            if mix_name is None:
                mix_name = "starcoder_mix"
            
            target_dict = midtrain_runs[mix_name]
            
            # Apply --only_tag filter for midtrain runs
            if args.only_tag and args.only_tag not in run.tags:
                continue
        elif "regular" in run.tags:
            regular_runs_count += 1
            # This is a baseline (non-midtrained) run
            target_dict = fixed_runs
        else:
            # Skip runs that don't match any pattern
            skipped_runs += 1
            print(f"  🚫 Skipping run '{run.name}' - no 'midtrained' or 'regular' tag. Tags: {run.tags}")
            continue
        
        # Extract seed from run name or config
        seed = run.config.get("seed", None)
        if seed is None:
            # Try to extract from run name (e.g., "run_seed_42")
            for tag in run.tags:
                if tag.startswith("seed_"):
                    try:
                        seed = int(tag.split("_")[1])
                        break
                    except (IndexError, ValueError):
                        pass
        # Default to 1337 if no seed found (represents the default/unnamed seed)
        if seed is None:
            seed = 1337
        
        # Add to appropriate dictionary by tag and seed
        for tag in run.tags:
            if tag.startswith("step-") or tag.startswith("final"):
                # Filter to only include 10k increments for step tags
                if tag.startswith("step-"):
                    try:
                        step_num = int(tag.split("-")[1])
                        # Only include if it's a multiple of 10k
                        if step_num % 10000 != 0:
                            continue
                    except (IndexError, ValueError):
                        # Skip malformed step tags
                        continue
                
                target_dict[tag][seed].append(run)

    print(f"🔍 Debug: Categorization summary:")
    print(f"  - Midtrained runs: {midtrained_runs_count}")
    print(f"  - Regular runs: {regular_runs_count}")
    print(f"  - Skipped runs: {skipped_runs}")
    print(f"  - Total fixed runs found: {sum(len(seeds) for seeds in fixed_runs.values())}")
    print(f"  - Total midtrain mixes found: {list(midtrain_runs.keys())}")

    # 4. Extract and average val_loss and val_c4_loss across seeds for each step
    fixed_results = {}
    midtrain_results = {}
    
    def process_runs_by_seed(runs_by_tag_and_seed):
        """Process runs grouped by tag and seed, return averaged results."""
        results = {}
        for tag, runs_by_seed in runs_by_tag_and_seed.items():
            val_losses = []
            val_c4_losses = []
            available_seeds = []
            run_ids = []
            
            for seed, runs_list in runs_by_seed.items():
                if not runs_list:
                    continue
                    
                # Take the best run for this seed (lowest val_loss)
                best_run = min(
                    runs_list,
                    key=lambda r: r.summary.get("val_loss", float("inf"))
                )
                
                val_loss = best_run.summary.get("val_loss", None)
                val_c4_loss = best_run.summary.get("val_c4_loss", None)
                
                if val_loss is not None:
                    val_losses.append(val_loss)
                    available_seeds.append(seed)
                    run_ids.append(f"{seed}:{best_run.id}")
                
                if val_c4_loss is not None:
                    val_c4_losses.append(val_c4_loss)
            
            # Calculate averages and standard deviations
            avg_val_loss = statistics.mean(val_losses) if val_losses else None
            avg_val_c4_loss = statistics.mean(val_c4_losses) if val_c4_losses else None
            
            # Calculate std dev
            std_val_loss = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0
            std_val_c4_loss = statistics.stdev(val_c4_losses) if len(val_c4_losses) > 1 else 0.0
            
            # Determine all possible seeds (we'll define this as 1337, plus any others found)
            all_found_seeds = set()
            for other_tag, other_runs_by_seed in runs_by_tag_and_seed.items():
                all_found_seeds.update(other_runs_by_seed.keys())
            
            missing_seeds = sorted(all_found_seeds - set(available_seeds))
            
            results[tag] = {
                "val_loss": avg_val_loss,
                "val_c4_loss": avg_val_c4_loss,
                "val_loss_std": std_val_loss,
                "val_c4_loss_std": std_val_c4_loss,
                "val_losses_raw": val_losses,  # Keep raw values for significance testing
                "val_c4_losses_raw": val_c4_losses,
                "available_seeds": sorted(available_seeds),
                "missing_seeds": missing_seeds,
                "run_ids": run_ids,
                "num_seeds": len(available_seeds)
            }
        
        return results
    
    # Process fixed runs
    fixed_results = process_runs_by_seed(fixed_runs)
    
    # Process midtrain runs by mix
    midtrain_results = {}
    for mix_name, mix_runs in midtrain_runs.items():
        midtrain_results[mix_name] = process_runs_by_seed(mix_runs)

    # Perform significance tests
    significance_results = perform_significance_tests(fixed_results, midtrain_results)

    # 5. Print results sorted by tag
    print("Fixed checkpoint results (averaged across seeds):")
    for tag in sorted(fixed_results):
        result = fixed_results[tag]
        val_loss_str = f"{result['val_loss']:.6f}" if result['val_loss'] is not None else "N/A"
        val_c4_loss_str = f"{result['val_c4_loss']:.6f}" if result['val_c4_loss'] is not None else "N/A"
        val_loss_std_str = f"±{result['val_loss_std']:.6f}" if result['val_loss_std'] > 0 else ""
        val_c4_loss_std_str = f"±{result['val_c4_loss_std']:.6f}" if result['val_c4_loss_std'] > 0 else ""
        
        print(f"{tag}: val_loss = {val_loss_str} {val_loss_std_str}, "
              f"val_c4_loss = {val_c4_loss_str} {val_c4_loss_std_str} "
              f"(seeds: {result['available_seeds']}, {result['num_seeds']} total)")
        if result['missing_seeds']:
            print(f"  Missing seeds: {result['missing_seeds']}")
    
    print("\nMidtrain checkpoint results (averaged across seeds):")
    for mix_name in sorted(midtrain_results.keys()):
        print(f"\n--- {mix_name.upper()} ---")
        
        # Show significance test results for final checkpoint
        if mix_name in significance_results and "final" in midtrain_results[mix_name]:
            sig_info = significance_results[mix_name]
            val_sig_marker = "🟢" if sig_info["val_loss_significant"] and sig_info["val_loss_direction"] == "better" else \
                            "🔴" if sig_info["val_loss_significant"] and sig_info["val_loss_direction"] == "worse" else "⚪"
            c4_sig_marker = "🟢" if sig_info["c4_loss_significant"] and sig_info["c4_loss_direction"] == "better" else \
                           "🔴" if sig_info["c4_loss_significant"] and sig_info["c4_loss_direction"] == "worse" else "⚪"
            
            print(f"📊 Significance vs baseline: Val loss {val_sig_marker} (p={sig_info['val_loss_p_value']:.4f}), "
                  f"C4 loss {c4_sig_marker} (p={sig_info.get('c4_loss_p_value', 'N/A')})")
        
        mix_results = midtrain_results[mix_name]
        for tag in sorted(mix_results):
            result = mix_results[tag]
            val_loss_str = f"{result['val_loss']:.6f}" if result['val_loss'] is not None else "N/A"
            val_c4_loss_str = f"{result['val_c4_loss']:.6f}" if result['val_c4_loss'] is not None else "N/A"
            val_loss_std_str = f"±{result['val_loss_std']:.6f}" if result['val_loss_std'] > 0 else ""
            val_c4_loss_std_str = f"±{result['val_c4_loss_std']:.6f}" if result['val_c4_loss_std'] > 0 else ""
            
            print(f"{tag}: val_loss = {val_loss_str} {val_loss_std_str}, "
                  f"val_c4_loss = {val_c4_loss_str} {val_c4_loss_std_str} "
                  f"(seeds: {result['available_seeds']}, {result['num_seeds']} total)")
            if result['missing_seeds']:
                print(f"  Missing seeds: {result['missing_seeds']}")

    # 6. Write results out to CSV
    print(f"🔍 Debug: About to write CSV. midtrain_results.keys() = {list(midtrain_results.keys())}")
    print(f"🔍 Debug: midtrain_results content: {list(midtrain_results.items())[:2]}")  # Show first 2 items
    
    with open(OUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Create header with std columns and significance testing
        header = ["step_tag", "fixed_val_loss_avg", "fixed_val_loss_std", "fixed_val_c4_loss_avg", "fixed_val_c4_loss_std", "fixed_num_seeds", "fixed_missing_seeds"]
        for mix_name in sorted(midtrain_results.keys()):
            header.extend([
                f"{mix_name}_val_loss_avg",
                f"{mix_name}_val_loss_std",
                f"{mix_name}_val_c4_loss_avg", 
                f"{mix_name}_val_c4_loss_std",
                f"{mix_name}_num_seeds",
                f"{mix_name}_missing_seeds"
            ])
        
        # Add significance columns for final checkpoints
        for mix_name in sorted(midtrain_results.keys()):
            header.extend([
                f"{mix_name}_val_loss_p_value",
                f"{mix_name}_val_loss_significant",
                f"{mix_name}_c4_loss_p_value", 
                f"{mix_name}_c4_loss_significant"
            ])
        
        writer.writerow(header)
        
        # Get the union of all tags from fixed and all midtrain mixes
        all_tags = set(fixed_results.keys())
        for mix_results in midtrain_results.values():
            all_tags.update(mix_results.keys())
        all_tags = sorted(all_tags)
        
        for tag in all_tags:
            fixed_result = fixed_results.get(tag, {})
            row = [
                tag,
                fixed_result.get("val_loss", "N/A"),
                fixed_result.get("val_loss_std", "N/A"),
                fixed_result.get("val_c4_loss", "N/A"),
                fixed_result.get("val_c4_loss_std", "N/A"),
                fixed_result.get("num_seeds", 0),
                ";".join(map(str, fixed_result.get("missing_seeds", [])))
            ]
            
            # Add results for each mix
            for mix_name in sorted(midtrain_results.keys()):
                mix_result = midtrain_results[mix_name].get(tag, {})
                row.extend([
                    mix_result.get("val_loss", "N/A"),
                    mix_result.get("val_loss_std", "N/A"),
                    mix_result.get("val_c4_loss", "N/A"),
                    mix_result.get("val_c4_loss_std", "N/A"),
                    mix_result.get("num_seeds", 0),
                    ";".join(map(str, mix_result.get("missing_seeds", [])))
                ])
            
            # Add significance test results (only for final checkpoint)
            for mix_name in sorted(midtrain_results.keys()):
                if tag == "final" and mix_name in significance_results:
                    sig_info = significance_results[mix_name]
                    row.extend([
                        sig_info.get("val_loss_p_value", "N/A"),
                        sig_info.get("val_loss_significant", "N/A"),
                        sig_info.get("c4_loss_p_value", "N/A"),
                        sig_info.get("c4_loss_significant", "N/A")
                    ])
                else:
                    row.extend(["N/A", "N/A", "N/A", "N/A"])
            
            writer.writerow(row)

    print(f"\n✅ Results saved to {OUT_FILE}")

def perform_significance_tests(fixed_results, midtrain_results):
    """Perform t-tests comparing midtrain results to fixed baseline."""
    significance_results = {}
    
    # Get baseline (fixed) final results
    baseline_final = fixed_results.get("final", {})
    baseline_val_losses = baseline_final.get("val_losses_raw", [])
    baseline_c4_losses = baseline_final.get("val_c4_losses_raw", [])
    
    if len(baseline_val_losses) < 2:
        print("⚠️  Not enough baseline samples for significance testing")
        return significance_results
    
    for mix_name, mix_results in midtrain_results.items():
        final_result = mix_results.get("final", {})
        test_val_losses = final_result.get("val_losses_raw", [])
        test_c4_losses = final_result.get("val_c4_losses_raw", [])
        
        if len(test_val_losses) < 2:
            continue
            
        # Perform t-tests
        try:
            # Val loss t-test
            val_t_stat, val_p_value = stats.ttest_ind(baseline_val_losses, test_val_losses)
            val_significant = val_p_value < 0.05
            val_direction = "better" if statistics.mean(test_val_losses) < statistics.mean(baseline_val_losses) else "worse"
            
            # C4 loss t-test (if available)
            c4_t_stat, c4_p_value, c4_significant, c4_direction = None, None, False, "N/A"
            if len(baseline_c4_losses) >= 2 and len(test_c4_losses) >= 2:
                c4_t_stat, c4_p_value = stats.ttest_ind(baseline_c4_losses, test_c4_losses)
                c4_significant = c4_p_value < 0.05
                c4_direction = "better" if statistics.mean(test_c4_losses) < statistics.mean(baseline_c4_losses) else "worse"
            
            significance_results[mix_name] = {
                "val_loss_p_value": val_p_value,
                "val_loss_significant": val_significant,
                "val_loss_direction": val_direction,
                "c4_loss_p_value": c4_p_value,
                "c4_loss_significant": c4_significant,
                "c4_loss_direction": c4_direction
            }
            
        except Exception as e:
            print(f"⚠️  Error in significance test for {mix_name}: {e}")
    
    return significance_results

if __name__ == "__main__":
    main()