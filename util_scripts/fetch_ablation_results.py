#!/usr/bin/env python3
"""
Extract val_loss and val_c4_loss timeseries from W&B groups for continuous pretraining ablations.

Fetches from run groups like <dataset>_cts_pretrain_ablation_70m_sc and extracts:
- All available seeds for each step
- Average loss values across all seeds for each step
- Report which seeds are missing for each step in the output CSV
- Step number vs. val loss and c4 loss timeseries

Supports four ablation types:
- cts_pretrain_ablation: Groups by step tags (step-1000, step-2000, etc.)
- mixture_weight_ablation: Groups by mixture weight percentages (10_percent, 80_percent, etc.)  
- timing_ablation: Groups by timing steps (from_50k, from_100k, etc.)
- math_cts_ablation: Groups by math_cts tags vs base only comparison

Automatically filters by default masking type for each dataset:
- pycode: unmasked_prompt
- lima: unmasked_prompt
- sciq: masked_prompt  
- gsm8k: masked_prompt
"""
import argparse
import os
import csv
from collections import defaultdict
import statistics

import wandb
from wandb import Api

def sort_tags(tags):
        """Sort tags with step numbers in numerical order."""
        step_tags = []
        other_tags = []
        
        for tag in tags:
            if tag.startswith("step-"):
                try:
                    step_num = int(tag.split("-")[1])
                    step_tags.append((step_num, tag))
                except (IndexError, ValueError):
                    other_tags.append(tag)
            else:
                other_tags.append(tag)
        
        # Sort step tags by number, other tags alphabetically
        step_tags.sort(key=lambda x: x[0])
        other_tags.sort()
        
        return [tag for _, tag in step_tags] + other_tags
        
def main():
    parser = argparse.ArgumentParser(
        description="Fetch val_loss and val_c4_loss timeseries from W&B groups for ablation studies"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name for the ablation (e.g., 'gsm8k', 'lima', 'pycode')"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="70m",
        help="Model size (default: '70m')"
    )
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="cts_pretrain_ablation",
        help="Ablation type: 'cts_pretrain_ablation', 'mixture_weight_ablation', 'timing_ablation', or 'math_cts_ablation'"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name (default: auto-generated as 'finetune-pythia-<model_size>')"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="pretraining-and-behaviour",
        help="W&B entity name (default: 'pretraining-and-behaviour')"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file name (default: auto-generated based on dataset)"
    )
    
    args = parser.parse_args()
    
    # 1. Authenticate (ensure WANDB_API_KEY is set in your environment)
    wandb.login()

    # Auto-generate project name if not specified
    if args.project is None:
        args.project = f"finetune-pythia-{args.model_size}"

    # Construct group name based on pattern: <dataset>_cts_pretrain_ablation_70m_sc
    # For math_cts_ablation, we use the same group as cts_pretrain_ablation since math_cts runs
    # were stored there but distinguished by the math_cts tag
    if args.ablation_type == "math_cts_ablation":
        GROUP = f"{args.dataset}_cts_pretrain_ablation_{args.model_size}_sc"
    else:
        GROUP = f"{args.dataset}_{args.ablation_type}_{args.model_size}_sc"
    
    # Default output file name if not specified
    if args.output_file is None:
        OUT_FILE = f"ablation_results_{args.dataset}_{args.model_size}_{args.ablation_type}.csv"
    else:
        OUT_FILE = args.output_file

    print(f"Fetching results from group: {GROUP}")
    print(f"Project: {args.entity}/{args.project}")
    print(f"Output file: {OUT_FILE}")

    api = Api()

    # 2. Fetch all runs in the specified group with debugging
    try:
        # First, try to get all runs and then filter manually to debug
        print(f"Fetching all runs from project {args.entity}/{args.project}...")
        all_runs = api.runs(f"{args.entity}/{args.project}")
        print(f"Total runs in project: {len(all_runs)}")
        
        # Filter manually by group
        filtered_runs = []
        group_names_found = set()
        
        for run in all_runs:
            if hasattr(run, 'group') and run.group:
                group_names_found.add(run.group)
                if run.group == GROUP:
                    filtered_runs.append(run)
        
        print(f"Found {len(filtered_runs)} runs in group '{GROUP}'")
        print(f"All group names found ({len(group_names_found)} total):")
        for group in sorted(group_names_found):
            if GROUP in group or args.dataset in group:  # Show relevant groups
                print(f"  - {group}")
        
        # If we found the group but no runs, there might be an API issue
        if GROUP in group_names_found:
            print(f"✓ Group '{GROUP}' exists in the project")
        else:
            print(f"✗ Group '{GROUP}' not found.")
            # Show similar group names that might be what we're looking for
            similar_groups = [g for g in group_names_found if args.dataset in g]
            if similar_groups:
                print(f"Similar groups found:")
                for group in similar_groups:
                    print(f"  - {group}")
        
        runs = filtered_runs
        
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return

    # If group filtering didn't work, try tag-based approach as backup
    if len(runs) == 0:
        print(f"Group filtering failed, trying tag-based approach...")
        try:
            # Look for runs with the dataset tag and ablation type tag
            filters = {
                "tags": {"$all": [args.dataset, args.ablation_type]}
            }
            runs = api.runs(f"{args.entity}/{args.project}", filters=filters)
            print(f"Found {len(runs)} runs with tags '{args.dataset}' and '{args.ablation_type}'")
            
            # Further filter by model size if needed
            if args.model_size:
                original_count = len(runs)
                runs = [run for run in runs if args.model_size in run.tags]
                print(f"After filtering by model size '{args.model_size}': {len(runs)} runs (was {original_count})")
                
        except Exception as e:
            print(f"Tag-based filtering also failed: {e}")
            return

    if len(runs) == 0:
        print(f"No runs found with either group '{GROUP}' or tags '{args.dataset}' + '{args.ablation_type}'. Please check the group name and project.")
        return

    # Filter by masking type based on dataset defaults
    masking_defaults = {
        "pycode": "unmasked_prompt",
        "lima": "unmasked_prompt", 
        "sciq": "masked_prompt",
        "gsm8k": "masked_prompt"
    }
    
    expected_masking = masking_defaults.get(args.dataset, None)
    if expected_masking:
        original_count = len(runs)
        filtered_runs = []
        for run in runs:
            if expected_masking in run.tags:
                filtered_runs.append(run)
        runs = filtered_runs
        print(f"Filtered by masking type '{expected_masking}': {len(runs)} runs (was {original_count})")
    else:
        print(f"No default masking type defined for dataset '{args.dataset}', skipping masking filter")

    if len(runs) == 0:
        print(f"No runs found after filtering by masking type '{expected_masking}' for dataset '{args.dataset}'.")
        return

    # 3. Organize runs by step tag/mixture weight/timing and seed
    if args.ablation_type == "mixture_weight_ablation":
        # Structure: {mixture_weight: {seed: [runs]}}
        runs_by_weight_and_seed = defaultdict(lambda: defaultdict(list))
        
        for run in runs:
            # Extract seed from run name, config, or tags
            seed = run.config.get("seed", None)
            if seed is None:
                # Try to extract from run name or tags (e.g., "seed_42")
                for tag in run.tags:
                    if tag.startswith("seed_"):
                        try:
                            seed = int(tag.split("_")[1])
                            break
                        except (IndexError, ValueError):
                            pass
                
                # Try to extract from run name if it contains seed pattern
                if seed is None and "seed" in run.name.lower():
                    import re
                    seed_match = re.search(r'seed[_-]?(\d+)', run.name.lower())
                    if seed_match:
                        seed = int(seed_match.group(1))
            
            # Default to 1337 if no seed found (represents the default/unnamed seed)
            if seed is None:
                seed = 1337
            
            # Extract mixture weight from checkpoint_dir
            checkpoint_dir = run.config.get("checkpoint_dir", "")
            mixture_weight = extract_mixture_weight_from_checkpoint(checkpoint_dir)
            
            if mixture_weight:
                runs_by_weight_and_seed[mixture_weight][seed].append(run)
            else:
                print(f"Warning: Could not extract mixture weight from checkpoint_dir: {checkpoint_dir}")
    
    elif args.ablation_type == "timing_ablation":
        # Structure: {timing_step: {seed: [runs]}}
        runs_by_timing_and_seed = defaultdict(lambda: defaultdict(list))
        
        for run in runs:
            # Extract seed from run name, config, or tags
            seed = run.config.get("seed", None)
            if seed is None:
                # Try to extract from run name or tags (e.g., "seed_42")
                for tag in run.tags:
                    if tag.startswith("seed_"):
                        try:
                            seed = int(tag.split("_")[1])
                            break
                        except (IndexError, ValueError):
                            pass
                
                # Try to extract from run name if it contains seed pattern
                if seed is None and "seed" in run.name.lower():
                    import re
                    seed_match = re.search(r'seed[_-]?(\d+)', run.name.lower())
                    if seed_match:
                        seed = int(seed_match.group(1))
            
            # Default to 1337 if no seed found (represents the default/unnamed seed)
            if seed is None:
                seed = 1337
            
            # Extract timing step from checkpoint_dir
            checkpoint_dir = run.config.get("checkpoint_dir", "")
            timing_step = extract_timing_from_checkpoint(checkpoint_dir)
            
            if timing_step:
                runs_by_timing_and_seed[timing_step][seed].append(run)
            else:
                print(f"Warning: Could not extract timing step from checkpoint_dir: {checkpoint_dir}")
    
    elif args.ablation_type == "math_cts_ablation":
        # Structure: {condition: {seed: [runs]}} where condition is "math_cts" or "base_only"
        runs_by_condition_and_seed = defaultdict(lambda: defaultdict(list))
        
        for run in runs:
            # Extract seed from run name, config, or tags
            seed = run.config.get("seed", None)
            if seed is None:
                # Try to extract from run name or tags (e.g., "seed_42")
                for tag in run.tags:
                    if tag.startswith("seed_"):
                        try:
                            seed = int(tag.split("_")[1])
                            break
                        except (IndexError, ValueError):
                            pass
                
                # Try to extract from run name if it contains seed pattern
                if seed is None and "seed" in run.name.lower():
                    import re
                    seed_match = re.search(r'seed[_-]?(\d+)', run.name.lower())
                    if seed_match:
                        seed = int(seed_match.group(1))
            
            # Default to 1337 if no seed found (represents the default/unnamed seed)
            if seed is None:
                seed = 1337
            
            # Categorize runs as math_cts or base_only based on tags
            if "math_cts" in run.tags:
                runs_by_condition_and_seed["math_cts"][seed].append(run)
            else:
                # Assume runs without math_cts tag are base_only
                runs_by_condition_and_seed["base_only"][seed].append(run)
    
    else:
        # Structure: {tag: {seed: [runs]}}
        runs_by_tag_and_seed = defaultdict(lambda: defaultdict(list))
        
        for run in runs:
            # Extract seed from run name, config, or tags
            seed = run.config.get("seed", None)
            if seed is None:
                # Try to extract from run name or tags (e.g., "seed_42")
                for tag in run.tags:
                    if tag.startswith("seed_"):
                        try:
                            seed = int(tag.split("_")[1])
                            break
                        except (IndexError, ValueError):
                            pass
                
                # Try to extract from run name if it contains seed pattern
                if seed is None and "seed" in run.name.lower():
                    import re
                    seed_match = re.search(r'seed[_-]?(\d+)', run.name.lower())
                    if seed_match:
                        seed = int(seed_match.group(1))
            
            # Default to 1337 if no seed found (represents the default/unnamed seed)
            if seed is None:
                seed = 1337
            
            # Add to appropriate dictionary by step tag and seed
            for tag in run.tags:
                if tag.startswith("step-") or tag.startswith("final"):
                    # For step tags, optionally filter to specific increments
                    if tag.startswith("step-"):
                        try:
                            step_num = int(tag.split("-")[1])
                            # Optionally, filter to only include certain step increments
                            # For now, include all steps - can be modified later
                        except (IndexError, ValueError):
                            # Skip malformed step tags
                            continue
                    
                    runs_by_tag_and_seed[tag][seed].append(run)

    # 4. Define processing functions
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
                    
                # Take the best run for this seed (lowest val_loss if available)
                valid_runs = [r for r in runs_list if r.summary.get("val_loss") is not None]
                if valid_runs:
                    best_run = min(valid_runs, key=lambda r: r.summary.get("val_loss", float("inf")))
                else:
                    # If no val_loss, just take the first run
                    best_run = runs_list[0]
                
                val_loss = best_run.summary.get("val_loss", None)
                val_c4_loss = best_run.summary.get("val_c4_loss", None)
                
                if val_loss is not None:
                    val_losses.append(val_loss)
                    available_seeds.append(seed)
                    run_ids.append(f"{seed}:{best_run.id}")
                
                if val_c4_loss is not None:
                    val_c4_losses.append(val_c4_loss)
            
            # Calculate averages
            avg_val_loss = statistics.mean(val_losses) if val_losses else None
            avg_val_c4_loss = statistics.mean(val_c4_losses) if val_c4_losses else None
            
            # Calculate std dev for additional info
            std_val_loss = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0
            std_val_c4_loss = statistics.stdev(val_c4_losses) if len(val_c4_losses) > 1 else 0.0
            
            # Determine all possible seeds (collect all seeds seen across all tags)
            all_found_seeds = set()
            for other_tag, other_runs_by_seed in runs_by_tag_and_seed.items():
                all_found_seeds.update(other_runs_by_seed.keys())
            
            missing_seeds = sorted(all_found_seeds - set(available_seeds))
            
            results[tag] = {
                "val_loss": avg_val_loss,
                "val_c4_loss": avg_val_c4_loss,
                "val_loss_std": std_val_loss,
                "val_c4_loss_std": std_val_c4_loss,
                "available_seeds": sorted(available_seeds),
                "missing_seeds": missing_seeds,
                "run_ids": run_ids,
                "num_seeds": len(available_seeds)
            }
        
        return results

    # Extract and average val_loss and val_c4_loss across seeds for each step/weight/timing/condition
    if args.ablation_type == "mixture_weight_ablation":
        # Process mixture weight ablation results
        results = process_mixture_weight_runs(runs_by_weight_and_seed)
    elif args.ablation_type == "timing_ablation":
        # Process timing ablation results
        results = process_timing_runs(runs_by_timing_and_seed)
    elif args.ablation_type == "math_cts_ablation":
        # Process math continuous pretraining ablation results
        results = process_math_cts_runs(runs_by_condition_and_seed)
    else:
        # Process continuous pretraining ablation results
        results = process_runs_by_seed(runs_by_tag_and_seed)

    # 5. Print results sorted by tag/weight/timing/condition (with step numbers or percentages in order)
    if args.ablation_type == "mixture_weight_ablation":
        sorted_keys = sort_mixture_weights(results.keys())
        print(f"\nMixture weight ablation results for {args.dataset} (averaged across seeds):")
    elif args.ablation_type == "timing_ablation":
        sorted_keys = sort_timing_steps(results.keys())
        print(f"\nTiming ablation results for {args.dataset} (averaged across seeds):")
    elif args.ablation_type == "math_cts_ablation":
        sorted_keys = sort_math_cts_conditions(results.keys())
        print(f"\nMath continuous pretraining ablation results for {args.dataset} (averaged across seeds):")
    else:
        sorted_keys = sort_tags(results.keys())
        print(f"\nContinuous pretraining ablation results for {args.dataset} (averaged across seeds):")
    
    print("=" * 80)

    for key in sorted_keys:
        result = results[key]
        val_loss_str = f"{result['val_loss']:.6f}" if result['val_loss'] is not None else "N/A"
        val_c4_loss_str = f"{result['val_c4_loss']:.6f}" if result['val_c4_loss'] is not None else "N/A"
        
        print(f"{key:15}: val_loss = {val_loss_str:>8} ± {result['val_loss_std']:.6f}, "
              f"val_c4_loss = {val_c4_loss_str:>8} ± {result['val_c4_loss_std']:.6f} "
              f"(seeds: {len(result['available_seeds'])}/{len(result['available_seeds']) + len(result['missing_seeds'])})")
        
        if result['missing_seeds']:
            print(f"                 Missing seeds: {result['missing_seeds']}")

    # 6. Write results out to CSV
    with open(OUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        if args.ablation_type == "mixture_weight_ablation":
            header = [
                "mixture_weight", 
                "percentage",
                "val_loss_avg", 
                "val_loss_std",
                "val_c4_loss_avg", 
                "val_c4_loss_std",
                "num_seeds", 
                "available_seeds",
                "missing_seeds",
                "run_ids"
            ]
        elif args.ablation_type == "timing_ablation":
            header = [
                "timing_step", 
                "step_number",
                "val_loss_avg", 
                "val_loss_std",
                "val_c4_loss_avg", 
                "val_c4_loss_std",
                "num_seeds", 
                "available_seeds",
                "missing_seeds",
                "run_ids"
            ]
        elif args.ablation_type == "math_cts_ablation":
            header = [
                "condition", 
                "val_loss_avg", 
                "val_loss_std",
                "val_c4_loss_avg", 
                "val_c4_loss_std",
                "num_seeds", 
                "available_seeds",
                "missing_seeds",
                "run_ids"
            ]
        else:
            header = [
                "step_tag", 
                "step_number",
                "val_loss_avg", 
                "val_loss_std",
                "val_c4_loss_avg", 
                "val_c4_loss_std",
                "num_seeds", 
                "available_seeds",
                "missing_seeds",
                "run_ids"
            ]
        writer.writerow(header)
        
        # Data rows
        for key in sorted_keys:
            result = results[key]
            
            if args.ablation_type == "mixture_weight_ablation":
                # Extract percentage number for easier analysis
                percentage = "N/A"
                if key.endswith("_percent"):
                    try:
                        percentage = int(key.split("_")[0])
                    except (IndexError, ValueError):
                        pass
                
                row = [
                    key,
                    percentage,
                    result.get("val_loss", "N/A"),
                    result.get("val_loss_std", "N/A"),
                    result.get("val_c4_loss", "N/A"),
                    result.get("val_c4_loss_std", "N/A"),
                    result.get("num_seeds", 0),
                    ";".join(map(str, result.get("available_seeds", []))),
                    ";".join(map(str, result.get("missing_seeds", []))),
                    ";".join(result.get("run_ids", []))
                ]
            elif args.ablation_type == "timing_ablation":
                # Extract step number for easier analysis
                step_number = "N/A"
                if key.startswith("from_") and key.endswith("k"):
                    try:
                        step_number = int(key[5:-1]) * 1000  # Convert from_50k to 50000
                    except (IndexError, ValueError):
                        pass
                elif key == "final":
                    step_number = "final"
                
                row = [
                    key,
                    step_number,
                    result.get("val_loss", "N/A"),
                    result.get("val_loss_std", "N/A"),
                    result.get("val_c4_loss", "N/A"),
                    result.get("val_c4_loss_std", "N/A"),
                    result.get("num_seeds", 0),
                    ";".join(map(str, result.get("available_seeds", []))),
                    ";".join(map(str, result.get("missing_seeds", []))),
                    ";".join(result.get("run_ids", []))
                ]
            elif args.ablation_type == "math_cts_ablation":
                # For math_cts_ablation, the key is the condition (base_only or math_cts)
                row = [
                    key,  # condition name
                    result.get("val_loss", "N/A"),
                    result.get("val_loss_std", "N/A"),
                    result.get("val_c4_loss", "N/A"),
                    result.get("val_c4_loss_std", "N/A"),
                    result.get("num_seeds", 0),
                    ";".join(map(str, result.get("available_seeds", []))),
                    ";".join(map(str, result.get("missing_seeds", []))),
                    ";".join(result.get("run_ids", []))
                ]
            else:
                # Extract step number for easier analysis
                step_number = "N/A"
                if key.startswith("step-"):
                    try:
                        step_number = int(key.split("-")[1])
                    except (IndexError, ValueError):
                        pass
                elif key == "final":
                    step_number = "final"
                
                row = [
                    key,
                    step_number,
                    result.get("val_loss", "N/A"),
                    result.get("val_loss_std", "N/A"),
                    result.get("val_c4_loss", "N/A"),
                    result.get("val_c4_loss_std", "N/A"),
                    result.get("num_seeds", 0),
                    ";".join(map(str, result.get("available_seeds", []))),
                    ";".join(map(str, result.get("missing_seeds", []))),
                    ";".join(result.get("run_ids", []))
                ]
            
            writer.writerow(row)

    print(f"\n✅ Results saved to {OUT_FILE}")
    
    # Print summary based on ablation type
    if args.ablation_type == "mixture_weight_ablation":
        print(f"📊 Summary: {len(results)} mixture weight conditions found")
    elif args.ablation_type == "timing_ablation":
        print(f"📊 Summary: {len(results)} timing conditions found")
    elif args.ablation_type == "math_cts_ablation":
        print(f"📊 Summary: {len(results)} conditions found (math_cts vs base_only)")
    else:
        print(f"📊 Summary: {len(results)} step checkpoints found")
    
    # Print summary statistics
    total_seeds_by_step = [len(result['available_seeds']) for result in results.values()]
    if total_seeds_by_step:
        print(f"🎯 Seeds per step: min={min(total_seeds_by_step)}, "
              f"max={max(total_seeds_by_step)}, "
              f"avg={statistics.mean(total_seeds_by_step):.1f}")

def extract_mixture_weight_from_checkpoint(checkpoint_dir):
    """Extract mixture weight percentage from checkpoint directory path."""
    if not isinstance(checkpoint_dir, str):
        return None
    
    import re
    # Look for patterns like "10_percent", "30_percent", etc.
    weight_pattern = r'(\d+)_percent'
    match = re.search(weight_pattern, checkpoint_dir)
    if match:
        return f"{match.group(1)}_percent"
    return None

def extract_timing_from_checkpoint(checkpoint_dir):
    """Extract timing step from checkpoint directory path."""
    if not isinstance(checkpoint_dir, str):
        return None
    
    import re
    # Look for patterns like "from_50k", "from_100k", etc.
    timing_pattern = r'from_(\d+)k'
    match = re.search(timing_pattern, checkpoint_dir)
    if match:
        return f"from_{match.group(1)}k"
    return None

def process_mixture_weight_runs(runs_by_weight_and_seed):
    """Process runs grouped by mixture weight and seed."""
    results = {}
    for weight, runs_by_seed in runs_by_weight_and_seed.items():
        val_losses = []
        val_c4_losses = []
        available_seeds = []
        run_ids = []
        
        for seed, runs_list in runs_by_seed.items():
            if not runs_list:
                continue
                
            # Take the best run for this seed (lowest val_loss if available)
            valid_runs = [r for r in runs_list if r.summary.get("val_loss") is not None]
            if valid_runs:
                best_run = min(valid_runs, key=lambda r: r.summary.get("val_loss", float("inf")))
            else:
                best_run = runs_list[0]
            
            val_loss = best_run.summary.get("val_loss", None)
            val_c4_loss = best_run.summary.get("val_c4_loss", None)
            
            if val_loss is not None:
                val_losses.append(val_loss)
                available_seeds.append(seed)
                run_ids.append(f"{seed}:{best_run.id}")
            
            if val_c4_loss is not None:
                val_c4_losses.append(val_c4_loss)
        
        # Calculate averages and std dev
        avg_val_loss = statistics.mean(val_losses) if val_losses else None
        avg_val_c4_loss = statistics.mean(val_c4_losses) if val_c4_losses else None
        std_val_loss = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0
        std_val_c4_loss = statistics.stdev(val_c4_losses) if len(val_c4_losses) > 1 else 0.0
        
        # Determine all possible seeds
        all_found_seeds = set()
        for other_weight, other_runs_by_seed in runs_by_weight_and_seed.items():
            all_found_seeds.update(other_runs_by_seed.keys())
        
        missing_seeds = sorted(all_found_seeds - set(available_seeds))
        
        results[weight] = {
            "val_loss": avg_val_loss,
            "val_c4_loss": avg_val_c4_loss,
            "val_loss_std": std_val_loss,
            "val_c4_loss_std": std_val_c4_loss,
            "available_seeds": sorted(available_seeds),
            "missing_seeds": missing_seeds,
            "run_ids": run_ids,
            "num_seeds": len(available_seeds)
        }
    
    return results

def process_timing_runs(runs_by_timing_and_seed):
    """Process runs grouped by timing step and seed."""
    results = {}
    for timing_step, runs_by_seed in runs_by_timing_and_seed.items():
        val_losses = []
        val_c4_losses = []
        available_seeds = []
        run_ids = []
        
        for seed, runs_list in runs_by_seed.items():
            if not runs_list:
                continue
                
            # Take the best run for this seed (lowest val_loss if available)
            valid_runs = [r for r in runs_list if r.summary.get("val_loss") is not None]
            if valid_runs:
                best_run = min(valid_runs, key=lambda r: r.summary.get("val_loss", float("inf")))
            else:
                best_run = runs_list[0]
            
            val_loss = best_run.summary.get("val_loss", None)
            val_c4_loss = best_run.summary.get("val_c4_loss", None)
            
            if val_loss is not None:
                val_losses.append(val_loss)
                available_seeds.append(seed)
                run_ids.append(f"{seed}:{best_run.id}")
            
            if val_c4_loss is not None:
                val_c4_losses.append(val_c4_loss)
        
        # Calculate averages and std dev
        avg_val_loss = statistics.mean(val_losses) if val_losses else None
        avg_val_c4_loss = statistics.mean(val_c4_losses) if val_c4_losses else None
        std_val_loss = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0
        std_val_c4_loss = statistics.stdev(val_c4_losses) if len(val_c4_losses) > 1 else 0.0
        
        # Determine all possible seeds
        all_found_seeds = set()
        for other_timing, other_runs_by_seed in runs_by_timing_and_seed.items():
            all_found_seeds.update(other_runs_by_seed.keys())
        
        missing_seeds = sorted(all_found_seeds - set(available_seeds))
        
        results[timing_step] = {
            "val_loss": avg_val_loss,
            "val_c4_loss": avg_val_c4_loss,
            "val_loss_std": std_val_loss,
            "val_c4_loss_std": std_val_c4_loss,
            "available_seeds": sorted(available_seeds),
            "missing_seeds": missing_seeds,
            "run_ids": run_ids,
            "num_seeds": len(available_seeds)
        }
    
    return results

def process_math_cts_runs(runs_by_condition_and_seed):
    """Process runs grouped by math_cts condition (math_cts vs base_only) and seed."""
    results = {}
    for condition, runs_by_seed in runs_by_condition_and_seed.items():
        val_losses = []
        val_c4_losses = []
        available_seeds = []
        run_ids = []
        
        for seed, runs_list in runs_by_seed.items():
            if not runs_list:
                continue
                
            # Take the best run for this seed (lowest val_loss if available)
            valid_runs = [r for r in runs_list if r.summary.get("val_loss") is not None]
            if valid_runs:
                best_run = min(valid_runs, key=lambda r: r.summary.get("val_loss", float("inf")))
            else:
                best_run = runs_list[0]
            
            val_loss = best_run.summary.get("val_loss", None)
            val_c4_loss = best_run.summary.get("val_c4_loss", None)
            
            if val_loss is not None:
                val_losses.append(val_loss)
                available_seeds.append(seed)
                run_ids.append(f"{seed}:{best_run.id}")
            
            if val_c4_loss is not None:
                val_c4_losses.append(val_c4_loss)
        
        # Calculate averages and std dev
        avg_val_loss = statistics.mean(val_losses) if val_losses else None
        avg_val_c4_loss = statistics.mean(val_c4_losses) if val_c4_losses else None
        std_val_loss = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0
        std_val_c4_loss = statistics.stdev(val_c4_losses) if len(val_c4_losses) > 1 else 0.0
        
        # Determine all possible seeds
        all_found_seeds = set()
        for other_condition, other_runs_by_seed in runs_by_condition_and_seed.items():
            all_found_seeds.update(other_runs_by_seed.keys())
        
        missing_seeds = sorted(all_found_seeds - set(available_seeds))
        
        results[condition] = {
            "val_loss": avg_val_loss,
            "val_c4_loss": avg_val_c4_loss,
            "val_loss_std": std_val_loss,
            "val_c4_loss_std": std_val_c4_loss,
            "available_seeds": sorted(available_seeds),
            "missing_seeds": missing_seeds,
            "run_ids": run_ids,
            "num_seeds": len(available_seeds)
        }
    
    return results

def sort_mixture_weights(weights):
    """Sort mixture weight percentages numerically."""
    weight_tuples = []
    other_weights = []
    
    for weight in weights:
        if weight.endswith("_percent"):
            try:
                percent_num = int(weight.split("_")[0])
                weight_tuples.append((percent_num, weight))
            except (IndexError, ValueError):
                other_weights.append(weight)
        else:
            other_weights.append(weight)
    
    weight_tuples.sort(key=lambda x: x[0])
    other_weights.sort()
    
    return [weight for _, weight in weight_tuples] + other_weights

def sort_timing_steps(timing_steps):
    """Sort timing steps numerically."""
    timing_tuples = []
    other_steps = []
    
    for step in timing_steps:
        if step.startswith("from_") and step.endswith("k"):
            try:
                step_num = int(step[5:-1])  # Extract number from "from_50k"
                timing_tuples.append((step_num, step))
            except (IndexError, ValueError):
                other_steps.append(step)
        else:
            other_steps.append(step)
    
    timing_tuples.sort(key=lambda x: x[0])
    other_steps.sort()
    
    return [step for _, step in timing_tuples] + other_steps

def sort_math_cts_conditions(conditions):
    """Sort math_cts conditions with base_only first, then math_cts."""
    sorted_conditions = []
    if "base_only" in conditions:
        sorted_conditions.append("base_only")
    if "math_cts" in conditions:
        sorted_conditions.append("math_cts")
    
    # Add any other conditions alphabetically
    other_conditions = [c for c in conditions if c not in ["base_only", "math_cts"]]
    other_conditions.sort()
    sorted_conditions.extend(other_conditions)
    
    return sorted_conditions

if __name__ == "__main__":
    main()
