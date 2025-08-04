#!/usr/bin/env python3
"""
Report missing seeds across different datasets and model sizes.
Checks for missing runs and seeds across pycode, gsm8k, lima, sciq datasets
and 70m, 160m, 410m model sizes.
"""
import argparse
import csv
from collections import defaultdict
import wandb
from wandb import Api

def get_intermediate_steps_for_mix(mix_type):
    """Get the appropriate intermediate step tags for each mix type based on their starting points."""
    # Define starting points for each mix (in thousands)
    mix_start_points = {
        "regular": 10,  # Regular starts from 10k
        "starcoder_mix": 10,  # Starcoder starts from 6k, but we use 10k increments, so first checkpoint is 10k
        "math_mix": 20,  # Math starts from 20k
        "flan_mix": 40,  # Flan starts from 40k
        "knowledgeqa_mix": 40,  # KnowledgeQA starts from 40k
        "dclm_mix": 40,  # DCLM starts from 40k
        "other_midtrain": 10  # Default to 10k for unclassified
    }
    
    start_k = mix_start_points.get(mix_type, 10)
    max_k = 60  # All go up to 60k
    
    # Generate step tags from start point to 60k in 10k increments
    steps = []
    for k in range(start_k, max_k + 1, 10):
        steps.append(f"step-{k}000")
    
    return steps

def check_ablation_seeds(api, args, dataset, model_size, ablation_type, expected_seeds):
    """Check missing seeds for a specific ablation type."""
    # Ablations only exist for 70m and 160m models
    if model_size not in ["70m", "160m"]:
        # Don't return anything for 410m - no missing seeds to report
        return None
    
    project = f"finetune-pythia-{model_size}"
    group = f"{dataset}_{ablation_type}_{model_size}_sc"
    
    print(f"\nChecking {dataset} {model_size} {ablation_type} (group: {group})")
    print(f"Project: {args.entity}/{project}")
    
    try:
        # Fetch runs from this group
        runs = api.runs(f"{args.entity}/{project}", filters={"group": group})
        print(f"  Found {len(runs)} total runs")
        
        # Filter by masking type based on dataset defaults
        if dataset in ["pycode", "lima"]:
            # For pycode and lima: use unmasked runs (default is unmasked)
            filtered_runs = [r for r in runs if "unmasked_prompt" in r.tags]
            masking_note = "(using unmasked runs)"
        elif dataset in ["gsm8k", "sciq"]:
            # For gsm8k and sciq: use masked runs (default is masked)
            filtered_runs = [r for r in runs if "masked_prompt" in r.tags]
            masking_note = "(using masked runs)"
        else:
            # Fallback: all runs
            filtered_runs = runs
            masking_note = "(all runs)"
        
        print(f"  Found {len(filtered_runs)} relevant runs {masking_note}")
        
        # Organize runs by seed (no need to distinguish regular vs midtrained for ablations)
        runs_by_seed = defaultdict(list)
        
        for run in filtered_runs:
            # Extract seed
            seed = run.config.get("seed", None)
            if seed is None:
                # Try to extract from tags
                for tag in run.tags:
                    if tag.startswith("seed_"):
                        try:
                            seed = int(tag.split("_")[1])
                            break
                        except (IndexError, ValueError):
                            pass
            
            # Default to 1337 if no seed found
            if seed is None:
                seed = 1337
            
            runs_by_seed[seed].append(run)
        
        # Check for missing seeds
        found_seeds = set(runs_by_seed.keys())
        missing_seeds = set(expected_seeds) - found_seeds
        
        print(f"  {ablation_type}: found seeds {sorted(found_seeds)}")
        if missing_seeds:
            print(f"  {ablation_type}: MISSING seeds {sorted(missing_seeds)}")
            
            # Only return a result if there are missing seeds
            return {
                "dataset": dataset,
                "model_size": model_size,
                "project": project,
                "group": group,
                "run_type": ablation_type,
                "total_runs": len(filtered_runs),
                "found_seeds": ";".join(map(str, sorted(found_seeds))),
                "missing_seeds": ";".join(map(str, sorted(missing_seeds))),
                "num_missing": len(missing_seeds),
                "all_seeds_present": len(missing_seeds) == 0
            }
        else:
            # No missing seeds, don't include in report
            return None
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "dataset": dataset,
            "model_size": model_size,
            "project": project,
            "group": group,
            "run_type": f"{ablation_type}_ERROR",
            "total_runs": 0,
            "found_seeds": "",
            "missing_seeds": f"ERROR: {str(e)}",
            "num_missing": len(expected_seeds),
            "all_seeds_present": False
        }

def check_intermediate_steps(api, args, dataset, model_size, expected_seeds):
    """Check missing seeds for intermediate checkpoint steps in non-ablation runs."""
    project = f"finetune-pythia-{model_size}"
    group = f"{dataset}_{args.group_suffix}_{model_size}"
    
    print(f"\n=== Checking intermediate steps for {dataset} {model_size} ===")
    
    try:
        # Fetch runs from this group
        runs = api.runs(f"{args.entity}/{project}", filters={"group": group})
        
        # Filter by masking type based on dataset defaults
        if dataset in ["pycode", "lima"]:
            # For pycode and lima: use unmasked runs (default is unmasked)
            filtered_runs = [r for r in runs if "unmasked_prompt" in r.tags]
            masking_note = "(using unmasked runs)"
        elif dataset in ["gsm8k", "sciq"]:
            # For gsm8k and sciq: use masked runs (default is masked)
            filtered_runs = [r for r in runs if "masked_prompt" in r.tags]
            masking_note = "(using masked runs)"
        else:
            # Fallback: exclude masked_prompt runs
            filtered_runs = [r for r in runs if "masked_prompt" not in r.tags]
            masking_note = "(fallback: excluding masked_prompt)"
        
        print(f"  Found {len(filtered_runs)} relevant runs {masking_note}")
        
        # Define the midtrain mix patterns (excluding flan_high_mix)
        midtrain_mix_tags = [
            "starcoder_mix",
            "math_mix", 
            "flan_mix",
            "knowledgeqa_mix",
            "dclm_mix"
        ]
        
        # Organize runs by step tag, run type, and seed
        intermediate_results = []
        
        # Process regular runs
        regular_runs_by_step_and_seed = defaultdict(lambda: defaultdict(list))
        midtrain_runs_by_mix_step_and_seed = {}
        for mix_tag in midtrain_mix_tags:
            midtrain_runs_by_mix_step_and_seed[mix_tag] = defaultdict(lambda: defaultdict(list))
        # Add catch-all for unclassified midtrain runs
        midtrain_runs_by_mix_step_and_seed["other_midtrain"] = defaultdict(lambda: defaultdict(list))
        
        for run in filtered_runs:
            # Extract seed
            seed = run.config.get("seed", None)
            if seed is None:
                # Try to extract from tags
                for tag in run.tags:
                    if tag.startswith("seed_"):
                        try:
                            seed = int(tag.split("_")[1])
                            break
                        except (IndexError, ValueError):
                            pass
            
            # Default to 1337 if no seed found
            if seed is None:
                seed = 1337
            
            # Look for intermediate step tags
            for tag in run.tags:
                if tag.startswith("step-"):
                    try:
                        # Extract step number, handling both "step-10000" and "step-00010000" formats
                        step_part = tag.split("-")[1]
                        step_num = int(step_part)
                        # Only include if it's a multiple of 10k
                        if step_num % 10000 != 0:
                            continue
                        # Normalize to consistent format "step-10000" (without leading zeros)
                        normalized_tag = f"step-{step_num}"
                    except (IndexError, ValueError):
                        # Skip malformed step tags
                        continue
                    
                    # Categorize by regular vs specific midtrain mix
                    if "regular" in run.tags:
                        regular_runs_by_step_and_seed[normalized_tag][seed].append(run)
                    elif "midtrained" in run.tags:
                        # Determine which specific mix this run belongs to
                        mix_found = False
                        for mix_tag in midtrain_mix_tags:
                            if mix_tag in run.tags:
                                midtrain_runs_by_mix_step_and_seed[mix_tag][normalized_tag][seed].append(run)
                                mix_found = True
                                break
                        
                        # If no specific mix tag found, it's other midtrain
                        if not mix_found:
                            midtrain_runs_by_mix_step_and_seed["other_midtrain"][normalized_tag][seed].append(run)
        
        # Check regular runs for all intermediate steps (10k-60k)
        regular_steps = get_intermediate_steps_for_mix("regular")
        for step_tag in regular_steps:
            found_seeds = set(regular_runs_by_step_and_seed[step_tag].keys())
            missing_seeds = set(expected_seeds) - found_seeds
            
            # Only add to results if there are missing seeds
            if missing_seeds:
                intermediate_results.append({
                    "dataset": dataset,
                    "model_size": model_size,
                    "project": project,
                    "group": group,
                    "run_type": f"regular_{step_tag}",
                    "total_runs": sum(len(runs_list) for runs_list in regular_runs_by_step_and_seed[step_tag].values()),
                    "found_seeds": ";".join(map(str, sorted(found_seeds))),
                    "missing_seeds": ";".join(map(str, sorted(missing_seeds))),
                    "num_missing": len(missing_seeds),
                    "all_seeds_present": len(missing_seeds) == 0
                })
                
                print(f"  Regular {step_tag}: found seeds {sorted(found_seeds)}")
                print(f"  Regular {step_tag}: MISSING seeds {sorted(missing_seeds)}")
        
        # Check midtrain runs for each mix type (with appropriate starting points)
        for mix_name, mix_runs_by_step in midtrain_runs_by_mix_step_and_seed.items():
            if not any(any(step_runs.values()) for step_runs in mix_runs_by_step.values()):
                continue  # Skip if no runs found for this mix
                
            mix_steps = get_intermediate_steps_for_mix(mix_name)
            for step_tag in mix_steps:
                found_seeds = set(mix_runs_by_step[step_tag].keys())
                missing_seeds = set(expected_seeds) - found_seeds
                
                # Only add to results if there are missing seeds
                if missing_seeds:
                    intermediate_results.append({
                        "dataset": dataset,
                        "model_size": model_size,
                        "project": project,
                        "group": group,
                        "run_type": f"{mix_name}_{step_tag}",
                        "total_runs": sum(len(runs_list) for runs_list in mix_runs_by_step[step_tag].values()),
                        "found_seeds": ";".join(map(str, sorted(found_seeds))),
                        "missing_seeds": ";".join(map(str, sorted(missing_seeds))),
                        "num_missing": len(missing_seeds),
                        "all_seeds_present": len(missing_seeds) == 0
                    })
                    
                    print(f"  {mix_name} {step_tag}: found seeds {sorted(found_seeds)}")
                    print(f"  {mix_name} {step_tag}: MISSING seeds {sorted(missing_seeds)}")
        
        return intermediate_results
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return [{
            "dataset": dataset,
            "model_size": model_size,
            "project": project,
            "group": f"{dataset}_{args.group_suffix}_{model_size}",
            "run_type": "intermediate_ERROR",
            "total_runs": 0,
            "found_seeds": "",
            "missing_seeds": f"ERROR: {str(e)}",
            "num_missing": len(expected_seeds),
            "all_seeds_present": False
        }]

def main():
    parser = argparse.ArgumentParser(
        description="Report missing seeds across datasets and model sizes"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="pretraining-and-behaviour",
        help="W&B entity name (default: 'pretraining-and-behaviour')"
    )
    parser.add_argument(
        "--expected_seeds",
        type=str,
        default="1337,895,3641,3181,9762",
        help="Comma-separated list of expected seeds (default: '1337,9762')"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="missing_seeds_report.csv",
        help="Output CSV file name (default: 'missing_seeds_report.csv')"
    )
    parser.add_argument(
        "--group_suffix",
        type=str,
        default="final_fixed",
        help="Group name suffix pattern (default: 'final_fixed')"
    )
    parser.add_argument(
        "--include_ablations",
        action="store_true",
        help="Include ablation experiments in the report"
    )
    parser.add_argument(
        "--include_intermediate",
        action="store_true",
        help="Include intermediate checkpoint steps (10k increments) for non-ablation runs"
    )
    
    args = parser.parse_args()
    
    # Configuration
    datasets = ["pycode", "gsm8k", "lima", "sciq"]
    model_sizes = ["70m", "160m", "410m"]
    expected_seeds = [int(s.strip()) for s in args.expected_seeds.split(",")]
    
    # Ablation types to check if requested
    ablation_types = ["cts_pretrain_ablation", "mixture_weight_ablation", "timing_ablation"] if args.include_ablations else []
    
    print(f"Expected seeds: {expected_seeds}")
    print(f"Checking datasets: {datasets}")
    print(f"Checking model sizes: {model_sizes}")
    if args.include_ablations:
        print(f"Including ablations: {ablation_types}")
    if args.include_intermediate:
        print("Including intermediate checkpoints (10k increments) for non-ablation runs")
    print("=" * 60)
    
    # Authenticate
    wandb.login()
    api = Api()
    
    # Results storage
    results = []
    
    for dataset in datasets:
        for model_size in model_sizes:
            project = f"finetune-pythia-{model_size}"
            group = f"{dataset}_{args.group_suffix}_{model_size}"
            
            print(f"\nChecking {dataset} {model_size} (group: {group})")
            print(f"Project: {args.entity}/{project}")
            
            try:
                # Fetch runs from this group
                runs = api.runs(f"{args.entity}/{project}", filters={"group": group})
                print(f"  Found {len(runs)} total runs")
                
                # Filter to relevant runs based on dataset's default masking behavior
                if dataset in ["pycode", "lima"]:
                    # For pycode and lima: use unmasked runs (default is unmasked)
                    filtered_runs = [r for r in runs if "unmasked_prompt" in r.tags]
                    masking_note = "(using unmasked runs)"
                elif dataset in ["gsm8k", "sciq"]:
                    # For gsm8k and sciq: use masked runs (default is masked)
                    filtered_runs = [r for r in runs if "masked_prompt" in r.tags]
                    masking_note = "(using masked runs)"
                else:
                    # Fallback: exclude masked_prompt runs
                    filtered_runs = [r for r in runs if "masked_prompt" not in r.tags]
                    masking_note = "(fallback: excluding masked_prompt)"
                
                print(f"  Found {len(filtered_runs)} relevant runs {masking_note}")
                
                # Define the midtrain mix patterns (excluding flan_high_mix)
                midtrain_mix_tags = [
                    "starcoder_mix",
                    "math_mix", 
                    "flan_mix",
                    "knowledgeqa_mix",
                    "dclm_mix"
                ]
                
                # Organize runs by checkpoint type and seed
                regular_runs_by_seed = defaultdict(list)
                midtrain_runs_by_mix_and_seed = {}
                for mix_tag in midtrain_mix_tags:
                    midtrain_runs_by_mix_and_seed[mix_tag] = defaultdict(list)
                # Add a catch-all for unclassified midtrain runs
                midtrain_runs_by_mix_and_seed["other_midtrain"] = defaultdict(list)
                
                for run in filtered_runs:
                    # Extract seed
                    seed = run.config.get("seed", None)
                    if seed is None:
                        # Try to extract from tags
                        for tag in run.tags:
                            if tag.startswith("seed_"):
                                try:
                                    seed = int(tag.split("_")[1])
                                    break
                                except (IndexError, ValueError):
                                    pass
                    
                    # Default to 1337 if no seed found
                    if seed is None:
                        seed = 1337
                    
                    # Categorize by regular vs specific midtrain mix
                    if "regular" in run.tags:
                        regular_runs_by_seed[seed].append(run)
                    elif "midtrained" in run.tags:
                        # Determine which specific mix this run belongs to
                        mix_found = False
                        for mix_tag in midtrain_mix_tags:
                            if mix_tag in run.tags:
                                midtrain_runs_by_mix_and_seed[mix_tag][seed].append(run)
                                mix_found = True
                                break
                        
                        # If no specific mix tag found, it's the original starcoder mix or other
                        if not mix_found:
                            midtrain_runs_by_mix_and_seed["other_midtrain"][seed].append(run)
                
                # Check for missing seeds
                found_regular_seeds = set(regular_runs_by_seed.keys())
                missing_regular_seeds = set(expected_seeds) - found_regular_seeds
                
                # Report findings for regular runs
                print(f"  Regular runs: found seeds {sorted(found_regular_seeds)}")
                if missing_regular_seeds:
                    print(f"  Regular runs: MISSING seeds {sorted(missing_regular_seeds)}")
                
                # Add regular results (only if there are missing seeds)
                if missing_regular_seeds:
                    results.append({
                        "dataset": dataset,
                        "model_size": model_size,
                        "project": project,
                        "group": group,
                        "run_type": "regular",
                        "total_runs": len([r for r in filtered_runs if "regular" in r.tags]),
                        "found_seeds": ";".join(map(str, sorted(found_regular_seeds))),
                        "missing_seeds": ";".join(map(str, sorted(missing_regular_seeds))),
                        "num_missing": len(missing_regular_seeds),
                        "all_seeds_present": len(missing_regular_seeds) == 0
                    })
                
                # Check and report for each midtrain mix
                for mix_name, mix_runs_by_seed in midtrain_runs_by_mix_and_seed.items():
                    # Only report mixes that have runs
                    if not any(mix_runs_by_seed.values()):
                        continue
                        
                    found_mix_seeds = set(mix_runs_by_seed.keys())
                    missing_mix_seeds = set(expected_seeds) - found_mix_seeds
                    
                    print(f"  {mix_name} runs: found seeds {sorted(found_mix_seeds)}")
                    if missing_mix_seeds:
                        print(f"  {mix_name} runs: MISSING seeds {sorted(missing_mix_seeds)}")
                    
                    # Only add to results if there are missing seeds
                    if missing_mix_seeds:
                        results.append({
                            "dataset": dataset,
                            "model_size": model_size,
                            "project": project,
                            "group": group,
                            "run_type": mix_name,
                            "total_runs": sum(len(runs) for runs in mix_runs_by_seed.values()),
                            "found_seeds": ";".join(map(str, sorted(found_mix_seeds))),
                            "missing_seeds": ";".join(map(str, sorted(missing_mix_seeds))),
                            "num_missing": len(missing_mix_seeds),
                            "all_seeds_present": len(missing_mix_seeds) == 0
                        })
                
            except Exception as e:
                print(f"  ERROR: {e}")
                # Add error entry to results
                results.append({
                    "dataset": dataset,
                    "model_size": model_size,
                    "project": project,
                    "group": group,
                    "run_type": "ERROR",
                    "total_runs": 0,
                    "found_seeds": "",
                    "missing_seeds": f"ERROR: {str(e)}",
                    "num_missing": len(expected_seeds),
                    "all_seeds_present": False
                })
            
            # Check ablations if requested
            if args.include_ablations:
                for ablation_type in ablation_types:
                    ablation_result = check_ablation_seeds(api, args, dataset, model_size, ablation_type, expected_seeds)
                    if ablation_result is not None:  # Only add if there are missing seeds
                        results.append(ablation_result)
            
            # Check intermediate steps if requested (only for non-ablation runs)
            if args.include_intermediate:
                intermediate_results = check_intermediate_steps(api, args, dataset, model_size, expected_seeds)
                results.extend(intermediate_results)
    
    # Write results to CSV
    print(f"\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    with open(args.output_file, "w", newline="") as csvfile:
        fieldnames = [
            "dataset", "model_size", "project", "group", "run_type",
            "total_runs", "found_seeds", "missing_seeds", "num_missing", "all_seeds_present"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    # Print summary
    missing_count = sum(1 for r in results if r["num_missing"] > 0 and r["run_type"] != "ERROR")
    total_count = len([r for r in results if r["run_type"] != "ERROR"])
    error_count = len([r for r in results if r["run_type"] == "ERROR"])
    
    print(f"📊 SUMMARY:")
    print(f"  Total dataset/model/type combinations: {total_count}")
    print(f"  Combinations with missing seeds: {missing_count}")
    print(f"  Combinations with errors: {error_count}")
    print(f"  Complete combinations: {total_count - missing_count}")
    
    print(f"\n🔍 MISSING SEED DETAILS:")
    for result in results:
        if result["num_missing"] > 0 and result["run_type"] != "ERROR":
            print(f"  {result['dataset']:8} {result['model_size']:4} {result['run_type']:15}: missing {result['missing_seeds']}")
    
    if error_count > 0:
        print(f"\n❌ ERROR DETAILS:")
        for result in results:
            if result["run_type"] == "ERROR":
                print(f"  {result['dataset']:8} {result['model_size']:4}: {result['missing_seeds']}")
    
    # Print breakdown by run type
    run_type_counts = defaultdict(int)
    for result in results:
        if result["run_type"] != "ERROR":
            run_type_counts[result["run_type"]] += 1
    
    print(f"\n📋 RUN TYPE BREAKDOWN:")
    for run_type, count in sorted(run_type_counts.items()):
        missing_for_type = sum(1 for r in results if r["run_type"] == run_type and r["num_missing"] > 0)
        print(f"  {run_type:15}: {count:2} combinations ({missing_for_type} with missing seeds)")
    
    print(f"\n✅ Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
