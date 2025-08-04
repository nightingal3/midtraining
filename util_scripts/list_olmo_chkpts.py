#!/usr/bin/env python3
"""
Script to select OLMo checkpoints with smart interval selection.
Handles both stage1-step{X}-tokens{Y} and stage2-ingredient{Z}-step{X}-tokens{Y} formats.
"""

import argparse
import sys
import os
import re
import random
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    print("Required package 'huggingface_hub' not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi, snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Select OLMo checkpoints with smart interval selection")
    
    # Repository info
    parser.add_argument("repo_id", type=str, help="Hugging Face repository ID")
    parser.add_argument("--token", type=str, help="Hugging Face API token for private repos")
    
    # Filtering options
    parser.add_argument("--pattern", type=str, help="Filter branches by regex pattern")
    parser.add_argument("--stage", type=str, choices=["1", "2", "both"], default="both",
                       help="Filter by stage (1, 2, or both)")
    parser.add_argument("--key-points", action="store_true", 
                       help="Select key training points (1T, 2T, 3T tokens)")
    parser.add_argument("--min-tokens", type=float, 
                       help="Minimum token count in billions (e.g., 5 for 5B)")
    parser.add_argument("--max-tokens", type=float, 
                       help="Maximum token count in billions (e.g., 20 for 20B)")
    
    # Selection options
    parser.add_argument("--count", type=int, default=8, 
                       help="Total number of checkpoints to select")
    parser.add_argument("--stage1-count", type=int,
                       help="Number of checkpoints to select from stage 1 (default: proportional)")
    parser.add_argument("--stage2-count", type=int,
                       help="Number of checkpoints to select from stage 2 (default: proportional)")
    parser.add_argument("--sort-by", choices=["name", "tokens"], default="tokens", 
                       help="Sort branches by name or token count")
    parser.add_argument("--save-list", type=str, help="Save full branch list to file")
    
    # Download options
    parser.add_argument("--download", action="store_true", help="Download selected branches")
    parser.add_argument("--output-dir", type=str, default="./downloaded_checkpoints_olmo",
                      help="Directory to save downloaded checkpoints")
    parser.add_argument("--selected-file", type=str, 
                      help="File with list of branches to download (one per line)")
    
    # Integration with evaluation
    parser.add_argument("--run-evals", action="store_true", 
                      help="Run evaluation script on downloaded checkpoints")
    parser.add_argument("--eval-script", type=str, default="./run_evals.sh",
                      help="Path to evaluation script")
    parser.add_argument("--eval-args", type=str, 
                      help="Additional arguments to pass to the evaluation script")
    
    return parser.parse_args()

def parse_olmo_branch(branch_name):
    """
    Parse OLMo branch name to extract stage, ingredient, step, and token information.
    Handles both formats:
    - stage1-step1000-tokens5B
    - stage2-ingredient1-step1000-tokens5B
    """
    # Try stage2 format first
    stage2_pattern = r'stage2-ingredient(\d+)-step(\d+)-tokens(\d+[KMBT]?)'
    match = re.search(stage2_pattern, branch_name)
    if match:
        ingredient = int(match.group(1))
        step = int(match.group(2))
        token_str = match.group(3)
        tokens = convert_token_str(token_str)
        
        return {
            "name": branch_name,
            "stage": 2,
            "ingredient": ingredient,
            "step": step,
            "tokens": tokens,
            "token_str": token_str
        }
    
    # Try stage1 format
    stage1_pattern = r'stage1-step(\d+)-tokens(\d+[KMBT]?)'
    match = re.search(stage1_pattern, branch_name)
    if match:
        step = int(match.group(1))
        token_str = match.group(2)
        tokens = convert_token_str(token_str)
        
        return {
            "name": branch_name,
            "stage": 1,
            "ingredient": None,
            "step": step,
            "tokens": tokens,
            "token_str": token_str
        }
    
    # Try generic token format as fallback
    token_pattern = r'tokens[_-]?(\d+[KMBT]?)'
    match = re.search(token_pattern, branch_name)
    if match:
        token_str = match.group(1)
        tokens = convert_token_str(token_str)
        
        # Try to infer stage from name
        stage = 1
        if "stage2" in branch_name:
            stage = 2
        elif "stage1" in branch_name:
            stage = 1
        
        return {
            "name": branch_name,
            "stage": stage,
            "ingredient": None,
            "step": None,
            "tokens": tokens,
            "token_str": token_str
        }
    
    # Couldn't parse
    return None

def convert_token_str(token_str):
    """Convert a token string like '5B' to its numeric value."""
    # Extract the numeric part and the unit
    match = re.match(r'(\d+\.?\d*)([KMBT]?)', token_str, re.IGNORECASE)
    if not match:
        return 0
        
    value = float(match.group(1))
    unit = match.group(2).upper() if match.group(2) else ''
    
    # Convert based on unit
    if unit == 'K':
        return value * 1e3
    elif unit == 'M':
        return value * 1e6
    elif unit == 'B':
        return value * 1e9
    elif unit == 'T':
        return value * 1e12
    else:
        return value

def format_token_count(tokens):
    """Format token count in human-readable form."""
    if tokens is None:
        return ""
    
    if tokens >= 1e12:
        return f"{tokens/1e12:.2f}T"
    elif tokens >= 1e9:
        return f"{tokens/1e9:.2f}B"
    elif tokens >= 1e6:
        return f"{tokens/1e6:.2f}M"
    elif tokens >= 1e3:
        return f"{tokens/1e3:.2f}K"
    else:
        return f"{tokens:.0f}"

def get_olmo_branches(repo_id, token=None, pattern=None, stage=None, min_tokens=None, max_tokens=None):
    """Get OLMo branches with parsed token information."""
    api = HfApi(token=token)
    
    try:
        # Get all branches
        repo_refs = api.list_repo_refs(repo_id)
        branches = list(repo_refs.branches)
        
        # Skip common non-checkpoint branches
        system_branches = ['main', 'master', 'dev', 'development', 'staging', 'production']
        branches = [b.name for b in branches if b.name not in system_branches]
        
        # Apply pattern filter if specified
        if pattern:
            try:
                regex = re.compile(pattern)
                branches = [b for b in branches if regex.search(b)]
            except re.error as e:
                print(f"Error in regex pattern: {e}")
                return []
        
        # Process branches with metadata
        branch_info = []
        for branch in branches:
            info = parse_olmo_branch(branch)
            if info:
                # Apply stage filter if specified
                if stage and stage != "both" and str(info["stage"]) != stage:
                    continue
                    
                # Apply token range filters if specified
                if min_tokens and (info["tokens"] is None or info["tokens"] < min_tokens * 1e9):
                    continue
                if max_tokens and (info["tokens"] is None or info["tokens"] > max_tokens * 1e9):
                    continue
                    
                branch_info.append(info)
        
        return branch_info
        
    except Exception as e:
        print(f"Error accessing repository: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def select_key_token_points(branch_info, target_points=None):
    """
    Select branches closest to key token points (1T, 2T, 3T by default)
    """
    if not branch_info:
        return []
    
    if target_points is None:
        # Default key points in tokens (1T, 2T, 3T)
        target_points = [1e12, 2e12, 3e12]
    
    selected = []
    for target in target_points:
        # Find branch closest to target token count
        closest = min(branch_info, key=lambda x: abs(x["tokens"] - target))
        
        # Only add if not already selected (might happen with sparse checkpoints)
        if closest not in selected:
            selected.append(closest)
    
    return selected

def select_by_interval(branches, count, key_branches=None):
    """Select branches at regular intervals, optionally including key branches."""
    if not branches:
        return []
    
    if len(branches) <= count:
        return branches
    
    # Start with key branches if provided
    result = []
    if key_branches:
        result = key_branches.copy()
        
    # Calculate how many more branches we need
    remaining = count - len(result)
    
    if remaining <= 0:
        return result[:count]  # We already have enough or too many
    
    # Filter out branches that are already selected
    filtered_branches = [b for b in branches if b not in result]
    
    if not filtered_branches:
        return result  # No more branches to select
    
    # Select remaining branches at regular intervals
    step = (len(filtered_branches) - 1) / remaining if remaining > 0 else 1
    
    for i in range(remaining):
        idx = min(int(i * step), len(filtered_branches) - 1)
        result.append(filtered_branches[idx])
    
    # Sort result by token count
    result.sort(key=lambda x: x["tokens"])
    
    return result

def download_branch(repo_id, branch, token=None, output_dir=None):
    """Download a branch using the Hugging Face snapshot_download method."""
    print(f"Downloading branch '{branch}'...")
    
    # Create a clean folder name
    folder_name = branch.replace("/", "_")
    target_dir = os.path.join(output_dir, folder_name)
    
    # Create output directory
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Download the branch
        snapshot_download(
            repo_id=repo_id,
            revision=branch,
            token=token,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Successfully downloaded to {target_dir}")
        return target_dir
        
    except Exception as e:
        print(f"Error downloading branch '{branch}': {e}")
        return None

def main():
    args = parse_args()
    
    # If selected-file is provided, read branches from file
    if args.selected_file:
        try:
            with open(args.selected_file, 'r') as f:
                selected_names = [line.strip().split(':')[-1].strip() for line in f if line.strip()]
            
            # Remove any numbering at the beginning (e.g., "1: branch-name")
            selected_names = [name.split(': ', 1)[-1] if ': ' in name else name for name in selected_names]
            
            print(f"Loaded {len(selected_names)} branches from {args.selected_file}")
            
            # Get all branches to match with loaded names
            all_branches = get_olmo_branches(args.repo_id, args.token)
            
            # Match selected names to branch info
            selected = []
            for name in selected_names:
                matches = [b for b in all_branches if b["name"] == name]
                if matches:
                    selected.append(matches[0])
                else:
                    # Create basic branch info if no match found
                    selected.append({"name": name, "stage": None, "tokens": None, "step": None})
        except Exception as e:
            print(f"Error reading file {args.selected_file}: {e}")
            return
    else:
        # Get branches from repository
        all_branches = get_olmo_branches(
            args.repo_id, 
            args.token, 
            args.pattern, 
            args.stage,
            args.min_tokens,
            args.max_tokens
        )
        
        if not all_branches:
            print("No branches found that match the criteria")
            return
            
        print(f"Found {len(all_branches)} branches in repository {args.repo_id}")
        
        # Group branches by stage
        stage1_branches = [b for b in all_branches if b["stage"] == 1]
        stage2_branches = [b for b in all_branches if b["stage"] == 2]
        
        print(f"  Stage 1: {len(stage1_branches)} branches")
        print(f"  Stage 2: {len(stage2_branches)} branches")
        
        # Sort branches by token count
        if args.sort_by == "tokens":
            stage1_branches.sort(key=lambda x: x["tokens"])
            stage2_branches.sort(key=lambda x: x["tokens"])
        else:
            stage1_branches.sort(key=lambda x: x["name"])
            stage2_branches.sort(key=lambda x: x["name"])
        
        # Save full branch list if requested
        if args.save_list:
            try:
                with open(args.save_list, 'w') as f:
                    f.write("Stage 1 Branches:\n")
                    f.write("-" * 80 + "\n")
                    for i, branch in enumerate(stage1_branches):
                        token_str = f" (Tokens: {format_token_count(branch['tokens'])})"
                        step_str = f" (Step: {branch['step']})" if branch["step"] is not None else ""
                        f.write(f"{i+1}: {branch['name']}{token_str}{step_str}\n")
                    
                    f.write("\nStage 2 Branches:\n")
                    f.write("-" * 80 + "\n")
                    for i, branch in enumerate(stage2_branches):
                        token_str = f" (Tokens: {format_token_count(branch['tokens'])})"
                        step_str = f" (Step: {branch['step']})" if branch["step"] is not None else ""
                        ingredient_str = f" (Ingredient: {branch['ingredient']})" if branch["ingredient"] is not None else ""
                        f.write(f"{i+1}: {branch['name']}{token_str}{step_str}{ingredient_str}\n")
                
                print(f"Saved full branch list to {args.save_list}")
            except Exception as e:
                print(f"Error saving branch list: {e}")
        
        # Select key points if requested
        key_branches = []
        if args.key_points:
            print("\nSelecting key token points (1T, 2T, 3T)...")
            key_branches = select_key_token_points(all_branches)
            print(f"Found {len(key_branches)} branches at key token points")
        
        # Determine how many branches to select from each stage
        total_count = args.count - len(key_branches)
        
        if total_count <= 0:
            # Just use key branches if we already have enough
            selected = key_branches
        else:
            # Calculate stage counts if not specified
            if args.stage1_count is None and args.stage2_count is None:
                # Distribute proportionally
                total_branches = len(stage1_branches) + len(stage2_branches)
                if total_branches > 0:
                    stage1_count = max(1, int(total_count * len(stage1_branches) / total_branches))
                    stage2_count = total_count - stage1_count
                else:
                    stage1_count = stage2_count = 0
            else:
                # Use specified counts
                stage1_count = args.stage1_count or 0
                stage2_count = args.stage2_count or 0
                
                # Adjust if total exceeds requested count
                if stage1_count + stage2_count > total_count:
                    # Reduce proportionally
                    scale = total_count / (stage1_count + stage2_count)
                    stage1_count = int(stage1_count * scale)
                    stage2_count = total_count - stage1_count
            
            print(f"\nSelecting {stage1_count} checkpoints from Stage 1 and {stage2_count} from Stage 2")
            
            # Select branches by interval from each stage
            stage1_selected = select_by_interval(stage1_branches, stage1_count, [b for b in key_branches if b["stage"] == 1])
            stage2_selected = select_by_interval(stage2_branches, stage2_count, [b for b in key_branches if b["stage"] == 2])
            
            # Combine selections
            selected = stage1_selected + stage2_selected
            
            # Add any key branches that weren't selected
            for branch in key_branches:
                if branch not in selected:
                    selected.append(branch)
            
            # Sort by token count
            selected.sort(key=lambda x: x["tokens"] if x["tokens"] is not None else 0)
    
    if not selected:
        print("No branches selected")
        return
    
    # Print selected branches
    print("\nSelected branches:")
    print("-" * 80)
    print(f"{'Branch Name':<50} {'Stage':<6} {'Tokens':<10} {'Step':<8}")
    print("-" * 80)
    for branch in selected:
        stage_str = f"{branch['stage']}" if branch["stage"] is not None else "?"
        token_str = format_token_count(branch["tokens"]) if branch["tokens"] is not None else "?"
        step_str = f"{branch['step']}" if branch["step"] is not None else "?"
        print(f"{branch['name']:<50} {stage_str:<6} {token_str:<10} {step_str:<8}")
    
    # Download branches if requested
    if args.download:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save selected branches to file (for future reference)
        selected_file = Path(args.output_dir) / "selected_branches.txt"
        try:
            with open(selected_file, 'w') as f:
                for branch in selected:
                    f.write(f"{branch['name']}\n")
            print(f"Saved selected branches to {selected_file}")
        except Exception as e:
            print(f"Error saving selected branches: {e}")
        
        # Ask for confirmation
        response = input(f"\nDownload {len(selected)} branches? [y/N] ")
        if response.lower() != "y":
            print("Download cancelled")
            return
        
        # Download each branch
        downloaded_dirs = []
        for branch in selected:
            target_dir = download_branch(
                args.repo_id, 
                branch["name"], 
                token=args.token,
                output_dir=args.output_dir
            )
            if target_dir:
                downloaded_dirs.append(target_dir)
        
        print(f"\nDownload complete! Successfully downloaded {len(downloaded_dirs)}/{len(selected)} branches.")
        print(f"Checkpoints saved to {args.output_dir}")
        
        # Run evaluation if requested
        if args.run_evals and downloaded_dirs:
            print("\nRunning evaluation on downloaded checkpoints...")
            # This part would need to be implemented if you want to run evaluations
    
    else:
        # Print command to download these branches later
        print("\nTo download these branches later, you can use:")
        selected_file = "selected_branches.txt"
        with open(selected_file, 'w') as f:
            for branch in selected:
                f.write(f"{branch['name']}\n")
        print(f"Saved selected branches to {selected_file}")
        print(f"python {sys.argv[0]} {args.repo_id} --selected-file {selected_file} --download")

if __name__ == "__main__":
    main()