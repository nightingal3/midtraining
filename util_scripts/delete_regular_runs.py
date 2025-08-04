#!/usr/bin/env python3
"""
Delete all runs tagged with 'regular' from a specific W&B group.
"""
import wandb
from wandb import Api

def delete_regular_runs():
    # Authenticate
    wandb.login()
    
    ENTITY = "pretraining-and-behaviour"
    PROJECT = "finetune-pythia-160m"
    GROUP = "pycode_final_fixed_160m"
    
    api = Api()
    
    # Fetch all runs in the group
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP})
    
    # Find runs tagged with 'regular'
    regular_runs = [r for r in runs if "regular" in r.tags]
    
    print(f"Found {len(regular_runs)} runs tagged with 'regular':")
    for run in regular_runs:
        print(f"  - {run.name} (ID: {run.id}) - Tags: {run.tags}")
    
    # Safety check
    if len(regular_runs) == 0:
        print("No runs found with 'regular' tag.")
        return
    
    # Confirm deletion
    response = input(f"\nAre you sure you want to delete these {len(regular_runs)} runs? (yes/no): ")
    if response.lower() != 'yes':
        print("Deletion cancelled.")
        return
    
    # Delete runs
    print(f"\nDeleting {len(regular_runs)} runs...")
    for i, run in enumerate(regular_runs, 1):
        try:
            run.delete()
            print(f"  {i}/{len(regular_runs)}: Deleted {run.name} (ID: {run.id})")
        except Exception as e:
            print(f"  {i}/{len(regular_runs)}: Failed to delete {run.name} (ID: {run.id}) - Error: {e}")
    
    print("\n✅ Deletion complete!")

if __name__ == "__main__":
    delete_regular_runs()