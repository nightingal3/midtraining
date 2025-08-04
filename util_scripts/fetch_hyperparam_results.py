#!/usr/bin/env python3
"""
Fetch best learning_rate and val_loss per 'step-XXXXX' tag from a W&B group,
print them out and save to CSV.
"""

import os
import csv
from collections import defaultdict

import wandb
from wandb import Api

def main():
    # 1. Authenticate (ensure WANDB_API_KEY is set in your environment)
    wandb.login()

    ENTITY  = "pretraining-and-behaviour"
    PROJECT = "finetune-pythia-410m"
    GROUP   = "param_sweep_revised_sciq_410m_revised"
    specific_dsets = None

    api = Api()

    # 2. Fetch all runs in the specified group
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=10000)

    # 3. Group runs by their "step-XXXXX" tag
    tagged_runs = defaultdict(list)
    for run in runs:
        if specific_dsets and specific_dsets not in run.tags:
            continue
        for tag in run.tags:
            if tag.startswith("step-") or tag.startswith("final"):
                tagged_runs[tag].append(run)

    # 4. Find best run per tag (min val_loss) and collect its lr and loss
    best_per_tag = {}
    for tag, runs_list in tagged_runs.items():
        best_run = min(
            runs_list,
            key=lambda r: r.summary.get("val_loss", float("inf"))
        )
        lr       = best_run.config["train"]["max_lr"]
        val_loss = best_run.summary.get("val_loss", None)
        best_per_tag[tag] = {"learning_rate": lr, "val_loss": val_loss}

    # 5. Print results sorted by tag
    for tag in sorted(best_per_tag):
        info = best_per_tag[tag]
        print(f"{tag}: learning_rate = {info['learning_rate']}, val_loss = {info['val_loss']}")

    # 6. Write results out to CSV
    out_file = f"best_lrs_per_step_{GROUP}.csv"
    if specific_dsets:
        out_file = f"best_lrs_per_step_{specific_dsets}.csv"
    with open(out_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step_tag", "learning_rate", "val_loss"])
        for tag in sorted(best_per_tag):
            info = best_per_tag[tag]
            writer.writerow([tag, info["learning_rate"], info["val_loss"]])

    print(f"\n✅ Results saved to {out_file}")

if __name__ == "__main__":
    main()
