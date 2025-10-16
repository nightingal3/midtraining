# Midtraining Bridges Pretraining and Posttraining Distributions

This repository contains the code and configuration files for the paper "Midtraining Bridges Pretraining and Posttraining Distributions" (2025)

## Overview
This repo provides scripts and configuration templates to run large-scale language model training with midtraining, pretraining, and ablation experiments. All training is managed via YAML config files and SLURM batch scripts.

## Trying a New Midtraining Config
To try a new midtraining experiment:
1. Create a new YAML config in `midtrain_configs/` (see existing examples for structure).
2. Edit the config to specify your desired data, model, and training parameters.

## Sharded mixed dataset behaviour
The repo includes a flexible sharded mixed dataset implementation (`litgpt.data.ShardedMixedDataset`) that:

- Discovers shards under a base data directory by type (main numbered shards `1`, `2`, ..., and typed shards like `c1`, `m1`, `q1`, `w1` etc.).
- Supports three mixing modes:
	- literal: specify exact shard weights via `literal_weights_str` (e.g. `main/1:0.8,q1:0.1,q2:0.1`).
	- weighted: give per-type weights via `mix_weights_str` (e.g. `main:0.8,math:0.2`) which are divided equally across shards of that type.
	- proportional: set `proportional_sampling` to compute weights proportional to shard sizes.
- Uses streaming loaders and `CombinedStreamingDataset` to sample from multiple shards with the computed weights.

Make sure your shard names match the config (literal weights require exact shard IDs like `w1` or `main/1`).

### Data types & prefixes

- Prefixes are simple labels used to group shards. Examples: `main` (numbered folders `1`, `2`), `c` → `c1` (code), `m` → `m1` (math), `w` → `w1` (web).
- Use `mix_weights_str` for per-type weights (divided across shards), `literal_weights_str` for exact shard IDs (e.g. `w1` or `main/1`), or enable `proportional_sampling` to weight by shard size.
- To add a new type: pick a short prefix, add it to `DATASET_TYPE_CONFIGS` in `litgpt/litgpt/data/sharded_mixed_dataset.py`, name shards `p1`,`p2`,... and use the label in your config.

## How midtraining is implemented
Midtraining experiments are implemented simply by resuming from an intermediate checkpoint and changing the dataset blend or mixing config in the YAML. In practice you:

- Point `out_dir` and `resume: true` to the checkpointed run you want to continue from.
- Update `mix_weights_str`, `literal_weights_str`, or `mix_config_path` in the new midtraining YAML to change the data blend.
- Launch training; the code will load the checkpoint and continue training with the new data mixture.


## Training Scripts
- `training_scripts/small_model_pretrain.sh`: Main SLURM script for launching pretraining or midtraining jobs. It supports array jobs and can be pointed to any config file.
- Other scripts in `training_scripts/` and `util_scripts/` provide evaluation, symlinking, and utility functions.

## Running with a Custom Config
You can run the small model pretrain script with a specific config file by passing it as an environment variable:

```bash
export model_config_file=/path/to/your_config.yaml
sbatch training_scripts/small_model_pretrain.sh
```

Alternatively, you can edit the script to directly set `model_config_file` to your config path.

---
For more details, see the example configs in `midtrain_configs/` and the comments in each script.

## Citation 
 
TODO

