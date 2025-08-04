import re
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# Helper to extract integer steps from strings like "step-00002000"
def parse_step(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None

# Define dataset names and file-paths for metrics and checkpoints
datasets = [
    ("GSM8K",
     "./c4_losses_after_gsm8k.csv",
     "./val_loss_comparison_gsm8k.csv"),
    ("PythonCode",
     "./c4_losses_after_pycode.csv",
     "./val_loss_comparison_pycode.csv"),
    ("SocialIQA",
     "./c4_losses_after_social_i_qa.csv",
     "./val_loss_comparison_siqa.csv"),
]

# Base paths to pretrained and finetuned checkpoints
pretrain_fixed_base      = "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/pretrained_chkpts/pythia_70m_128b_fixed"
pretrain_midtrain_base   = "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/pretrained_chkpts/pythia_70m_128b_fixed_midtrain_spikefix"
finetuned_base           = "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/finetuned_chkpts/pythia-70m"

# Gather distance data for each dataset
distance_data = {}

for name, c4_path, val_path in datasets:
    # Load and filter steps >= 10000
    c4 = pd.read_csv(c4_path);  c4['step_int'] = c4['step'].apply(parse_step)
    c4 = c4[c4['step_int'] >= 10000]
    
    steps = sorted(c4['step_int'].unique())
    
    dist_reg = []
    dist_md  = []
    
    for step in steps:
        step_str = f"step-{step:08d}"
        
        # Paths for pretrain checkpoints
        pre_fixed = os.path.join(pretrain_fixed_base, step_str)
        pre_mid   = os.path.join(pretrain_midtrain_base, step_str)
        
        # Paths for finetuned checkpoints
        fine_reg  = os.path.join(finetuned_base, name, f"{step_str}_reg")
        fine_md   = os.path.join(finetuned_base, name, f"{step_str}_md")
        
        # Load models (ensure local caching or proper environment)
        model_pre_fixed = AutoModelForCausalLM.from_pretrained(pre_fixed, torch_dtype=torch.float32)
        model_pre_mid   = AutoModelForCausalLM.from_pretrained(pre_mid,   torch_dtype=torch.float32)
        model_fine_reg  = AutoModelForCausalLM.from_pretrained(fine_reg,  torch_dtype=torch.float32)
        model_fine_md   = AutoModelForCausalLM.from_pretrained(fine_md,   torch_dtype=torch.float32)
        
        # Get final layer weight name (last transformer block MLP output)
        layer_idx = model_pre_fixed.config.n_layers - 1
        layer_name = f"transformer.h.{layer_idx}.mlp.fc_out.weight"
        
        # Extract and flatten
        w_pre_fixed = model_pre_fixed.state_dict()[layer_name].cpu().numpy().ravel()
        w_pre_mid   = model_pre_mid.state_dict()[layer_name].cpu().numpy().ravel()
        w_fine_reg  = model_fine_reg.state_dict()[layer_name].cpu().numpy().ravel()
        w_fine_md   = model_fine_md.state_dict()[layer_name].cpu().numpy().ravel()
        
        # Compute Euclidean distances
        dist_reg.append(np.linalg.norm(w_pre_fixed - w_fine_reg))
        dist_md.append(np.linalg.norm(w_pre_mid   - w_fine_md))
    
    # Store in a DataFrame
    distance_data[name] = pd.DataFrame({
        'step': steps,
        'dist_reg': dist_reg,
        'dist_md': dist_md
    })

# Now plot: 3 rows × 3 cols (C4 loss, In-Domain val loss, Weight distance)
fig, axes = plt.subplots(nrows=3, ncols=3, sharex='col', figsize=(18, 12))

for i, (name, c4_path, val_path) in enumerate(datasets):
    # Load and filter metrics again
    c4 = pd.read_csv(c4_path);  c4['step_int'] = c4['step'].apply(parse_step);  c4 = c4[c4['step_int'] >= 10000]
    val = pd.read_csv(val_path); val['step_int'] = val['step_tag'].apply(parse_step); val = val[val['step_int'] >= 10000]
    df_c4 = c4[['step_int','reg_loss','md_loss']].rename(columns={'step_int':'step'})
    df_val= val[['step_int','fixed_val_loss','midtrain_val_loss']].rename(columns={'step_int':'step'})
    df_dist = distance_data[name]
    
    # 1) C4 forgetting
    ax0 = axes[i,0]
    ax0.plot(df_c4['step'], df_c4['reg_loss'], label='reg_loss')
    ax0.plot(df_c4['step'], df_c4['md_loss'],  label='md_loss', linestyle='--')
    ax0.set_ylabel('C4 Loss')
    ax0.set_title(f'{name} – Forgetting')
    ax0.legend(fontsize='small')
    
    # 2) In-domain performance
    ax1 = axes[i,1]
    ax1.plot(df_val['step'], df_val['fixed_val_loss'],    label='fixed_val')
    ax1.plot(df_val['step'], df_val['midtrain_val_loss'], label='midtrain_val', linestyle='--')
    ax1.set_ylabel('Val Loss')
    ax1.set_title(f'{name} – In-Domain')
    ax1.legend(fontsize='small')
    
    # 3) Weight distance
    ax2 = axes[i,2]
    ax2.plot(df_dist['step'], df_dist['dist_reg'], label='reg_dist')
    ax2.plot(df_dist['step'], df_dist['dist_md'],  label='md_dist', linestyle='--')
    ax2.set_ylabel('Euclidean Distance')
    ax2.set_title(f'{name} – Layer Distortion')
    ax2.legend(fontsize='small')

# common x-label
for ax in axes[-1]:
    ax.set_xlabel('Step')

plt.tight_layout()
plt.show()
