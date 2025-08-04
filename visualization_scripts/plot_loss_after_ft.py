import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import matplotlib as mpl

# —––––– helper to extract integer steps —–––––
def parse_step(x):
    if x == 'final':
        return args.final_step  # Use the specified final step
    m = re.search(r"step-(\d+)", str(x))
    return int(m.group(1)) if m else None

def smooth_series(series, window=3):
    """Apply simple moving average smoothing"""
    if len(series) < window:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

# —––––– parse command line arguments —–––––
parser = argparse.ArgumentParser(description='Plot finetune loss comparison')
parser.add_argument(
    '--csv_file',
    type=str,
    required=True,
    help='Path to the CSV file with loss results'
)
parser.add_argument(
    '--starts_at',
    type=int,
    default=0,
    help='Starting step to filter data (default: 0)'
)
parser.add_argument(
    '--output',
    type=str,
    default='finetune_loss_comparison.png',
    help='Output filename (default: finetune_loss_comparison.png)'
)
parser.add_argument(
    '--smooth',
    action='store_true',
    help='Apply smoothing to the loss curves'
)
parser.add_argument(
    '--smooth_window',
    type=int,
    default=3,
    help='Window size for smoothing (default: 3)'
)
parser.add_argument(
    '--hide_middle',
    action='store_true',
    help='Hide checkpoints between 30k-60k steps (deprecated - now filters to 10k multiples)'
)
parser.add_argument(
    '--final_step',
    type=int,
    default=61035,
    help='Step number for final checkpoint (default: 61035)'
)
args = parser.parse_args()

# Load the CSV file
if not os.path.exists(args.csv_file):
    raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

df = pd.read_csv(args.csv_file)
print(f"Loaded {len(df)} rows from {args.csv_file}")

# Extract step numbers and filter
df['step_int'] = df['step_tag'].apply(parse_step)
df = df[df['step_int'] >= args.starts_at].copy()

# Filter to only multiples of 10k and final checkpoint
df = df[(df['step_int'] % 10000 == 0) | (df['step_int'] == args.final_step)].copy()

# Exclude step-50000 for pycode and gsm8k datasets due to high variance outliers
if 'pycode' in args.csv_file.lower():
    df = df[df['step_int'] != 50000].copy()
    print("Excluded step-50000 for pycode dataset due to high variance")
elif 'gsm8k' in args.csv_file.lower():
    df = df[df['step_int'] != 50000].copy()
    print("Excluded step-50000 for gsm8k dataset due to high variance")

print(f"After filtering to 10k multiples + final: {len(df)} rows")

print(f"Remaining steps: {sorted(df['step_int'].unique())}")

# Sort by step for proper line plots (final will be at the end due to inf value)
df = df.sort_values('step_int')

# Identify all mix columns dynamically
all_columns = df.columns.tolist()
mix_names = []

# List of mixes to exclude
excluded_mixes = ['flan_high_mix']

# For pycode datasets, only include starcoder_mix to reduce noise
if 'pycode' in args.csv_file.lower():
    print("Pycode dataset detected - only showing fixed baseline vs starcoder_mix")
    # Find all mix types first
    all_mix_types = []
    for col in all_columns:
        if col.endswith('_val_loss_avg') and not col.startswith('fixed_'):
            mix_name = col.replace('_val_loss_avg', '')
            all_mix_types.append(mix_name)
    
    # Exclude everything except starcoder_mix
    for mix_type in all_mix_types:
        if mix_type != 'starcoder_mix':
            excluded_mixes.append(mix_type)

# For gsm8k datasets, only include math_mix to reduce noise
elif 'gsm8k' in args.csv_file.lower():
    print("GSM8K dataset detected - only showing fixed baseline vs math_mix")
    # Find all mix types first
    all_mix_types = []
    for col in all_columns:
        if col.endswith('_val_loss_avg') and not col.startswith('fixed_'):
            mix_name = col.replace('_val_loss_avg', '')
            all_mix_types.append(mix_name)
    
    # Exclude everything except math_mix
    for mix_type in all_mix_types:
        if mix_type != 'math_mix':
            excluded_mixes.append(mix_type)

for col in all_columns:
    if col.endswith('_val_loss_avg') and not col.startswith('fixed_'):
        mix_name = col.replace('_val_loss_avg', '')
        if mix_name not in excluded_mixes:
            mix_names.append(mix_name)

print(f"Found {len(mix_names)} mix columns (excluding {excluded_mixes}): {mix_names}")

# Set global font sizes for publication-quality figures
mpl.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'legend.title_fontsize': 24
})

# Create two subplots with less horizontal aspect
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))

# Define a more appealing color palette
if 'pycode' in args.csv_file.lower():
    # For pycode: grey baseline, red starcoder
    colors = ['#707070', '#E74C3C']  # Grey, red
elif 'gsm8k' in args.csv_file.lower():
    # For gsm8k: grey baseline, red math_mix
    colors = ['#707070', '#E74C3C']  # Grey, red
else:
    # For other datasets: use a broader appealing palette
    colors = ['#2E8B57', '#FF6B35', '#4A90E2', '#8E44AD', '#E67E22', '#27AE60', '#E74C3C']
    # Sea green, orange-red, blue, purple, orange, green, red

def plot_loss_type(ax, loss_type, title, ylabel):
    # Plot fixed baseline with shaded error region
    fixed_col = f'fixed_{loss_type}_avg'
    fixed_std_col = f'fixed_{loss_type}_std'
    if fixed_col in df.columns and fixed_std_col in df.columns:
        # Filter out N/A values and convert to numeric
        valid_mask = (df[fixed_col].notna() & (df[fixed_col] != 'N/A') & 
                     df[fixed_std_col].notna() & (df[fixed_std_col] != 'N/A'))
        valid_df = df[valid_mask].copy()
        if len(valid_df) > 0:
            valid_df[fixed_col] = pd.to_numeric(valid_df[fixed_col])
            valid_df[fixed_std_col] = pd.to_numeric(valid_df[fixed_std_col])
            x_vals = valid_df['step_int']
            y_vals = valid_df[fixed_col]
            y_std = valid_df[fixed_std_col]
            
            # Apply smoothing if requested (but not to error bars)
            if args.smooth:
                y_vals = smooth_series(y_vals, args.smooth_window)
            
            # Plot line
            ax.plot(x_vals, y_vals, label='Fixed Baseline', color=colors[0], linewidth=2)
            
            # Add shaded error region
            ax.fill_between(x_vals, y_vals - y_std, y_vals + y_std, 
                           color=colors[0], alpha=0.2)

    # Plot each mix with shaded error regions
    for i, mix_name in enumerate(mix_names):
        mix_col = f'{mix_name}_{loss_type}_avg'
        mix_std_col = f'{mix_name}_{loss_type}_std'
        if mix_col in df.columns and mix_std_col in df.columns:
            # Filter out N/A values and convert to numeric
            valid_mask = (df[mix_col].notna() & (df[mix_col] != 'N/A') & 
                         df[mix_std_col].notna() & (df[mix_std_col] != 'N/A'))
            valid_df = df[valid_mask].copy()
            if len(valid_df) > 0:
                valid_df[mix_col] = pd.to_numeric(valid_df[mix_col])
                valid_df[mix_std_col] = pd.to_numeric(valid_df[mix_std_col])
                x_vals = valid_df['step_int']
                y_vals = valid_df[mix_col]
                y_std = valid_df[mix_std_col]
                
                # Apply smoothing if requested (but not to error bars)
                if args.smooth:
                    y_vals = smooth_series(y_vals, args.smooth_window)
                
                # Format mix name for display
                display_name = mix_name.replace('_mix', '').replace('_', ' ').title()
                
                # Plot line
                ax.plot(x_vals, y_vals, label=display_name, color=colors[i + 1], 
                       linewidth=2, linestyle='--')
                
                # Add shaded error region
                ax.fill_between(x_vals, y_vals - y_std, y_vals + y_std, 
                               color=colors[i + 1], alpha=0.2)

    # Set labels and title
    ax.set_ylabel(ylabel, fontsize=26)
    ax.set_title(title, fontsize=20)
    # Remove per-axes legend; will add a single combined legend later
    ax.grid(False)

    # Custom x-axis labels to show 'final' properly
    x_ticks = ax.get_xticks()
    x_labels = []
    for tick in x_ticks:
        if abs(tick - args.final_step) < 1000:  # Close to final step
            x_labels.append('final')
        else:
            x_labels.append(f'{int(tick)}')
    ax.set_xticklabels(x_labels, fontsize=22)


# Plot top panel: in-domain validation loss (shorter title)
plot_loss_type(ax1, 'val_loss', 'In-Domain Performance', 'Validation Loss')

# Plot bottom panel: C4 validation loss (shorter title)
plot_loss_type(ax2, 'val_c4_loss', 'C4 Forgetting', 'C4 Validation Loss')

# Set x-axis label only on bottom plot
ax2.set_xlabel('Step', fontsize=26)


# Add a single combined legend at the bottom center
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1
labels = labels1
# Only add handles/labels from ax2 if not already present
for h, l in zip(handles2, labels2):
    if l not in labels:
        handles.append(h)
        labels.append(l)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=len(labels), fontsize=20, frameon=False)

plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig(args.output, dpi=300)
# also save a pdf
plt.savefig(args.output.replace('.png', '.pdf'), dpi=300)
print(f"Plot saved to {args.output}")