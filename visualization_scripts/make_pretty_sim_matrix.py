import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Load and clean
df = pd.read_csv("token_similarity_corrected_07_20.csv", comment='#', index_col=0)
for rm in ["Movie Reviews", "Social IQA", "TREC", "starcoder-cts", "StarCoder-CTS"]:
    if rm in df.index: df = df.drop(index=rm)
    if rm in df.columns: df = df.drop(columns=rm)

# Select subset
subset = ["C4", "StarCoder", "Math Combined", "PyCode", "GSM8K"]
df_small = df.loc[subset, subset]

# Plot compact version
labels = subset
n = len(labels)
fig, ax = plt.subplots(figsize=(4, 4))  # smaller overall
cmap = "viridis"
c = ax.pcolor(df_small.values, cmap=cmap, edgecolors=None, linewidths=0)  # remove heavy grid

# Ticks and labels with bigger font
ax.set_xticks(np.arange(0.5, n, 1))
ax.set_yticks(np.arange(0.5, n, 1))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight='medium')
ax.set_yticklabels(labels, fontsize=10, fontweight='medium')

# Annotate with larger text
norm = colors.Normalize(vmin=np.nanmin(df_small.values), vmax=np.nanmax(df_small.values))
for i in range(n):
    for j in range(n):
        val = df_small.values[i, j]
        rgba = plt.cm.get_cmap(cmap)(norm(val))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "black" if lum > 0.6 else "white"
        ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

# Clean up visuals
ax.set_frame_on(False)
ax.tick_params(length=0)  # remove tick marks
cb = fig.colorbar(c, ax=ax, fraction=0.08, pad=0.02)
cb.ax.tick_params(labelsize=8)
cb.set_label("Similarity", fontsize=9)

plt.tight_layout()
fig.savefig("cleaned_similarity_matrix_mini_compact.png", dpi=300, bbox_inches="tight")
fig.savefig("cleaned_similarity_matrix_mini_compact.pdf", format="pdf", bbox_inches="tight")
