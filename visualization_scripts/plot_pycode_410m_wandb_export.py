import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "./visualization_scripts/from_wandb/pycode_410m.csv"
df = pd.read_csv(csv_path)

# Extract steps and losses for each run
steps = df["trainer/global_step"]
starcoder_loss = df["dulcet-forest-1280 - loss"]
base_loss = df["sparkling-shadow-779 - loss"]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, base_loss, label="Base", color="#707070", linewidth=2)
plt.plot(steps, starcoder_loss, label="Starcoder", color="#E74C3C", linewidth=2)

plt.xlabel("Step", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.title("Representative train loss curve (Pythia 410m, Pycode)", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig("pycode_410m_wandb_export.png", dpi=300)
plt.savefig("pycode_410m_wandb_export.pdf", dpi=300)
print("Saved to pycode_410m_wandb_export.png and .pdf")
