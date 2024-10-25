import argparse
import glob
import json
import math
import os
import re

import matplotlib.pyplot as plt


def extract_step(filename):
    match = re.search(r"step-(\d{8})", filename)
    return int(match.group(1)) if match else None


def extract_accuracy(data, keys):
    return {key: data.get(key, {}).get("acc,none") for key in keys}


def load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)


def filter_steps(accuracies):
    for key in accuracies:
        if accuracies[key]:
            max_step = max(step for step, _ in accuracies[key])
            accuracies[key] = [
                (step, acc)
                for step, acc in accuracies[key]
                if step % 1000 == 0 or step == max_step
            ]
    return accuracies


def plot_accuracies(accuracies, keys):
    num_plots = len(keys)
    rows = math.ceil(math.sqrt(num_plots))
    cols = math.ceil(num_plots / rows)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle("Accuracy over training", fontsize=16)

    if num_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for i, key in enumerate(keys):
        ax = axs[i]
        if key in accuracies and accuracies[key]:
            steps = [step for step, acc in accuracies[key]]
            values = [acc for step, acc in accuracies[key]]

            # Plot pretraining steps in blue
            ax.plot(
                steps[:-1], values[:-1], marker="o", color="blue", label="Pretraining"
            )

            # Plot final step in red
            ax.plot(
                steps[-1],
                values[-1],
                marker="o",
                color="red",
                label="Final decay (fw only)",
            )

            ax.set_title(key)
            ax.set_xlabel("Steps")
            ax.set_ylabel("Accuracy")
            ax.grid(True)

            # Set x-axis ticks to show all step values
            ax.set_xticks(steps)
            ax.set_xticklabels(steps, rotation=45, ha="right")

            # Add legend
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No data for {key}", ha="center", va="center")
            ax.set_title(key)

    # Remove any unused subplots
    for i in range(num_plots, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig("accuracy_over_time.png", dpi=300)


def main(base_dir):
    keys_to_extract = [
        "arc_easy",
        "commonsense_qa",
        "hellaswag",
        "logiqa2",
        "mathqa",
        "mmlu",
        "piqa",
        "sciq",
    ]

    checkpoint_files = glob.glob(os.path.join(base_dir, "step-*", "results.json"))

    accuracies = {key: [] for key in keys_to_extract}

    for file in checkpoint_files:
        step = extract_step(file)
        if step is None:
            continue

        data = load_results(file)
        acc_values = extract_accuracy(data, keys_to_extract)

        for key, value in acc_values.items():
            if value is not None:
                accuracies[key].append((step, value))

    for key in accuracies:
        accuracies[key].sort(key=lambda x: x[0])
        if not accuracies[key]:
            print(f"Warning: No data collected for key '{key}'")

    accuracies = filter_steps(accuracies)

    plot_accuracies(accuracies, keys_to_extract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot accuracy over time from checkpoint results."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/users/nightingal3/manifold/all_in_one_pretraining/training_analysis/pythia-1b-fw/evals",
        help="Base directory containing checkpoint folders (default: ./checkpoints)",
    )
    args = parser.parse_args()

    main(args.base_dir)
