#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# =============================================================================
# 1) ARGPARSE + GLOBAL DEVICE
# =============================================================================

parser = argparse.ArgumentParser(
    description="Compute CKA trajectory across multiple Pythia checkpoints using a chosen probe."
)
parser.add_argument(
    "--hf_root", type=str, required=True,
    help="Root directory containing Pythia HF checkpoints (subfolders: step-XXXXXX/)."
)
parser.add_argument(
    "--probe", type=str, required=True, choices=["humaneval", "gsm8k"],
    help="Probe dataset to use: 'humaneval' (code) or 'gsm8k' (math)."
)
parser.add_argument(
    "--layer_idx", type=int, default=-1,
    help="Hidden layer index to extract (0-based). Use -1 for final layer."
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="Batch size for activation extraction."
)
parser.add_argument(
    "--max_length", type=int, default=256,
    help="Maximum token length for probe sequences."
)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 2) PROBE DATASET CLASSES
# =============================================================================

class HumanEvalCodeDataset(Dataset):
    """
    Loads the HumanEval 'test' split (164 code examples) and tokenizes the
    'canonical_solution' field.
    """
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ds = load_dataset("openai/openai_humaneval", split="test")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        code = self.ds[idx]["canonical_solution"]
        enc = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


class GSM8KDataset(Dataset):
    """
    Loads the GSM8K 'validation' split (1000 math problems) and tokenizes
    "question + answer" together.
    """
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ds = load_dataset("gsm8k", "main", split="validation")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        q = self.ds[idx]["question"]
        a = self.ds[idx]["answer"]
        text = f"{q} {a}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


def build_probe_loader(probe_name, tokenizer, batch_size, max_length):
    """
    Returns a DataLoader for the specified probe dataset.
    """
    if probe_name == "humaneval":
        dataset = HumanEvalCodeDataset(tokenizer, max_length)
    elif probe_name == "gsm8k":
        dataset = GSM8KDataset(tokenizer, max_length)
    else:
        raise ValueError(f"Unknown probe: {probe_name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# =============================================================================
# 3) CKA IMPLEMENTATION
# =============================================================================

def center_matrix(A: np.ndarray) -> np.ndarray:
    """
    Column‐center the matrix A of shape [n_examples, hidden_dim].
    """
    return A - A.mean(axis=0, keepdims=True)

def linear_CKA(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute linear CKA between activation matrices A and B
    (each shape [n_examples, hidden_dim]).
    """
    A_cent = center_matrix(A)
    B_cent = center_matrix(B)
    n = A_cent.shape[0]
    HSIC_AB = np.linalg.norm(A_cent.T @ B_cent, 'fro')**2 / ((n - 1)**2)
    HSIC_AA = np.linalg.norm(A_cent.T @ A_cent, 'fro')**2 / ((n - 1)**2)
    HSIC_BB = np.linalg.norm(B_cent.T @ B_cent, 'fro')**2 / ((n - 1)**2)
    return HSIC_AB / np.sqrt(HSIC_AA * HSIC_BB + 1e-12)

def get_hidden_activations(model, dataloader, layer_idx):
    """
    Extract hidden states at `layer_idx` from `model` for all examples in `dataloader`.
    Returns an [n_examples, hidden_dim] NumPy array.
    """
    model.eval()
    accum = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden = outputs.hidden_states[layer_idx]  # [B, seq_len, hidden_dim]
            pooled = hidden.mean(dim=1)                # [B, hidden_dim]
            accum.append(pooled.cpu().numpy())
    return np.concatenate(accum, axis=0)  # [n_examples, hidden_dim]


# =============================================================================
# 4) MAIN: ITERATE OVER CHECKPOINTS, COMPUTE CKA TRAJECTORY
# =============================================================================

def main():
    hf_root = args.hf_root.rstrip("/")
    probe    = args.probe
    layer_idx = args.layer_idx
    batch_size = args.batch_size
    max_length = args.max_length

    # Gather all "step-*" subdirectories under hf_root
    subdirs = sorted(
        [d for d in os.listdir(hf_root) if d.startswith("step-")],
        key=lambda x: int(x.split("-")[-1])
    )
    if not subdirs:
        raise FileNotFoundError(f"No 'step-*' folders found under {hf_root}")
    print("Found checkpoints:", subdirs)

    # Load Pythia tokenizer from the first checkpoint
    first_ckpt = subdirs[0]
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(hf_root, first_ckpt), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build probe DataLoader
    probe_loader = build_probe_loader(probe, tokenizer, batch_size, max_length)
    print(f"Probe '{probe}' size:", len(probe_loader.dataset))

    # Extract H0 from the first checkpoint
    model0 = AutoModelForCausalLM.from_pretrained(
        os.path.join(hf_root, first_ckpt),
        local_files_only=True
    ).to(DEVICE)
    H0 = get_hidden_activations(model0, probe_loader, layer_idx=layer_idx)
    print(f"H0 shape: {H0.shape}")

    # Loop over each checkpoint and compute CKA vs. H0
    steps = []
    cka_scores = []
    for ckpt in tqdm(subdirs, desc="Computing CKA"):
        print(f"Processing checkpoint: {ckpt}")
        steps.append(int(ckpt.split("-")[-1]) // 1000)  # in thousands
        if ckpt == first_ckpt:
            cka_scores.append(1.0)
            continue

        model_t = AutoModelForCausalLM.from_pretrained(
            os.path.join(hf_root, ckpt),
            local_files_only=True
        ).to(DEVICE)
        Ht = get_hidden_activations(model_t, probe_loader, layer_idx=layer_idx)
        score = linear_CKA(H0, Ht)
        cka_scores.append(score)
        del model_t
        torch.cuda.empty_cache()

    # Plot the trajectory
    plt.figure(figsize=(7, 4))
    plt.plot(steps, cka_scores, marker="o", linestyle="-", color="tab:orange")
    plt.xlabel("Pretraining Step (×1000)")
    plt.ylabel(f"CKA vs {subdirs[0]} (Layer {layer_idx})")
    plt.title(f"Pythia CKA Trajectory on '{probe}' Probe")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cka_trajectory_{probe}_layer{layer_idx}.png")


if __name__ == "__main__":
    main()
