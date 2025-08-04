#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# 1) ARGPARSE + GLOBAL DEVICE
# =============================================================================

parser = argparse.ArgumentParser(
    description="Compute a layer-wise CKA similarity matrix between base and fine-tuned models."
)
parser.add_argument(
    "--base_model", type=str, required=True,
    help="Path to the 'before fine-tuning' HF model directory (e.g. hf_ckpts/vanilla/step0)."
)
parser.add_argument(
    "--ft_model", type=str, required=True,
    help="Path to the 'after fine-tuning' HF model directory (e.g. hf_ckpts/vanilla/finetune_v0)."
)
parser.add_argument(
    "--probe", type=str, required=True, choices=["humaneval", "gsm8k"],
    help="Which probe to use: 'humaneval' or 'gsm8k'."
)
parser.add_argument(
    "--max_layer", type=int, required=True,
    help="Number of transformer layers (e.g. 12 for Pythia-70M)."
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="Batch size for extraction."
)
parser.add_argument(
    "--max_length", type=int, default=256,
    help="Max token length for tokenization."
)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 2) PROBE DATASET CLASSES
# =============================================================================

class HumanEvalCodeDataset(Dataset):
    """ HumanEval test (164 examples) """
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
    """ GSM8K validation (1000 examples) """
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

def build_probe_loader(probe_name, tokenizer):
    """
    Returns DataLoader for the probe dataset (supporting 'humaneval' and 'gsm8k').
    """
    if probe_name == "humaneval":
        ds = HumanEvalCodeDataset(tokenizer, args.max_length)
    elif probe_name == "gsm8k":
        ds = GSM8KDataset(tokenizer, args.max_length)
    else:
        raise ValueError(f"Unknown probe: {probe_name}")
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False)


# =============================================================================
# 3) CKA FUNCTIONS
# =============================================================================

def center_matrix(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)

def linear_CKA(A: np.ndarray, B: np.ndarray) -> float:
    A_cent = center_matrix(A)
    B_cent = center_matrix(B)
    n = A_cent.shape[0]
    HSIC_AB = np.linalg.norm(A_cent.T @ B_cent, 'fro')**2 / ((n - 1)**2)
    HSIC_AA = np.linalg.norm(A_cent.T @ A_cent, 'fro')**2 / ((n - 1)**2)
    HSIC_BB = np.linalg.norm(B_cent.T @ B_cent, 'fro')**2 / ((n - 1)**2)
    return HSIC_AB / np.sqrt(HSIC_AA * HSIC_BB + 1e-12)

def get_layer_activations(model, dataloader, layer_idx):
    """
    Extract [batch, seq_len, hidden_dim] for one layer;
    then average-pool over seq_len to get [batch, hidden_dim].
    Accumulate over all batches to return [n_examples, hidden_dim].
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
    return np.concatenate(accum, axis=0)            # [n_examples, hidden_dim]


# =============================================================================
# 4) MAIN: BUILD CKA MATRIX
# =============================================================================

def main():
    base_dir = args.base_model.rstrip("/")
    ft_dir   = args.ft_model.rstrip("/")
    probe    = args.probe
    max_layer = args.max_layer

    # Verify both directories exist
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base model dir not found: {base_dir}")
    if not os.path.isdir(ft_dir):
        raise FileNotFoundError(f"Fine-tuned model dir not found: {ft_dir}")

    # 4a) Load Pythia tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4b) Build probe DataLoader
    probe_loader = build_probe_loader(probe, tokenizer)
    n_examples = len(probe_loader.dataset)
    print(f"Probe '{probe}' has {n_examples} examples.")

    # 4c) Load both models
    model_base = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True).to(DEVICE)
    model_ft   = AutoModelForCausalLM.from_pretrained(ft_dir,   local_files_only=True).to(DEVICE)

    # 4d) Prepare an empty CKA matrix of size (num_layers+1) x (num_layers+1)
    # We consider hidden_states indices: 0…max_layer inclusive.
    L = max_layer + 1
    cka_matrix = np.zeros((L, L), dtype=float)

    # 4e) Extract and store activations layer by layer
    #    Save each [n_examples x hidden_dim] matrix in a list for reuse.
    base_activations = []
    ft_activations = []
    for layer_idx in tqdm(range(L), desc="Extracting activations"):
        print(f"Processing layer {layer_idx}...")
        H_base = get_layer_activations(model_base, probe_loader, layer_idx)
        H_ft   = get_layer_activations(model_ft,   probe_loader, layer_idx)
        base_activations.append(H_base)
        ft_activations.append(H_ft)

    # 4f) Compute pairwise CKA across all layers
    for i in range(L):
        for j in range(L):
            cka_matrix[i, j] = linear_CKA(base_activations[i], ft_activations[j])

    # 4g) Plot heatmap
    layer_labels = [f"L{i}" for i in range(L)]
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cka_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="CKA Similarity")
    plt.xticks(ticks=np.arange(L), labels=layer_labels, rotation=45)
    plt.yticks(ticks=np.arange(L), labels=layer_labels)
    plt.xlabel("Fine‐Tuned Model Layers")
    plt.ylabel("Base Model Layers")
    plt.title(f"Layer‐wise CKA: {os.path.basename(base_dir)} vs {os.path.basename(ft_dir)} on '{probe}'")

    # Annotate each cell with the CKA value (optional)
    for i in range(L):
        for j in range(L):
            plt.text(j, i, f"{cka_matrix[i,j]:.2f}", ha="center", va="center", color="white" if cka_matrix[i,j]<0.5 else "black", fontsize=6)

    plt.tight_layout()
    plt.savefig(f"cka_matrix_{os.path.basename(base_dir)}_vs_{os.path.basename(ft_dir)}_{probe}.png", dpi=300)


if __name__ == "__main__":
    main()
