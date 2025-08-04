#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from itertools import islice
import json
import os
import random

def extract_text(sample, dataset):
    """
    Return a single 'text' string for each sample across multiple dataset types.
    """
    if dataset in ("openai/gsm8k",):
        return sample["question"] + " " + sample["answer"]
    if dataset in ("allenai/social_i_qa", "social_i_qa"):
        choices = [sample[f"answer{opt}"] for opt in ["A","B","C"]]
        num_to_choice = {"1": "A", "2": "B", "3": "C"}
        correct_label = num_to_choice[sample["label"]]
        correct = sample[f"answer{correct_label}"]
        return sample["context"] + " " + sample["question"] + " " + correct
    if dataset in ("bigcode/starcoderdata", "starcoder-python"):
        return sample["content"]
    if dataset in ("allenai/c4", "c4"):
        return sample["text"]
    if dataset in ("mattymchen/mr","mr"):
        sentiment = "positive" if sample["label"] == 1 else "negative"
        return sample["text"] + " " + sentiment
    if dataset.startswith("nyu-mll/glue") and sample.get("sentence1") and sample.get("sentence2"):
        entail = "entailment" if sample["label"] == 1 else "not_entailment"
        return sample["sentence1"] + " " + sample["sentence2"] + " " + entail
    if dataset in ("CogComp/trec","trec"):
        return sample["text"] + " " + str(sample["coarse_label"])
    # Check if this is a local dataset format (instruction + input + output)
    if "instruction" in sample and "input" in sample and "output" in sample:
        return sample["instruction"] + " " + sample["input"] + " " + sample["output"]
    # Fallback: concatenate any available text‑like fields
    parts = []
    for key in ("input","instruction","question","text","content"):
        if key in sample and isinstance(sample[key], str):
            parts.append(sample[key])
    return " ".join(parts).strip()

def load_local_jsonl(file_path, sample_size=100):
    """
    Load samples from a local JSONL file.
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        
    # Shuffle the lines for random sampling
    random.shuffle(all_lines)
    
    for line in all_lines[:sample_size]:
        line = line.strip()
        if line:
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                continue
                
    return samples

def is_local_dataset(dataset_name):
    """
    Check if the dataset name refers to a local file path.
    """
    return os.path.exists(dataset_name) or (not "/" in dataset_name and os.path.exists(f"{dataset_name}/train.jsonl"))

def compute_avg_embedding(dataset_name, model, split="train", sample_size=100, streaming=True, local_data_dir=None):
    """
    Stream the dataset or load local data, extract text, encode to embeddings, and average them.
    """
    texts = []
    
    if is_local_dataset(dataset_name) or local_data_dir:
        # Handle local dataset
        if local_data_dir:
            # If local_data_dir is provided, look for the dataset there
            file_path = os.path.join(local_data_dir, dataset_name, f"{split}.jsonl")
        else:
            # Direct path or relative path
            if os.path.isfile(dataset_name):
                file_path = dataset_name
            else:
                file_path = os.path.join(dataset_name, f"{split}.jsonl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local dataset file not found: {file_path}")
            
        print(f"Loading local dataset from: {file_path}")
        samples = load_local_jsonl(file_path, sample_size)
        
        for sample in samples:
            txt = extract_text(sample, dataset_name)
            if txt:
                texts.append(txt)
    else:
        # Handle Hugging Face dataset (original code)
        if dataset_name == "allenai/c4":
            ds = load_dataset(dataset_name, "en", split=split, streaming=streaming, trust_remote_code=True)
        elif dataset_name == "openai/gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split=split, streaming=streaming, trust_remote_code=True)
        elif dataset_name == "mattymchen/mr":
            ds = load_dataset("mattymchen/mr", split="test", streaming=streaming, trust_remote_code=True)
        elif dataset_name == "nyu-mll/glue":
            ds = load_dataset("nyu-mll/glue", "rte", split=split, streaming=streaming, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=True)

        ds = ds.shuffle(buffer_size=10_000, seed=42)  # Approx shuffle
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)
    
    if not texts:
        raise ValueError(f"No valid texts extracted from dataset: {dataset_name}")
        
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return np.mean(embeddings, axis=0)

def cosine_sim(a, b):
    """Compute cosine similarity via dot product and norms."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_display_name(dataset_name, local_data_dir=None):
    """
    Get a clean display name for the dataset.
    """
    if is_local_dataset(dataset_name) or local_data_dir:
        if local_data_dir and not os.path.isfile(dataset_name):
            # This is a dataset name within local_data_dir
            return dataset_name
        else:
            # This is a direct path - get the parent directory name
            return os.path.basename(os.path.dirname(dataset_name)) if os.path.isfile(dataset_name) else os.path.basename(dataset_name.rstrip('/'))
    else:
        return dataset_name

def main(datasets, sample_size, model_name, output_file, local_data_dir):
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load SentenceTransformer model for document embeddings
    model = SentenceTransformer(model_name)
    names = datasets
    display_names = [get_display_name(name, local_data_dir) for name in names]
    embs = []
    
    for ds in names:
        print(f"→ Processing {ds} …")
        emb = compute_avg_embedding(ds, model, sample_size=sample_size, local_data_dir=local_data_dir)
        embs.append(emb)
        
    N = len(names)
    sim = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):  # Only compute upper triangle, excluding diagonal
            sim[i, j] = cosine_sim(embs[i], embs[j])
            
    # Plot similarity matrix with pcolor and save to file
    fig, ax = plt.subplots(figsize=(8, 8))
    c = ax.pcolor(sim, cmap="viridis", edgecolors='k', linewidths=0.5)
    ax.set_xticks(np.arange(0.5, N, 1)); ax.set_yticks(np.arange(0.5, N, 1))
    ax.set_xticklabels(display_names, rotation=45, ha="right"); ax.set_yticklabels(display_names)
    
    # Add similarity values as text labels
    for i in range(N):
        for j in range(i + 1, N):
            ax.text(j + 0.5, i + 0.5, f"{sim[i, j]:.2f}", 
                    ha="center", va="center", color="white", fontsize=8)
    
    fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✔ Saved similarity matrix to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Hugging Face datasets and local datasets via doc embeddings and cosine similarity"
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="List of dataset IDs (e.g., allenai/c4 openai/gsm8k) or local dataset names/paths"
    )
    parser.add_argument(
        "--sample_size", type=int, default=1000,
        help="Number of examples per dataset to sample (default: 1000)"
    )
    parser.add_argument(
        "--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model for embeddings (default: all-mpnet-base-v2)"
    )
    parser.add_argument(
        "--output_file", type=str, default="similarity_matrix.png",
        help="Filename for the saved similarity matrix image"
    )
    parser.add_argument(
        "--local_data_dir", type=str, default=None,
        help="Base directory for local datasets (e.g., ../manifold_data/knowledgeqa_formatted_revised/)"
    )
    args = parser.parse_args()
    main(args.datasets, args.sample_size, args.model_name, args.output_file, args.local_data_dir)