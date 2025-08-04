#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from itertools import islice
import json
import os
import random
import glob
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from collections import defaultdict

def extract_text(sample, dataset):
    """Return a single 'text' string for each sample across multiple dataset types."""
    if dataset in ("openai/gsm8k",):
        return sample["question"] + " " + sample.get("answer", sample.get("solution", ""))
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
    if dataset in ("GAIR/lima",):
        return " ".join(sample["conversations"])
    if dataset in ("math_combined",):
        return sample.get("text", "")
    if dataset in ("flan_combined",):
        return sample.get("text", "")
    if dataset.startswith("nyu-mll/glue") or dataset == "nyu-mll/glue":
        if sample.get("sentence1") and sample.get("sentence2"):
            entail = "entailment" if sample["label"] == 1 else "not_entailment"
            return sample["sentence1"] + " " + sample["sentence2"] + " " + entail
    if dataset in ("ai2_arc",):
        # ARC format: question + choices + correct answer
        question = sample.get("question", "")
        choices = sample.get("choices", {})
        if isinstance(choices, dict):
            choice_text = " ".join([f"{label}) {text}" for label, text in zip(choices.get("label", []), choices.get("text", []))])
            answer_key = sample.get("answerKey", "")
            return f"{question} {choice_text} Answer: {answer_key}"
        return question
    if dataset in ("sciq",):
        # SciQ format: question + correct answer
        question = sample.get("question", "")
        correct_answer = sample.get("correct_answer", "")
        return f"{question} {correct_answer}"
    if dataset in ("pycode",):
        # For code datasets, return the code content
        return sample.get("func_code_string", sample.get("content", sample.get("code", "")))
    
    # Check if this is a local dataset format (instruction + input + output)
    if "instruction" in sample and "input" in sample and "output" in sample:
        return sample["instruction"] + " " + sample["input"] + " " + sample["output"]
    
    # Fallback
    parts = []
    for key in ("input","instruction","question","text","content","code","func_code_string"):
        if key in sample and isinstance(sample[key], str):
            parts.append(sample[key])
    return " ".join(parts).strip()

def load_math_combined(sample_size=100):
    """Load and combine math datasets."""
    print("Loading math datasets...")
    
    # Load OpenMath
    openmath = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
    openmath_samples = []
    for i, example in enumerate(islice(openmath, sample_size // 3)):
        text = f"{example['question']}\n\n{example['expected_answer']}"
        openmath_samples.append({"text": text})
    
    # Load MathInstruct  
    mathinstruct = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
    mathinstruct_samples = []
    for i, example in enumerate(islice(mathinstruct, sample_size // 3)):
        text = f"{example['instruction']}\n\n{example['output']}"
        mathinstruct_samples.append({"text": text})
    
    # Load MATH-plus
    mathplus = load_dataset("TIGER-Lab/MATH-plus", split="train", streaming=True)
    mathplus_samples = []
    for i, example in enumerate(islice(mathplus, sample_size // 3)):
        text = f"{example['instruction']}\n\n{example['output']}"
        mathplus_samples.append({"text": text})
    
    all_samples = openmath_samples + mathinstruct_samples + mathplus_samples
    random.shuffle(all_samples)
    return all_samples[:sample_size]

def load_flan_combined(sample_size=100):
    """Load and combine FLAN datasets."""
    print("Loading FLAN datasets...")
    flan_dir = "/data/tir/projects/tir3/users/mengyan3/manifold_data/datasets/flan/"
    
    train_files = sorted(glob.glob(os.path.join(flan_dir, "*_train.jsonl")))
    if not train_files:
        # Try alternative: load from HuggingFace directly
        print("FLAN files not found locally, trying HuggingFace...")
        try:
            # Load a subset of FLAN datasets from HF
            ds = load_dataset("tau/scrolls", "qasper", split="train", streaming=True)
            samples = []
            for example in islice(ds, sample_size):
                if example.get("input") and example.get("output"):
                    text = example["input"] + "\n\n" + example["output"]
                    samples.append({"text": text})
            if samples:
                return samples
        except:
            pass
        
        # Last resort fallback
        print("WARNING: Using placeholder FLAN data - results will be invalid!")
        return [{"text": f"FLAN instruction {i} with output {i}"} for i in range(sample_size)]
    
    raw = load_dataset("json", data_files={"train": train_files}, split="train", streaming=True)
    raw = raw.shuffle(buffer_size=10_000, seed=42)
    
    samples = []
    for example in islice(raw, sample_size):
        if example.get("inputs") and example.get("targets"):
            text = example["inputs"].strip() + "\n\n" + example["targets"].strip()
            samples.append({"text": text})
    
    return samples

def get_dataset_texts(dataset_name, sample_size):
    """Extract texts from dataset."""
    texts = []
    
    if dataset_name == "allenai/c4":
        print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name, "en", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "bigcode/starcoderdata":
        print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "math_combined":
        samples = load_math_combined(sample_size)
        for sample in samples:
            txt = extract_text(sample, dataset_name)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "flan_combined":
        samples = load_flan_combined(sample_size)
        for sample in samples:
            txt = extract_text(sample, dataset_name)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "pycode":
        print(f"Loading {dataset_name}...")
        # Based on your reference, this seems to be a local code dataset
        # Let's create a fallback that loads from HuggingFace if local file not found
        try:
            file_path = "../manifold_data/all_in_one_pretraining/datasets/just_Nan-Do/code-search-net-python/Nan-Do/code-search-net-python/train.json"
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    random.shuffle(data)
                    samples = data[:sample_size]
                else:
                    samples = [data]
                
                for sample in samples:
                    txt = extract_text(sample, dataset_name)
                    if txt:
                        texts.append(txt)
            else:
                # Fallback: use CodeSearchNet Python from HuggingFace
                print("  Local pycode file not found, using CodeSearchNet Python from HF...")
                ds = load_dataset("code_search_net", "python", split="train", streaming=True)
                ds = ds.shuffle(buffer_size=10_000, seed=42)
                for example in islice(ds, sample_size):
                    if example.get("func_code_string"):
                        texts.append(example["func_code_string"])
        except Exception as e:
            print(f"  Error loading pycode: {e}")
            # Create minimal fallback
            for i in range(min(sample_size, 100)):
                texts.append(f"def function_{i}():\n    return {i}")
                
    elif dataset_name == "allenai/social_i_qa":
        print(f"Loading {dataset_name}...")
        try:
            ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                txt = extract_text(example, dataset_name)
                if txt and len(txt.strip()) > 0:
                    texts.append(txt)
        except Exception as e:
            print(f"  Error loading social_i_qa: {e}")
            print("  Using placeholder data for Social IQA...")
            for i in range(min(sample_size, 100)):
                texts.append(f"Context: Person {i} is in a social situation. Question: What will happen next? Answer: They will respond appropriately.")
                
    elif dataset_name == "nyu-mll/glue":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("nyu-mll/glue", "rte", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "ai2_arc":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "sciq":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("sciq", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "GAIR/lima":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("GAIR/lima", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)
    
    elif dataset_name == "openai/gsm8k":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name)
            if txt:
                texts.append(txt)

    return texts

def select_important_parameters(model, layer_selection="key_layers"):
    """
    Select which parameters to include in gradient computation.
    
    Options:
    - "all": All parameters (very expensive)
    - "key_layers": Embeddings + output layer + layer norms  
    - "embeddings_only": Just embedding layers
    - "output_only": Just the language modeling head
    - "by_layer_type": Specific layer types
    """
    
    selected_params = []
    param_info = []
    
    if layer_selection == "all":
        for name, param in model.named_parameters():
            if param.requires_grad:
                selected_params.append(param)
                param_info.append(name)
                
    elif layer_selection == "key_layers":
        # Focus on layers that are most important for task adaptation
        key_patterns = [
            'embed',  # embedding layers
            'lm_head',  # language modeling head
            'ln',  # layer norms
            'layernorm',  # alternative layer norm naming
            'norm',  # general normalization
        ]
        
        for name, param in model.named_parameters():
            if param.requires_grad and any(pattern in name.lower() for pattern in key_patterns):
                selected_params.append(param)
                param_info.append(name)
                
    elif layer_selection == "embeddings_only":
        for name, param in model.named_parameters():
            if param.requires_grad and 'embed' in name.lower():
                selected_params.append(param)
                param_info.append(name)
                
    elif layer_selection == "output_only":
        for name, param in model.named_parameters():
            if param.requires_grad and ('lm_head' in name.lower() or 'output' in name.lower()):
                selected_params.append(param)
                param_info.append(name)
                
    elif layer_selection == "first_last":
        # First few and last few layers
        all_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        n_params = len(all_params)
        
        # Take first 20% and last 20% of parameters
        first_n = max(1, n_params // 5)
        last_n = max(1, n_params // 5)
        
        selected_indices = list(range(first_n)) + list(range(n_params - last_n, n_params))
        
        for idx in selected_indices:
            name, param = all_params[idx]
            selected_params.append(param)
            param_info.append(name)
    
    print(f"Selected {len(selected_params)} parameter groups out of {sum(1 for _ in model.parameters())} total:")
    for info in param_info[:5]:  # Show first 5
        print(f"  - {info}")
    if len(param_info) > 5:
        print(f"  ... and {len(param_info) - 5} more")
    
    return selected_params

def compute_dataset_gradients(texts, model, tokenizer, selected_params, max_length=256, 
                            max_samples=50, device='cuda'):
    """
    Compute average gradients for a dataset.
    
    This implements ∇L_i from your theory.
    """
    model.train()  # Enable gradient computation
    
    # Sample texts to keep computation manageable
    if len(texts) > max_samples:
        sampled_texts = random.sample(texts, max_samples)
    else:
        sampled_texts = texts
    
    print(f"  Computing gradients on {len(sampled_texts)} samples...")
    print(f"  Sample text preview: {sampled_texts[0][:100]}...")
    
    all_gradients = []
    valid_samples = 0
    
    for i, text in enumerate(sampled_texts):
        if i % 10 == 0:
            print(f"    Processing sample {i+1}/{len(sampled_texts)}")
        
        try:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                             truncation=True, padding=True).to(device)
            
            # Skip very short sequences
            if inputs["input_ids"].shape[1] < 5:
                print(f"    Skipping sample {i}: too short")
                continue
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass with labels for language modeling loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    Skipping sample {i}: invalid loss {loss}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Collect gradients from selected parameters
            sample_gradients = []
            total_grad_norm = 0
            
            for param in selected_params:
                if param.grad is not None:
                    grad_data = param.grad.flatten().detach().cpu().numpy()
                    sample_gradients.append(grad_data)
                    total_grad_norm += np.linalg.norm(grad_data)**2
            
            if sample_gradients and total_grad_norm > 0:
                # Concatenate all gradients into a single vector
                gradient_vector = np.concatenate(sample_gradients)
                all_gradients.append(gradient_vector)
                valid_samples += 1
            else:
                print(f"    Skipping sample {i}: no valid gradients")
            
        except Exception as e:
            print(f"    Error processing sample {i}: {e}")
            continue
    
    if not all_gradients:
        raise ValueError("No gradients were successfully computed!")
    
    print(f"  Successfully computed gradients for {valid_samples}/{len(sampled_texts)} samples")
    
    # Average gradients across samples
    mean_gradient = np.mean(all_gradients, axis=0)
    
    # Sanity check: gradient should not be all zeros
    grad_norm = np.linalg.norm(mean_gradient)
    print(f"  Average gradient norm: {grad_norm:.6f}")
    
    if grad_norm < 1e-8:
        print("  WARNING: Very small gradient norm!")
    
    return mean_gradient

def compute_gradient_alignment(grad_i, grad_j):
    """
    Compute gradient alignment as defined in your theory.
    
    Returns:
    - g_ij: normalized dot product ∇L_j^T ∇L_i / ||∇L_j||^2  
    - alpha_ij: misalignment measure (1 - cosine similarity)
    - cosine_sim: standard cosine similarity
    """
    
    # Compute norms
    norm_i = np.linalg.norm(grad_i)
    norm_j = np.linalg.norm(grad_j)
    
    if norm_i == 0 or norm_j == 0:
        return 0.0, 1.0, 0.0
    
    # Dot product
    dot_product = np.dot(grad_i, grad_j)
    
    # g_ij as defined in your theory: ∇L_j^T ∇L_i / ||∇L_j||^2
    g_ij = dot_product / (norm_j ** 2)
    
    # Standard cosine similarity
    cosine_sim = dot_product / (norm_i * norm_j)
    
    # α_ij = 1 - g_ij (misalignment measure from your theory)
    # But we'll use cosine similarity version: α_ij = 1 - cosine_sim
    alpha_ij = 1.0 - cosine_sim
    
    return g_ij, alpha_ij, cosine_sim

def compute_raw_gradient_similarities(all_datasets, model, tokenizer, sample_size=100, 
                                    layer_selection="key_layers", device='cuda'):
    """
    Simple function to compute raw gradient cosine similarities between datasets.
    No bridging analysis, just the basic similarity matrix.
    """
    
    # Select which parameters to use for gradient computation
    selected_params = select_important_parameters(model, layer_selection)
    
    # Load texts for all datasets
    print("Loading dataset texts...")
    all_texts = {}
    for dataset in all_datasets:
        print(f"→ Loading {dataset}")
        texts = get_dataset_texts(dataset, sample_size)
        if not texts:
            raise ValueError(f"No texts found for dataset: {dataset}")
        all_texts[dataset] = texts
        print(f"  Loaded {len(texts)} texts")
        
        # Quick sample check
        sample_text = texts[0][:150] + "..." if len(texts[0]) > 150 else texts[0]
        print(f"  Sample: {sample_text}")
    
    # Compute gradients for each dataset
    print("\nComputing gradients for each dataset...")
    dataset_gradients = {}
    
    for dataset in all_datasets:
        print(f"→ Computing gradients for {dataset}")
        gradients = compute_dataset_gradients(
            all_texts[dataset], model, tokenizer, selected_params, 
            max_samples=min(50, len(all_texts[dataset])), device=device
        )
        dataset_gradients[dataset] = gradients
    
    # Compute similarity matrix
    print("\nComputing gradient similarity matrix...")
    N = len(all_datasets)
    cosine_matrix = np.zeros((N, N))
    
    for i, dataset_i in enumerate(all_datasets):
        for j, dataset_j in enumerate(all_datasets):
            if i == j:
                cosine_matrix[i, j] = 1.0
            else:
                grad_i = dataset_gradients[dataset_i]
                grad_j = dataset_gradients[dataset_j]
                
                # Simple cosine similarity
                norm_i = np.linalg.norm(grad_i)
                norm_j = np.linalg.norm(grad_j)
                
                if norm_i > 0 and norm_j > 0:
                    cosine_sim = np.dot(grad_i, grad_j) / (norm_i * norm_j)
                    cosine_matrix[i, j] = cosine_sim
                else:
                    cosine_matrix[i, j] = 0.0
                
                print(f"  {dataset_i} vs {dataset_j}: {cosine_sim:.3f}")
    
    return {
        'cosine_matrix': cosine_matrix,
        'dataset_gradients': dataset_gradients
    }
    """
    Quick embedding-based similarity for comparison.
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Computing embedding-based similarity for comparison...")
        st_model = SentenceTransformer('all-mpnet-base-v2')
        
        dataset_embeddings = {}
        for dataset in all_datasets:
            texts = all_texts[dataset][:20]  # Small sample for speed
            embeddings = st_model.encode(texts)
            mean_embedding = np.mean(embeddings, axis=0)
            dataset_embeddings[dataset] = mean_embedding
        
        # Compute embedding similarity matrix
        n = len(all_datasets)
        emb_sim_matrix = np.zeros((n, n))
        
        for i, ds1 in enumerate(all_datasets):
            for j, ds2 in enumerate(all_datasets):
                if i == j:
                    emb_sim_matrix[i, j] = 1.0
                else:
                    emb1 = dataset_embeddings[ds1]
                    emb2 = dataset_embeddings[ds2]
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    emb_sim_matrix[i, j] = sim
        
        print("Embedding-based similarities:")
        for i, ds1 in enumerate(all_datasets):
            for j, ds2 in enumerate(all_datasets):
                if i < j:  # Only upper triangle
                    print(f"  {ds1} ↔ {ds2}: {emb_sim_matrix[i, j]:.3f}")
        
        return emb_sim_matrix
        
    except Exception as e:
        print(f"Could not compute embedding baseline: {e}")
        return None
def compute_gradient_alignment_matrix(all_datasets, model, tokenizer, sample_size=100, 
                                    layer_selection="key_layers", device='cuda'):
    """
    Compute the full gradient alignment matrix between all datasets.
    
    This implements the g_ij matrix from your theory.
    """
    
    # Select which parameters to use for gradient computation
    selected_params = select_important_parameters(model, layer_selection)
    
    # Load texts for all datasets
    print("Loading dataset texts...")
    all_texts = {}
    for dataset in all_datasets:
        print(f"→ Loading {dataset}")
        texts = get_dataset_texts(dataset, sample_size)
        if not texts:
            raise ValueError(f"No texts found for dataset: {dataset}")
        all_texts[dataset] = texts
        print(f"  Loaded {len(texts)} texts")
        
        # Print sample for verification
        sample_text = texts[0][:200] + "..." if len(texts[0]) > 200 else texts[0]
        print(f"  Sample: {sample_text}")
        
        # Check text quality
        avg_length = np.mean([len(text.split()) for text in texts[:10]])
        print(f"  Average word count (first 10): {avg_length:.1f}")
        
        if avg_length < 5:
            print(f"  WARNING: Very short texts for {dataset}")
    
    # Compare with embedding baseline
    emb_sim_matrix = compare_with_embedding_baseline(all_texts, all_datasets)
    
    # Compute gradients for each dataset
    print("\nComputing gradients for each dataset...")
    dataset_gradients = {}
    
    for dataset in all_datasets:
        print(f"→ Computing gradients for {dataset}")
        gradients = compute_dataset_gradients(
            all_texts[dataset], model, tokenizer, selected_params, 
            max_samples=min(50, len(all_texts[dataset])), device=device
        )
        dataset_gradients[dataset] = gradients
    
    # Compute alignment matrix
    print("\nComputing gradient alignment matrix...")
    N = len(all_datasets)
    
    g_matrix = np.zeros((N, N))  # g_ij values from your theory
    alpha_matrix = np.zeros((N, N))  # α_ij misalignment values
    cosine_matrix = np.zeros((N, N))  # Standard cosine similarities
    
    for i, dataset_i in enumerate(all_datasets):
        for j, dataset_j in enumerate(all_datasets):
            if i == j:
                g_matrix[i, j] = 1.0
                alpha_matrix[i, j] = 0.0  
                cosine_matrix[i, j] = 1.0
            else:
                grad_i = dataset_gradients[dataset_i]
                grad_j = dataset_gradients[dataset_j]
                
                g_ij, alpha_ij, cosine_ij = compute_gradient_alignment(grad_i, grad_j)
                
                g_matrix[i, j] = g_ij
                alpha_matrix[i, j] = alpha_ij
                cosine_matrix[i, j] = cosine_ij
                
                print(f"  {dataset_i} vs {dataset_j}: g_ij={g_ij:.3f}, α_ij={alpha_ij:.3f}, cos={cosine_ij:.3f}")
    
    # Sanity checks
    print(f"\nSanity checks:")
    print(f"  Cosine similarity range: [{cosine_matrix[cosine_matrix != 1.0].min():.3f}, {cosine_matrix[cosine_matrix != 1.0].max():.3f}]")
    print(f"  Gradient norm analysis:")
    
    grad_norms = {}
    for i, dataset in enumerate(all_datasets):
        grad_norm = np.linalg.norm(dataset_gradients[dataset])
        grad_norms[dataset] = grad_norm
        print(f"    {dataset}: {grad_norm:.6f}")
    
    # Check for problematic gradient norm ratios
    max_norm = max(grad_norms.values())
    min_norm = min(grad_norms.values())
    print(f"  Max/Min gradient norm ratio: {max_norm/min_norm:.2f}")
    
    if max_norm/min_norm > 100:
        print("  WARNING: Large gradient norm differences detected!")
        print("  This could indicate numerical issues or very different dataset complexities.")
    
    # Check gradient alignment calculation details for a few pairs
    print(f"\nDetailed alignment analysis:")
    test_pairs = [
        ("allenai/c4", "GAIR/lima"),
        ("flan_combined", "GAIR/lima"), 
        ("math_combined", "GAIR/lima")
    ]
    
    for ds1, ds2 in test_pairs:
        i = all_datasets.index(ds1)
        j = all_datasets.index(ds2)
        
        grad_i = dataset_gradients[ds1]
        grad_j = dataset_gradients[ds2]
        
        norm_i = np.linalg.norm(grad_i)
        norm_j = np.linalg.norm(grad_j)
        dot_prod = np.dot(grad_i, grad_j)
        cosine_sim = dot_prod / (norm_i * norm_j)
        
        print(f"  {ds1} vs {ds2}:")
        print(f"    ||grad_i||: {norm_i:.6f}")
        print(f"    ||grad_j||: {norm_j:.6f}")
        print(f"    dot product: {dot_prod:.6f}")
        print(f"    cosine sim: {cosine_sim:.6f}")
        print(f"    α (1-cosine): {1-cosine_sim:.6f}")
    
    return {
        'g_matrix': g_matrix,
        'alpha_matrix': alpha_matrix, 
        'cosine_matrix': cosine_matrix,
        'dataset_gradients': dataset_gradients,
        'embedding_similarity': emb_sim_matrix
    }

def analyze_bridge_candidates(alpha_matrix, all_datasets, pretrain_datasets, downstream_datasets):
    """
    Find optimal bridging paths using your theoretical framework.
    
    Looks for cases where α_12 + α_23 < α_13 (triangle inequality)
    """
    
    bridge_analysis = {}
    
    print("\n" + "="*60)
    print("BRIDGE ANALYSIS BASED ON GRADIENT ALIGNMENT THEORY")
    print("="*60)
    
    for pretrain_ds in pretrain_datasets:
        for downstream_ds in downstream_datasets:
            
            i = all_datasets.index(pretrain_ds)
            j = all_datasets.index(downstream_ds)
            
            alpha_13 = alpha_matrix[i, j]  # Direct path misalignment
            
            print(f"\nAnalyzing path: {pretrain_ds} → {downstream_ds}")
            print(f"Direct misalignment (α_13): {alpha_13:.3f}")
            
            candidates = []
            
            # Check all potential bridge datasets
            for bridge_ds in all_datasets:
                if bridge_ds == pretrain_ds or bridge_ds == downstream_ds:
                    continue
                
                k = all_datasets.index(bridge_ds)
                alpha_12 = alpha_matrix[i, k]  # Pretrain → Bridge
                alpha_23 = alpha_matrix[k, j]  # Bridge → Downstream
                
                bridge_path_cost = alpha_12 + alpha_23
                
                if bridge_path_cost < alpha_13:
                    benefit = alpha_13 - bridge_path_cost
                    candidates.append({
                        'bridge': bridge_ds,
                        'benefit': benefit,
                        'alpha_12': alpha_12,
                        'alpha_23': alpha_23,
                        'alpha_13': alpha_13,
                        'total_bridge_cost': bridge_path_cost
                    })
                    
                    print(f"  ✓ {bridge_ds}: α_12 + α_23 = {alpha_12:.3f} + {alpha_23:.3f} = {bridge_path_cost:.3f} < {alpha_13:.3f}")
                    print(f"    Benefit: {benefit:.3f}")
                else:
                    print(f"  ✗ {bridge_ds}: α_12 + α_23 = {alpha_12:.3f} + {alpha_23:.3f} = {bridge_path_cost:.3f} ≥ {alpha_13:.3f}")
            
            # Sort by benefit
            candidates.sort(key=lambda x: x['benefit'], reverse=True)
            bridge_analysis[f"{pretrain_ds} → {downstream_ds}"] = candidates
            
            if candidates:
                print(f"  → Best bridge: {candidates[0]['bridge']} (benefit: {candidates[0]['benefit']:.3f})")
            else:
                print(f"  → No beneficial bridges found")
    
    return bridge_analysis

def create_visualization(results, all_datasets, display_names, pretrain_datasets, 
                        downstream_datasets, bridge_analysis, layer_selection, output_file):
    """Create comprehensive visualization of gradient alignment results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Gradient Alignment Analysis ({layer_selection} layers)', fontsize=16)
    
    matrices = [
        (results['cosine_matrix'], 'Cosine Similarity', 'viridis'),
        (results['alpha_matrix'], 'Misalignment (α)', 'viridis_r'),
        (results['g_matrix'], 'Gradient Alignment (g)', 'RdBu'),
    ]
    
    # Plot first three matrices
    for idx, (matrix, title, cmap) in enumerate(matrices):
        ax = axes[idx // 2, idx % 2]
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(display_names)))
        ax.set_yticks(range(len(display_names)))
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_yticklabels(display_names)
        ax.set_title(title)
        
        # Add values to cells
        for i in range(len(all_datasets)):
            for j in range(len(all_datasets)):
                if i != j:  # Skip diagonal
                    color = 'white' if cmap == 'viridis' else 'black'
                    ax.text(j, i, f'{matrix[i,j]:.2f}', 
                           ha='center', va='center', color=color, fontsize=8)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add dividing lines
        n_pretrain = len(pretrain_datasets)
        ax.axhline(y=n_pretrain-0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(x=n_pretrain-0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    # Bridge analysis summary
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('Bridge Recommendations', fontsize=12, pad=20)
    
    y_pos = 0.9
    for path, candidates in bridge_analysis.items():
        if candidates and y_pos > 0.1:
            ax.text(0.05, y_pos, f"{path}:", fontweight='bold', fontsize=10, 
                   transform=ax.transAxes)
            y_pos -= 0.08
            
            for i, candidate in enumerate(candidates[:2]):  # Top 2
                bridge_text = (f"  {i+1}. {candidate['bridge']}\n"
                             f"      Benefit: {candidate['benefit']:.3f}")
                ax.text(0.1, y_pos, bridge_text, fontsize=8, 
                       transform=ax.transAxes)
                y_pos -= 0.12
            y_pos -= 0.05
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✔ Saved gradient alignment analysis to {output_file}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Compute gradient alignment similarity matrix')
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-small',
                       help='HuggingFace model name (use smaller models for faster computation)')
    parser.add_argument('--sample_size', type=int, default=200,
                       help='Number of samples per dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for computation')
    parser.add_argument('--layer_selection', choices=['all', 'key_layers', 'embeddings_only', 'output_only', 'first_last'],
                       default='key_layers', help='Which layers to include in gradient computation')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--raw_gradients', action='store_true',
                       help='Just compute and display raw gradient similarities, skip bridging analysis')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Fixed datasets (same as your script)
    pretrain_datasets = [
        "allenai/c4",
        "bigcode/starcoderdata", 
        "math_combined",
        "flan_combined"
    ]
    
    downstream_datasets = [
        "GAIR/lima",
        "openai/gsm8k",
        "allenai/social_i_qa", 
        "nyu-mll/glue",
        "ai2_arc",
        "sciq",
        "pycode",
    ]
    
    all_datasets = pretrain_datasets + downstream_datasets
    display_names = [
        "C4", "StarCoder", "Math Combined", "FLAN Combined",  # pretrain
        "LIMA", "GSM8K", "Social IQA", "GLUE-RTE", "ARC-Challenge", "SciQ", "PyCode"  # downstream
    ]
    
    print(f"Analyzing datasets: {all_datasets}")
    print(f"Using model: {args.model}")
    print(f"Layer selection: {args.layer_selection}")
    
    # Load model and tokenizer
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )
    
    if device == 'cpu':
        model = model.to(device)
    
    if args.raw_gradients:
        # Use simpler computation for raw gradients
        print("\n" + "="*60)
        print("COMPUTING RAW GRADIENT SIMILARITIES")
        print("="*60)
        
        results = compute_raw_gradient_similarities(
            all_datasets, model, tokenizer, args.sample_size, args.layer_selection, device
        )
        
        # Print raw gradient similarities
        print("\n" + "="*60)
        print("RAW GRADIENT SIMILARITIES (COSINE)")
        print("="*60)
        
        cosine_matrix = results['cosine_matrix']
        
        # Print similarity matrix in a nice format
        print(f"\n{'Dataset':<20}", end="")
        for name in display_names:
            print(f"{name:<12}", end="")
        print()
        print("-" * (20 + 12 * len(display_names)))
        
        for i, ds1 in enumerate(all_datasets):
            name1 = display_names[i]
            print(f"{name1:<20}", end="")
            for j, ds2 in enumerate(all_datasets):
                if i == j:
                    print(f"{'1.00':<12}", end="")
                else:
                    print(f"{cosine_matrix[i,j]:<12.3f}", end="")
            print()
        
        # Print most/least similar pairs
        mask = ~np.eye(len(all_datasets), dtype=bool)
        cosine_vals = cosine_matrix[mask]
        
        print(f"\nSimilarity range: [{cosine_vals.min():.3f}, {cosine_vals.max():.3f}]")
        print(f"Mean similarity: {cosine_vals.mean():.3f}")
        
        # Find most similar pairs
        print(f"\nMost similar pairs:")
        # Create list of (similarity, i, j) tuples for sorting
        similarities = []
        for i in range(len(all_datasets)):
            for j in range(i+1, len(all_datasets)):  # Only upper triangle
                similarities.append((cosine_matrix[i, j], i, j))
        
        similarities.sort(reverse=True)  # Sort by similarity descending
        
        for sim, i, j in similarities[:10]:  # Top 10
            print(f"  {display_names[i]:<15} ↔ {display_names[j]:<15}: {sim:.3f}")
        
        # Find least similar pairs  
        print(f"\nLeast similar pairs:")
        for sim, i, j in similarities[-10:]:  # Bottom 10
            print(f"  {display_names[i]:<15} ↔ {display_names[j]:<15}: {sim:.3f}")
        
        return
    
    # Regular analysis with full computation and bridging
    # Compute gradient alignment matrix
    results = compute_gradient_alignment_matrix(
        all_datasets, model, tokenizer, args.sample_size, args.layer_selection, device
    )
    
    # Analyze bridge candidates
    bridge_analysis = analyze_bridge_candidates(
        results['alpha_matrix'], all_datasets, pretrain_datasets, downstream_datasets
    )
    
    # Create visualization
    output_file = f"gradient_alignment_analysis_{args.layer_selection}_{args.model.replace('/', '_')}.png"
    create_visualization(
        results, all_datasets, display_names, pretrain_datasets, 
        downstream_datasets, bridge_analysis, args.layer_selection, output_file
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    cosine_matrix = results['cosine_matrix']
    alpha_matrix = results['alpha_matrix']
    
    # Exclude diagonal for statistics
    mask = ~np.eye(len(all_datasets), dtype=bool)
    cosine_vals = cosine_matrix[mask]
    alpha_vals = alpha_matrix[mask]
    
    print(f"Cosine similarity range: [{cosine_vals.min():.3f}, {cosine_vals.max():.3f}]")
    print(f"Mean cosine similarity: {cosine_vals.mean():.3f}")
    print(f"Misalignment (α) range: [{alpha_vals.min():.3f}, {alpha_vals.max():.3f}]")
    print(f"Mean misalignment: {alpha_vals.mean():.3f}")
    
    # Most/least similar pairs
    n = len(all_datasets)
    max_idx = np.unravel_index(cosine_vals.argmax(), (n, n))
    min_idx = np.unravel_index(cosine_vals.argmin(), (n, n))
    
    print(f"\nMost similar pair: {all_datasets[max_idx[0]]} ↔ {all_datasets[max_idx[1]]} ({cosine_matrix[max_idx]:.3f})")
    print(f"Least similar pair: {all_datasets[min_idx[0]]} ↔ {all_datasets[min_idx[1]]} ({cosine_matrix[min_idx]:.3f})")

if __name__ == "__main__":
    main()