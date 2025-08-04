import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import seaborn as sns
from tqdm import tqdm
import os

# =============================================================================
# ARGPARSE + DEVICE
# =============================================================================

parser = argparse.ArgumentParser(description="CKA analysis for midtraining study")
parser.add_argument("--model_size", type=str, required=True, choices=["70m", "160m", "410m"], 
                   help="Model size to analyze")
parser.add_argument("--probe", type=str, required=True, choices=["humaneval", "gsm8k", "c4", "apps"], help="Probe dataset")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
parser.add_argument("--output_dir", type=str, default="./cka_results", help="Output directory for plots")

args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_paths(model_size):
    """Get model paths based on model size"""
    base_dir = "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining"
    
    if model_size == "70m":
        return {
            "base": f"{base_dir}/pretrained_chkpts/pythia_70m_128b_fixed_hf/final",
            "base_sft": f"{base_dir}/finetuned_chkpts/pythia-70m/pycode/final_regular_895_hf/final",
            "midtrained": f"{base_dir}/pretrained_chkpts/pythia_70m_128b_fixed_midtrain_spikefix_hf/final",
            "midtrained_sft": f"{base_dir}/finetuned_chkpts/pythia-70m/pycode/final_md_895_hf/final"
        }
    elif model_size == "160m":
        return {
            "base": f"{base_dir}/pretrained_chkpts/pythia_160m_128b_hf/final",
            "base_sft": f"{base_dir}/finetuned_chkpts/pythia-160m/pycode/final_regular_895_hf/final",
            "midtrained": f"{base_dir}/pretrained_chkpts/pythia_160m_128b_midtrain_from_6k_starcoder_hf/final",
            "midtrained_sft": f"{base_dir}/finetuned_chkpts/pythia-160m/pycode/final_md_895_hf/final"
        }
    elif model_size == "410m":
        return {
            "base": f"{base_dir}/pretrained_chkpts/pythia_410m_128b_fixed_hf/final",
            "base_sft": f"{base_dir}/finetuned_chkpts/pythia-410m/pycode/final_regular_895_hf/final",
            "midtrained": f"{base_dir}/pretrained_chkpts/pythia_410m_128b_midtrain_from_6k_starcoder_hf/final",
            "midtrained_sft": f"{base_dir}/finetuned_chkpts/pythia-410m/pycode/final_md_895_hf/final"
        }
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

# =============================================================================
# PROBE DATASETS (from your original script)
# =============================================================================

class HumanEvalCodeDataset(Dataset):
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
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ds = load_dataset("gsm8k", "main", split="test")

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

class C4Dataset(Dataset):
    def __init__(self, tokenizer, max_length, num_samples=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        # Use streaming to avoid downloading the entire dataset
        print(f"Loading {num_samples} samples from C4 validation set (streaming)...")
        self.ds = load_dataset("c4", "en", split="validation", streaming=True)
        
        # Collect the first num_samples into a list for indexing
        self.samples = []
        for i, sample in enumerate(self.ds):
            if i >= num_samples:
                break
            self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} C4 samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        # Take first 200 words to keep it manageable
        words = text.split()[:200]
        text = " ".join(words)
        
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

class APPSDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Load APPS dataset (training split has ~5000 examples)
        self.ds = load_dataset("codeparrot/apps", split="train", trust_remote_code=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Combine problem and solution
        problem = self.ds[idx]["question"]
        solutions = self.ds[idx]["solutions"]
        
        # Use first solution if available, otherwise just the problem
        if solutions and len(solutions) > 0:
            import json
            try:
                solution_list = json.loads(solutions)
                if solution_list and len(solution_list) > 0:
                    text = f"Problem: {problem}\n\nSolution: {solution_list[0]}"
                else:
                    text = f"Problem: {problem}"
            except:
                text = f"Problem: {problem}"
        else:
            text = f"Problem: {problem}"
        
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
    if probe_name == "humaneval":
        dataset = HumanEvalCodeDataset(tokenizer, max_length)
    elif probe_name == "gsm8k":
        dataset = GSM8KDataset(tokenizer, max_length)
    elif probe_name == "c4":
        dataset = C4Dataset(tokenizer, max_length)
    elif probe_name == "apps":
        dataset = APPSDataset(tokenizer, max_length)
    else:
        raise ValueError(f"Unknown probe: {probe_name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def center_matrix(A):
    """Center the matrix A"""
    return A - A.mean(axis=0, keepdims=True)

def linear_CKA(A, B):
    """Compute linear CKA between activation matrices A and B"""
    A_cent = center_matrix(A)
    B_cent = center_matrix(B)
    n = A_cent.shape[0]
    
    HSIC_AB = np.linalg.norm(A_cent.T @ B_cent, 'fro')**2 / ((n - 1)**2)
    HSIC_AA = np.linalg.norm(A_cent.T @ A_cent, 'fro')**2 / ((n - 1)**2)
    HSIC_BB = np.linalg.norm(B_cent.T @ B_cent, 'fro')**2 / ((n - 1)**2)
    
    return HSIC_AB / np.sqrt(HSIC_AA * HSIC_BB + 1e-12)

def get_layer_activations(model, dataloader, device):
    """Extract activations from all layers for all examples"""
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Prepare mask for weighted pooling
            # attention_mask: [batch, seq_len]
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            batch_activations = []
            
            # Pool each layer's hidden states over the unmasked tokens
            for hidden_state in outputs.hidden_states:
                # hidden_state: [batch, seq_len, hidden]
                # Sum only over non-padding tokens
                tok_sum = (hidden_state * mask).sum(dim=1)       # [batch, hidden]
                lengths = mask.sum(dim=1).clamp(min=1e-5)       # [batch, 1]
                pooled = (tok_sum / lengths).cpu().numpy()     # [batch, hidden]
                batch_activations.append(pooled)
            
            all_activations.append(batch_activations)
    
    # Now concatenate over batches for each layer
    num_layers = len(all_activations[0])
    layer_activations = []
    for layer_idx in range(num_layers):
        layer_data = np.concatenate([batch[layer_idx] for batch in all_activations], axis=0)
        layer_activations.append(layer_data)
    
    return layer_activations


def compute_cka_matrix(model1_activations, model2_activations):
    """Compute CKA matrix between two models' activations"""
    num_layers = len(model1_activations)
    cka_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            cka_score = linear_CKA(model1_activations[i], model2_activations[j])
            cka_matrix[i, j] = cka_score
    
    return cka_matrix

# =============================================================================
# MAIN CKA ANALYSIS
# =============================================================================

def main():
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model paths based on size
    model_paths = get_model_paths(args.model_size)
    
    # Load tokenizer (from base model)
    tokenizer = AutoTokenizer.from_pretrained(model_paths["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build probe dataloader
    probe_loader = build_probe_loader(args.probe, tokenizer, args.batch_size, args.max_length)
    print(f"Probe '{args.probe}' size:", len(probe_loader.dataset))
    
    # Load all 4 models
    print("Loading models...")
    
    def debug_and_load_model(model_path, name):
        """Debug config and load model"""
        print(f"\nDebugging {name} at {model_path}")
        
        # Check if config exists
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"  Config found - hidden_size: {config.get('hidden_size', 'NOT_FOUND')}")
            print(f"  Model type: {config.get('model_type', 'NOT_FOUND')}")
        else:
            print(f"  WARNING: No config.json found at {config_path}")
        
        # Try loading with explicit local_files_only to avoid HF hub confusion
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path, 
                local_files_only=True,
                trust_remote_code=True
            ).to(DEVICE)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"  Size mismatch detected! Trying manual config load...")
                # Load config explicitly and create model
                from transformers import AutoConfig, GPTNeoXForCausalLM
                config = AutoConfig.from_pretrained(model_path, local_files_only=True)
                print(f"  Loaded config hidden_size: {config.hidden_size}")
                
                # Create model with config, then load weights
                model = GPTNeoXForCausalLM(config)
                state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                return model.to(DEVICE)
            else:
                raise
    
    models = {
        "base": debug_and_load_model(model_paths["base"], "base"),
        "base_sft": debug_and_load_model(model_paths["base_sft"], "base_sft"),
        "midtrained": debug_and_load_model(model_paths["midtrained"], "midtrained"),
        "midtrained_sft": debug_and_load_model(model_paths["midtrained_sft"], "midtrained_sft")
    }
    
    # Define the 4 key comparisons
    comparisons = [
        ("base", "midtrained", "Base vs Midtrained\n(Effect of midtraining)"),
        ("base", "base_sft", "Base vs Base-SFT\n(Base model finetuning changes)"),
        ("midtrained", "midtrained_sft", "Midtrained vs Midtrained-SFT\n(Midtrained model finetuning changes)"),
        ("base_sft", "midtrained_sft", "Base-SFT vs Midtrained-SFT\n(Final models comparison)")
    ]
    
    # First, extract activations from all models
    print("Extracting activations from all models...")
    model_activations = {}
    
    for model_name, model in models.items():
        print(f"Extracting activations for {model_name}...")
        model_activations[model_name] = get_layer_activations(model, probe_loader, DEVICE)
    
    # Run CKA for each comparison
    for model1_name, model2_name, title in comparisons:
        print(f"\nComputing CKA: {model1_name} vs {model2_name}")
        
        # Compute CKA matrix using our homemade implementation
        cka_matrix = compute_cka_matrix(
            model_activations[model1_name], 
            model_activations[model2_name]
        )
        
        # Dynamic sizing based on matrix dimensions
        matrix_size = max(cka_matrix.shape)
        if matrix_size <= 8:
            fig_size = (8, 6)
            annot_fontsize = 10
        elif matrix_size <= 12:
            fig_size = (12, 10)
            annot_fontsize = 11  # Increased for better readability
        else:
            fig_size = (14, 12)
            annot_fontsize = 10  # Increased for better readability
        
        # Create and save plot
        plt.figure(figsize=fig_size)
        ax = sns.heatmap(
            cka_matrix, 
            annot=True, 
            fmt='.2f',
            annot_kws={'size': annot_fontsize},  # Control annotation font size
            cmap='viridis',
            xticklabels=[f"L{i}" for i in range(cka_matrix.shape[1])],
            yticklabels=[f"L{i}" for i in range(cka_matrix.shape[0])],
            cbar=True,
            square=True
        )
        
        # Highlight diagonal elements in red
        min_dim = min(cka_matrix.shape)
        for i in range(min_dim):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))
        
        plt.title(f"{title}\n{args.model_size.upper()} Model - Probe: {args.probe}")
        plt.xlabel(f"{model2_name.replace('_', '-')} Layers")
        plt.ylabel(f"{model1_name.replace('_', '-')} Layers")
        plt.tight_layout()
        
        # Save with descriptive filename
        filename = f"cka_{model1_name}_vs_{model2_name}_{args.probe}_{args.model_size}.png"
        out_png = os.path.join(args.output_dir, filename)
        out_pdf = out_png.replace('.png', '.pdf')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename} and PDF version")
    
    # Create a summary figure with all 4 comparisons - dynamic sizing
    # Determine size based on largest matrix
    max_matrix_size = 0
    for model_name in model_activations:
        max_matrix_size = max(max_matrix_size, len(model_activations[model_name]))
    

    # Row format: 1x4
    if max_matrix_size <= 8:
        fig_size = (24, 6)
        annot_fontsize = 8
    elif max_matrix_size <= 12:
        fig_size = (32, 8)
        annot_fontsize = 8
    else:
        fig_size = (40, 10)
        annot_fontsize = 7

    fig, axes = plt.subplots(1, 4, figsize=fig_size)
    axes = axes.flatten()

    for idx, (model1_name, model2_name, title) in enumerate(comparisons):
        cka_matrix = compute_cka_matrix(
            model_activations[model1_name],
            model_activations[model2_name]
        )
        plt.sca(axes[idx])
        ax = sns.heatmap(
            cka_matrix,
            annot=True,
            fmt='.2f',
            annot_kws={'size': annot_fontsize},
            cmap='viridis',
            xticklabels=[f"L{i}" for i in range(cka_matrix.shape[1])],
            yticklabels=[f"L{i}" for i in range(cka_matrix.shape[0])],
            cbar=False,
            square=True
        )
        min_dim = min(cka_matrix.shape)
        for i in range(min_dim):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
        plt.title(title, fontsize=10)

    plt.suptitle(f"CKA Analysis - Pythia {args.model_size.upper()} (Probe: {args.probe})", fontsize=14)
    plt.tight_layout()
    summary_png = os.path.join(args.output_dir, f"cka_summary_{args.probe}_{args.model_size}.png")
    summary_pdf = summary_png.replace('.png', '.pdf')
    plt.savefig(summary_png, dpi=300, bbox_inches='tight')
    plt.savefig(summary_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nKey insights to look for:")
    print("1. Base vs Midtrained: Which layers change during midtraining?")
    print("2. Base vs Base-SFT: How much does base model change during finetuning?") 
    print("3. Midtrained vs Midtrained-SFT: How much does midtrained model change during finetuning?")
    print("4. Compare #2 vs #3: Midtraining should require smaller changes during finetuning!")

if __name__ == "__main__":
    main()