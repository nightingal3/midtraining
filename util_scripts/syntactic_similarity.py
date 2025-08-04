import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from itertools import islice
import json
import os
import random
import glob
import argparse
from collections import Counter, defaultdict
import re
from tqdm import tqdm


def get_ngrams(text, n):
    """Extract n-grams from text."""
    if n == 1:
        # Word unigrams
        words = text.lower().split()
        return words
    else:
        # Word n-grams
        words = text.lower().split()
        if len(words) < n:
            return []
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def get_char_ngrams(text, n):
    """Extract character n-grams from text."""
    text = text.lower()
    if len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def compute_multilevel_ngram_similarity(texts1, texts2, ngram_sizes=[1, 2, 3, 4], ngram_type='word', weights=None):
    """
    Compute similarity based on multiple n-gram levels (BLEU-style).
    
    Args:
        texts1, texts2: Lists of texts from two datasets
        ngram_sizes: List of n-gram sizes to use (e.g., [1, 2, 3, 4])
        ngram_type: 'word' or 'char' for word or character n-grams
        weights: Optional weights for each n-gram level (default: uniform)
    
    Returns:
        Dictionary with individual n-gram similarities and combined score
    """
    print(f"    Computing multi-level {ngram_type} n-gram similarity (sizes: {ngram_sizes})...")
    
    if weights is None:
        weights = [1.0 / len(ngram_sizes)] * len(ngram_sizes)
    elif len(weights) != len(ngram_sizes):
        raise ValueError("Length of weights must match length of ngram_sizes")
    
    similarities = {}
    weighted_scores = []
    
    for i, n in enumerate(ngram_sizes):
        print(f"      Computing {ngram_type} {n}-grams...")
        
        # Extract n-grams from both datasets
        if ngram_type == 'word':
            ngrams1 = []
            ngrams2 = []
            for text in texts1:
                ngrams1.extend(get_ngrams(text, n))
            for text in texts2:
                ngrams2.extend(get_ngrams(text, n))
        elif ngram_type == 'char':
            ngrams1 = []
            ngrams2 = []
            for text in texts1:
                ngrams1.extend(get_char_ngrams(text, n))
            for text in texts2:
                ngrams2.extend(get_char_ngrams(text, n))
        
        # Count n-gram frequencies
        counter1 = Counter(ngrams1)
        counter2 = Counter(ngrams2)
        
        # Get all unique n-grams
        all_ngrams = set(counter1.keys()) | set(counter2.keys())
        
        if not all_ngrams:
            print(f"        Warning: No {ngram_type} {n}-grams found")
            similarity = 0.0
        else:
            # Create frequency vectors
            vec1 = np.array([counter1.get(ngram, 0) for ngram in all_ngrams])
            vec2 = np.array([counter2.get(ngram, 0) for ngram in all_ngrams])
            
            # Normalize to probabilities
            vec1 = vec1 / (vec1.sum() + 1e-10)
            vec2 = vec2 / (vec2.sum() + 1e-10)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
        
        similarities[f'{ngram_type}_{n}gram'] = similarity
        weighted_scores.append(weights[i] * similarity)
        
        print(f"        {ngram_type.capitalize()} {n}-gram similarity: {similarity:.3f}")
        print(f"        Dataset 1: {len(counter1)} unique, Dataset 2: {len(counter2)} unique")
        print(f"        Overlap: {len(set(counter1.keys()) & set(counter2.keys()))} shared")
    
    # Compute weighted average (BLEU-style)
    combined_score = sum(weighted_scores)
    similarities['combined_score'] = combined_score
    
    print(f"    Combined multi-level score: {combined_score:.3f}")
    print(f"    Weights used: {weights}")
    
    return similarities


def compute_ngram_frequency_similarity(texts1, texts2, n=2, ngram_type='word'):
    """
    Compute similarity based on n-gram frequency distributions.
    
    Args:
        texts1, texts2: Lists of texts from two datasets
        n: N-gram size (default=2 for bigrams)
        ngram_type: 'word' or 'char' for word or character n-grams
    
    Returns:
        Cosine similarity between n-gram frequency vectors
    """
    print(f"    Computing {ngram_type} {n}-gram frequency similarity...")
    
    # Extract n-grams from both datasets
    if ngram_type == 'word':
        ngrams1 = []
        ngrams2 = []
        for text in texts1:
            ngrams1.extend(get_ngrams(text, n))
        for text in texts2:
            ngrams2.extend(get_ngrams(text, n))
    elif ngram_type == 'char':
        ngrams1 = []
        ngrams2 = []
        for text in texts1:
            ngrams1.extend(get_char_ngrams(text, n))
        for text in texts2:
            ngrams2.extend(get_char_ngrams(text, n))
    else:
        raise ValueError("ngram_type must be 'word' or 'char'")
    
    # Count n-gram frequencies
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)
    
    # Get all unique n-grams
    all_ngrams = set(counter1.keys()) | set(counter2.keys())
    
    if not all_ngrams:
        print(f"    Warning: No {ngram_type} {n}-grams found")
        return 0.0
    
    # Create frequency vectors
    vec1 = np.array([counter1.get(ngram, 0) for ngram in all_ngrams])
    vec2 = np.array([counter2.get(ngram, 0) for ngram in all_ngrams])
    
    # Normalize to probabilities
    vec1 = vec1 / (vec1.sum() + 1e-10)
    vec2 = vec2 / (vec2.sum() + 1e-10)
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm1 * norm2)
    
    print(f"    {ngram_type.capitalize()} {n}-gram similarity: {similarity:.3f}")
    print(f"    Dataset 1: {len(counter1)} unique {ngram_type} {n}-grams")
    print(f"    Dataset 2: {len(counter2)} unique {ngram_type} {n}-grams") 
    print(f"    Overlap: {len(set(counter1.keys()) & set(counter2.keys()))} shared {ngram_type} {n}-grams")
    
    return similarity


def compute_character_overlap_similarity(texts1, texts2):
    """
    Compute similarity based on character-level overlap patterns.
    
    Args:
        texts1, texts2: Lists of texts from two datasets
    
    Returns:
        Dictionary with various character-level similarity metrics
    """
    print("    Computing character overlap similarity...")
    
    # Combine all texts for each dataset
    combined1 = ' '.join(texts1)
    combined2 = ' '.join(texts2)
    
    # Character frequency distributions
    char_counter1 = Counter(combined1.lower())
    char_counter2 = Counter(combined2.lower())
    
    # All unique characters
    all_chars = set(char_counter1.keys()) | set(char_counter2.keys())
    
    # Character frequency vectors
    char_vec1 = np.array([char_counter1.get(char, 0) for char in all_chars])
    char_vec2 = np.array([char_counter2.get(char, 0) for char in all_chars])
    
    # Normalize to probabilities
    char_vec1 = char_vec1 / (char_vec1.sum() + 1e-10)
    char_vec2 = char_vec2 / (char_vec2.sum() + 1e-10)
    
    # Cosine similarity
    char_cosine = np.dot(char_vec1, char_vec2) / (np.linalg.norm(char_vec1) * np.linalg.norm(char_vec2) + 1e-10)
    
    # Jaccard similarity (character set overlap)
    chars1 = set(combined1.lower())
    chars2 = set(combined2.lower())
    char_jaccard = len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0.0
    
    # Special character patterns
    special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?`~')
    special1 = chars1 & special_chars
    special2 = chars2 & special_chars
    special_jaccard = len(special1 & special2) / len(special1 | special2) if special1 | special2 else 0.0
    
    # Punctuation patterns
    punct_chars = set('.,;:!?()[]{}"\'-')
    punct1 = chars1 & punct_chars
    punct2 = chars2 & punct_chars
    punct_jaccard = len(punct1 & punct2) / len(punct1 | punct2) if punct1 | punct2 else 0.0
    
    # Digit patterns
    digit_chars = set('0123456789')
    digit1 = chars1 & digit_chars
    digit2 = chars2 & digit_chars
    digit_jaccard = len(digit1 & digit2) / len(digit1 | digit2) if digit1 | digit2 else 0.0
    
    results = {
        'char_frequency_cosine': char_cosine,
        'char_set_jaccard': char_jaccard,
        'special_char_jaccard': special_jaccard,
        'punctuation_jaccard': punct_jaccard,
        'digit_jaccard': digit_jaccard
    }
    
    print(f"    Character frequency cosine: {char_cosine:.3f}")
    print(f"    Character set Jaccard: {char_jaccard:.3f}")
    print(f"    Special characters Jaccard: {special_jaccard:.3f}")
    print(f"    Punctuation Jaccard: {punct_jaccard:.3f}")
    print(f"    Digit Jaccard: {digit_jaccard:.3f}")
    
    return results


# Import the dataset loading functions from the original script
def extract_text(sample, dataset, use_instruction_format=True):
    """
    Return a single 'text' string for each sample across multiple dataset types.
    """
    if dataset in ("openai/gsm8k",):
        if use_instruction_format:
            return f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        else:
            return sample["question"] + " " + sample["answer"]
            
    if dataset in ("allenai/social_i_qa", "social_i_qa"):
        choices = [sample[f"answer{opt}"] for opt in ["A","B","C"]]
        num_to_choice = {"1": "A", "2": "B", "3": "C"}
        correct_label = num_to_choice[sample["label"]]
        correct = sample[f"answer{correct_label}"]
        
        if use_instruction_format:
            choices_text = "\n".join([f"{opt}: {sample[f'answer{opt}']}" for opt in ["A","B","C"]])
            return f"Context: {sample['context']}\nQuestion: {sample['question']}\nChoices:\n{choices_text}\nAnswer: {correct_label}"
        else:
            return sample["context"] + " " + sample["question"] + " " + correct
            
    if dataset in ("bigcode/starcoderdata", "starcoder-python"):
        return sample["content"]
        
    if dataset in ("allenai/c4", "c4"):
        return sample["text"]
        
    if dataset.startswith("nyu-mll/glue") and sample.get("sentence1") and sample.get("sentence2"):
        entail = "entailment" if sample["label"] == 1 else "not_entailment"
        if use_instruction_format:
            return f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\nRelationship: {entail}"
        else:
            return sample["sentence1"] + " " + sample["sentence2"] + " " + entail
            
    if dataset in ("ai2_arc", "arc_challenge"):
        choices_text = " ".join(sample["choices"]["text"])
        correct_idx = sample["answerKey"]
        if use_instruction_format:
            choices_formatted = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(sample["choices"]["text"])])
            return f"Question: {sample['question']}\nChoices:\n{choices_formatted}\nAnswer: {correct_idx}"
        else:
            return sample["question"] + " " + choices_text + " " + correct_idx
            
    if dataset in ("sciq", "allenai/sciq"):
        if use_instruction_format:
            if sample.get("input") and sample["input"].strip():
                return f"Question: {sample['instruction']}\nInput: {sample['input']}\nAnswer: {sample['output']}"
            else:
                return f"Question: {sample['instruction']}\nAnswer: {sample['output']}"
        else:
            input_text = sample.get("input", "")
            if input_text and input_text.strip():
                return sample["instruction"] + " " + input_text + " " + sample["output"]
            else:
                return sample["instruction"] + " " + sample["output"]
            
    if dataset in ("pycode",):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        
        if use_instruction_format:
            if instruction and input_text and output_text:
                return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
            elif instruction and output_text:
                return f"Instruction: {instruction}\nCode: {output_text}"
            else:
                return instruction + " " + input_text + " " + output_text
        else:
            return instruction + " " + input_text + " " + output_text
            
    if dataset in ("math_combined",):
        text = sample.get("text", "")
        if use_instruction_format and "\n\n" in text:
            parts = text.split("\n\n", 1)
            if len(parts) == 2:
                return f"Problem: {parts[0]}\nSolution: {parts[1]}"
        return text
        
    if dataset in ("flan_combined",):
        text = sample.get("text", "")
        if use_instruction_format and "\n\n" in text:
            parts = text.split("\n\n", 1)
            if len(parts) == 2:
                return f"Instruction: {parts[0]}\nResponse: {parts[1]}"
        return text
        
    if dataset in ("GAIR/lima",):
        conversations = sample["conversations"]
        if use_instruction_format and len(conversations) >= 2:
            formatted_parts = []
            for i, conv in enumerate(conversations):
                role = "Human" if i % 2 == 0 else "Assistant"
                formatted_parts.append(f"{role}: {conv}")
            return "\n".join(formatted_parts)
        else:
            return " ".join(conversations)
    
    if dataset in ("mattymchen/mr", "mr"):
        if use_instruction_format:
            return f"Question: {sample['instruction']}\nAnswer: {sample['output']}"
        else:
            return sample["instruction"] + " " + sample["output"]
    
    if dataset in ("CogComp/trec", "trec"):
        if use_instruction_format:
            return f"Question: {sample['instruction']}\nAnswer: {sample['output']}"
        else:
            return sample["instruction"] + " " + sample["output"]
    
    # Fallback
    parts = []
    for key in ("input","instruction","question","text","content","code"):
        if key in sample and isinstance(sample[key], str):
            parts.append(sample[key])
    return " ".join(parts).strip()


def load_local_json(file_path, sample_size=100):
    """Load samples from a local JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        random.shuffle(data)
        return data[:sample_size]
    else:
        return [data]


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
    
    # Combine all samples
    all_samples = openmath_samples + mathinstruct_samples + mathplus_samples
    random.shuffle(all_samples)
    return all_samples[:sample_size]


def load_flan_combined(sample_size=100):
    """Load and combine FLAN datasets."""
    print("Loading FLAN datasets...")
    flan_dir = "/data/tir/projects/tir3/users/mengyan3/manifold_data/datasets/flan/"
    
    train_files = sorted(glob.glob(os.path.join(flan_dir, "*_train.jsonl")))
    if not train_files:
        print("No FLAN files found locally, using fallback...")
        return [{"text": f"FLAN instruction {i} with output {i}"} for i in range(sample_size)]
    
    raw = load_dataset("json", data_files={"train": train_files}, split="train", streaming=True)
    raw = raw.shuffle(buffer_size=10_000, seed=42)
    
    samples = []
    for example in islice(raw, sample_size):
        if example.get("inputs") and example.get("targets"):
            text = example["inputs"].strip() + "\n\n" + example["targets"].strip()
            samples.append({"text": text})
    
    return samples


def get_dataset_texts(dataset_name, sample_size, use_instruction_format=True):
    """Extract texts from dataset."""
    texts = []
    
    if dataset_name == "allenai/c4":
        print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name, "en", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "bigcode/starcoderdata":
        print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "math_combined":
        samples = load_math_combined(sample_size)
        for sample in samples:
            txt = extract_text(sample, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "flan_combined":
        samples = load_flan_combined(sample_size)
        for sample in samples:
            txt = extract_text(sample, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "openai/gsm8k":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "pycode":
        print(f"Loading {dataset_name}...")
        file_path = "../manifold_data/all_in_one_pretraining/datasets/just_Nan-Do/code-search-net-python/Nan-Do/code-search-net-python/train.json"
        if os.path.exists(file_path):
            samples = load_local_json(file_path, sample_size)
            for sample in samples:
                txt = extract_text(sample, dataset_name, use_instruction_format)
                if txt:
                    texts.append(txt)
        else:
            print("  Local pycode file not found, using placeholder...")
            for i in range(min(sample_size, 100)):
                if use_instruction_format:
                    texts.append(f"Instruction: Write a function that returns {i}\nCode: def function_{i}():\n    return {i}")
                else:
                    texts.append(f"def function_{i}():\n    return {i}")
                
    elif dataset_name == "allenai/social_i_qa":
        print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "nyu-mll/glue":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("nyu-mll/glue", "rte", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "ai2_arc":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)
                
    elif dataset_name == "allenai/sciq":
        print(f"Loading {dataset_name} from processed JSON...")
        json_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/all_in_one_pretraining/datasets/just_allenai/sciq/allenai/sciq/train.json"
        if os.path.exists(json_path):
            samples = load_local_json(json_path, sample_size)
            for sample in samples:
                txt = extract_text(sample, dataset_name, use_instruction_format)
                if txt:
                    texts.append(txt)
        else:
            print(f"  JSON file not found, falling back to HuggingFace...")
            ds = load_dataset("allenai/sciq", split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                txt = extract_text(example, dataset_name, use_instruction_format)
                if txt:
                    texts.append(txt)

    elif dataset_name == "GAIR/lima":
        print(f"Loading {dataset_name}...")
        ds = load_dataset("GAIR/lima", split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(buffer_size=10_000, seed=42)
        for example in islice(ds, sample_size):
            txt = extract_text(example, dataset_name, use_instruction_format)
            if txt:
                texts.append(txt)

    elif dataset_name == "mattymchen/mr":
        print(f"Loading {dataset_name} from processed JSON...")
        json_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/all_in_one_pretraining/datasets/just_mattymchen/mr/mattymchen/mr/train.json"
        if os.path.exists(json_path):
            samples = load_local_json(json_path, sample_size)
            for sample in samples:
                txt = extract_text(sample, dataset_name, use_instruction_format)
                if txt:
                    texts.append(txt)
        else:
            print(f"  JSON file not found, falling back to HuggingFace...")
            ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                txt = extract_text(example, dataset_name, use_instruction_format)
                if txt:
                    texts.append(txt)
                
    elif dataset_name == "CogComp/trec":
        print(f"Loading {dataset_name} from processed JSON...")
        json_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/all_in_one_pretraining/datasets/just_CogComp/trec/CogComp/trec/train.json"
        if os.path.exists(json_path):
            samples = load_local_json(json_path, sample_size)
            for sample in samples:
                txt = extract_text(sample, dataset_name, use_instruction_format)
                if txt:
                    texts.append(txt)
        else:
            print(f"  JSON file not found, falling back to HuggingFace...")
            ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                txt = extract_text(example, dataset_name, use_instruction_format)
                if txt:
                    texts.append(txt)

    return texts


def compute_syntactic_similarity_matrix(all_datasets, method='ngram', sample_size=100, use_instruction_format=True, ngram_size=2, ngram_type='word', max_n=4):
    """
    Compute similarity matrix using syntactic features.
    
    Args:
        method: 'ngram', 'char_overlap', 'multilevel', or 'bleu'
        ngram_size: Size of n-grams (for single ngram method)
        ngram_type: 'word' or 'char' (for ngram methods)
        max_n: Maximum n-gram size (for multilevel and bleu methods)
    """
    N = len(all_datasets)
    sim = np.zeros((N, N))
    
    print(f"Computing syntactic similarity using method: {method}")
    if method == 'ngram':
        print(f"Using {ngram_type} {ngram_size}-grams")
    elif method in ['multilevel', 'bleu']:
        print(f"Using {ngram_type} 1-{max_n} grams")
    print(f"Using instruction format: {use_instruction_format}")
    
    # Load texts for all datasets
    all_texts = {}
    for i, ds in enumerate(all_datasets):
        print(f"→ Loading texts for {ds} ({i+1}/{N})...")
        all_texts[ds] = get_dataset_texts(ds, sample_size, use_instruction_format)
        print(f"  Loaded {len(all_texts[ds])} texts")
    
    print("Computing syntactic similarity matrix...")
    for i in range(N):
        for j in range(N):
            if i == j:
                sim[i, j] = 1.0
            else:
                print(f"Computing {method} similarity between {all_datasets[i]} and {all_datasets[j]}")
                
                if method == 'ngram':
                    similarity = compute_ngram_frequency_similarity(
                        all_texts[all_datasets[i]], 
                        all_texts[all_datasets[j]], 
                        n=ngram_size,
                        ngram_type=ngram_type
                    )
                elif method == 'multilevel':
                    ngram_sizes = list(range(1, max_n + 1))
                    multilevel_results = compute_multilevel_ngram_similarity(
                        all_texts[all_datasets[i]], 
                        all_texts[all_datasets[j]],
                        ngram_sizes=ngram_sizes,
                        ngram_type=ngram_type
                    )
                    similarity = multilevel_results['combined_score']
                elif method == 'bleu':
                    bleu_results = compute_bleu_style_similarity(
                        all_texts[all_datasets[i]], 
                        all_texts[all_datasets[j]],
                        max_n=max_n,
                        ngram_type=ngram_type
                    )
                    similarity = bleu_results['bleu_score']
                elif method == 'char_overlap':
                    char_results = compute_character_overlap_similarity(
                        all_texts[all_datasets[i]], 
                        all_texts[all_datasets[j]]
                    )
                    # Use character frequency cosine as main similarity
                    similarity = char_results['char_frequency_cosine']
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                sim[i, j] = similarity
                print(f"  Similarity: {similarity:.3f}")
    
    return sim


def create_similarity_plot(sim, all_datasets, display_names, pretrain_datasets, downstream_datasets, method_name, output_file):
    """Create and save the similarity matrix plot."""
    N = len(all_datasets)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    c = ax.pcolor(sim, cmap="viridis", edgecolors='k', linewidths=0.5)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(0.5, N, 1))
    ax.set_yticks(np.arange(0.5, N, 1))
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_yticklabels(display_names)
    
    # Add similarity values as text labels
    for i in range(N):
        for j in range(N):
            if i != j:  # Don't show 1.00 for diagonal
                ax.text(j + 0.5, i + 0.5, f"{sim[i, j]:.2f}", 
                        ha="center", va="center", color="white", fontsize=8)
    
    # Add dividing lines to separate pretrain from downstream
    n_pretrain = len(pretrain_datasets)
    ax.axhline(y=n_pretrain, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=n_pretrain, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    # Add labels for sections
    ax.text(n_pretrain/2, -0.5, 'Pretrain/Midtrain', ha='center', va='top', fontweight='bold')
    ax.text(n_pretrain + len(downstream_datasets)/2, -0.5, 'Downstream', ha='center', va='top', fontweight='bold')
    ax.text(-0.5, n_pretrain/2, 'Pretrain/\nMidtrain', ha='right', va='center', fontweight='bold', rotation=90)
    ax.text(-0.5, n_pretrain + len(downstream_datasets)/2, 'Downstream', ha='right', va='center', fontweight='bold', rotation=90)
    
    plt.title(f"Dataset Similarity Matrix ({method_name}): Pretrain/Midtrain vs Downstream")
    fig.colorbar(c, ax=ax, label='Similarity')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✔ Saved similarity matrix to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compute syntactic dataset similarity matrix')
    parser.add_argument('--method', choices=['ngram', 'char_overlap', 'multilevel', 'bleu'], default='ngram',
                       help='Syntactic similarity method')
    parser.add_argument('--ngram_size', type=int, default=2,
                       help='Size of n-grams (for single ngram method)')
    parser.add_argument('--ngram_type', choices=['word', 'char'], default='word',
                       help='Type of n-grams: word or character')
    parser.add_argument('--max_n', type=int, default=4,
                       help='Maximum n-gram size (for multilevel and bleu methods)')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of samples per dataset')
    parser.add_argument('--use_instruction_format', action='store_true',
                       help='Use instruction format for datasets where applicable')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Dataset configuration
    pretrain_datasets = [
        "allenai/c4",
        "bigcode/starcoderdata",
        "math_combined",
        "flan_combined"
    ]
    
    downstream_datasets = [
        "openai/gsm8k",
        "pycode", 
        "allenai/social_i_qa",
        "GAIR/lima",
        "allenai/sciq",
        "mattymchen/mr",
        "CogComp/trec"
    ]
    
    all_datasets = pretrain_datasets + downstream_datasets
    
    display_names = [
        "C4", "StarCoder", "Math Combined", "FLAN Combined",  # pretrain
        "GSM8K", "PyCode", "Social IQA", "LIMA", "SciQ", "Movie Reviews", "TREC"  # downstream
    ]
    
    print(f"Using syntactic method: {args.method}")
    if args.method == 'ngram':
        print(f"N-gram configuration: {args.ngram_type} {args.ngram_size}-grams")
    elif args.method in ['multilevel', 'bleu']:
        print(f"Multi-level configuration: {args.ngram_type} 1-{args.max_n} grams")
    
    # Compute similarity matrix
    sim = compute_syntactic_similarity_matrix(
        all_datasets, 
        method=args.method,
        sample_size=args.sample_size,
        use_instruction_format=args.use_instruction_format,
        ngram_size=args.ngram_size,
        ngram_type=args.ngram_type,
        max_n=args.max_n
    )
    
    # Create method name and output file
    format_suffix = "_instruct" if args.use_instruction_format else "_plain"
    
    if args.method == 'ngram':
        method_name = f"Syntactic N-gram ({args.ngram_type} {args.ngram_size}-grams, {'instructional' if args.use_instruction_format else 'plain'})"
        output_file = f"dataset_similarity_matrix_syntactic_ngram_{args.ngram_type}{args.ngram_size}{format_suffix}.png"
    elif args.method == 'multilevel':
        method_name = f"Syntactic Multi-level ({args.ngram_type} 1-{args.max_n} grams, {'instructional' if args.use_instruction_format else 'plain'})"
        output_file = f"dataset_similarity_matrix_syntactic_multilevel_{args.ngram_type}1to{args.max_n}{format_suffix}.png"
    elif args.method == 'bleu':
        method_name = f"Syntactic BLEU-style ({args.ngram_type} 1-{args.max_n} grams, {'instructional' if args.use_instruction_format else 'plain'})"
        output_file = f"dataset_similarity_matrix_syntactic_bleu_{args.ngram_type}1to{args.max_n}{format_suffix}.png"
    elif args.method == 'char_overlap':
        method_name = f"Syntactic Character Overlap ({'instructional' if args.use_instruction_format else 'plain'})"
        output_file = f"dataset_similarity_matrix_syntactic_char_overlap{format_suffix}.png"
    
    # Create and save the plot
    create_similarity_plot(sim, all_datasets, display_names, pretrain_datasets, 
                          downstream_datasets, method_name, output_file)
    
    print(f"\nSyntactic similarity analysis complete!")
    print(f"Method: {method_name}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()  # downstream
    
    print(f"Using syntactic method: {args.method}")
    if args.method == 'ngram':
        print(f"N-gram configuration: {args.ngram_type} {args.ngram_size}-grams")
    
    # Compute similarity matrix
    sim = compute_syntactic_similarity_matrix(
        all_datasets, 
        method=args.method,
        sample_size=args.sample_size,
        use_instruction_format=args.use_instruction_format,
        ngram_size=args.ngram_size,
        ngram_type=args.ngram_type
    )
    
    # Create method name and output file
    if args.method == 'ngram':
        method_name = f"Syntactic N-gram ({args.ngram_type} {args.ngram_size}-grams, {'instructional' if args.use_instruction_format else 'plain'})"
        format_suffix = "_instruct" if args.use_instruction_format else "_plain"
        output_file = f"dataset_similarity_matrix_syntactic_ngram_{args.ngram_type}{args.ngram_size}{format_suffix}.png"
    elif args.method == 'char_overlap':
        method_name = f"Syntactic Character Overlap ({'instructional' if args.use_instruction_format else 'plain'})"
        format_suffix = "_instruct" if args.use_instruction_format else "_plain"
        output_file = f"dataset_similarity_matrix_syntactic_char_overlap{format_suffix}.png"
    
    # Create and save the plot
    create_similarity_plot(sim, all_datasets, display_names, pretrain_datasets, 
                          downstream_datasets, method_name, output_file)
    
    print(f"\nSyntactic similarity analysis complete!")
    print(f"Method: {method_name}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()