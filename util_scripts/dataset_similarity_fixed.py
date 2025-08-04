#!/usr/bin/env python3
"""
Information-Theoretic Dataset Formatting Similarity Analysis

This script computes formatting similarity between datasets using information-theoretic
measures on character n-grams at multiple scales.
"""

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
import csv
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import zlib


def get_char_ngrams(text, n):
    """Extract character n-grams from text."""
    if len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def compute_jensen_shannon_similarity(texts1, texts2, ngram_type='char', n=3):
    """
    Compute formatting similarity using Jensen-Shannon divergence.
    This is symmetric and bounded, making it ideal for formatting comparisons.
    """
    print(f"        Computing Jensen-Shannon similarity (char {n}-grams)...")
    
    # Extract character n-grams
    ngrams1 = []
    ngrams2 = []
    for text in texts1:
        ngrams1.extend(get_char_ngrams(text, n))
    for text in texts2:
        ngrams2.extend(get_char_ngrams(text, n))
    
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)
    all_ngrams = list(set(counter1.keys()) | set(counter2.keys()))
    
    if not all_ngrams:
        print(f"        Warning: No character {n}-grams found")
        return 0.0
    
    # Build probability distributions
    total1 = sum(counter1.values())
    total2 = sum(counter2.values())
    
    if total1 == 0 or total2 == 0:
        return 0.0
    
    p1 = np.array([counter1.get(ngram, 0) / total1 for ngram in all_ngrams])
    p2 = np.array([counter2.get(ngram, 0) / total2 for ngram in all_ngrams])
    
    # Compute Jensen-Shannon distance
    js_distance = jensenshannon(p1, p2)
    
    # Convert to similarity (JS distance is in [0, 1])
    similarity = 1 - js_distance
    
    print(f"        JS distance: {js_distance:.3f}, Similarity: {similarity:.3f}")
    print(f"        Dataset 1: {len(counter1)} unique, Dataset 2: {len(counter2)} unique")
    print(f"        Overlap: {len(set(counter1.keys()) & set(counter2.keys()))} shared")
    
    return similarity


def compute_cross_entropy_similarity(texts1, texts2, ngram_type='char', n=3, smoothing=1e-10):
    """
    Compute formatting similarity using cross-entropy between n-gram distributions.
    """
    print(f"        Computing cross-entropy similarity (char {n}-grams)...")
    
    # Extract character n-grams
    ngrams1 = []
    ngrams2 = []
    for text in texts1:
        ngrams1.extend(get_char_ngrams(text, n))
    for text in texts2:
        ngrams2.extend(get_char_ngrams(text, n))
    
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)
    all_ngrams = list(set(counter1.keys()) | set(counter2.keys()))
    
    if not all_ngrams:
        return 0.0
    
    # Build probability distributions with smoothing
    total1 = sum(counter1.values()) + smoothing * len(all_ngrams)
    total2 = sum(counter2.values()) + smoothing * len(all_ngrams)
    
    p1 = np.array([(counter1.get(ngram, 0) + smoothing) / total1 for ngram in all_ngrams])
    p2 = np.array([(counter2.get(ngram, 0) + smoothing) / total2 for ngram in all_ngrams])
    
    # Compute symmetric cross-entropy
    h12 = -np.sum(p1 * np.log(p2))
    h21 = -np.sum(p2 * np.log(p1))
    avg_cross_entropy = (h12 + h21) / 2
    
    # Convert to similarity
    similarity = np.exp(-avg_cross_entropy)
    
    print(f"        Avg cross-entropy: {avg_cross_entropy:.3f}, Similarity: {similarity:.3f}")
    
    return similarity


def compute_kl_divergence_similarity(texts1, texts2, ngram_type='char', n=3, smoothing=1e-6):
    """
    Compute formatting similarity using KL divergence between n-gram distributions.
    """
    print(f"        Computing KL divergence similarity (char {n}-grams)...")
    
    # Extract character n-grams
    ngrams1 = []
    ngrams2 = []
    for text in texts1:
        ngrams1.extend(get_char_ngrams(text, n))
    for text in texts2:
        ngrams2.extend(get_char_ngrams(text, n))
    
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)
    all_ngrams = list(set(counter1.keys()) | set(counter2.keys()))
    
    if not all_ngrams:
        return 0.0
    
    # Build probability distributions with smoothing
    total1 = sum(counter1.values()) + smoothing * len(all_ngrams)
    total2 = sum(counter2.values()) + smoothing * len(all_ngrams)
    
    p1 = np.array([(counter1.get(ngram, 0) + smoothing) / total1 for ngram in all_ngrams])
    p2 = np.array([(counter2.get(ngram, 0) + smoothing) / total2 for ngram in all_ngrams])
    
    # Symmetric KL divergence
    kl12 = entropy(p1, p2)
    kl21 = entropy(p2, p1)
    avg_kl = (kl12 + kl21) / 2
    
    # Convert to similarity
    similarity = np.exp(-avg_kl)
    
    print(f"        Avg KL divergence: {avg_kl:.3f}, Similarity: {similarity:.3f}")
    
    return similarity


def compute_multilevel_info_theoretic_similarity(texts1, texts2, method='jensen_shannon', 
                                               ngram_sizes=[2, 3, 4, 5], weights=None):
    """
    Compute multilevel information-theoretic formatting similarity.
    """
    print(f"    Computing multilevel {method} similarity (char n-grams: {ngram_sizes})...")
    
    if weights is None:
        weights = [1.0 / len(ngram_sizes)] * len(ngram_sizes)
    elif len(weights) != len(ngram_sizes):
        raise ValueError("Length of weights must match length of ngram_sizes")
    
    similarities = {}
    weighted_scores = []
    
    for i, n in enumerate(ngram_sizes):
        print(f"      Computing character {n}-grams...")
        
        if method == 'jensen_shannon':
            similarity = compute_jensen_shannon_similarity(texts1, texts2, ngram_type='char', n=n)
        elif method == 'cross_entropy':
            similarity = compute_cross_entropy_similarity(texts1, texts2, ngram_type='char', n=n)
        elif method == 'kl_divergence':
            similarity = compute_kl_divergence_similarity(texts1, texts2, ngram_type='char', n=n)
        else:
            raise ValueError(f"Method {method} not supported for multilevel")
        
        similarities[f'char_{n}gram'] = similarity
        weighted_scores.append(weights[i] * similarity)
        print(f"        Character {n}-gram similarity: {similarity:.3f}")
    
    # Compute weighted average
    combined_score = sum(weighted_scores)
    similarities['combined_score'] = combined_score
    
    print(f"    Combined multilevel score: {combined_score:.3f}")
    print(f"    Weights used: {weights}")
    
    return similarities


def compute_compression_based_similarity(texts1, texts2):
    """
    Use compression ratios to measure formatting similarity.
    """
    print("    Computing compression-based formatting similarity...")
    
    # Combine texts for each dataset
    combined1 = '\n'.join(texts1)
    combined2 = '\n'.join(texts2)
    
    # Compress individual datasets
    compressed1 = zlib.compress(combined1.encode('utf-8'))
    compressed2 = zlib.compress(combined2.encode('utf-8'))
    
    # Compress concatenated datasets
    combined_both = combined1 + '\n' + combined2
    compressed_both = zlib.compress(combined_both.encode('utf-8'))
    
    # Calculate normalized compression distance (NCD)
    len1 = len(compressed1)
    len2 = len(compressed2)
    len_both = len(compressed_both)
    
    # NCD formula: (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    ncd = (len_both - min(len1, len2)) / max(len1, len2)
    
    # Convert to similarity
    similarity = max(0, 1 - ncd)
    
    print(f"        NCD: {ncd:.3f}, Similarity: {similarity:.3f}")
    
    return similarity


# Dataset loading functions - comprehensive from the semantic similarity script
def extract_text_simple(sample, dataset, use_instruction_format=True):
    """
    Extract text from sample based on dataset type.
    Now includes all datasets from the semantic similarity script.
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
        
    if dataset in ("knowledgeqa_combined",):
        text = sample.get("text", "")
        if use_instruction_format and len(text.split()) > 10:
            words = text.split()
            mid_point = len(words) // 2
            return f"Instruction: {' '.join(words[:mid_point])}\nResponse: {' '.join(words[mid_point:])}"
        return text
        
    if dataset in ("dclm_combined",):
        return sample.get("text", "")
        
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
    
    # Fallback: try common text fields
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


def load_knowledgeqa_combined(sample_size=100):
    """Load and combine knowledgeQA datasets."""
    print("Loading knowledgeQA datasets...")
    sft_data_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/knowledgeqa_formatted_revised"
    
    # Load different splits of knowledgeQA
    splits = ["triviaqa", "dclm", "dialogue", "openhermes", "ultrachat"]
    all_samples = []
    
    for split in splits:
        split_dir = os.path.join(sft_data_path, split)
        train_file = os.path.join(split_dir, "train.jsonl")
        
        if os.path.exists(train_file):
            print(f"  Loading {split} split...")
            try:
                with open(train_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= sample_size // len(splits):
                            break
                        try:
                            record = json.loads(line.strip())
                            instruction = record.get("instruction", "")
                            input_text = record.get("input", "")
                            output_text = record.get("output", "")
                            
                            combined_text = instruction
                            if input_text:
                                combined_text += " " + input_text
                            if output_text:
                                combined_text += " " + output_text
                            
                            all_samples.append({"text": combined_text.strip()})
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"    Error loading {split}: {e}")
                continue
        else:
            print(f"    {split} file not found, skipping...")
    
    if not all_samples:
        print("  No knowledgeQA files found, using placeholder...")
        all_samples = [{"text": f"Knowledge QA example {i} with answer {i}"} for i in range(sample_size)]
    
    random.shuffle(all_samples)
    return all_samples[:sample_size]


def load_dclm_combined(sample_size=100):
    """Load DCLM dataset."""
    print("Loading DCLM dataset...")
    try:
        dataset = load_dataset(
            "allenai/dolmino-mix-1124",
            name="dclm",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        dataset = dataset.with_format("python")
        dataset = dataset.shuffle(buffer_size=10_000, seed=42)
        
        samples = []
        for example in islice(dataset, sample_size):
            text = example.get("text", "")
            if text:
                samples.append({"text": text})
        
        return samples
    except Exception as e:
        print(f"  Error loading DCLM: {e}, using placeholder...")
        return [{"text": f"DCLM high-quality web text example {i}"} for i in range(sample_size)]


def load_dataset_texts(dataset_name, sample_size=1000, use_instruction_format=True):
    """
    Load texts from a dataset - comprehensive version with all datasets.
    Now includes all datasets from the semantic similarity script.
    """
    print(f"Loading {dataset_name}...")
    texts = []
    
    try:
        if dataset_name == "allenai/c4":
            ds = load_dataset(dataset_name, "en", split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "bigcode/starcoderdata":
            ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "math_combined":
            samples = load_math_combined(sample_size)
            for sample in samples:
                text = extract_text_simple(sample, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "flan_combined":
            samples = load_flan_combined(sample_size)
            for sample in samples:
                text = extract_text_simple(sample, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "knowledgeqa_combined":
            samples = load_knowledgeqa_combined(sample_size)
            for sample in samples:
                text = extract_text_simple(sample, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "dclm_combined":
            samples = load_dclm_combined(sample_size)
            for sample in samples:
                text = extract_text_simple(sample, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "openai/gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "pycode":
            file_path = "../manifold_data/all_in_one_pretraining/datasets/just_Nan-Do/code-search-net-python/Nan-Do/code-search-net-python/train.json"
            if os.path.exists(file_path):
                samples = load_local_json(file_path, sample_size)
                for sample in samples:
                    text = extract_text_simple(sample, dataset_name, use_instruction_format)
                    if text and len(text.strip()) > 0:
                        texts.append(text)
            else:
                print("  Local pycode file not found, using placeholder...")
                for i in range(min(sample_size, 100)):
                    if use_instruction_format:
                        texts.append(f"Instruction: Write a function that returns {i}\nCode: def function_{i}():\n    return {i}")
                    else:
                        texts.append(f"def function_{i}():\n    return {i}")
                        
        elif dataset_name == "allenai/social_i_qa":
            ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "nyu-mll/glue":
            ds = load_dataset("nyu-mll/glue", "rte", split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "ai2_arc":
            ds = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    
        elif dataset_name == "allenai/sciq":
            json_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/all_in_one_pretraining/datasets/just_allenai/sciq/allenai/sciq/train.json"
            if os.path.exists(json_path):
                samples = load_local_json(json_path, sample_size)
                for sample in samples:
                    text = extract_text_simple(sample, dataset_name, use_instruction_format)
                    if text and len(text.strip()) > 0:
                        texts.append(text)
            else:
                print(f"  JSON file not found, falling back to HuggingFace...")
                ds = load_dataset("allenai/sciq", split="train", streaming=True, trust_remote_code=True)
                ds = ds.shuffle(buffer_size=10_000, seed=42)
                for example in islice(ds, sample_size):
                    text = extract_text_simple(example, dataset_name, use_instruction_format)
                    if text and len(text.strip()) > 0:
                        texts.append(text)

        elif dataset_name == "GAIR/lima":
            ds = load_dataset("GAIR/lima", split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)

        elif dataset_name == "mattymchen/mr":
            json_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/all_in_one_pretraining/datasets/just_mattymchen/mr/mattymchen/mr/train.json"
            if os.path.exists(json_path):
                samples = load_local_json(json_path, sample_size)
                for sample in samples:
                    text = extract_text_simple(sample, dataset_name, use_instruction_format)
                    if text and len(text.strip()) > 0:
                        texts.append(text)
            else:
                print(f"  JSON file not found, falling back to HuggingFace...")
                ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
                ds = ds.shuffle(buffer_size=10_000, seed=42)
                for example in islice(ds, sample_size):
                    text = extract_text_simple(example, dataset_name, use_instruction_format)
                    if text and len(text.strip()) > 0:
                        texts.append(text)
                        
        elif dataset_name == "CogComp/trec":
            json_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/all_in_one_pretraining/datasets/just_CogComp/trec/CogComp/trec/train.json"
            if os.path.exists(json_path):
                samples = load_local_json(json_path, sample_size)
                for sample in samples:
                    text = extract_text_simple(sample, dataset_name, use_instruction_format)
                    if text and len(text.strip()) > 0:
                        texts.append(text)
            else:
                print(f"  JSON file not found, falling back to HuggingFace...")
                ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
                ds = ds.shuffle(buffer_size=10_000, seed=42)
                for example in islice(ds, sample_size):
                    text = extract_text_simple(example, dataset_name, use_instruction_format)
                    if text and len(text.strip()) > 0:
                        texts.append(text)
        else:
            # Try loading as-is for any other dataset
            ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            for example in islice(ds, sample_size):
                text = extract_text_simple(example, dataset_name, use_instruction_format)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                
    except Exception as e:
        print(f"  Error loading {dataset_name}: {e}")
        # Create placeholder data for demonstration
        for i in range(min(sample_size, 100)):
            if "code" in dataset_name.lower():
                texts.append(f"def function_{i}():\n    # This is function {i}\n    return {i}")
            elif "math" in dataset_name.lower():
                texts.append(f"Problem: What is {i} + {i}?\nSolution: {i} + {i} = {i*2}")
            else:
                texts.append(f"This is sample text {i} from dataset {dataset_name}.")
    
    print(f"  Loaded {len(texts)} texts")
    return texts


def compute_formatting_similarity_matrix(datasets, method='multilevel_jensen_shannon', 
                                       sample_size=1000, ngram_sizes=[2, 3, 4, 5], weights=None):
    """
    Compute formatting similarity matrix using information-theoretic measures.
    """
    N = len(datasets)
    sim = np.zeros((N, N))
    
    print(f"Computing formatting similarity using method: {method}")
    print(f"N-gram sizes: {ngram_sizes}")
    print(f"Sample size: {sample_size}")
    
    # Load texts for all datasets
    all_texts = {}
    for i, ds in enumerate(datasets):
        print(f"→ Loading texts for {ds} ({i+1}/{N})...")
        all_texts[ds] = load_dataset_texts(ds, sample_size)
    
    print("Computing similarity matrix...")
    # Only compute upper triangle (including diagonal)
    total_pairs = N * (N + 1) // 2
    completed_pairs = 0
    
    for i in range(N):
        for j in range(i, N):  # Only compute i <= j
            if i == j:
                sim[i, j] = 1.0
                print(f"[{completed_pairs+1}/{total_pairs}] Diagonal: {datasets[i]} = 1.000")
            else:
                print(f"[{completed_pairs+1}/{total_pairs}] Computing similarity between {datasets[i]} and {datasets[j]}")
                
                if method.startswith('multilevel_'):
                    base_method = method.replace('multilevel_', '')
                    result = compute_multilevel_info_theoretic_similarity(
                        all_texts[datasets[i]], 
                        all_texts[datasets[j]],
                        method=base_method,
                        ngram_sizes=ngram_sizes,
                        weights=weights
                    )
                    similarity = result['combined_score']
                elif method == 'jensen_shannon':
                    similarity = compute_jensen_shannon_similarity(
                        all_texts[datasets[i]], 
                        all_texts[datasets[j]],
                        n=3
                    )
                elif method == 'cross_entropy':
                    similarity = compute_cross_entropy_similarity(
                        all_texts[datasets[i]], 
                        all_texts[datasets[j]],
                        n=3
                    )
                elif method == 'kl_divergence':
                    similarity = compute_kl_divergence_similarity(
                        all_texts[datasets[i]], 
                        all_texts[datasets[j]],
                        n=3
                    )
                elif method == 'compression':
                    similarity = compute_compression_based_similarity(
                        all_texts[datasets[i]], 
                        all_texts[datasets[j]]
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Set both (i,j) and (j,i) since the measure is symmetric
                sim[i, j] = similarity
                sim[j, i] = similarity
                print(f"  Similarity: {similarity:.3f}")
            
            completed_pairs += 1
    
    return sim


def create_similarity_plot(sim, datasets, display_names, method_name, output_file):
    """Create and save the similarity matrix plot."""
    N = len(datasets)
    
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
    
    plt.title(f"Dataset Formatting Similarity Matrix\n({method_name})")
    fig.colorbar(c, ax=ax, label='Formatting Similarity')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✔ Saved similarity matrix to {output_file}")


def save_similarity_csv(sim, datasets, display_names, method_name, output_file):
    """Save the similarity matrix as a CSV file."""
    csv_file = output_file.replace('.png', '.csv')
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header with metadata
        writer.writerow([f"# Dataset Formatting Similarity Matrix - {method_name}"])
        writer.writerow([f"# Method: {method_name}"])
        writer.writerow([])
        
        # Write column headers
        header = ["Dataset"] + display_names
        writer.writerow(header)
        
        # Write similarity matrix rows
        for i, dataset_name in enumerate(display_names):
            row = [dataset_name] + [f"{sim[i, j]:.6f}" for j in range(len(datasets))]
            writer.writerow(row)
    
    print(f"✔ Saved similarity matrix to {csv_file}")
    return csv_file


def main():
    parser = argparse.ArgumentParser(description='Compute information-theoretic dataset formatting similarity')
    parser.add_argument('--method', 
                       choices=['jensen_shannon', 'cross_entropy', 'kl_divergence', 'compression',
                               'multilevel_jensen_shannon', 'multilevel_cross_entropy', 'multilevel_kl_divergence'], 
                       default='multilevel_jensen_shannon',
                       help='Information-theoretic similarity method')
    parser.add_argument('--ngram_sizes', nargs='+', type=int, default=[2, 3, 4, 5],
                       help='Character n-gram sizes for multilevel methods')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                       help='Weights for n-gram levels (must match ngram_sizes length)')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of samples per dataset')
    parser.add_argument('--datasets', nargs='+', 
                       default=[
                           # Pretrain/Midtrain datasets
                           "allenai/c4", "bigcode/starcoderdata", "math_combined", 
                           "flan_combined", "dclm_combined", "knowledgeqa_combined",
                           # Downstream datasets
                           "openai/gsm8k", "pycode", "allenai/social_i_qa", 
                           "GAIR/lima", "allenai/sciq", "mattymchen/mr", "CogComp/trec"
                       ],
                       help='List of datasets to compare')
    parser.add_argument('--display_names', nargs='+', default=None,
                       help='Display names for datasets (must match datasets length)')
    parser.add_argument('--use_instruction_format', action='store_true',
                       help='Use instruction format for datasets where applicable')
    parser.add_argument('--output', type=str, default='formatting_similarity_matrix.png',
                       help='Output file for the similarity matrix plot')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Validate arguments
    if args.weights and len(args.weights) != len(args.ngram_sizes):
        raise ValueError("Length of weights must match length of ngram_sizes")
    
    if args.display_names and len(args.display_names) != len(args.datasets):
        raise ValueError("Length of display_names must match length of datasets")
    
    # Use dataset names as display names if not provided
    if args.display_names and len(args.display_names) != len(args.datasets):
        raise ValueError("Length of display_names must match length of datasets")
    
    # Default display names from the semantic similarity script
    default_display_names = {
        "allenai/c4": "C4",
        "bigcode/starcoderdata": "StarCoder", 
        "math_combined": "Math Combined",
        "flan_combined": "FLAN Combined",
        "dclm_combined": "DCLM",
        "knowledgeqa_combined": "KnowledgeQA",
        "openai/gsm8k": "GSM8K",
        "pycode": "PyCode",
        "allenai/social_i_qa": "Social IQA",
        "GAIR/lima": "LIMA",
        "allenai/sciq": "SciQ",
        "mattymchen/mr": "Movie Reviews",
        "CogComp/trec": "TREC"
    }
    
    display_names = args.display_names or [default_display_names.get(ds, ds.split('/')[-1]) for ds in args.datasets]
    
    print(f"Analyzing formatting similarity for datasets: {args.datasets}")
    print(f"Method: {args.method}")
    if args.method.startswith('multilevel'):
        print(f"N-gram sizes: {args.ngram_sizes}")
        print(f"Weights: {args.weights or 'uniform'}")
    
    # Compute similarity matrix
    sim = compute_formatting_similarity_matrix(
        args.datasets,
        method=args.method,
        sample_size=args.sample_size,
        ngram_sizes=args.ngram_sizes,
        weights=args.weights
    )
    
    # Create method name for display
    if args.method.startswith('multilevel'):
        base_method = args.method.replace('multilevel_', '').replace('_', ' ').title()
        method_display = f"Multilevel {base_method} (char {args.ngram_sizes[0]}-{args.ngram_sizes[-1]} grams)"
    else:
        method_display = args.method.replace('_', ' ').title()
    
    # Create and save outputs
    create_similarity_plot(sim, args.datasets, display_names, method_display, args.output)
    save_similarity_csv(sim, args.datasets, display_names, method_display, args.output)
    
    print(f"\nFormatting similarity analysis complete!")
    print(f"Method: {method_display}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output: {args.output}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean similarity: {np.mean(sim[np.triu_indices_from(sim, k=1)]):.3f}")
    print(f"Min similarity: {np.min(sim[np.triu_indices_from(sim, k=1)]):.3f}")
    print(f"Max similarity: {np.max(sim[np.triu_indices_from(sim, k=1)]):.3f}")


if __name__ == "__main__":
    main()