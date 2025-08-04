import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from itertools import islice
import json, os, random, glob, argparse, csv
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

# Midtrain mixture composition: specialty_data_percentage
MIXTURE_COMPOSITION = {
    'StarCoder': 0.20,       # 20% StarCoder + 80% C4
    'FLAN Combined': 0.05,   # 5% FLAN + 95% C4  
    'Math Combined': 0.12,   # 12% Math + 88% C4
    'DCLM': 0.20,           # 20% DCLM + 80% C4
    'KnowledgeQA': 0.20     # 20% KnowledgeQA + 80% C4
}

# Utility to map dataset names
def map_name(ds):
    name_map = {
        'bigcode/starcoderdata': 'StarCoder',
        'flan_combined': 'FLAN Combined',
        'math_combined': 'Math Combined',
        'dclm_combined': 'DCLM',
        'knowledgeqa_combined': 'KnowledgeQA',
        'allenai/c4': 'C4'
    }
    return name_map.get(ds, ds)

def is_midtrain_mix(dataset_name):
    """Check if dataset is a midtrain mixture (contains C4)."""
    mapped_name = map_name(dataset_name)
    return mapped_name in MIXTURE_COMPOSITION

def get_specialty_proportion(dataset_name):
    """Get the specialty data proportion for midtrain mixes."""
    mapped_name = map_name(dataset_name)
    return MIXTURE_COMPOSITION.get(mapped_name, 1.0)

def get_c4_proportion(dataset_name):
    """Get the C4 proportion for midtrain mixes."""
    if is_midtrain_mix(dataset_name):
        return 1.0 - get_specialty_proportion(dataset_name)
    return 0.0

# n-gram extraction
def get_ngrams(text, n):
    words = text.lower().split()
    if n == 1:
        return words
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)] if len(words)>=n else []

def get_char_ngrams(text, n):
    text = text.lower()
    return [text[i:i+n] for i in range(len(text)-n+1)] if len(text)>=n else []

# Mixture-aware similarity functions
def compute_mixture_aware_token_similarity(dataset1_data, dataset2_data, dataset1_name, dataset2_name, 
                                         c4_data, tokenizer, use_mixture_weights=False, vocab_size_limit=50000):
    """
    Compute token similarity accounting for actual dataset compositions.
    
    Args:
        dataset1_data, dataset2_data: Pure specialty data for each dataset
        dataset1_name, dataset2_name: Dataset names
        c4_data: Pure C4 data
        tokenizer: HuggingFace tokenizer
        use_mixture_weights: Whether to account for mixture compositions
    """
    print("    Computing mixture-aware token similarity...")
    
    # Determine actual data compositions
    if use_mixture_weights:
        if is_midtrain_mix(dataset1_name):
            c4_prop1 = get_c4_proportion(dataset1_name)
            spec_prop1 = get_specialty_proportion(dataset1_name)
            print(f"      {map_name(dataset1_name)}: {c4_prop1:.1%} C4 + {spec_prop1:.1%} specialty")
            # Create mixed dataset1: combine C4 and specialty data
            mixed_data1 = create_mixed_dataset(c4_data, dataset1_data, c4_prop1, spec_prop1)
        else:
            mixed_data1 = dataset1_data
            print(f"      {map_name(dataset1_name)}: 100% pure")
            
        if is_midtrain_mix(dataset2_name):
            c4_prop2 = get_c4_proportion(dataset2_name)
            spec_prop2 = get_specialty_proportion(dataset2_name)
            print(f"      {map_name(dataset2_name)}: {c4_prop2:.1%} C4 + {spec_prop2:.1%} specialty")
            # Create mixed dataset2: combine C4 and specialty data
            mixed_data2 = create_mixed_dataset(c4_data, dataset2_data, c4_prop2, spec_prop2)
        else:
            mixed_data2 = dataset2_data
            print(f"      {map_name(dataset2_name)}: 100% pure")
    else:
        # No mixture weighting - use pure data
        mixed_data1 = dataset1_data
        mixed_data2 = dataset2_data
    
    # Tokenize the (potentially mixed) datasets
    tokens1 = tokenize_texts(mixed_data1, tokenizer)
    tokens2 = tokenize_texts(mixed_data2, tokenizer)
    
    # Compute similarity on the tokenized data
    return compute_token_similarity_from_tokens(tokens1, tokens2, vocab_size_limit)

def create_mixed_dataset(c4_data, specialty_data, c4_proportion, specialty_proportion):
    """
    Create a mixed dataset by sampling from C4 and specialty data according to proportions.
    """
    # Calculate how many samples to take from each
    total_samples = len(specialty_data)
    c4_samples = int(total_samples * c4_proportion / specialty_proportion)
    
    # Sample from C4 and combine
    sampled_c4 = random.sample(c4_data, min(c4_samples, len(c4_data)))
    mixed_data = sampled_c4 + specialty_data
    random.shuffle(mixed_data)
    
    return mixed_data

def tokenize_texts(texts, tokenizer):
    """Tokenize a list of texts and return all tokens."""
    tokens = []
    batch_size = 32
    
    print(f"      Tokenizing {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), batch_size), desc="      Tokenization"):
        batch = texts[i:i+batch_size]
        batch = [text[:10000] if len(text) > 10000 else text for text in batch]
        try:
            encoded = tokenizer(batch, 
                              add_special_tokens=False, 
                              truncation=True, 
                              max_length=512,
                              padding=False,
                              return_attention_mask=False)
            for token_ids in encoded['input_ids']:
                tokens.extend(token_ids)
        except Exception as e:
            print(f"        Warning: Failed to tokenize batch: {e}")
            for text in batch:
                try:
                    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
                    tokens.extend(token_ids)
                except:
                    continue
    
    return tokens

def compute_token_similarity_from_tokens(tokens1, tokens2, vocab_size_limit):
    """Compute similarity metrics from pre-tokenized data."""
    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)
    
    total_tokens1 = sum(freq1.values())
    total_tokens2 = sum(freq2.values())
    unique_tokens1 = len(freq1)
    unique_tokens2 = len(freq2)
    
    print(f"      Dataset 1: {total_tokens1:,} total tokens, {unique_tokens1:,} unique tokens")
    print(f"      Dataset 2: {total_tokens2:,} total tokens, {unique_tokens2:,} unique tokens")
    
    freq1_top = dict(freq1.most_common(vocab_size_limit))
    freq2_top = dict(freq2.most_common(vocab_size_limit))
    
    vocab1 = set(freq1_top.keys())
    vocab2 = set(freq2_top.keys())
    
    vocab_jaccard = len(vocab1 & vocab2) / len(vocab1 | vocab2) if vocab1 | vocab2 else 0.0
    vocab_overlap_ratio = len(vocab1 & vocab2) / min(len(vocab1), len(vocab2)) if min(len(vocab1), len(vocab2)) > 0 else 0.0
    
    all_tokens = vocab1 | vocab2
    vec1 = np.array([freq1_top.get(token, 0) for token in all_tokens])
    vec2 = np.array([freq2_top.get(token, 0) for token in all_tokens])
    
    # Normalize to probabilities
    vec1_norm = vec1 / (np.sum(vec1) + 1e-10)
    vec2_norm = vec2 / (np.sum(vec2) + 1e-10)
    
    # Cosine similarity
    cosine_sim = np.dot(vec1_norm, vec2_norm) / (np.linalg.norm(vec1_norm) * np.linalg.norm(vec2_norm) + 1e-10)
    
    # Jensen-Shannon similarity
    avg_vec = 0.5 * (vec1_norm + vec2_norm)
    kl1 = np.sum(vec1_norm * np.log(vec1_norm / (avg_vec + 1e-10) + 1e-10))
    kl2 = np.sum(vec2_norm * np.log(vec2_norm / (avg_vec + 1e-10) + 1e-10))
    js_divergence = 0.5 * (kl1 + kl2)
    js_similarity = 1.0 / (1.0 + js_divergence)
    
    # Combined score
    combined_score = (0.4 * cosine_sim + 0.3 * vocab_jaccard + 0.3 * js_similarity)
    
    print(f"      Token-based metrics:")
    print(f"        Vocabulary Jaccard: {vocab_jaccard:.3f}")
    print(f"        Vocabulary overlap ratio: {vocab_overlap_ratio:.3f}")
    print(f"        Token frequency cosine: {cosine_sim:.3f}")
    print(f"        JS similarity: {js_similarity:.3f}")
    print(f"        Combined score: {combined_score:.3f}")
    
    return {
        'vocab_jaccard': vocab_jaccard,
        'vocab_overlap_ratio': vocab_overlap_ratio,
        'token_freq_cosine': cosine_sim,
        'js_similarity': js_similarity,
        'combined_score': combined_score
    }

def compute_mixture_aware_ngram_similarity(dataset1_data, dataset2_data, dataset1_name, dataset2_name,
                                         c4_data, use_mixture_weights=False, n=2, ngram_type='word'):
    """Compute n-gram similarity accounting for actual dataset compositions."""
    print(f"    Computing mixture-aware {ngram_type} {n}-gram similarity...")
    
    # Determine actual data compositions
    if use_mixture_weights:
        if is_midtrain_mix(dataset1_name):
            c4_prop1 = get_c4_proportion(dataset1_name)
            spec_prop1 = get_specialty_proportion(dataset1_name)
            print(f"      {map_name(dataset1_name)}: {c4_prop1:.1%} C4 + {spec_prop1:.1%} specialty")
            mixed_data1 = create_mixed_dataset(c4_data, dataset1_data, c4_prop1, spec_prop1)
        else:
            mixed_data1 = dataset1_data
            print(f"      {map_name(dataset1_name)}: 100% pure")
            
        if is_midtrain_mix(dataset2_name):
            c4_prop2 = get_c4_proportion(dataset2_name)
            spec_prop2 = get_specialty_proportion(dataset2_name)
            print(f"      {map_name(dataset2_name)}: {c4_prop2:.1%} C4 + {spec_prop2:.1%} specialty")
            mixed_data2 = create_mixed_dataset(c4_data, dataset2_data, c4_prop2, spec_prop2)
        else:
            mixed_data2 = dataset2_data
            print(f"      {map_name(dataset2_name)}: 100% pure")
    else:
        mixed_data1 = dataset1_data
        mixed_data2 = dataset2_data
    
    # Extract n-grams from mixed datasets
    if ngram_type == 'word':
        ngrams1 = [gram for text in mixed_data1 for gram in get_ngrams(text, n)]
        ngrams2 = [gram for text in mixed_data2 for gram in get_ngrams(text, n)]
    elif ngram_type == 'char':
        ngrams1 = [gram for text in mixed_data1 for gram in get_char_ngrams(text, n)]
        ngrams2 = [gram for text in mixed_data2 for gram in get_char_ngrams(text, n)]
    else:
        raise ValueError("ngram_type must be 'word' or 'char'")
    
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)
    
    all_ngrams = set(counter1.keys()) | set(counter2.keys())
    
    if not all_ngrams:
        print(f"    Warning: No {ngram_type} {n}-grams found")
        return 0.0
    
    vec1 = np.array([counter1.get(ngram, 0) for ngram in all_ngrams])
    vec2 = np.array([counter2.get(ngram, 0) for ngram in all_ngrams])
    
    vec1 = vec1 / (vec1.sum() + 1e-10)
    vec2 = vec2 / (vec2.sum() + 1e-10)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm1 * norm2)
    
    print(f"    Mixture-aware {ngram_type.capitalize()} {n}-gram similarity: {similarity:.3f}")
    print(f"    Dataset 1: {len(counter1)} unique {ngram_type} {n}-grams")
    print(f"    Dataset 2: {len(counter2)} unique {ngram_type} {n}-grams") 
    print(f"    Overlap: {len(set(counter1.keys()) & set(counter2.keys()))} shared {ngram_type} {n}-grams")
    
    return similarity

def compute_mixture_aware_character_overlap_similarity(dataset1_data, dataset2_data, dataset1_name, dataset2_name,
                                                     c4_data, use_mixture_weights=False):
    """Compute character overlap similarity accounting for actual dataset compositions."""
    print("    Computing mixture-aware character overlap similarity...")
    
    # Determine actual data compositions
    if use_mixture_weights:
        if is_midtrain_mix(dataset1_name):
            c4_prop1 = get_c4_proportion(dataset1_name)
            spec_prop1 = get_specialty_proportion(dataset1_name)
            print(f"      {map_name(dataset1_name)}: {c4_prop1:.1%} C4 + {spec_prop1:.1%} specialty")
            mixed_data1 = create_mixed_dataset(c4_data, dataset1_data, c4_prop1, spec_prop1)
        else:
            mixed_data1 = dataset1_data
            print(f"      {map_name(dataset1_name)}: 100% pure")
            
        if is_midtrain_mix(dataset2_name):
            c4_prop2 = get_c4_proportion(dataset2_name)
            spec_prop2 = get_specialty_proportion(dataset2_name)
            print(f"      {map_name(dataset2_name)}: {c4_prop2:.1%} C4 + {spec_prop2:.1%} specialty")
            mixed_data2 = create_mixed_dataset(c4_data, dataset2_data, c4_prop2, spec_prop2)
        else:
            mixed_data2 = dataset2_data
            print(f"      {map_name(dataset2_name)}: 100% pure")
    else:
        mixed_data1 = dataset1_data
        mixed_data2 = dataset2_data
    
    combined1 = ' '.join(mixed_data1)
    combined2 = ' '.join(mixed_data2)
    
    char_counter1 = Counter(combined1.lower())
    char_counter2 = Counter(combined2.lower())
    
    all_chars = set(char_counter1.keys()) | set(char_counter2.keys())
    
    char_vec1 = np.array([char_counter1.get(char, 0) for char in all_chars])
    char_vec2 = np.array([char_counter2.get(char, 0) for char in all_chars])
    
    char_vec1 = char_vec1 / (char_vec1.sum() + 1e-10)
    char_vec2 = char_vec2 / (char_vec2.sum() + 1e-10)
    
    char_cosine = np.dot(char_vec1, char_vec2) / (np.linalg.norm(char_vec1) * np.linalg.norm(char_vec2) + 1e-10)
    
    print(f"    Mixture-aware character frequency cosine: {char_cosine:.3f}")
    
    return char_cosine

def compute_mixture_aware_multilevel_ngram_similarity(dataset1_data, dataset2_data, dataset1_name, dataset2_name,
                                                    c4_data, use_mixture_weights=False, ngram_sizes=[1,2,3,4], ngram_type='word'):
    """Compute multilevel n-gram similarity accounting for actual dataset compositions."""
    weights = [1.0/len(ngram_sizes)] * len(ngram_sizes)
    combined = 0.0
    
    for w, n in zip(weights, ngram_sizes):
        sim = compute_mixture_aware_ngram_similarity(
            dataset1_data, dataset2_data, dataset1_name, dataset2_name,
            c4_data, use_mixture_weights, n=n, ngram_type=ngram_type
        )
        combined += w * sim
    
    return combined

# Dataset loading functions - load PURE specialty data
def extract_text(sample, dataset, use_instruction_format=True):
    """Return a single 'text' string for each sample across multiple dataset types."""
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

def load_pure_math_data(sample_size=100):
    """Load PURE math data (not mixed with C4)."""
    print("Loading pure math datasets...")
    
    # Load OpenMath
    openmath = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
    openmath_samples = []
    for i, example in enumerate(islice(openmath, sample_size // 3)):
        text = f"{example['question']}\n\n{example['expected_answer']}"
        openmath_samples.append(text)
    
    # Load MathInstruct
    mathinstruct = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
    mathinstruct_samples = []
    for i, example in enumerate(islice(mathinstruct, sample_size // 3)):
        text = f"{example['instruction']}\n\n{example['output']}"
        mathinstruct_samples.append(text)
    
    # Load MATH-plus
    mathplus = load_dataset("TIGER-Lab/MATH-plus", split="train", streaming=True)
    mathplus_samples = []
    for i, example in enumerate(islice(mathplus, sample_size // 3)):
        text = f"{example['instruction']}\n\n{example['output']}"
        mathplus_samples.append(text)
    
    # Combine all samples
    all_samples = openmath_samples + mathinstruct_samples + mathplus_samples
    random.shuffle(all_samples)
    return all_samples[:sample_size]

def load_pure_flan_data(sample_size=100):
    """Load PURE FLAN data (not mixed with C4)."""
    print("Loading pure FLAN datasets...")
    flan_dir = "/data/tir/projects/tir3/users/mengyan3/manifold_data/datasets/flan/"
    
    train_files = sorted(glob.glob(os.path.join(flan_dir, "*_train.jsonl")))
    if not train_files:
        print("No FLAN files found locally, using fallback...")
        return [f"FLAN instruction {i} with output {i}" for i in range(sample_size)]
    
    raw = load_dataset("json", data_files={"train": train_files}, split="train", streaming=True)
    raw = raw.shuffle(buffer_size=10_000, seed=42)
    
    samples = []
    for example in islice(raw, sample_size):
        if example.get("inputs") and example.get("targets"):
            text = example["inputs"].strip() + "\n\n" + example["targets"].strip()
            samples.append(text)
    
    return samples

def load_pure_knowledgeqa_data(sample_size=100):
    """Load PURE knowledgeQA data (not mixed with C4)."""
    print("Loading pure knowledgeQA datasets...")
    sft_data_path = "/data/tir/projects/tir3/users/mengyan3/manifold_data/knowledgeqa_formatted_revised"
    
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
                            
                            all_samples.append(combined_text.strip())
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"    Error loading {split}: {e}")
                continue
        else:
            print(f"    {split} file not found, skipping...")
    
    if not all_samples:
        print("  No knowledgeQA files found, using placeholder...")
        all_samples = [f"Knowledge QA example {i} with answer {i}" for i in range(sample_size)]
    
    random.shuffle(all_samples)
    return all_samples[:sample_size]

def load_pure_dclm_data(sample_size=100):
    """Load PURE DCLM data (not mixed with C4)."""
    print("Loading pure DCLM dataset...")
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
                samples.append(text)
        
        return samples
    except Exception as e:
        print(f"  Error loading DCLM: {e}, using placeholder...")
        return [f"DCLM high-quality web text example {i}" for i in range(sample_size)]

def get_dataset_texts(dataset_name, sample_size, use_instruction_format=True):
    """Load PURE specialty data for each dataset (no C4 mixing here)."""
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
        texts = load_pure_math_data(sample_size)
                
    elif dataset_name == "flan_combined":
        texts = load_pure_flan_data(sample_size)
                
    elif dataset_name == "knowledgeqa_combined":
        texts = load_pure_knowledgeqa_data(sample_size)
                
    elif dataset_name == "dclm_combined":
        texts = load_pure_dclm_data(sample_size)
                
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

    return texts

def compute_syntactic_similarity_matrix(all_datasets,
                                         method='ngram',
                                         sample_size=100,
                                         use_instruction_format=True,
                                         ngram_size=2,
                                         ngram_type='word',
                                         max_n=4,
                                         tokenizer=None,
                                         use_mixture_weights=False):
    """
    Compute similarity matrix accounting for actual dataset compositions.
    """
    N = len(all_datasets)
    sim = np.zeros((N, N))

    # Load all datasets (pure specialty data)
    print("Loading pure specialty data for all datasets...")
    all_texts = {}
    for i, ds in enumerate(all_datasets):
        print(f"→ Loading pure texts for {ds} ({i+1}/{N})...")
        all_texts[ds] = get_dataset_texts(ds, sample_size, use_instruction_format)
        print(f"  Loaded {len(all_texts[ds])} pure texts")

    # Load C4 data separately (needed for mixing)
    c4_data = None
    if use_mixture_weights:
        print("\n→ Loading C4 data for mixture compositions...")
        c4_data = get_dataset_texts("allenai/c4", sample_size, use_instruction_format)
        print(f"  Loaded {len(c4_data)} C4 texts")

    print(f"\nComputing similarity matrix with mixture awareness: {'ENABLED' if use_mixture_weights else 'DISABLED'}")
    
    if use_mixture_weights:
        print("Mixture compositions:")
        for dataset, spec_prop in MIXTURE_COMPOSITION.items():
            c4_prop = 1.0 - spec_prop
            print(f"  {dataset}: {c4_prop:.1%} C4 + {spec_prop:.1%} specialty")
    
    total_comparisons = N * (N - 1)  # Excluding diagonal
    completed = 0
    
    for i in range(N):
        for j in range(N):
            if i == j:
                sim[i, j] = 1.0
                continue

            completed += 1
            A, B = all_datasets[i], all_datasets[j]
            print(f"\n[{completed}/{total_comparisons}] Computing similarity: {A} → {B}")

            if method == 'ngram':
                sim[i, j] = compute_mixture_aware_ngram_similarity(
                    all_texts[A], all_texts[B], A, B, c4_data, use_mixture_weights,
                    n=ngram_size, ngram_type=ngram_type
                )

            elif method == 'char_overlap':
                sim[i, j] = compute_mixture_aware_character_overlap_similarity(
                    all_texts[A], all_texts[B], A, B, c4_data, use_mixture_weights
                )

            elif method == 'multilevel':
                sim[i, j] = compute_mixture_aware_multilevel_ngram_similarity(
                    all_texts[A], all_texts[B], A, B, c4_data, use_mixture_weights,
                    ngram_sizes=list(range(1, max_n+1)), ngram_type=ngram_type
                )
                
            elif method == 'token':
                if tokenizer is None:
                    raise ValueError("Tokenizer required for token-based similarity")
                result = compute_mixture_aware_token_similarity(
                    all_texts[A], all_texts[B], A, B, c4_data, tokenizer, use_mixture_weights
                )
                sim[i, j] = result['combined_score']

            else:
                raise ValueError(f"Unknown method: {method}")

            print(f"  Similarity {A} → {B}: {sim[i,j]:.3f}")

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

def save_similarity_csv(sim, all_datasets, display_names, method_name, args, png_output_file):
    """Save the similarity matrix as a CSV file."""
    # Derive CSV filename from PNG output file by changing extension
    import os
    csv_file = os.path.splitext(png_output_file)[0] + '.csv'
    
    # Write CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header with metadata
        writer.writerow([f"# Dataset Similarity Matrix - {method_name}"])
        writer.writerow([f"# Sample size: {args.sample_size}"])
        writer.writerow([f"# Method: {args.method}"])
        if args.method == 'ngram':
            writer.writerow([f"# N-gram type: {args.ngram_type}, size: {args.ngram_size}"])
        elif args.method == 'multilevel':
            writer.writerow([f"# N-gram type: {args.ngram_type}, range: 1-{args.max_n}"])
        elif args.method == 'token':
            writer.writerow([f"# Tokenizer: {args.tokenizer_name}"])
        writer.writerow([f"# Instruction format: {args.use_instruction_format}"])
        writer.writerow([f"# Mixture awareness: {'ENABLED' if args.enable_mixture_weights else 'DISABLED'}"])
        if args.enable_mixture_weights:
            writer.writerow([f"# Mixture compositions: {MIXTURE_COMPOSITION}"])
        writer.writerow([])  # Empty row for separation
        
        # Write column headers
        header = ["Dataset"] + display_names
        writer.writerow(header)
        
        # Write similarity matrix rows
        for i, dataset_name in enumerate(display_names):
            row = [dataset_name] + [f"{sim[i, j]:.6f}" for j in range(len(all_datasets))]
            writer.writerow(row)
    
    print(f"✔ Saved similarity matrix to {csv_file}")
    return csv_file

def main():
    parser = argparse.ArgumentParser(description='Compute mixture-aware dataset similarity matrix')
    parser.add_argument('--method', choices=['ngram', 'char_overlap', 'multilevel', 'token'], default='ngram',
                       help='Syntactic similarity method')
    parser.add_argument('--ngram_size', type=int, default=2,
                       help='Size of n-grams (for single ngram method)')
    parser.add_argument('--ngram_type', choices=['word', 'char'], default='word',
                       help='Type of n-grams: word or character')
    parser.add_argument('--max_n', type=int, default=4,
                       help='Maximum n-gram size (for multilevel method)')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of samples per dataset')
    parser.add_argument('--use_instruction_format', action='store_true',
                       help='Use instruction format for datasets where applicable')
    parser.add_argument('--enable_mixture_weights', action='store_true',
                       help='Enable mixture-aware similarity calculation based on actual training compositions')
    parser.add_argument('--output_file', type=str, default='dataset_similarity_matrix_syntactic.png',
                       help='Output file for the similarity matrix plot')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/pythia-70m',
                       help='Tokenizer to use for token-based similarity')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Dataset configuration
    pretrain_datasets = [
        "allenai/c4",
        "bigcode/starcoderdata",
        "math_combined",
        "flan_combined",
        "dclm_combined",
        "knowledgeqa_combined"
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
        "C4", "StarCoder", "Math Combined", "FLAN Combined", "DCLM", "KnowledgeQA",  # pretrain
        "GSM8K", "PyCode", "Social IQA", "LIMA", "SciQ", "Movie Reviews", "TREC"  # downstream
    ]
    
    print(f"Using syntactic method: {args.method}")
    if args.method == 'ngram':
        print(f"N-gram configuration: {args.ngram_type} {args.ngram_size}-grams")
    elif args.method == 'multilevel':
        print(f"Multi-level configuration: {args.ngram_type} 1-{args.max_n} grams")
    elif args.method == 'token':
        print(f"Token-based configuration: Using {args.tokenizer_name}")
    
    print(f"Mixture awareness: {'ENABLED' if args.enable_mixture_weights else 'DISABLED'}")
    if args.enable_mixture_weights:
        print("Training mixture compositions:")
        for dataset, spec_prop in MIXTURE_COMPOSITION.items():
            c4_prop = 1.0 - spec_prop
            print(f"  {dataset}: {c4_prop:.1%} C4 + {spec_prop:.1%} specialty data")
    
    # Load tokenizer if needed
    tokenizer = None
    if args.method == 'token':
        print(f"Loading tokenizer: {args.tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        print(f"✓ Loaded tokenizer with vocabulary size: {len(tokenizer)}")
    
    # Compute similarity matrix
    sim = compute_syntactic_similarity_matrix(
        all_datasets, 
        method=args.method,
        sample_size=args.sample_size,
        use_instruction_format=args.use_instruction_format,
        ngram_size=args.ngram_size,
        ngram_type=args.ngram_type,
        max_n=args.max_n,
        tokenizer=tokenizer,
        use_mixture_weights=args.enable_mixture_weights
    )
    
    # Create method name and output file
    format_suffix = "_instruct" if args.use_instruction_format else "_plain"
    mixture_suffix = "_mixaware" if args.enable_mixture_weights else "_pure"
    
    if args.method == 'ngram':
        method_name = f"Mixture-Aware Syntactic N-gram ({args.ngram_type} {args.ngram_size}-grams, {'instructional' if args.use_instruction_format else 'plain'}, {'MIXTURE AWARE' if args.enable_mixture_weights else 'PURE DATA'})"
        default_output_file = f"dataset_similarity_matrix_syntactic_ngram_{args.ngram_type}{args.ngram_size}{format_suffix}{mixture_suffix}.png"
    elif args.method == 'multilevel':
        method_name = f"Mixture-Aware Syntactic Multi-level ({args.ngram_type} 1-{args.max_n} grams, {'instructional' if args.use_instruction_format else 'plain'}, {'MIXTURE AWARE' if args.enable_mixture_weights else 'PURE DATA'})"
        default_output_file = f"dataset_similarity_matrix_syntactic_multilevel_{args.ngram_type}1to{args.max_n}{format_suffix}{mixture_suffix}.png"
    elif args.method == 'char_overlap':
        method_name = f"Mixture-Aware Syntactic Character Overlap ({'instructional' if args.use_instruction_format else 'plain'}, {'MIXTURE AWARE' if args.enable_mixture_weights else 'PURE DATA'})"
        default_output_file = f"dataset_similarity_matrix_syntactic_char_overlap{format_suffix}{mixture_suffix}.png"
    elif args.method == 'token':
        method_name = f"Mixture-Aware Token-based ({args.tokenizer_name}, {'instructional' if args.use_instruction_format else 'plain'}, {'MIXTURE AWARE' if args.enable_mixture_weights else 'PURE DATA'})"
        default_output_file = f"dataset_similarity_matrix_token_based_{args.tokenizer_name.replace('/', '-')}{format_suffix}{mixture_suffix}.png"
    
    # Use provided output file or default naming
    output_file = args.output_file if args.output_file != 'dataset_similarity_matrix_syntactic.png' else default_output_file
    
    # Create and save the plot
    create_similarity_plot(sim, all_datasets, display_names, pretrain_datasets, 
                          downstream_datasets, method_name, output_file)
    
    # Save CSV output
    csv_file = save_similarity_csv(sim, all_datasets, display_names, method_name, args, output_file)
    
    print(f"\nMixture-aware similarity analysis complete!")
    print(f"Method: {method_name}")
    print(f"Sample size: {args.sample_size}")
    print(f"Mixture awareness: {'ENABLED' if args.enable_mixture_weights else 'DISABLED'}")
    print(f"Output: {output_file}")
    print(f"CSV: {csv_file}")

if __name__ == "__main__":
    main()