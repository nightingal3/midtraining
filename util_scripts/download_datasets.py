# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import argparse
import json
import os
from collections import defaultdict
from functools import partial
import string
import dotenv
from datasets import load_dataset, concatenate_datasets, Dataset
import random
import re
from pathlib import Path

dotenv.load_dotenv(dotenv_path="./configs/.env")
MANIFOLD_DIR = os.environ["MANIFOLD_DIR"]

default_qa_template = {
    "instruction": "{instruction}. {generated_choices_str}\nAnswer:",
    "output": "{output}"
}


def make_mcq_string(choices):
    alpha_order = string.ascii_uppercase
    choices_str = "\n".join([f"{alpha_order[i]}. {choice}" for i, choice in enumerate(choices)])
    return choices_str


datasets = {
    # TODO: natural instructions already downloaded, unzip and upload to manifold manually
    "pretraining": ["HuggingFaceFW/fineweb-edu", "bigcode/starcoderdata", "allenai/c4"],
    "instruction": [
        "databricks/databricks-dolly-15k",
        "GAIR/lima",
        "tatsu-lab/alpaca",
        "hkust-nlp/deita-10k-v0",
        "allenai/tulu-v2-sft-mixture",
    ],
    "sft": [
        "openai/gsm8k",
        "EleutherAI/asdiv",
        "allenai/ai2_arc",
        "allenai/sciq",
        "Rowan/hellaswag",
        "microsoft/wiki_qa",
        "ybisk/piqa",
        "Nan-Do/code-search-net-python",
        "mattymchen/mr",       # movie-review sentiment
        "nyu_mll/glue",       # GLUE benchmark
        "CogComp/trec"
    ],
    "sft_reasoning": [
        "lighteval/MATH",
        "openai/gsm8k",
        "EleutherAI/asdiv",
        "allenai/ai2_arc",
        "allenai/sciq",
        "ChilleD/SVAMP",
        "microsoft/wiki_qa",
        "nvidia/OpenMathInstruct-1",
        "allenai/social_i_qa",  # Social commonsense QA
    ],
    "blank": [
        "HuggingFaceFW/fineweb-edu"
    ],
    "mmlu": [
        "cais/mmlu",
    ]
}


def mcq_to_text(answers: list, labels: list, correct_label: str):
    labels_to_answers = {lab: ans for ans, lab in zip(answers, labels)}
    return labels_to_answers[correct_label]


# some datasets have different formats, store them in a common way with instruction/input(context)/output
def format_row(sample: dict, dataset: str, mmlu_with_mcq: bool = False) -> dict:
    if dataset == "databricks/databricks-dolly-15k":
        return {
            "instruction": sample["instruction"],
            "input": sample["context"],
            "output": sample["response"],
        }
    elif dataset == "tatsu-lab/alpaca":
        return {
            "instruction": sample["instruction"],
            "input": sample["input"],
            "output": sample["output"],
        }
    elif dataset == "openai/gsm8k":
        return {
            "instruction": sample["question"],
            "input": "",
            "output": sample["answer"],
        }
    elif dataset == "EleutherAI/asdiv":
        return {
            "instruction": sample["question"],
            "input": sample["body"],
            "output": sample["answer"],
        }
    elif dataset == "GAIR/lima":
        return {
            "instruction": sample["conversations"][0],
            "input": "",
            "output": "".join(sample["conversations"][1:]),
        }
    elif dataset == "allenai/sciq":
        return {
            "instruction": sample["question"],
            "input": sample["support"],
            "output": sample["correct_answer"],
            "choices": random.sample(
                [sample["distractor1"], sample["distractor2"], sample["distractor3"], sample["correct_answer"]],
                k=4
            )
        }
    elif dataset == "allenai/ai2_arc":
        return {
            "instruction": sample["question"],
            "input": "",
            "output": mcq_to_text(
                sample["choices"]["text"],
                sample["choices"]["label"],
                sample["answerKey"],
            ),
            "choices": sample["choices"]["text"]
        }
    elif dataset == "Rowan/hellaswag":
        try:
            return {
                "instruction": sample["ctx"],
                "input": "",
                "output": mcq_to_text(
                    sample["endings"],
                    ["0", "1", "2", "3"],
                    sample["label"],
                ),
            }
        except:
            # some don't have labels
            return {}
    elif dataset == "microsoft/wiki_qa":
        # this dataset seems to have many negative examples. Just taking the positive ones
        return {
            "instruction": sample["question"],
            "input": "",
            "output": sample["answer"],
        }
    elif dataset == "ybisk/piqa":
        if sample["label"] != -1:
            return {
                "instruction": sample["goal"],
                "input": "",
                "output": mcq_to_text(
                    [sample["sol1"], sample["sol2"]],
                    [0, 1],
                    sample["label"]
                ),
            }
        else:
            return {}
    elif dataset == "allenai/tulu-v2-sft-mixture":
        # NOTE: I'm excluding multi-turn conversations for now
        return {
            "instruction": sample["messages"][0]["content"],
            "input": "",
            "output": sample["messages"][1]["content"],
            "label": len(sample["messages"]) == 2,
        }
    elif dataset == "hkust-nlp/deita-10k-v0":
        # NOTE: TODO: since this dataset only has multi-turn convos, just take the first interaction.
        return {
            "instruction": sample["conversations"][0]["value"],
            "input": "",
            "output": sample["conversations"][1]["value"],
            "label": True,
        }
    elif dataset == "lighteval/MATH":
        return {
            "instruction": sample["problem"],
            "input": "",
            "output": sample["solution"]
        }
    elif dataset == "ChilleD/SVAMP":
        return {
            "instruction": sample["question_concat"],
            "input": "",
            "output": sample["Answer"]
        }
    elif dataset == "nvidia/OpenMathInstruct-1":
        return {
            "instruction": sample["question"],
            "input": sample["generated_solution"],
            "output": sample["expected_answer"]
        }
    elif dataset == "HuggingFaceFW/fineweb-edu":
        return {
            "instruction": sample["question"],
            "input": "",
            "output": sample["answer"]
        }
    elif dataset == "bigcode/starcoderdata":
        return {
            "instruction": "",
            "input": "",
            "output": sample["content"]
        }
    elif dataset == "allenai/c4":
        return {
            "instruction": "",
            "input": "",
            "output": sample["text"]
        }
    elif dataset == "cais/mmlu":
        mcq_string = make_mcq_string(sample["choices"])
        if mmlu_with_mcq:
            return {
                "instruction": sample["question"] + "\n" + mcq_string,
                "input": "",
                "output": mcq_to_text(
                    sample["choices"],
                    [0, 1, 2, 3],
                    sample["answer"],
                ),
                "choices": sample["choices"]
            }
        else:
            return {
                "instruction": sample["question"],
                "input": "",
                "output": mcq_to_text(
                    sample["choices"],
                    [0, 1, 2, 3],
                    sample["answer"],
                ),
            }
    elif dataset == "Nan-Do/code-search-net-python":
        return {
            "instruction": "",
            "input": "",
            "output": sample["original_string"],
        }

    # Added: Social IQA
    elif dataset == "allenai/social_i_qa":
        return {
            "instruction": sample["question"],
            "input": sample["context"],
            "choices": [
                sample["answerA"],
                sample["answerB"],
                sample["answerC"],
            ],
            "output": mcq_to_text(
                [sample["answerA"], sample["answerB"], sample["answerC"]],
                ["1", "2", "3"],
                sample["label"],
            ),
        }

    # Added: MR sentiment
    elif dataset == "mattymchen/mr":
        # Debug: Print the label to see what we're getting
        print(f"DEBUG MR: label = {sample['label']}, type = {type(sample['label'])}")
        
        # Map numeric labels to text labels
        label_mapping = {
            0: "negative",
            1: "positive"
        }
        
        mapped_output = label_mapping.get(sample["label"], str(sample["label"]))
        print(f"DEBUG MR: mapped output = '{mapped_output}'")

        return {
            "instruction": sample.get("text", sample.get("sentence", "")),
            "input": "",
            "output": mapped_output,
        }

    # Added: GLUE benchmark
    elif dataset == "nyu_mll/glue":
        return {
            "instruction": sample.get("sentence1", ""),
            "input": sample.get("sentence2", ""),
            "output": str(sample["label"]),
        }

    elif dataset == "CogComp/trec":
        label_mapping = {
            0: "abbreviation",
            1: "entity",
            2: "description and abstract concept",
            3: "human",
            4: "location",
            5: "numeric",
        }
        return {
            "instruction": sample["text"],
            "input": "",
            "output": label_mapping.get(sample["coarse_label"], ""),
        }

    return sample


def preprocess_dataset(dataset_name, split="train", n_items=1000, qa_insertion_prob=0.5):
    # Load the dataset
    dataset = load_dataset(dataset_name, "CC-MAIN-2024-10", split=split, streaming=True)

    def extract_qa_pair(text):
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) < 2:
            return None, None  # Not enough sentences to extract a pair
        start_idx = random.randint(0, len(sentences) - 2)
        return sentences[start_idx], sentences[start_idx + 1]

    def process_example(example):
        if random.random() < qa_insertion_prob:
            question, answer = extract_qa_pair(example['text'])
            if question and answer:
                example['question'] = question
                example['answer'] = answer
            else:
                example['question'] = ""
                example['answer'] = ""
        else:
            example['question'] = ""
            example['answer'] = ""
        return example

    processed_dataset = (
        dataset
        .take(n_items)
        .map(process_example)
    )

    return processed_dataset


def main(args: argparse.Namespace) -> None:
    if args.dataset:
        to_load = [(args.dataset, f"just_{args.dataset}")]
    else:
        dataset_tuples = [
            (dataset, type_)
            for type_, datasets_list in datasets.items()
            for dataset in datasets_list
        ]
        to_load = (
            dataset_tuples
            if args.type == "all"
            else [
                (dataset, type_)
                for dataset, type_ in dataset_tuples
                if type_ == args.type
            ]
        )

    final_cols = ["instruction", "input", "output"]
    all_rows = defaultdict(list)

    for dataset, dataset_type in to_load:
        print(f"Downloading {dataset}...")
        if dataset == "openai/gsm8k":
            dataset_dict = load_dataset("openai/gsm8k", "main")
        elif dataset == "allenai/ai2_arc":
            dataset_dict = load_dataset("allenai/ai2_arc", "ARC-Easy")
        elif dataset == "HuggingFaceFW/fineweb-edu":
            dataset_dict = preprocess_dataset("HuggingFaceFW/fineweb-edu",
                                              split="train",
                                              n_items=100000,
                                              qa_insertion_prob=1)
            dataset_dict = {"train": list(dataset_dict)}
        elif dataset == "cais/mmlu":
            dataset_dict = load_dataset("cais/mmlu", "all")
            if args.mmlu_data_source == "auxiliary":
                dataset_dict["train"] = dataset_dict["auxiliary_train"]
                del dataset_dict["auxiliary_train"]
            else:
                dataset_dict["train"] = concatenate_datasets(
                    dataset_dict["validation"][:1000],
                    dataset_dict["dev"]
                )
                del dataset_dict["dev"]
                dataset_dict["validation"] = dataset_dict["validation"][1000:]
        elif dataset == "bigcode/starcoderdata":
            from tqdm import tqdm
            from itertools import islice

            dataset_stream = load_dataset(
                "bigcode/starcoderdata",
                split="train",
                streaming=True,
            )

            # hack to advance to a later offset
            CHUNK_SIZE = 5_000_000
            CHUNK_END = 25
            OFFSET = 10000
            SEEK_FROM = (CHUNK_END - 1) * CHUNK_SIZE + OFFSET

            it = iter(dataset_stream)
            pbar = tqdm(total=SEEK_FROM)
            for _ in range(SEEK_FROM):
                try:
                    next(it)
                except StopIteration:
                    break
                pbar.update(1)
            pbar.close()

            TARGET_SIZE = 6000
            FETCH_SIZE = 100_000
            py_examples = []
            while len(py_examples) < TARGET_SIZE:
                batch = list(islice(it, FETCH_SIZE))
                if not batch:
                    break
                py_examples.extend(
                    b for b in batch if b["max_stars_repo_path"].endswith(".py")
                )

            train_data = py_examples[:5000]
            val_data = py_examples[5000:6000]
            dataset_dict = {
                "train": Dataset.from_list(train_data),
                "val": Dataset.from_list(val_data),
            }
        elif dataset == "allenai/c4":
            dataset_stream = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
            )
            dataset_stream = dataset_stream.shuffle(seed=42, buffer_size=10000)
            train_data = list(dataset_stream.take(5000))
            val_data = list(dataset_stream.skip(5000).take(1000))
            dataset_dict = {
                "train": Dataset.from_list(train_data),
                "val": Dataset.from_list(val_data),
            }
        elif dataset == "nyu-mll/glue":
            dataset_dict = load_dataset("nyu-mll/glue", "mnli")
        else:
            dataset_dict = load_dataset(dataset, trust_remote_code=True)

    
        # Auto split the train set to get 80/10/10 ratio
        # First split: 80% train, 20% temp (which will be split into 10% val + 10% test)
        train_temp_split = dataset_dict["test"].train_test_split(test_size=0.2, seed=42)
        # Second split: split the 20% temp into 10% val and 10% test
        val_test_split = train_temp_split["test"].train_test_split(test_size=0.5, seed=42)
        
        dataset_dict = {
            "train": train_temp_split["train"],  # 80% of original
            "val": val_test_split["train"],      # 10% of original  
            "test": val_test_split["test"]       # 10% of original
        }
        print(f"Auto-split {dataset}: train={len(dataset_dict['train'])}, val={len(dataset_dict['val'])}, test={len(dataset_dict['test'])}")

        # Add debug for MR dataset
        if dataset == "mattymchen/mr":
            print(f"DEBUG: Auto-split created splits: {list(dataset_dict.keys())}")
            for split_name, split_data in dataset_dict.items():
                labels = [item["label"] for item in split_data]
                from collections import Counter
                print(f"Split '{split_name}' label distribution: {Counter(labels)}")
                print(f"Split '{split_name}' size: {len(split_data)}")

        # Write the data for each split
        for split, data in dataset_dict.items():
            split_name = split if split != "validation" else "val"
            data_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/{dataset}/{split_name}.json"
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            formatted_dataset = data.map(
                partial(format_row, dataset=dataset,
                        mmlu_with_mcq=(args.mmlu_format == "with_mcq"))
            )
            
            # drop extra columns
            filtered_dataset = [
                dict(x)
                for x in formatted_dataset.remove_columns(
                    [
                        col
                        for col in formatted_dataset.column_names
                        if col not in final_cols and col != "label"
                    ]
                )
            ]

            # Special handling for MR dataset - don't filter by label since 0/1 are actual classes
            if dataset == "mattymchen/mr":
                # Keep the filtered_dataset as is, just remove the label column if present
                for item in filtered_dataset:
                    item.pop("label", None)
            else:
                # filter by label if present (for other datasets where label indicates validity)
                if filtered_dataset and "label" in filtered_dataset[0]:
                    filtered_dataset = [item for item in filtered_dataset if item.get("label", True)]
                    for item in filtered_dataset:
                        item.pop("label", None)

            # special handling for some large datasets
            if dataset == "nvidia/OpenMathInstruct-1" and args.concat_all:
                filtered_dataset = filtered_dataset[:50000]
            elif dataset == "Nan-Do/code-search-net-python":
                filtered_dataset = filtered_dataset[:10000]
            elif dataset == "bigcode/starcoderdata":
                train_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/{dataset}/train.json"
                val_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/{dataset}/val.json"
                os.makedirs(os.path.dirname(train_path), exist_ok=True)
                os.makedirs(os.path.dirname(val_path), exist_ok=True)
                if split == "train":
                    with open(train_path, "w") as f:
                        json.dump(filtered_dataset, f)
                else:
                    with open(val_path, "w") as f:
                        json.dump(filtered_dataset, f)
                continue
            elif dataset == "allenai/c4":
                train_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/{dataset}/train.json"
                val_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/{dataset}/val.json"
                os.makedirs(os.path.dirname(train_path), exist_ok=True)
                os.makedirs(os.path.dirname(val_path), exist_ok=True)
                if split == "train":
                    with open(train_path, "w") as f:
                        json.dump(filtered_dataset, f)
                else:
                    with open(val_path, "w") as f:
                        json.dump(filtered_dataset, f)
                continue

            print(f"Dataset {dataset}, {split} prepared with {len(filtered_dataset)} rows, written to {data_path}")

            if args.concat_all:
                all_rows[split_name].extend(filtered_dataset)
            else:
                Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)
                with open(data_path, "w") as f:
                    json.dump(filtered_dataset, f)

    if args.concat_all:
        for split, rows in all_rows.items():
            data_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/concat/{split}.json"
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            print(f"Total rows for {split}: {len(rows)}")
            with open(data_path, "w") as f:
                json.dump(rows, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download the instruction tuning/SFT datasets for this project"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["all", "sft", "instruction", "pretraining", "flan", "sft_reasoning", "blank", "mmlu"],
        default="all",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="just download a single dataset. Note: will save the dataset in the type subdir specified",
    )
    parser.add_argument(
        "--concat_all",
        action="store_true",
        help="Concatenate all datasets into one (for SFT/smaller datasets)",
   
    )
    parser.add_argument(
        "--auto_split",  # Change from --auto-split to --auto_split
        action="store_true",
        help="Auto split the dataset into train/val/test",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of the dataset to use for validation (if --auto-split is set)",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.05,
        help="Fraction of the dataset to use for test (if --auto-split is set)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset",
    )
    parser.add_argument(
        "--mmlu_format",
        type=str,
        choices=["with_mcq", "without_mcq"],
        default="without_mcq",
        help="Format for MMLU dataset",
    )
    parser.add_argument(
        "--mmlu_data_source",
        type=str,
        choices=["auxiliary", "dev_val"],
        default="dev_val",
        help="Data source for MMLU dataset",
    )

    args = parser.parse_args()
    main(args)