# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import argparse
import json
import os
from collections import defaultdict
from functools import partial
import string
import dotenv
from datasets import load_dataset, concatenate_datasets
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
    "pretraining": ["HuggingFaceFW/fineweb-edu"],
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
        return sample  # TODO
    elif dataset == "allenai/sciq":
        return {
            "instruction": sample["question"],
            "input": sample["support"],
            "output": sample["correct_answer"],
            "choices": random.shuffle([sample["distractor1"], sample["distractor2"], sample["distractor3"], sample["correct_answer"]])
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
                    [sample["sol1"], sample["sol2"]], [0, 1], sample["label"]
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
        # NOTE: TODO: since this dataset only has multi-turn convos, just take the first interaction. This should make sense by itself as well in most cases
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

    return sample

def preprocess_dataset(dataset_name, split="train", n_items=1000, qa_insertion_prob=0.5):
    # Load the dataset
    dataset = load_dataset(dataset_name, "CC-MAIN-2024-10", split=split, streaming=True)
    
    def extract_qa_pair(text):
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        if len(sentences) < 2:
            return None, None  # Not enough sentences to extract a pair
        
        # Randomly select two consecutive sentences
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
            for type_, datasets in datasets.items()
            for dataset in datasets
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
            # load a smaller subset
            dataset_dict = preprocess_dataset("HuggingFaceFW/fineweb-edu", split="train", n_items=100000, qa_insertion_prob=1)
            dataset_dict = {"train": list(dataset_dict)}
        elif dataset == "cais/mmlu":
            dataset_dict = load_dataset("cais/mmlu", "all")
            if args.mmlu_data_source == "auxiliary":
                dataset_dict["train"] = dataset_dict["auxiliary_train"]
                del dataset_dict["auxiliary_train"]
            else:
                dataset_dict["train"] = concatenate_datasets(dataset_dict["validation"][:1000], dataset_dict["dev"])
                del dataset_dict["dev"]
                dataset_dict["validation"] = dataset_dict["validation"][1000:]

        else:
            dataset_dict = load_dataset(dataset, trust_remote_code=True)

        for split in dataset_dict.keys():
            split_name = split if split != "validation" else "val"

            data_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/{dataset}/{split_name}.json"
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            formatted_dataset = dataset_dict[split].map(
                partial(format_row, dataset=dataset)
            )

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

            if "label" in filtered_dataset[0].keys():
                filtered_dataset = [
                    entry for entry in filtered_dataset if entry["label"] == True
                ]
                # remove the label now
                for item in filtered_dataset:
                    del item["label"]

            if dataset == "nvidia/OpenMathInstruct-1" and args.concat_all: 
                # cap this to 20k rows to not overwhelm the other datasets
                filtered_dataset = filtered_dataset[:50000]

            print(
                f"Dataset {dataset}, {split} prepared with {len(filtered_dataset)} rows"
            )

            if args.concat_all:
                all_rows[split_name].extend(filtered_dataset)
            else:
                # mkdir -p 
                Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)
                with open(data_path, "w") as f:
                    json.dump(filtered_dataset, f)

    if args.concat_all:
        for split in all_rows.keys():
            data_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/concat/{split}.json"
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            print(f"Total rows for {split}: ", len(all_rows[split]))
            with open(data_path, "w") as f:
                json.dump(all_rows[split], f)


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
        "--mmlu_data_source",
        type=str,
        choices=["auxiliary", "main"],
        help="use the mmlu auxiliary data or slice part of the dev data",
        default="auxiliary",
    )
    parser.add_argument(
        "--mmlu_format",
        type=str,
        choices=["with_mcq", "plain"],
        help="include the multiple choices in the question or not",
        default="with_mcq",
    )

    args = parser.parse_args()

    main(args)
