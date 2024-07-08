# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import argparse
import json
import os
from collections import defaultdict
from functools import partial

import dotenv
from datasets import load_dataset


dotenv.load_dotenv(dotenv_path="./configs/.env")
MANIFOLD_DIR = os.environ["MANIFOLD_DIR"]

datasets = {
    # TODO: natural instructions already downloaded, unzip and upload to manifold manually
    "pretraining": ["HuggingFaceFW/fineweb-edu"],
    "instruction": ["databricks/databricks-dolly-15k", "GAIR/lima", "tatsu-lab/alpaca"],
    "sft": [
        "openai/gsm8k",
        "EleutherAI/asdiv",
        "allenai/ai2_arc",
        "allenai/sciq",
        "Rowan/hellaswag",
        "microsoft/wiki_qa",
        "ybisk/piqa",
    ],
}


def mcq_to_text(answers: list, labels: list, correct_label: str):
    labels_to_answers = {lab: ans for ans, lab in zip(answers, labels)}
    return labels_to_answers[correct_label]


# some datasets have different formats, store them in a common way with instruction/input(context)/output
def format_row(sample: dict, dataset: str) -> dict:
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
            "input": "",
            "output": sample["correct_answer"],
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

    return sample


def main(args: argparse.Namespace) -> None:
    dataset_tuples = [
        (dataset, type_) for type_, datasets in datasets.items() for dataset in datasets
    ]
    to_load = (
        dataset_tuples
        if args.type == "all"
        else [
            (dataset, type_) for dataset, type_ in dataset_tuples if type_ == args.type
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
        else:
            dataset_dict = load_dataset(dataset, trust_remote_code=True)

        for split in dataset_dict.keys():
            data_path = f"{MANIFOLD_DIR}/all_in_one_pretraining/datasets/{dataset_type}/{dataset}/{split}.json"
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
                        if col not in final_cols
                    ]
                )
                if len(dict(x)) > 0
            ]

            if dataset == "microsoft/wiki_qa":
                filtered_dataset = [
                    dict(x)
                    for x in formatted_dataset.remove_columns(
                        [
                            col
                            for col in formatted_dataset.column_names
                            if col not in final_cols and col != "label"
                        ]
                    )
                    if len(dict(x)) > 0
                ]
                filtered_dataset = [
                    entry for entry in filtered_dataset if entry["label"] == 1
                ]
                # remove the label now
                for item in filtered_dataset:
                    del item["label"]
            print(
                f"Dataset {dataset}, {split} prepared with {len(filtered_dataset)} rows"
            )

            if 1 in filtered_dataset:
                breakpoint()

            if args.concat_all:
                all_rows[split].extend(filtered_dataset)
            else:
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
        choices=["all", "sft", "instruction", "pretraining"],
        default="all",
    )
    parser.add_argument(
        "--concat_all",
        action="store_true",
        help="Concatenate all datasets into one (for SFT/smaller datasets)",
    )

    args = parser.parse_args()
    main(args)
