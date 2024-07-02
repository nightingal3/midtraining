# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import argparse
import json
import os
from functools import partial

import dotenv
from datasets import load_dataset


dotenv.load_dotenv(dotenv_path="./configs/.env")
MANIFOLD_DIR = os.environ["MANIFOLD_DIR"]

datasets = {
    # TODO: natural instructions already downloaded, unzip and upload to manifold manually
    "pretraining": ["HuggingFaceFW/fineweb-edu"],
    "instruction": ["databricks/databricks-dolly-15k", "GAIR/lima", "tatsu-lab/alpaca"],
    "sft": ["openai/gsm8k", "EleutherAI/asdiv", "allenai/ai2_arc", "allenai/sciq"],
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
            ]

            with open(data_path, "w") as f:
                json.dump(filtered_dataset, f)


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

    args = parser.parse_args()
    main(args)
