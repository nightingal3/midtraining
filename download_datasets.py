# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import argparse
import os
from datasets import load_dataset
from functools import partial
import json

datasets = {
    # TODO: natural instructions already downloaded, unzip and upload to manifold manually
    "instruction": ["databricks/databricks-dolly-15k", "GAIR/lima", "tatsu-lab/alpaca"],
    "sft": [],
}


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
    elif dataset == "GAIR/lima":
        return sample  # TODO

    return sample


def main(args: argparse.Namespace) -> None:
    to_load = (
        datasets[args.type]
        if args.type != "all"
        else [item for sublist in datasets.values() if sublist for item in sublist]
    )

    final_cols = ["instruction", "input", "output"]
    for dataset in to_load:
        print(f"Downloading {dataset}...")
        dataset_dict = load_dataset(dataset)
        for split in dataset_dict.keys():
            data_path = f"./datasets/{dataset}/{split}.json"
            formatted_dataset = dataset_dict[split].map(partial(format_row, dataset=dataset))
            filtered_dataset = [dict(x) for x in formatted_dataset.remove_columns([col for col in formatted_dataset.column_names if col not in final_cols])]
            
            with open(data_path, "w") as f:
                json.dump(filtered_dataset, f)

        # TODO: upload to manifold if not exists there


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download the instruction tuning/SFT datasets for this project"
    )
    parser.add_argument(
        "--type", type=str, choices=["all", "sft", "instruction"], default="all"
    )

    args = parser.parse_args()
    main(args)
