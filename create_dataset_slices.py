import argparse
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split


def create_initial_slices(tulu_data_path: str, combined_sft_data_path: str) -> None:
    num_samples = [1000, 2500, 5000, 10000, 25000, 50000]

    with open(tulu_data_path, "r") as f:
        tulu_data = json.load(f)
    with open(combined_sft_data_path, "r") as f:
        combined_sft_data = json.load(f)

    random.seed(1234)
    # use 5% of tulu to form train/val sets
    tulu_train, tulu_val = train_test_split(tulu_data, test_size=0.1, random_state=1234)
    tulu_val, tulu_test = train_test_split(tulu_val, test_size=0.5, random_state=1234)
    with open(
        f"../manifold/all_in_one_pretraining/datasets/tulu/train.json",
        "w",
    ) as f:
        json.dump(tulu_train, f)
    with open(
        f"../manifold/all_in_one_pretraining/datasets/tulu/val.json",
        "w",
    ) as f:
        json.dump(tulu_val, f)
    with open(
        f"../manifold/all_in_one_pretraining/datasets/tulu/test.json",
        "w",
    ) as f:
        json.dump(tulu_test, f)

    for num in num_samples:
        print(num)
        tulu_slice = random.sample(tulu_train, num)
        combined_sft_slice = random.sample(combined_sft_data, num)

        # mkdir -p
        Path(f"../manifold/all_in_one_pretraining/datasets/tulu/{num}").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"../manifold/all_in_one_pretraining/datasets/sft/concat/{num}").mkdir(
            parents=True, exist_ok=True
        )

        with open(
            f"../manifold/all_in_one_pretraining/datasets/tulu/{num}/train.json",
            "w",
        ) as f:
            json.dump(tulu_slice, f)
        with open(
            f"../manifold/all_in_one_pretraining/datasets/sft/concat/{num}/train.json",
            "w",
        ) as f:
            json.dump(combined_sft_slice, f)

        with open(
            f"../manifold/all_in_one_pretraining/datasets/tulu/{num}/val.json",
            "w",
        ) as f:
            json.dump(tulu_val, f)
        with open(
            f"../manifold/all_in_one_pretraining/datasets/tulu/{num}/test.json",
            "w",
        ) as f:
            json.dump(tulu_test, f)


def create_slice_from_val(val_dataset_path: str, num_samples=1000) -> None:
    # Create an unseen training set for final SFT stage from val set
    with open(val_dataset_path, "r") as f:
        val = json.load(f)
    with open(
        "../manifold/all_in_one_pretraining/datasets/sft/concat/test.json", "r"
    ) as f:
        orig_test = json.load(f)

    random.seed(1234)
    sft_train, sft_val = train_test_split(
        val, train_size=num_samples, random_state=1234
    )

    Path(f"../manifold/all_in_one_pretraining/datasets/sft_final/{num_samples}").mkdir(
        parents=True, exist_ok=True
    )
    with open(
        f"../manifold/all_in_one_pretraining/datasets/sft_final/{num_samples}/train.json",
        "w",
    ) as f:
        json.dump(sft_train, f)
    with open(
        f"../manifold/all_in_one_pretraining/datasets/sft_final/{num_samples}/val.json",
        "w",
    ) as f:
        json.dump(sft_val, f)
    with open(
        f"../manifold/all_in_one_pretraining/datasets/sft_final/{num_samples}/test.json",
        "w",
    ) as f:
        json.dump(orig_test, f)

    print(
        f"Final sft sets created at ../manifold/all_in_one_pretraining/datasets/sft_final/{num_samples}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["initial", "final_sft"], default="initial")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    tulu_data_path = "../manifold/all_in_one_pretraining/datasets/tulu/allenai/tulu-v2-sft-mixture/train.json"
    combined_sft_data_path = (
        "../manifold/all_in_one_pretraining/datasets/sft/concat/train.json"
    )
    if args.mode == "initial":
        create_initial_slices(tulu_data_path, combined_sft_data_path)
    else:
        create_slice_from_val(
            "../manifold/all_in_one_pretraining/datasets/sft/concat/val.json",
            args.num_samples,
        )
