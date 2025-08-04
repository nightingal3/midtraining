# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import argparse
import os
import pathlib

import dotenv

from huggingface_hub import HfApi, Repository

from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv(dotenv_path="./configs/.env")
MANIFOLD_DIR = os.environ["MANIFOLD_DIR"]
# MODELS_PATH = f"{MANIFOLD_DIR}/all_in_one_pretraining/models"
# TODO: note: I don't think we can use git pull on manifold, save locally
MODELS_PATH = "./models"
STEP_INTERVAL = 10000
EARLIEST_STEP = 50000


def download_model_branches(model_name: str) -> None:
    api = HfApi()
    branches = api.list_repo_refs(model_name).branches
    for branch in branches:
        branch_name = branch.name
        if "step" in branch_name or "main" in branch_name:
            # save the checkpoint
            if "step" in branch_name:
                step_num = int(branch_name.split("step")[-1])
                if step_num % STEP_INTERVAL != 0 or step_num < EARLIEST_STEP:
                    continue

            step_path = os.path.join(MODELS_PATH, model_name, branch_name)
            print(f"Downloading {model_name} branch: {branch_name}")
            if not os.path.exists(step_path) or not any(
                pathlib.Path(step_path).iterdir()
            ):
                pathlib.Path(step_path).mkdir(parents=True, exist_ok=True)
                repo = Repository(step_path, clone_from=model_name)
                repo.git_pull()


def main():
    to_download = [
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
    ]

    for model in to_download:
        pathlib.Path(f"{MODELS_PATH}/{model}").mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model}...")
        download_model_branches(model)


if __name__ == "__main__":
    main()
