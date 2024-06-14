# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pathlib
from huggingface_hub import HfApi, Repository

from manifold.clients.python import ManifoldClient

MODELS_PATH = "./models/"
MANIFOLD_PATH = "all_in_one_pretraining/models/"

def dir_to_manifold(dir_path: str, subdir_in_manifold: str) -> None:
    client = ManifoldClient("coin/tree")
        
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            try:
                client.sync_put(f"coin/tree/{MANIFOLD_PATH}", )
            except Exception as e:
                print(e)

def download_model_branches(model_name: str) -> None:
    api = HfApi()
    client = ManifoldClient("coin/tree")
    branches = api.list_repo_refs(model_name).branches
    for branch in branches:
        branch_name = branch.name
        if "step" in branch_name or "main" in branch_name:
            # save the checkpoint
            step_path = os.path.join(MODELS_PATH, model_name, branch_name)
            print(f"Downloading {model_name} branch: {branch_name}")
            if not os.path.exists(step_path):
                repo = Repository(step_path, clone_from=model_name)
                repo.git_pull()

            # add to manifold if not present there
            dir_to_manifold(step_path, f"{model_name}/{branch_name}")


def main():
    to_download = ["EleutherAI/pythia-1b", "EleutherAI/pythia-2.8b"]

    for model in to_download:
        if not os.path.exists(f"{MODELS_PATH}/{model}"):
            pathlib.Path(f"{MODELS_PATH}/{model}").mkdir(parents=True, exist_ok=False)
            print(f"Downloading {model}...")
            download_model_branches(model)


if __name__ == "__main__":
    main()
