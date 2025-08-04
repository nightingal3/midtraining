from tasksource import list_tasks, load_task
import os
import pathlib
import json
import shutil

BASE_PATH = "/data/users/nightingal3/manifold/all_in_one_pretraining/datasets/tasksource_w_choices"
excludes = [
    "mmlu",
    "glue",
    "super_glue",
    "bigbench",
    "math_qa",
    "commonsense_qa",
    "logiqa",
    "logiqa-2.0-nli",
    "PARADISE" # this dataset seems to be malformatted?,
    "balanced-copa", # needs extra context, doesn't make sense
    "e-CARE,
    "implicatures",
    "model_written_evals",
    "osst2_pairwise_rlhf_reward",
    "path-naturalness-prediction",
    "ReSQ",
    "wiqa"
]

def mcq_to_text(answers: list, labels: list, correct_label: str):
    labels_to_answers = {lab: ans for ans, lab in zip(answers, labels)}
    return labels_to_answers[correct_label]

def get_choice_cols_and_labels(sample):
    all_choices_and_labels = []
    for key in sample:
        if "choice" in key:
            int_val = int(key.split("choice")[-1])
            all_choices_and_labels.append((sample[key], int_val))

    return all_choices_and_labels

def to_inst_format(sample):
    # convert to litgpt's sft format
    choice_vals_and_labels = get_choice_cols_and_labels(sample)
    choice_vals = [x[0] for x in choice_vals_and_labels]
    choice_labels = [x[1] for x in choice_vals_and_labels]
    return {
            "instruction": sample["inputs"],
            "input": "",
            "output": mcq_to_text(choice_vals, choice_labels, sample["labels"]),
            "choices": choice_vals
        }

final_cols = ["instruction", "input", "output", "choices"]

exclude_strings = ["mmlu", "bigbench"]

df = list_tasks()

for i, id in enumerate(df[df.task_type=="Classification"].id):
    if any(x in id for x in excludes):
        if os.path.exists(os.path.join(BASE_PATH, id)):
            print("Deleting ", os.path.join(BASE_PATH, id))
            shutil.rmtree(os.path.join(BASE_PATH, id))
        continue
    
    print(f"Task {i}: {id}")
    try:
        dataset = load_task(id, trust_remote_code=True)
    except:
        print(f"Skipping {id} as it failed to load")
        continue

    for split in dataset.keys():
        breakpoint()
        data_path = os.path.join(BASE_PATH, id, split + ".json")
        if os.path.exists(data_path):
            print(f"Skipping {data_path} as it already exists")
            continue

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        split = dataset[split]
        split = split.map(to_inst_format)
        filtered_split = [
                dict(x)
                for x in split.remove_columns(
                    [
                        col
                        for col in split.column_names
                        if col not in final_cols and col != "label"
                    ]
                )
            ]
        with open(data_path, "w") as f:
            json.dump(filtered_split, f)
