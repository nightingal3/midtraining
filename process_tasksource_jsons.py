import json
import os
import pathlib
import argparse
import string
import random
# the original tasksource format needs some postproc. Because the tasks are different I need to explain the task a bit.

default_qa_template = {
    "instruction": "{instruction}. {generated_choices_str}\nAnswer:",
    "output": "{output}"
}
paired_preference_template = {
        "instruction": "User: {instruction}.\n Which is the best response? {generated_choices_str}\nAnswer:",
        "output": "{output}"
}

question_wrappers = {
    "arct": {
        "instruction": "{instruction}. This implies {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "balanced-copa": {
        "instruction": "{instruction}. This is most likely because {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "cicero": {
        "instruction": "Dialogue: {instruction}. Which of the options is true? {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "cloth": {
        "instruction": "{instruction}. [MASK] is most likely {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "cosmos_qa": default_qa_template,
    "cycic_multiplechoice": default_qa_template,
    "definite_pronoun_resolution": default_qa_template,
    "dgen": {
        "instruction": "{instruction}. **blank** is most likely {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "dream": default_qa_template,
    "ekar_english": {
        "instruction": "Find the equivalent analogy. {instruction}. {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "fig-qa": {
        "instruction": "{instruction}. That is to say: {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "hellaswag": default_qa_template,
    "lsat-ar": default_qa_template,
    "lsat-rc": default_qa_template,
    "medmcqa": default_qa_template,
    "MedQA-USMLE-4-options-hf": default_qa_template,
    "model-written-evals": default_qa_template,
    "mutual": {
        "instruction": "Dialogue: {instruction}. How does the dialogue likely continue? {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
    "NeQA": default_qa_template,
    "openbookqa": default_qa_template,
    "piqa": default_qa_template,
    "prost": default_qa_template,
    "qasc": default_qa_template,
    "quail": default_qa_template,
    "quartz": default_qa_template,
    "quote-repetition": default_qa_template,
    "race-c": default_qa_template,
    "reclor": default_qa_template,
    "redefine-math": default_qa_template,
    "ReSQ": default_qa_template,
    "ScienceQA_text_only": default_qa_template,
    "sciq": default_qa_template,
    "SHP": paired_preference_template,
    "spartqa-mchoice": default_qa_template,
    "SpaRTUN": default_qa_template,
    "synthetic-instruct-gptj-pairwise": paired_preference_template,
    "UltraFeedback-paired": paired_preference_template,
    "webgpt_comparisons": paired_preference_template,
    "winodict": {
        "instruction": "{instruction}. Which word should go in the blank? {generated_choices_str}\nAnswer:",
        "output": "{output}"
    },
}


def make_mcq_string(choices, output):
    random.shuffle(choices)
    alpha_order = string.ascii_uppercase
    choices_str = "\n".join([f"{alpha_order[i]}. {choice}" for i, choice in enumerate(choices)])
    return choices_str


def process_directory(directory, out_path, concat_all=False):
    print(f"Processing directory: {directory}")
    task_name = directory.split("/")[-1]
    print("Task name: ", task_name)
    if task_name not in question_wrappers:
        return

    new_samples_overall = {}
    for filename in os.listdir(directory):
        out_path_dir = os.path.join(out_path, task_name)
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Process the file here
            print(f"  Processing file: {filename}")
            with open(file_path, "r") as f:
                data = json.load(f)
            
            new_samples = []
            for sample in data:
                sample_instruction = sample["instruction"]
                sample_output = sample["output"]
                sample_choices = sample["choices"]
                sample_input = sample["input"]

                mcq_string = make_mcq_string(sample_choices, sample_output)


                new_sample = {
                    "instruction": question_wrappers[task_name]["instruction"].format(instruction=sample_instruction, generated_choices_str=mcq_string),
                    "output": question_wrappers[task_name]["output"].format(output=sample_output),
                    "input": sample_input
                }
                new_samples.append(new_sample)
            
            
            pathlib.Path(out_path_dir).mkdir(parents=True, exist_ok=True)

            if not concat_all:
                with open(os.path.join(out_path_dir, filename), "w") as f:
                    json.dump(new_samples, f)
                    print(f"Wrote {len(new_samples)} samples to {os.path.join(out_path_dir, filename)}")

            else:
                new_samples_overall[filename] = new_samples
        
    if concat_all:
        return new_samples_overall
                

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/data/users/nightingal3/manifold/all_in_one_pretraining/datasets/tasksource_w_choices")
    parser.add_argument("--output_dir", type=str, default="/data/users/nightingal3/manifold/all_in_one_pretraining/datasets/tasksource_processed")
    parser.add_argument("--concat_all", action="store_true", help="Concatenate all files into a single file rather than dividing across datasets")
    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.root_dir):
        # Process each subdirectory
        new_dict = process_directory(dirpath, args.output_dir, concat_all=args.concat_all)
        if args.concat_all:
            if new_dict is not None:
                pathlib.Path(os.path.join(args.output_dir, "concat")).mkdir(parents=True, exist_ok=True)
                for filename, new_samples in new_dict.items():
                    if os.path.exists(os.path.join(args.output_dir, "concat", filename)):
                        with open(os.path.join(args.output_dir, "concat", filename), "r") as f:
                            prev_data = json.load(f)
                    else:
                        prev_data = []

                
                    prev_data.extend(new_samples)
                    with open(os.path.join(args.output_dir, "concat",filename), "w") as f:
                        json.dump(prev_data, f)
                        print(f"Wrote {len(prev_data)} samples to {os.path.join(args.output_dir, filename)}")

                        
