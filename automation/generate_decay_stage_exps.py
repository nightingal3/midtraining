import os
from ast import literal_eval

import pandas as pd


train_types = ["train_mixed", "train_seq"]
num_additional_steps = [200, 4500, 10000]
pretraining_data_dirs = ["/mnt/mffuse/scaling_mates/data/fineweb/sample-100BT/train", "/mnt/mffuse/scaling_mates/data/mates-fineweb-500000"]

exps = []

for train_type in train_types:
    if train_type == "train_mixed":
        _command = "torchx run mast.py:{train_type} --nnodes 1 --name {exp_name} -- --max_additional_steps {max_additional_steps} --checkpoint_dir /mnt/mffuse/scaling_mates/out/pythia-1b/fineweb/sample-100BT --decay_lr --step '{step}' --out_dir {out_dir} --sft_template {sft_template} --pretraining_data_dir {pretraining_data_dir} --data_ratios {data_ratios} --instruction_data_paths 'stackexchange /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/code/train.jsonl 0.33,openhermes /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/openhermes/train.jsonl 0.33,trivia /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/triviaqa/train.jsonl 0.33,ultrachat /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/ultrachat/train.jsonl 0.33,dialogue /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/dialogue/train.jsonl 0.33'"
        command = "torchx run mast.py:{train_type} --nnodes 1 --name {exp_name} -- --max_additional_steps {max_additional_steps} --checkpoint_dir /mnt/mffuse/scaling_mates/tc_out/pythia-6.9b_fineweb/ --decay_lr --step '{step}' --out_dir {out_dir} --sft_template {sft_template} --pretraining_data_dir {pretraining_data_dir} --data_ratios {data_ratios} --instruction_data_paths 'stackexchange /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/code/train.jsonl 0.33,openhermes /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/openhermes/train.jsonl 0.33,trivia /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/triviaqa/train.jsonl 0.33,ultrachat /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/ultrachat/train.jsonl 0.33,dialogue /mnt/mffuse/all_in_one_pretraining/datasets/knowledgeqa_formatted/dialogue/train.jsonl 0.33' --model_name pythia-6.9b"

        #data_ratios = ["'[0.75, 0.25]'", "'[0.5, 0.5]'", "'[0.9, 0.1]'", "'[0.25, 0.75]'", "'[1, 0]'"]
        #data_ratios = ["'[0.25, 0.75]'", "'[1, 0]'"]
        data_ratios = ["'[0, 0.41, 0.15, 0.04, 0.23, 0.15]'", 
    "'[0.5, 0.205, 0.075, 0.02, 0.115, 0.075]'"]
        sft_templates = ["default"]

        for d_ratio in data_ratios:
            for template in sft_templates:
                for n_steps in num_additional_steps:
                    for pretraining_dir in pretraining_data_dirs:
                        step_num = 100000
                        step_arg = f"-00{step_num}"
                        data_ratio_lst = literal_eval(d_ratio[1:-1])

                        #data_ratio_str = f"{data_ratio_lst[0]}-{data_ratio_lst[1]}".replace(".", "-")
                        data_ratio_str = "0_pre" if data_ratio_lst[0] == 0 else "0-5_pre"
                        pretraining_data_name = "mates" if "mates-fineweb-500000" in pretraining_dir else "fw"
                        data_name = f"{data_ratio_str}_{pretraining_data_name}"
                        exp_name = f"decay_7b_{n_steps}step_{data_name}_sft_input_{template}_cpm"
                        out_dir = f"/mnt/mffuse/all_in_one_pretraining/out/pythia-7b/decay_from_{step_num}_with_sft_and_input_{data_name}_{template}_for_{n_steps}steps_cpm"
                        curr_cmd = command.format(train_type=train_type, exp_name=exp_name, max_additional_steps=n_steps, step=step_arg, out_dir=out_dir, pretraining_data_dir=pretraining_dir, data_ratios=d_ratio, sft_template=template)
                        
                        exps.append(
                            {
                                "exp_name": exp_name,
                                "steps": n_steps,
                                "template": template,
                                "d_ratio": data_ratio_str,
                                "pretraining_data": pretraining_dir,
                                "torchx_cmd": curr_cmd
                            }
                        )
    else:
        _command = "torchx run mast.py:{train_type} --nnodes 1 --name {exp_name} -- --max_additional_steps {max_additional_steps} --checkpoint_dir /mnt/mffuse/scaling_mates/out/pythia-1b/fineweb/sample-100BT --decay_lr --step '{step}' --out_dir {out_dir} --pretraining_data_dir {pretraining_data_dir} "
        command = "torchx run mast.py:{train_type} --nnodes 1 --name {exp_name} -- --max_additional_steps {max_additional_steps} --checkpoint_dir /mnt/mffuse/scaling_mates/tc_out/pythia-6.9b_fineweb/ --decay_lr --step '{step}' --out_dir {out_dir} --pretraining_data_dir {pretraining_data_dir} --model_name pythia-6.9b"

        for n_steps in num_additional_steps:
            for pretraining_dir in pretraining_data_dirs:
                step_num = 100000
                step_arg = f"-00{step_num}"
                data_name = "mates" if "mates-fineweb-500000" in pretraining_dir else "fw"
                exp_name = f"decay_train_7b_{n_steps}step_{data_name}_seq_cpm"
                out_dir = f"/mnt/mffuse/all_in_one_pretraining/out/pythia-7b/decay_from_{step_num}_{data_name}_for_{n_steps}steps"
                curr_cmd = command.format(train_type=train_type, exp_name=exp_name, max_additional_steps=n_steps, step=step_arg, out_dir=out_dir, pretraining_data_dir=pretraining_dir)
                exps.append(
                    {
                        "exp_name": exp_name,
                        "steps": n_steps,
                        "d_ratio": "1_0",
                        "pretraining_data": pretraining_dir,
                        "torchx_cmd": curr_cmd
                    }
                )

exp_df = pd.DataFrame(exps)
exp_df.to_csv("./automation/decay_stage_exps_pythia.csv", index=False)
