import os

import pandas as pd

num_steps = [1000, 10000]

torchx_cmd = "torchx run mast.py:sft --name {exp_name} -- --pretrained_checkpoint_dir {pretrained_checkpoint_dir} --instruction_data_json {instruction_data_json} --run_id {exp_name}"
exps = []
for n_s in num_steps:
    if n_s == 1000:
        paths = [
            "/mnt/mffuse/out/pythia-1b_seq_train_d_10000-ljcp0f9r/final/",
            "/mnt/mffuse/all_in_one_pretraining/out/pythia-1b_mixtrained_from_140000_id1721024117/final/",
            "/mnt/mffuse/all_in_one_pretraining/out/pythia-1b_mixtrained_from_140000_id1721025074/final/",
            "/mnt/mffuse/all_in_one_pretraining/out/pythia-1b_mixtrained_from_140000_id1721026286/final",
        ]

    elif n_s == 10000:
        paths = [
            "/mnt/mffuse/out/pythia_1b_seq_train_d_steps_100k-ddcr595/final/",
            "/mnt/mffuse/all_in_one_pretraining/out/pythia-1b_mixtrained_from_140000_id1720935358/final/",
            "/mnt/mffuse/all_in_one_pretraining/out/pythia-1b_mixtrained_from_140000_id1720976918/final/",
            "/mnt/mffuse/all_in_one_pretraining/out/pythia-1b_mixtrained_from_140000_id1720980661/final/",
        ]
    instruction_data_json = (
        f"/mnt/mffuse/all_in_one_pretraining/datasets/sft_final/{n_s}/"
    )

    for path in paths:
        exp_name = path.split("/")[path.split("/").index("final") - 1]
        exps.append(
            {
                "exp_name": f"{exp_name}_sft_{n_s}steps",
                "steps": n_s,
                "pretrained_checkpoint_dir": path,
                "instruction_data_json": instruction_data_json,
                "torchx_cmd": torchx_cmd.format(
                    exp_name=f"{exp_name}_sft_{n_s}steps",
                    pretrained_checkpoint_dir=path,
                    instruction_data_json=instruction_data_json,
                ),
            }
        )

exp_df = pd.DataFrame(exps)
exp_df.to_csv("./automation/sft_exps.csv", index=False)
