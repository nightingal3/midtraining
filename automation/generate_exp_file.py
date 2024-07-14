import os

import pandas as pd


exp_types = ["seq", "mixed"]
num_steps = [10000, 100000, 1000000]
models = ["pythia-1b", "pythia-2.8b", "meta-llama/Meta-Llama-3-8B"]
pct_sfts = [0.1, 0.25, 0.5]
scenarios = ["a", "c", "d"]

torchx_cmd = {
    "seq": "torchx run mast.py:train_seq --name {exp_name} --nnodes 2 -- --steps_to_train {steps_to_train} --instruction_data_json {inst_data_json} --run_id {exp_name}",
    "mixed": "torchx run mast.py:train_mixed --name {exp_name} --nnodes 2 -- --steps_to_train {steps_to_train} --instruction_data_json {inst_data_json} --run_id {exp_name}",
}

exps = []
for model in models:
    for exp_type in exp_types:
        for num_step in num_steps:
            for pct_sft in pct_sfts:
                sft_size = int(num_step * pct_sft)
                pretrain_size = num_step
                for scenario in scenarios:
                    if scenario == "a":
                        if exp_type == "seq":
                            continue  # mixed type
                        pretrain_size = int(num_step * (1 - pct_sft))
                    elif scenario == "c":
                        if exp_type == "mixed":
                            continue  # seq type
                        pretrain_size = int(num_step * (1 + pct_sft))
                    elif scenario == "d":
                        if exp_type == "mixed":
                            continue  # seq type
                        pretrain_size = num_step

                    inst_data_json = f"/mnt/mffuse/all_in_one_pretraining/datasets/sft/concat/{sft_size}/"
                    if not os.path.exists(
                        inst_data_json.replace(
                            "/mnt/mffuse", "/data/users/nightingal3/manifold"
                        )
                    ):
                        print(
                            "Instruction slice of correct size doesn't exist, continuing..."
                        )
                        continue
                    if exp_type == "mixed":
                        pct_sft_str = str(pct_sft).replace(".", "_")
                        exp_name = f"{model}_{exp_type}_train_{scenario}_{pretrain_size}_sft{pct_sft_str}"
                    else:
                        exp_name = (
                            f"{model}_{exp_type}_train_{scenario}_{pretrain_size}"
                        )
                    print(exp_name)
                    exps.append(
                        {
                            "model": model,
                            "exp_type": exp_type,
                            "scenario": scenario,
                            "pretrain_size": pretrain_size,
                            "sft_size": sft_size,
                            "exp_name": exp_name,
                            "inst_data_json": inst_data_json,
                            "torchx_cmd": torchx_cmd[exp_type].format(
                                steps_to_train=pretrain_size,
                                inst_data_json=inst_data_json,
                                exp_name=exp_name,
                            ),
                        }
                    )

df_exp = pd.DataFrame(exps)
# remove exps with the same exp name
df_exp = df_exp.drop_duplicates(subset=["exp_name"])
df_exp.to_csv("./automation/sft_exps.tsv", sep=",", index=False)
