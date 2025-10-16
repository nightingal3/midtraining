import wandb

# ── CONFIGURE ──────────────────────────────────────────────────────────────
ENTITY  = "pretraining-and-behaviour"
PROJECT = "finetune-pythia-70m"
GROUP   = "pycode_final_fixed_70m"
OLD_TAG       = "flan_high_mix"
NEW_TAG       = "starcoder_mix"
CHKPT_PREFIX  = "/projects/bfcu/mliu7/all_in_one_pretrainingpretrained_chkpts/pythia_70m_128b_fixed_midtrain_spikefix"
DRY_RUN       = False
# ───────────────────────────────────────────────────────────────────────────

api = wandb.Api()
filters = {"tags": OLD_TAG}
if GROUP:
    filters["group"] = GROUP

for run in api.runs(f"{ENTITY}/{PROJECT}", filters):
    chkpt = run.config.get("checkpoint_dir", "")
    if not chkpt.startswith(CHKPT_PREFIX):
        continue

    print(f"Found run {run.id} with tags {run.tags}")
    new_tags = [t for t in run.tags if t != OLD_TAG]
    if NEW_TAG not in new_tags:
        new_tags.append(NEW_TAG)
    print(f" → would update to: {new_tags}")

    if not DRY_RUN:
        # mutate, then push
        run.tags = new_tags
        run.update()   # ← no args here :contentReference[oaicite:0]{index=0}
        print(f" ✔ updated run {run.id}")
