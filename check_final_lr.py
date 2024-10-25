import torch

# Load the checkpoint
checkpoint = torch.load(
    "/data/users/nightingal3/manifold/all_in_one_pretraining/out/pythia-7b/decay_from_100000_fw_for_200steps/final/lit_model.pth"
)

# Access the optimizer state
breakpoint()
optimizer_state = checkpoint["optimizer_state_dict"]

# Extract the learning rate
if "param_groups" in optimizer_state:
    lr = optimizer_state["param_groups"][0]["lr"]
    print(f"The final learning rate was: {lr}")
else:
    print("Couldn't find learning rate in the optimizer state.")
