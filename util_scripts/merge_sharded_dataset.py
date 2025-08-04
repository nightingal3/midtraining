from litdata.processing.functions import merge_datasets
import os

# List of directories containing the dataset shards

to_merge = "starcoder"

if to_merge == "starcoder":
    input_dirs = [
        f"/compute/babel-14-29/mengyan3/starcoder_python/{i}" for i in range(1, 6)
    ] + [
        f"/compute/babel-14-29/mengyan3/starcoder_ctd_pythia/{i}" for i in range(6, 25)
    ]
else:
    input_dirs = [
        f"/compute/babel-14-29/mengyan3/c4_ctd_pythia_2/{i}" for i in range(35, 51)
    ] + [
        f"/compute/babel-14-29/mengyan3/c4_ctd_pythia/{i}" for i in range(18, 36)
    ] + [
        f"/compute/babel-14-29/mengyan3/c4_pythia/{i}" for i in range(1, 17) 
    ]

# Output directory to store the merged dataset
output_dir = "/compute/babel-14-29/mengyan3/starcoder_merged"

# Optional: Number of workers for multithreading
max_workers = 8

# Optional: Storage options for cloud providers (if needed)

# filter out chunks without index.json
input_dirs = [d for d in input_dirs if os.path.exists(os.path.join(d, "index.json"))]


# Call the merge_datasets utility
merge_datasets(
    input_dirs=input_dirs,
    output_dir=output_dir,
)