from litdata import merge_datasets

in_dir = [
    f"../manifold/scaling_mates/data/fineweb/sample-350BT/train/{i}"
    for i in range(0, 10)
]
print(in_dir)
merged_out_dir = "../manifold/all_in_one_pretraining/datasets/fineweb/fineweb-350B"

merge_datasets(input_dirs=in_dir, output_dir=merged_out_dir)
