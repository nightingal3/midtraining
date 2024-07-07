import torchx.components.fb.conda as conda
import torchx.specs as specs


env_vars = {
    "DISABLE_NFS": "1",
    "DISABLE_OILFS": "1",
    "MANIFOLDFS_BUCKET": "coin",
    "LD_PRELOAD": "/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so",
    "TRITON_LIBCUDA_PATH": "/usr/local/fbcode/platform010/lib/libcuda.so",
}

additional_packages = [
    "torchx_conda_mount:stable",
    "manifold.manifoldfs:prod",
    "conda_mast_core:stable",
]


def train_singlegpu(
    *script_args: str,
    name: str = "mixed_train",
    script: str = "mixed_pretraining.sh",
    nnodes: int = 1,
    nproc_per_node: int = 1,
    h: str = "tc_any",
    run_as_root: bool = True,
    env: Optional[dict] = {},
    tb_log: bool = True,
    out_dir: str = "/mnt/mffuse/out/${app_id}",
) -> specs.AppDef:
    kwargs = {
        "name": name,
        "h": h,
        "run_as_root": run_as_root,
        "env": {**env_vars, **env} if env else env_vars,
    }

    # args = [
    #     "--nnodes",
    #     str(nnodes),
    #     "--nproc-per-node",
    #     str(nproc_per_node),
    #     "--no-python",
    #     "--conda_pack_ignore_missing_files=True",
    #     "./run.sh",
    #     script,
    #     *script_args,
    #     f"--out_dir={out_dir}",
    #     f"--tb_log={tb_log}",
    # ]

    args = [
        "--nnodes",
        str(nnodes),
        "--nproc-per-node",
        str(nproc_per_node),
        "--no-python",
        "./run_new.sh",
        *script_args,
        f"--out_dir={out_dir}",
        f"--tb_log={tb_log}",
    ]

    job_spec = conda.torchrun(*args, **kwargs)

    packages = [job_spec.roles[0].image, *additional_packages]
    job_spec.roles[0].image = ";".join(packages)

    return job_spec
