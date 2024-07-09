import torchx.components.fb.conda as conda
import torchx.components.fb.interactive_lib as interactive_lib
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


def train(
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

    args = [
        "--nnodes",
        str(nnodes),
        "--nproc-per-node",
        str(nproc_per_node),
        "--no-python",
        "./run.sh",
        script,
        *script_args,
        f"--out_dir={out_dir}",
        f"--tb_log={tb_log}",
    ]

    job_spec = conda.torchrun(*args, **kwargs)

    packages = [job_spec.roles[0].image, *additional_packages]
    job_spec.roles[0].image = ";".join(packages)

    return job_spec


def train_interactive(
    *script_args: str,
    script: str = "mixed_pretraining.sh",
    nnodes: int = 1,
    nproc_per_node: int = 1,
    name: str = "train_interactive",
    h: str = "tc_any",
    run_as_root: bool = True,
    out_dir: str = "/mnt/mffuse/out/${app_id}",
    tb_log: bool = True,
) -> specs.AppDef:
    # Set env variable to disable mounting
    # additional_env = {"DISABLE_MOUNT": "1"}
    # define a training job spec
    job_spec = train(
        *script_args,
        script=script,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        name=name,
        h=h,
        run_as_root=run_as_root,
        env=additional_env,
        out_dir=out_dir,
        tb_log=tb_log,
    )
    # wrap the job spec with as_interactive
    return interactive_lib.as_interactive(
        # the job spec to wrap
        job_spec=job_spec,
        # reserve MAST host for 4 hours
        interactive_duration_hrs=4,
        # prerun the mount command at node startup
        prerun_commands={"torchrun": "source /packages/torchx_conda_mount/mount.sh"},
    )
