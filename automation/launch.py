import argparse
import concurrent.futures
import random
import subprocess
import time

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_file", default="./automation/pretrain_exps.csv")
    parser.add_argument("--constraint", default=None, help="filter exp df to a subset")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="don't execute, just print torchx command",
    )
    parser.add_argument(
        "--stagger",
        action="store_true",
        help="stagger the time at which jobs are submitted",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="submit jobs sequentially instead of in parallel",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.exp_file)
    if args.constraint:
        df = df.query(args.constraint)

    if args.dry_run:
        for row in df.to_dict(orient="records"):
            print(row["torchx_cmd"])
    else:
        if args.sequential:
            for row in df.to_dict(orient="records"):
                torchx_cmd = row["torchx_cmd"]
                print(f"Running {torchx_cmd}")
                deploy_path = "/data/users/nightingal3/all_in_one_pretraining/deploy"

                subprocess.run(torchx_cmd, shell=True, cwd=deploy_path)

                if args.stagger:
                    delay = 300
                    print("Waiting...")
                    print(f"Staggering job submission by {delay} seconds")
                    time.sleep(delay)
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for row in df.to_dict(orient="records"):
                    torchx_cmd = row["torchx_cmd"]
                    deploy_path = (
                        "/data/users/nightingal3/all_in_one_pretraining/deploy"
                    )
                    if args.stagger:
                        delay = random.randint(120, 3600)
                        print(f"Staggering job submission by {delay} seconds")
                        time.sleep(delay)

                    future = executor.submit(
                        subprocess.run, torchx_cmd, shell=True, cwd=deploy_path
                    )
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    future.result()
