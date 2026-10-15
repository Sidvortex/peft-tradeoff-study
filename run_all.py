"""Run every configuration in the study, each in a fresh process.

    python run_all.py                       # all runs
    python run_all.py --only lora_r16 qlora_r16
    python run_all.py --skip full_ft        # e.g. if you are short on GPU hours
    python run_all.py --set data.n_train=2000   # forwarded to every run

Runs already present in the results file are skipped, so you can resume after a
Kaggle session times out.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.config import base_parser, load_config


def main() -> None:
    parser = base_parser(__doc__, "configs/study.yaml")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--rerun", action="store_true", help="ignore existing results")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    done = set()
    results_file = Path(cfg["train"]["results_file"])
    if results_file.exists() and not args.rerun:
        done = {json.loads(l)["run"] for l in results_file.read_text().splitlines() if l.strip()}

    for run in cfg["runs"]:
        name = run["name"]
        if args.only and name not in args.only:
            continue
        if name in args.skip or name in done:
            print(f"-- skipping {name}")
            continue
        print(f"\n===== {name} =====", flush=True)
        cmd = [sys.executable, "train_one.py", "--config", args.config, "--run", name]
        if args.set:
            cmd += ["--set", *args.set]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! {name} failed with exit code {rc} (often out-of-memory); continuing")

    print("\nAll done. Now run:  python report.py")


if __name__ == "__main__":
    main()
