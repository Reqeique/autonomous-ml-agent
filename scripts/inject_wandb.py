#!/usr/bin/env python3
"""Inject W&B credentials from the CI/agent environment into a COPY of the
notebook so the Kaggle kernel can log online, without ever committing secrets.

Why: Kaggle kernels do NOT inherit GitHub Actions env vars. The agent has
WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT available in its shell, but they
would not reach the kernel at runtime. This script bakes them into the
notebook source right before `kaggle kernels push`.

Usage (run from repo root, agent-side):
    python scripts/inject_wandb.py --kernel kernels/perpos-enwik8 \
        --out /tmp/kernel_push

Required env: WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT  (else exit 1, so W&B
is mandatory — the kernel must never train unlogged).
Optional env: WANDB_SWEEP_ID.

Behavior:
    - Reads the notebook at <kernel>/train.ipynb
    - Prepends a self-contained `import os` + `os.environ[...] = ...` block to
      the first CODE cell so the values are present before the notebook's
      mandatory W&B guard runs (the block imports os itself, since the first
      code cell's own `import os` may come later in the same cell)
    - Writes the modified notebook + kernel-metadata.json into --out
    - Never touches the committed notebook in the repo
"""

import argparse
import json
import os
import shutil
import sys


def die(msg):
    print(f"[inject_wandb] ERROR: {msg}", flush=True)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, help="path to kernel folder with train.ipynb + kernel-metadata.json")
    p.add_argument("--out", required=True, help="output folder to stage the push payload")
    args = p.parse_args()

    kernel_dir = os.path.abspath(args.kernel)
    out_dir = os.path.abspath(args.out)

    nb_path = os.path.join(kernel_dir, "train.ipynb")
    meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
    for f in (nb_path, meta_path):
        if not os.path.exists(f):
            die(f"missing {f}")

    required = {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"}
    missing = [k for k in sorted(required) if not os.environ.get(k)]
    if missing:
        die(
            "W&B is mandatory and these env vars are missing: " + ", ".join(missing)
            + ". Set them (they are available in the GHA workflow env) or abort."
        )

    os.makedirs(out_dir, exist_ok=True)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    injected = ["import os"]
    for k in sorted(required):
        injected.append(f'os.environ[{k!r}] = {os.environ[k]!r}')
    if os.environ.get("WANDB_SWEEP_ID"):
        injected.append(f'os.environ["WANDB_SWEEP_ID"] = {os.environ["WANDB_SWEEP_ID"]!r}')
    block = "\n".join(injected)

    # Find first code cell (skip markdown headers).
    target = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            target = i
            break
    if target is None:
        die("no code cell found in notebook")

    header = "# --- injected W&B creds by scripts/inject_wandb.py (ephemeral, not committed) ---\n"
    src = "".join(nb["cells"][target]["source"])
    nb["cells"][target]["source"] = (header + block + "\n\n" + src).splitlines(keepends=True)

    out_nb = os.path.join(out_dir, "train.ipynb")
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    shutil.copy2(meta_path, os.path.join(out_dir, "kernel-metadata.json"))

    print(f"[inject_wandb] staged injected kernel to {out_dir} (keys present: {sorted(required)})", flush=True)
    print("[inject_wandb] next: kaggle kernels push -p " + out_dir, flush=True)


if __name__ == "__main__":
    main()
