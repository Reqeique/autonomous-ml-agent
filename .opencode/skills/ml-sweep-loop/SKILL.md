---
name: ml-sweep-loop
description: Monitor, steer, and fold in results from the parallel PerPos GPU sweep. Read W&B sweep runs, track convergence toward SOTA, adjust the bayes search space, terminate dead arms, and open PRs folding winning hyperparams into the canonical config. Never pushes Kaggle kernels (the sweep agents run them). Use for the GPU-sweep optimization loop, distinct from ml-training-loop (which owns the TPU kernel).
---

## Goal
Close the GPU-sweep loop that `ml-training-loop` deliberately does NOT own:

1. **Monitor** - pull all runs from the shared W&B sweep (`WANDB_SWEEP_ID`), rank by BPC under budget, detect stagnation.
2. **Steer** - evolve the sweep search space (narrow ranges around best arms, add/replace params), terminate dead arms, rebalance across the parallel agents. Steer = edit `sweep_config.json` + create/replace the W&B sweep id, then record the new id. Do NOT re-push kernels.
3. **Fold winners** - when a config beats `state/baseline.json` -> `current_best` under the SAME budget, open a PR updating the canonical config with the winning architecture hyperparams.
4. **Report** - summarize convergence, best config, next sweep, with W&B + PR links.

This skill runs in the `opencode-ml-sweep` workflow (separate concurrency group + state file from `opencode-ml-agent`). It is a READER + STEERER + PR-AUTHOR, never a kernel pusher.

## Constraint Enforcers (identical to ml-training-loop, re-stated)
- **The budget is LOCKED and FINAL.** `limit_chars=10M, block_size=512, stride=256, batch_size=384` (see `state/baseline.json` -> `committed_budget`). The GPU sweep kernel inherits this (`PERPOS_LIMIT_CHARS` env override exists but must stay 10M). Never fold in a config that raises these to cut BPC - that is the cheat.
- **Architecture-first.** BPC gains must come from the novel PerPos parts (ChunkCompressorEmbedding, InterpolatedFloatEmbedding, PerPos chunk decoder + small token decoder), NOT from compute inflation (epochs, limit_chars, block_size, batch, decoder layers, dim_ff).
- **Decoder CAPS (HARD, enforced by the kernel's `coerce_config`):** `token_dec_layers <= 2`, `token_dec_dim_ff <= 256`, `chunk_dec_layers <= 8`, `chunk_dec_dim_ff <= 2048`. If a winning config exceeds these, the kernel clamps it - do NOT fold an over-cap config in as-is; clamp it in the PR and note the adjustment.
- **Time-to-convergence is a first-class metric.** Only fold in a config that improves BPC under budget OR matches BPC much faster. A 2x-time config for +0.01 BPC is a regression.
- **Rejection rule:** a config whose only edge is more compute is NOT a valid improvement - reject it in the report.

## Monitor
- Sweep id/entity/project come from env. The `opencode-ml-sweep` workflow sets DEDICATED vars (`SWEEP_WANDB_ENTITY`, `SWEEP_WANDB_PROJECT`, `SWEEP_WANDB_SWEEP_ID`) AND mirrors them into `WANDB_ENTITY`/`WANDB_PROJECT`/`WANDB_SWEEP_ID`. Prefer the `SWEEP_WANDB_*` names; they point at the **GPU project** (`ml-playground-perpos-gpu` under the `-custom` entity). Do NOT trust the repo-wide `WANDB_PROJECT` var (that is the TPU agent's `-bpc` project) - always resolve the entity/project/sweep from `SWEEP_WANDB_*` first and verify the sweep actually lives there before acting. The canonical baseline is `state/baseline.json`.
- Query via `wandb.Api()`:
  ```python
  import wandb, os
  entity = os.environ.get("SWEEP_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY")
  project = os.environ.get("SWEEP_WANDB_PROJECT") or os.environ.get("WANDB_PROJECT")
  sweep_id = os.environ.get("SWEEP_WANDB_SWEEP_ID") or os.environ.get("WANDB_SWEEP_ID")
  api = wandb.Api()
  sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
  runs = list(sweep.runs)   # state, summary, config per run
  ```
- If the resolved `sweep_id` returns "Could not find sweep" under the resolved project, LIST sweeps in the project via `api.sweeps(f"{entity}/{project}")` to discover the real active sweep id before deciding - do not act on an unrelated sweep (e.g. the TPU project's sweep).
- Per run capture: `val/best_bpc` (or `val/loss` -> BPC), `train/CE`, `params/count`, `params/{group}`, `hb/r{epoch}`, `_runtime`, `_step`, `epochs`, and `config` (the sampled hyperparams incl. batch_size/lr/d_model/chunk_size/...).
- Compare every run's effective budget (`limit_chars`, `block_size`, `batch_size`) to the committed budget. Flag and IGNORE any run that inflated compute - it is not a candidate for folding.
- Track `state/sweep_state.json` (the sweep agent's own state file, separate from `pending_training.json`):
  ```json
  {
    "sweep_id": "<current>",
    "created_at": "<iso>",
    "goal": "SOTA BPC under committed budget via PerPos architecture",
    "best": { "bpc": 0.0, "run": "run-id", "config": {...}, "observed_at": "<iso>" },
    "stagnation": { "detected": false, "since_run": 0, "best_bpc_when_detected": 0.0 },
    "kernels_running": ["account-a/perpos-gpu-sweep-a", "..."],
    "notes": []
  }
  ```
- If `state/sweep_state.json` is absent, initialize it; never touch `state/pending_training.json` (owned by ml-training-loop/TPU agent).

## Steer (no kernel pushes)
Steering means evolving the SEARCH, not the compute:
1. **Rank** runs by `val/best_bpc`; take the top N (e.g. 5) that respect the budget.
2. **Stagnation:** if the best BPC has not improved for >= 2 consecutive monitor cycles or >= some runs (e.g. last 25 runs) while >= 4 agents run, mark `stagnation.detected` in `sweep_state.json`.
3. **Narrow the space:** edit `kernels/perpos-gpu-sweep/_private_live/sweep_config.json` - for each important param (d_model, chunk_size, quantized_dim, lr, weight_decay, dropout, compressor_emb_dim, token_local_window, ...) center the `distribution` ranges on the best arm(s). Drop params that the analysis shows are inert. NEVER add budget fields to the sweep space (limit_chars/block_size/batch are fixed in CONFIG/FIXED).
4. **Create the successor sweep** programmatically:
   ```python
   import wandb, json
   cfg = json.load(open("kernels/perpos-gpu-sweep/_private_live/sweep_config.json"))
   new_id = wandb.sweep(cfg, project=project, entity=entity)
   ```
   Update `state/sweep_state.json` -> `sweep_id` and the workflow repo variables `SWEEP_WANDB_SWEEP_ID` and `WANDB_SWEEP_ID` (via `gh variable set SWEEP_WANDB_SWEEP_ID --repo <owner>/autonomous-ml-agent -b <new_id> && gh variable set WANDB_SWEEP_ID --repo <owner>/autonomous-ml-agent -b <new_id>`), so the NEXT orchestrator push joins the successor sweep. Also open a PR with the sweep_config.json edit.
   - If the current sweep still has room (runs << run_cap, improving) you may keep it and only PR the narrowed config for the NEXT sweep - note the decision.
5. **Terminate dead arms:** W&B client-side Hyperband (`hb/r*`) already prunes inside each agent; do not kill kernels. If a specific kernel's agent is failing repeatedly, flag its slug in `sweep_state.json` notes for the orchestrator.

## Fold winners (PR only)
- When the sweep's best BPC beats `state/baseline.json` -> `current_best` -> `val_bpc` (currently 1.8833) UNDER THE SAME COMMITTED BUDGET:
  1. Take the winning run's `config`, CLAMP any over-cap decoder values, verify budget fields are unchanged.
  2. Open a PR titled e.g. `sweep: fold <run-id> into canonical config (BPC <old> -> <new>)`.
  3. In the PR, update the canonical config (the `CONFIG`/`FIXED` dict in `kernels/perpos-gpu-sweep/_private_live/train_src.py` AND the mirrored dict in the TPU kernel `kernels/perpos-enwik8/train.ipynb`) with the winning architecture hyperparams. Do NOT touch budget fields.
  4. Record the new best in `state/baseline.json` -> `current_best` (same PR or a follow-up) and in `sweep_state.json`.
- Do NOT push the PR's kernel to Kaggle - that is the orchestrator's job (it will pick up the new config on the next push).

## Tools & Commands
- W&B: `wandb.Api()` (reads `WANDB_API_KEY` env, set by the workflow from secrets). Sweep url pattern: `https://wandb.ai/<SWEEP_WANDB_ENTITY>/<SWEEP_WANDB_PROJECT>/sweeps/<SWEEP_WANDB_SWEEP_ID>`.
- GitHub: `gh variable set WANDB_SWEEP_ID --repo <owner>/autonomous-ml-agent-private -b <id>`; `gh pr create` for fold/steer PRs.
- Kaggle: only READ kernels for status/health (`kaggle kernels status <account>/<slug>`) using the per-account token env (`KAGGLE_API_TOKEN_N`); NEVER `kaggle kernels push`.
- State files: `state/sweep_state.json` (this skill), `state/baseline.json` (shared read), `state/pending_training.json` (hands-off).
- Workflow: `gh workflow run opencode-ml-sweep.yml` to trigger a manual monitor pass.

## Failure Modes to Watch
- **Quota exhaustion / queueing:** if several kernels sit in `Queued` for hours, the T4 quota is spent - do not create more sweep kernels; just record and report.
- **W&B not logging:** verify `WANDB_API_KEY`/`WANDB_ENTITY`/`WANDB_PROJECT` were injected; a run with no `val/best_bpc` is likely broken - flag it.
- **Budget inflation sneaks in:** any run with `limit_chars != 10M` or `batch_size != 384` (or the kernel's own defaults if it differs) is not foldable - reject, do not report as a win.
- **Empty sweep after successor:** if `WANDB_SWEEP_ID` points at a finished sweep with no runs, treat as "create successor now" not "error".

## Definition of Done (one cycle)
- `state/sweep_state.json` updated with fresh best + stagnation status + notes.
- If steering: `sweep_config.json` narrowed + successor sweep id created + `WANDB_SWEEP_ID` variable updated + PR opened.
- If folding: PR opened with clamped winning config (BPC old -> new in title), `baseline.json` current_best updated.
- Report emitted: sweep URL, best run URL, top-5 configs, stagnation verdict, links to any PRs.
- No Kaggle kernel was pushed by this skill.
