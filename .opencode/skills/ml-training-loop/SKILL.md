---
name: ml-training-loop
description: Iterative ML model design, W&B hyperparam tuning/sweeps, Kaggle GPU training, result analysis, and retweaking. Supports handoff for completion-triggered retrigger to save compute/tokens. Use for autonomous training cycles.
---

## Goal
Drive a closed-loop, autonomous ML training cycle:

1. **Design** — small, focused improvement to model, features, or hyperparams.
2. **Tune** — short W&B sweep or single run if sweep overhead is too high.
3. **Train on Kaggle** — push a notebook kernel that uses free GPU, logs to W&B, saves artifacts to `/kaggle/working`.
4. **Hand off** — record state, exit immediately. A cheap watcher retriggers the agent on completion.
5. **Analyze** — when retriggered with a completion payload, pull metrics/artifacts, decide next move, open a PR, optionally push the next iteration (one push max per invocation).

## Efficient Handoff Workflow (critical for cost)
After successfully pushing a Kaggle kernel (`kaggle kernels push ...`):

1. Record state — write/update `state/pending_training.json`:
   ```json
   {
     "kernel": "youruser/kernel-slug",
     "pushed_at": "2026-08-08T03:00:00Z",
     "wandb_run": "youruser/project/run-id-or-sweep",
     "goal": "one-sentence intent for this run",
     "iteration": 3
   }
   ```
2. Optionally open/update a GitHub Issue titled `Training in progress: youruser/kernel-slug` with label `training-pending` containing the same JSON in the body.
3. Commit the state file (small JSON is fine) — the watcher also reads the issue label.
4. **Exit cleanly** with a short summary. Do not long-poll, sleep, or watch training. The watcher wakes the agent again.

When invoked with completion context (workflow prompt mentions "Training-completion notification", repository_dispatch payload, or kernel slug):

1. Skip straight to **Monitor/Analyze**: `kaggle kernels status <slug>`, `kaggle kernels output <slug>` to pull `/kaggle/working` artifacts, query W&B API for the run metrics.
2. Decide: meaningful improvement possible? If yes → edit code, push next kernel (max one per invocation), update state, exit. If no → open a PR summarizing findings, close the training issue, delete `state/pending_training.json`, exit.
3. Always emit links: kernel URL, W&B run URL, PR URL.

## Quotas & Free-Tier Discipline
- Kaggle: 30 GPU-hours/week per account. Prefer T4, ≤4 h/run, aggressive early stopping.
- W&B: free tier has run/sweep caps; sweep only on cheap models.
- GitHub Actions: public repo = unlimited minutes. Private repo = 2000 min/mo free.
- LLM tokens: design/prompt compact. Avoid long file dumps in prompts.

## IP Rotation (passive)
- Every `runs-on: ubuntu-latest` job = fresh VM, new public IP from GitHub's Azure pool.
- Watcher every ~20 min + agent on each event = many distinct IPs naturally.
- The agent logs the IP via `curl -s ifconfig.me || curl -s ipinfo.io/ip || echo unknown` for visibility. Do not try to force-rotate inside one job.

## State File Hygiene
- Schema is permissive: missing optional fields are OK; `kernel` is required.
- On startup, the agent should:
  - If `state/pending_training.json` exists → treat as "still training", do not push another kernel; just summarize status and exit.
  - If absent → proceed with design + push if quotas allow.
- The agent, on analyze, deletes the file after closing the issue / opening the PR.

## Tools & Commands
- Kaggle CLI: `kaggle kernels push -p kernels/baseline`, `kaggle kernels status <slug>`, `kaggle kernels output <slug> -p ./artifacts`.
- W&B: `wandb login` (uses `WANDB_API_KEY` env), then `wandb sweep`, `wandb agent`, or just `wandb.init()` in the notebook.
- gh: `gh workflow run kaggle-watcher.yml` for a manual check; `gh workflow run opencode-ml-agent.yml` to retrigger the agent with a custom prompt.
- State: prefer committing small JSON; fall back to GitHub Issue label `training-pending` if JSON not desired.
- Logging helpers in this repo:
  - `scripts/check_kaggle_quotas.py` — best-effort, may need login.
  - `scripts/test_keys.ps1` — pwsh, verifies W&B/Kaggle/NIM/Zen keys (masked output).
  - `scripts/retrigger_with_fallback.sh` — when this run hits a primary-model rate limit, re-dispatch the agent with the fallback model.
  - `kernels/baseline/train.ipynb` — minimal starter notebook using `kaggle_secrets` for W&B key.

## Model Chain (auto-tested via `scripts/test_keys.ps1`)

Primary → fallback policy is encoded as workflow inputs and honored by the agent:

| Order | Model (workflow `model:` input) | Provider | Notes |
| --- | --- | --- | --- |
| 1 (primary, free) | `opencode/deepseek-v4-flash-free` | OpenCode Zen | Free; data may be used to improve. |
| 2 (paid fallback) | `nvidia/z-ai/glm-5.2` | NVIDIA NIM | Uses `NIM_API_KEY` → `NVIDIA_API_KEY`. |
| 3 (last resort) | `nvidia/minimaxai/minimax-m3` | NVIDIA NIM | Same key as #2. |

If the primary model rate-limits, the agent should re-dispatch itself:

```sh
gh workflow run opencode-ml-agent.yml \
  -f prompt="Primary model rate-limited. Resume ml-training-loop work using the fallback model." \
  -f model=nvidia/z-ai/glm-5.2 \
  -f fallback_model=nvidia/minimaxai/minimax-m3
```

Or invoke `scripts/retrigger_with_fallback.sh "Primary rate-limited; resume ml-training-loop"`.

## Kernel Prep Checklist
Before `kaggle kernels push`:
- [ ] `kernel-metadata.json` has `code_file`, `language: python`, `kernel_type: notebook`, `enable_gpu: true`, `enable_internet: true`, `accelerator: nvidiaT4`, `dataset_sources`/`competition_sources` if needed, `keywords`, `title`.
- [ ] Notebook installs deps in a single cell, sets `WANDB_API_KEY` via `kaggle_secrets.UserSecretsClient().get_secret("WANDB_API_KEY")` (also pass `WANDB_API_KEY` env from the runner so it works on first run before secret exists).
- [ ] Saves models/plots to `/kaggle/working/` so `kaggle kernels output` retrieves them.
- [ ] Push command: `kaggle kernels push -p kernels/baseline` from repo root.
- [ ] After push, `kaggle kernels status <user>/<slug>` should return `running` within ~30 s.

## Failure Modes to Watch
- Kernel stuck in `Queueing` for hours → usually GPU quota exhausted. Stop pushing.
- W&B not logging → check `WANDB_API_KEY` propagation; fall back to logging JSON to `/kaggle/working/metrics.jsonl` and pull from output.
- Watcher grep regex misses "complete" / "error" → improve parser; `kaggle kernels status` returns plain text like `"<user>/<slug>" has status "complete"`.

## Definition of Done (one iteration)
- PR opened with diff (or state cleared if no change worth shipping).
- Pending state cleared (`state/pending_training.json` deleted).
- "training-pending" GitHub Issues closed.
- Summary comment on the issue or PR with W&B run URL + kernel URL.
