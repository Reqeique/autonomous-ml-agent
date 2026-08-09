---
name: ml-training-loop
description: Iterative ML model design, W&B hyperparam tuning/sweeps, Kaggle GPU training, result analysis, and retweaking. Supports handoff for completion-triggered retrigger to save compute/tokens. Use for autonomous training cycles.
---

## Goal
Drive a closed-loop, autonomous ML training cycle:

1. **Design** — small, focused improvement to model, features, or hyperparams.
2. **Tune** — short W&B sweep or single run if sweep overhead is too high.
3. **Train on Kaggle** — push a notebook kernel that uses free TPU **v5e-8**, logs to W&B, saves artifacts to `/kaggle/working`.
4. **Hand off** — record state, exit immediately. A cheap watcher retriggers the agent on completion.
5. **Analyze** — when retriggered with a completion payload, pull metrics/artifacts, decide next move, open a PR, optionally push the next iteration (one push max per invocation).

## Purpose: Architecture-First (Anti-Drift Guardrail)
The only legitimate source of BPC gains is the **architecture** — the novel PerPos parts (ChunkCompressorEmbedding, InterpolatedFloatEmbedding, PerPos chunk decoder + small token decoder). Epochs, dataset size, and context length are **budget levers, not loss levers**.

- **Never** increase epochs / `limit_chars` / `block_size` to lower loss. That is compute inflation: it burns the ~9 h TPU session budget and crowds out sweeps/ablations that could actually move the architecture.
- **Find the minimal budget that converges, once.** Across runs, search for the dataset-size × epoch combination that minimizes **first-time-to-convergence** (wall-clock to reach a target BPC). When found, LOCK it as a constant; subsequent runs run at that budget.
- **Time-to-convergence is a first-class metric.** Every run must report: wall-clock minutes, steps, epochs, and BPC at a matched epoch count. A config that takes 2× the time for +0.01 BPC is a regression, not an improvement.
- **Push rule:** a diff whose only effect is more compute (epochs↑, `limit_chars`↑, `block_size`↑) with no efficiency argument is NOT a valid improvement — reject it, do not push it.
- Keep the novel parts intact. Do not silently swap them for a generic architecture to game the metric.

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
- Kaggle: TPU v5e-8 free tier (~20 h/week per account, ≤9 h per session). Kernel must hard-assert TPU and W&B; no silent CPU fallback.
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
- Kaggle CLI: `kaggle kernels push -p kernels/perpos-enwik8`, `kaggle kernels status <slug>`, `kaggle kernels output <slug> -p ./artifacts`.
- W&B injection (REQUIRED before every push): `python scripts/inject_wandb.py --kernel kernels/perpos-enwik8 --out /tmp/kernel_push` then `kaggle kernels push -p /tmp/kernel_push`. See `AGENTS.md`. W&B is mandatory — if injection fails, do not push.
- W&B: `wandb login` (uses `WANDB_API_KEY` env), then `wandb sweep`, `wandb agent`, or just `wandb.init()` in the notebook.
- gh: `gh workflow run kaggle-watcher.yml` for a manual check; `gh workflow run opencode-ml-agent.yml` to retrigger the agent with a custom prompt.
- State: prefer committing small JSON; fall back to GitHub Issue label `training-pending` if JSON not desired.
- Logging helpers in this repo:
  - `scripts/inject_wandb.py` — stages an ephemeral notebook copy with W&B creds baked in (never commits secrets).
  - `scripts/check_kaggle_quotas.py` — best-effort, may need login.
  - `scripts/test_keys.ps1` — pwsh, verifies W&B/Kaggle/NIM/Zen keys (masked output).
  - `scripts/retrigger_with_fallback.sh` — when this run hits a primary-model rate limit, re-dispatch the agent with the fallback model.
  - `kernels/perpos-enwik8/train.ipynb` — canonical enwik8 kernel (TPU v5e-8, hard-requires W&B).

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
- [ ] `kernel-metadata.json` has `code_file`, `language: python`, `kernel_type: notebook`, `enable_tpu: true`, `enable_internet: true`, `accelerator: TpuV5E8`, `dataset_sources`/`competition_sources` if needed, `keywords`, `title`.
- [ ] Notebook installs deps in a single cell; reads W&B from env (`WANDB_API_KEY`/`WANDB_ENTITY`/`WANDB_PROJECT`) which `scripts/inject_wandb.py` bakes into the pushed copy.
- [ ] Saves models/plots to `/kaggle/working/` so `kaggle kernels output` retrieves them.
- [ ] Inject creds into an ephemeral copy and push from it:
  ```sh
  python scripts/inject_wandb.py --kernel kernels/perpos-enwik8 --out /tmp/kernel_push
  kaggle kernels push -p /tmp/kernel_push
  ```
- [ ] After push, `kaggle kernels status <user>/<slug>` should return `running` within ~30 s.

## Failure Modes to Watch
- Kernel stuck in `Queueing` for hours → usually TPU quota exhausted. Stop pushing.
- W&B not logging → check `scripts/inject_wandb.py` ran and all three env vars were set; the notebook raises otherwise (by design).
- Watcher grep regex misses "complete" / "error" → improve parser; `kaggle kernels status` returns plain text like `"<user>/<slug>" has status "complete"`.

## Definition of Done (one iteration)
- PR opened with diff (or state cleared if no change worth shipping).
- Pending state cleared (`state/pending_training.json` deleted).
- "training-pending" GitHub Issues closed.
- Summary comment on the issue or PR with W&B run URL + kernel URL.
- Every pushed run reports, alongside final BPC: wall-clock minutes, steps, epochs, and BPC at a matched epoch count (time-to-convergence). A run with no architecture change and no time-to-convergence improvement is a budget leak, not a result.
