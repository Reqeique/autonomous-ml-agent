# AGENTS.md — operating notes for any agent in this repo

This repo is designed to be driven by an LLM coding agent (OpenCode via
`.github/workflows/opencode-ml-agent.yml`) but the same rules apply to a human.

## Goal

Reach SOTA **BPC (bits per character)** for autoregressive char-level text
generation on enwik8 and text8, building on the user's prior
`HierarchicalCharTransformer / PerPos AR` work.

**Metric**: `BPC = avg_token_cross_entropy_in_nats / ln(2)`.
Lower is better. SOTA ballpark: enwik8 ≤1.05 BPC, text8 ≤1.05 BPC.

**Current baseline** (Sep 2025 sweep): val_loss=2.4216 nats → **BPC=3.493**
on enwik8 (10M chars subset, val_frac=0.1). Every recent run crashed at
epoch ≤3 — investigate first (probably Kaggle's 12h session cap, TPU quota,
or OOM). See `state/baseline.json` for the full context the agent must read
at startup, **before** deciding what to push next.

## Where the work lives

- `kernels/perpos-enwik8/{train.ipynb,kernel-metadata.json}` — canonical kernel:
  TF/Keras TPU v3-8, byte-level UTF-8 vocab=257, transformer with
  `ChunkCompressorEmbedding → InterpolatedFloatEmbedding → PerPos chunk
  decoder + small token decoder`. The notebook reads W&B entity/project/sweep
  from env vars (`WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_SWEEP_ID`); it carries
  no hardcoded team/project/sweep strings.
- `reference/shakespeare_pytorch_ref.ipynb` — small PyTorch dev loop on Tiny
  Shakespeare. Used for cheap local architecture probes; **not** pushed to
  Kaggle (its dataset is irrelevant to enwik8/text8).
- `state/baseline.json` — goal + current best BPC + historical sweep
  context (kept here as the agent's source of truth) + next iteration
  directions. Read first.
- See `README.md` for the full architecture (workflows, watcher, state).

## Read first

- `state/baseline.json` — what's the current best and what's the next direction.
- `.opencode/skills/ml-training-loop/SKILL.md` — the loop. State handoff rules
  there are non-negotiable; they exist to keep Actions minutes and tokens low.

## Conventions

- **State**: `state/pending_training.json` is the single source of truth for
  "is a training currently in flight?" It must be deleted/updated by whoever
  finishes analyzing the run.
- **Issues**: optional mirror. Label `training-pending` = open training;
  `training-complete` / `training-failed` are added on close.
- **Kernels**: one folder per kernel under `kernels/<name>/` containing
  `train.ipynb` + `kernel-metadata.json`. The `id` in the metadata must match
  `<KAGGLE_USERNAME>/<slug>`.
- **PRs**: every meaningful code change ships through a PR so the watcher can
  comment and the workflow runs from a clean checkout.

## Don'ts

- Don't long-poll Kaggle inside a single agent invocation — exit after `push`
  and let the watcher retrigger.
- Don't commit `~/.kaggle/kaggle.json`, `state/pending_training.json` with real
  slugs during active runs if you don't want public logs to leak them — though
  the public repo + public runners trade-off here is fine for slugs only.
- Don't push more than one kernel per agent run unless quotas clearly allow it.

## Useful gh one-liners

```sh
# Manual dispatch with a custom prompt
gh workflow run opencode-ml-agent.yml -f prompt="Your prompt here"

# Force a watcher check right now
gh workflow run kaggle-watcher.yml

# Tail the latest agent run
gh run list --workflow=opencode-ml-agent.yml --limit=1 --json databaseId,status,conclusion -q '.[0].databaseId' \
  | xargs -I{} gh run watch {}

# Clear pending state (e.g., stuck run)
git rm state/pending_training.json && git commit -m "chore: clear stuck pending training" && git push
```
