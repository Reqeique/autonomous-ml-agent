# Autonomous ML Agent — SOTA BPC for char-level text generation

A two-workflow system that runs an OpenCode-driven ML design/analyze loop against
Kaggle (free TPU/GPU) and W&B (free tracking) while spending almost nothing on
GitHub Actions minutes or LLM tokens.

## Goal & current baseline

Reach SOTA **BPC (bits per character)** for autoregressive char-level text
generation on enwik8 (target ≤1.05) and text8 (target ≤1.05).

`BPC = avg_token_cross_entropy_in_nats / ln(2)` — lower is better.

Current baseline:

- val_loss = 2.4216 nats → **BPC ≈ 3.493** on enwik8 (10M chars, val_frac=0.1).
- All recent runs crashed at epoch ≤3 (likely Kaggle 12h cap / TPU quota /
  OOM). First iteration priority: figure out why before pushing more.

W&B entity / project / sweep ids are **not** committed to the public repo; they
are kept in `state/baseline.json` (historical reference) and supplied to the
agent run via GitHub Actions env at runtime. See `state/baseline.json` for the
full context the agent reads on startup.

## How it works

```
   ┌────────────────────┐        completion        ┌──────────────────────┐
   │ opencode-ml-agent  │ ─── kaggle kernels push ─▶│  Kaggle TPU kernel   │
   │  (LLM, heavy)      │                          │  /kaggle/working     │
   └─────────┬──────────┘                          └──────────┬───────────┘
             │ state/pending_training.json                    │
             ▼                                               │
   ┌────────────────────┐  repo_dispatch (kaggle-training-complete)
   │ kaggle-watcher     │◀── polls every 20 min ─────────────┘
   │  (no LLM, cheap)   │
   └────────────────────┘
```

- **opencode-ml-agent.yml** — heavy LLM run. Triggers: `workflow_dispatch`,
  daily `schedule`, or `repository_dispatch` (`kaggle-training-complete`,
  `ml-agent-trigger`). Designs one improvement, pushes one kernel, exits.
- **kaggle-watcher.yml** — every 20 min. Reads `state/pending_training.json`,
  asks `kaggle kernels status`, and dispatches the agent on completion/failure.
  No LLM, seconds of runtime.

## Files

- `.github/workflows/opencode-ml-agent.yml` — main agent workflow.
- `.github/workflows/kaggle-watcher.yml` — cheap polling workflow.
- `.opencode/skills/ml-training-loop/SKILL.md` — skill the agent loads.
- `state/baseline.json` — current best BPC, sweep/project/entity context, next
  iteration directions.
- `state/pending_training.json.example` — schema reference for in-flight runs
  (real file is created/committed by the agent at runtime).
- `kernels/perpos-enwik8/{train.ipynb,kernel-metadata.json}` — canonical
  kernel (TF/Keras TPU v3-8, enwik8, byte vocab=257). Attaches to a W&B
  sweep via env vars (`WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_SWEEP_ID`) set
  at runtime; the notebook itself carries no hardcoded entity/project/sweep.
- `reference/shakespeare_pytorch_ref.ipynb` — small PyTorch dev loop on Tiny
  Shakespeare (cheap local architecture probes; not pushed to Kaggle).

## Remotes (dual-repo setup, both on Reqeique)

- **Public** (`origin`) → `https://github.com/Reqeique/autonomous-ml-agent` — where
  Actions run (unlimited minutes).
- **Private** (`private`) → `https://github.com/Reqeique/autonomous-ml-agent-private` —
  mirror of the same tree, for private OpenCode config + experiment notes.

```sh
git remote add origin   https://github.com/Reqeique/autonomous-ml-agent.git
git remote add private https://github.com/Reqeique/autonomous-ml-agent-private.git
git push origin main && git push private main   # both auth via gh → Reqeique
```

## Setup

1. Add secrets on the **public** repo (Settings → Secrets and variables → Actions):
   - `OPENCODE_ZEN_API_KEY` — OpenCode Zen (free primary model).
   - `NIM_API_KEY` — NVIDIA NIM (fallback models).
   - `WANDB_API_KEY` — W&B run logging; also store as a Kaggle user secret.
   - `KAGGLE_API_TOKEN` — single-token auth (Kaggle CLI v2 reads `KAGGLE_API_TOKEN`
     directly; no username-prefixed `kaggle.json` needed for `kaggle kernels push`).
   - `KAGGLE_USERNAME` — Kaggle account username, only used for the
     `kernel-metadata.json` `id` field (`<username>/<slug>`).
2. Set `WANDB_ENTITY` / `WANDB_PROJECT` / `WANDB_SWEEP_ID` as repo variables
   (Settings → Secrets and variables → Actions → Variables). These are
   **not** secrets but are kept out of the committed README/AGENTS.md to
   avoid publishing personal usernames in the public mirror.
3. Enable Actions PR creation: Settings → Actions → General → "Workflow
   permissions" → tick **"Allow GitHub Actions to create and approve pull
   requests"** and set the default to **Read and write**. Without this the
   agent can push branches but cannot open PRs from them (smoke-test run #4
   hit exactly this).
4. Manually trigger the agent once to push the baseline:
   ```sh
   gh workflow run opencode-ml-agent.yml \
     -f prompt="Use ml-training-loop. Push the baseline kernel in kernels/perpos-enwik8, write state/pending_training.json, exit. Do not analyze."
   ```
4. Confirm the watcher picks it up:
   ```sh
   gh workflow run kaggle-watcher.yml
   gh run list --workflow=kaggle-watcher.yml --limit=3
   ```

## Secrets

| Name | Purpose |
| --- | --- |
| `OPENCODE_ZEN_API_KEY` | OpenCode Zen primary LLM provider. |
| `NIM_API_KEY` | NVIDIA NIM fallback LLM provider. |
| `KAGGLE_API_TOKEN` | Kaggle CLI auth (token-only, new format). |
| `KAGGLE_USERNAME` | Only used for kernel `id` paths. |
| `WANDB_API_KEY` | W&B run logging; also load into Kaggle user secret. |

W&B entity/project/sweep ids and runner IP are kept out of the public
notebook and out of the workflow's stdout — set them as environment
variables at the workflow level rather than echoing them.

## Costs

- **Actions minutes**: public repo = unlimited; watcher job is ~10–30 s.
- **LLM tokens**: only on `workflow_dispatch`, daily cron, or completion events.
  Free primary provider (OpenCode Zen); fallbacks via NVIDIA NIM.
- **Kaggle**: TPU v3-8 quota (30h/week, 12h-session cap per run).
- **W&B**: free tier.

## Manual retrigger

```sh
gh workflow run opencode-ml-agent.yml \
  -f prompt="Use ml-training-loop. Analyze the most recent training and propose the next change."
```

## License

MIT.
