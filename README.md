# Autonomous ML Agent (free-tier, low-token)

A two-workflow system that runs an OpenCode-driven ML design/analyze loop against
Kaggle (free GPU) and W&B (free tracking) while spending almost nothing on
GitHub Actions minutes or LLM tokens.

## How it works

```
   ┌────────────────────┐        completion        ┌──────────────────────┐
   │ opencode-ml-agent  │ ─── kaggle kernels push ─▶│  Kaggle GPU kernel   │
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
- `state/pending_training.json.example` — schema reference (real file is created
  at runtime by the agent and committed).
- `kernels/baseline/{train.ipynb,kernel-metadata.json}` — starter kernel.

## Remotes (dual-repo setup)

- **Public** (`origin`) → `https://github.com/Reqeique/autonomous-ml-agent` — where
  Actions run (unlimited minutes).
- **Private** (`private`) → `https://github.com/OrbeHQ/autonomous-ml-agent` —
  mirror of the same tree, for private OpenCode config + experiment notes.

```sh
git remote add origin   https://github.com/Reqeique/autonomous-ml-agent.git  # public, Actions
git remote add private https://github.com/OrbeHQ/autonomous-ml-agent.git    # private mirror
git push origin main && git push private main
```

## Setup

1. Add secrets on the **public** repo (Settings → Secrets and variables → Actions):
   - `ANTHROPIC_API_KEY`
   - `KAGGLE_USERNAME`, `KAGGLE_KEY`
   - `WANDB_API_KEY`
2. Edit `kernels/baseline/kernel-metadata.json` — set `id` to `<your-kaggle-username>/<slug>`.
3. Manually trigger the agent once to push the baseline:
   ```sh
   gh workflow run opencode-ml-agent.yml \
     -f prompt="Use ml-training-loop. Push the baseline kernel in kernels/baseline, write state/pending_training.json, exit. Do not analyze."
   ```
5. Confirm the watcher picks it up:
   ```sh
   gh workflow run kaggle-watcher.yml
   gh run list --workflow=kaggle-watcher.yml --limit=3
   ```

## Secrets (GitHub Actions → Settings → Secrets)

| Name | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | LLM provider for OpenCode. |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle CLI auth (from `kaggle.json`). |
| `WANDB_API_KEY` | W&B run logging; also store as a Kaggle user secret for redundancy. |

## Costs

- **Actions minutes**: public repo = unlimited; watcher job is ~10–30 s.
- **LLM tokens**: only on `workflow_dispatch`, daily cron, or completion events.
- **Kaggle GPU**: 30 h/week per account, T4.
- **W&B**: free tier.

## Manual retrigger

```sh
gh workflow run opencode-ml-agent.yml \
  -f prompt="Use ml-training-loop. Analyze the most recent training and propose the next change."
```

## License

MIT.
