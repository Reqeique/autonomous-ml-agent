# AGENTS.md — operating notes for any agent in this repo

This repo is designed to be driven by an LLM coding agent (OpenCode via
`.github/workflows/opencode-ml-agent.yml`) but the same rules apply to a human.

## Read first

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
