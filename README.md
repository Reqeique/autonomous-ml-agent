# autonomous-ml-agent

Automated ML training orchestration via GitHub Actions. Triggered on schedule
or manually; pulls a training plan from a private mirror, runs an LLM coding
agent to assess state and propose a tweak, pushes a training kernel to
Kaggle, and tracks results through a polling watcher that hands off back to
the agent on completion.

## Contents

- `.github/workflows/` — two GitHub Actions: a heavy agent run and a cheap
  status watcher. These are the only files committed to this public repo.

Everything else (skills, kernels, state, agent operating notes) lives in a
private mirror pulled by the workflows at runtime via a stored PAT. This
keeps the architecture, dataset choices, and tuning logic out of the public
view while still benefiting from free Actions minutes.

## Manual trigger

```sh
gh workflow run opencode-ml-agent.yml -f prompt="Your prompt here"
```

## License

MIT.
