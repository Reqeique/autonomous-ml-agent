#!/usr/bin/env bash
# Re-dispatch opencode-ml-agent.yml with a fallback model.
# Usage: ./scripts/retrigger_with_fallback.sh "reason for switching"
set -euo pipefail
REASON="${1:-Switching to fallback model.}"
PRIMARY="${OPENCODE_PRIMARY_MODEL:-opencode/deepseek-v4-flash-free}"
FALLBACK="${OPENCODE_FALLBACK_MODEL:-nvidia/z-ai/glm-5.2}"
echo "[retrigger] reason: $REASON"
echo "[retrigger] primary: $PRIMARY  fallback: $FALLBACK"
gh workflow run opencode-ml-agent.yml \
  --repo "${GITHUB_REPOSITORY:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}" \
  -f prompt="Primary model ($PRIMARY) was rate-limited. Reason: $REASON. Use ml-training-loop with the fallback model. Resume the previous in-flight training cycle if state/pending_training.json still points to a non-terminal kernel; otherwise do the usual design/push/handoff or analyze path." \
  -f model="$FALLBACK" \
  -f fallback_model="nvidia/minimaxai/minimax-m3"
echo "[retrigger] dispatched"
