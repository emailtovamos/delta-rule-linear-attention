#!/bin/zsh
# Serial WikiText-103 125M runs, one variant+seed at a time.
# Resumable: completed runs (results/<name>/log.json exists) are skipped,
# so re-running this script continues where it left off.
# Graceful stop: `touch results/STOP_WIKI125M` halts after the current run.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/Library/Python/3.9/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"

STOP=results/STOP_WIKI125M
rm -f "$STOP"
exec > >(tee -a results/wiki125m_serial.log) 2>&1

# 2x2 factorial cells at the original seeds first, then the remaining four
# variants (completes the AE-required 8-variant table), then seeds 1234/2024.
# Seeds 1234/2024 were added 25 Aug 2026 after elevated seed variance had been
# observed (two of three seed-level interactions already known): post-hoc
# exploratory repeats, not a confirmatory design, and not used for paper claims.
# Reordered 26 Aug 2026: AE-required variants ran ahead of seeds 1234/2024.
RUNS=(
  scalar_static:42 static_channel:42 scalar_static_delta:42 static_channel_delta:42
  scalar_static:123 static_channel:123 scalar_static_delta:123 static_channel_delta:123
  scalar_static:7 static_channel:7 scalar_static_delta:7 static_channel_delta:7
  standard:42 gla:42 deltanet:42 kda:42
  standard:123 gla:123 deltanet:123 kda:123
  standard:7 gla:7 deltanet:7 kda:7
  scalar_static:1234 static_channel:1234 scalar_static_delta:1234 static_channel_delta:1234
  scalar_static:2024 static_channel:2024 scalar_static_delta:2024 static_channel_delta:2024
)

for run in "${RUNS[@]}"; do
  v=${run%:*}; s=${run#*:}; n="wiki_${v}_125m_s${s}"
  if [ -f "results/$n/log.json" ]; then
    echo "already done: $n"
    continue
  fi
  if [ -f "$STOP" ]; then
    echo "STOP file found — halting before $n ($(date))"
    exit 0
  fi
  if [ -d "results/$n" ]; then
    echo "clearing partial dir from interrupted run: results/$n"
    rm -rf "results/$n"
  fi
  echo "================================================================"
  echo "START $n  $(date)"
  echo "================================================================"
  caffeinate -dims uv run python scripts/train.py \
    --attention "$v" --name "$n" --size medium --dataset wikitext --seed "$s" \
    || { echo "FAILED: $n ($(date))"; exit 1; }
  echo "DONE  $n  $(date)"
done
echo "ALL 32 RUNS COMPLETE $(date)"
