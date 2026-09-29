#!/bin/bash
# Evaluate mini-D6 runs (selection block, 46 ICs, M = 5, 1 TU): paired primary score against each
# baseline given, the unpaired score (spread-skill), and the tail census.
# usage: eval_mini.sh <run> <baseline> [<baseline> ...]   (names without the D6mini_ prefix)
R=/export/scratch2/rik/time_series_TO/.claude/worktrees/overnight/lib/RikFlow
L=/ufs/rik/.claude/jobs/5801b24a/tmp/logs
O=$R/analysis/output
cd "$R" || exit 1
export TMPDIR=/export/scratch2/rik/tmp
V=$1; shift
for B in "$@"; do
  julia --startup-file=no --project=analysis analysis/score_d6.jl --paired "$O/D6mini_$V" "$O/D6mini_$B" > "$L/eval_mini_${V}_vs_$B.log" 2>&1
  echo "paired $V vs $B rc=$?"
done
[ -f "$O/d6_scores_D6mini_$V.jld2" ] || D6_OUT=$O/D6mini_$V julia --startup-file=no --project=analysis analysis/score_d6.jl > "$L/eval_mini_score_$V.log" 2>&1
echo "score $V rc=$?"
julia --startup-file=no --project=training analysis/tail_census.jl "D6mini_$V" > "$L/eval_mini_census_$V.log" 2>&1
echo "census $V rc=$?"
