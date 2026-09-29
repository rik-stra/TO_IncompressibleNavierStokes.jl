#!/bin/bash
# Mini-D6 paired scores restricted to the full D6's 19 selection ICs (odd ordinals 89-125: window
# 52.5-70.75 TU, stride 2 within the mini's all-ordinal set). Separates an IC-subset effect from an
# M / horizon / hardware effect when the mini-D6 and the 3 TU check disagree.
# usage: eval_sub19.sh <A> <B>   (D6mini_ names without the prefix)
R=/export/scratch2/rik/time_series_TO/.claude/worktrees/overnight/lib/RikFlow
L=/ufs/rik/.claude/jobs/5801b24a/tmp/logs
cd "$R" || exit 1
export TMPDIR=/export/scratch2/rik/tmp D6_T_MIN=52.5 D6_T_MAX=70.75 D6_STRIDE=2
julia --startup-file=no --project=analysis analysis/score_d6.jl --paired "analysis/output/D6mini_$1" "analysis/output/D6mini_$2" > "$L/eval_sub19_$1_vs_$2.log" 2>&1
echo "sub19 $1 vs $2 rc=$?"
