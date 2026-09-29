#!/bin/bash
# Evaluate D6sel3_<V> (3 TU, 19 selection ICs, M = 10) against the full-D6 LinReg1 and LinReg7 on the same ICs/seeds.
R=/export/scratch2/rik/time_series_TO/.claude/worktrees/overnight/lib/RikFlow
L=/ufs/rik/.claude/jobs/5801b24a/tmp/logs
V=${1:-LinReg7_ar2}
O=$R/analysis/output
cd "$R" || exit 1
export TMPDIR=/export/scratch2/rik/tmp D6_T_MIN=52.5 D6_T_MAX=70.75
for B in D6_LinReg7 D6_LinReg1; do
  julia --startup-file=no --project=analysis analysis/score_d6.jl --paired "$O/D6sel3_$V" "$O/$B" > "$L/eval_sel3_${V}_vs_$B.log" 2>&1
  echo "paired vs $B rc=$?"
done
for D in "D6sel3_$V" D6_LinReg7 D6_LinReg1; do
  D6_OUT=$O/$D julia --startup-file=no --project=analysis analysis/score_d6.jl > "$L/eval_sel3_score_$D.log" 2>&1
  echo "score $D rc=$?"
done
unset D6_T_MIN D6_T_MAX
julia --startup-file=no --project=training analysis/tail_census.jl "D6sel3_$V" D6_LinReg7 D6_LinReg1 --ics "D6sel3_$V" > "$L/eval_sel3_census_$V.log" 2>&1
echo "census rc=$?"
