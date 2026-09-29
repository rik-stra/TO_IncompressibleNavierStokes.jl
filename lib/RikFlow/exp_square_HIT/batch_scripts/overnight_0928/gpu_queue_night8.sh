#!/bin/bash
# Overnight GPU queue v8 (2026-09-29 ~05:15). The 3 TU check did not confirm Splice1_E0x7_ar2's lead
# (§13g), so the decisive test is M = 10 on all 46 selection ICs for LinReg1 and the splice: members
# 1-5 are hard links to the M = 5 runs (same seeds), run_d6.jl skips existing members, so only 6-10 run.
R=/export/scratch2/rik/time_series_TO/.claude/worktrees/overnight/lib/RikFlow
L=/ufs/rik/.claude/jobs/5801b24a/tmp/logs
cd "$R/exp_square_HIT" || exit 1
export TMPDIR=/export/scratch2/rik/tmp JULIA_CUDA_SOFT_MEMORY_LIMIT=4GiB
while pgrep -f "tools/run_d6.jl" > /dev/null; do sleep 60; done
run() {  # closure model outname [members] [dir prefix]
  local M=${4:-5} P=${5:-D6mini}
  local out="$R/analysis/output/${P}_$3" n=$((46 * ${4:-5}))
  if [ -d "$out" ] && [ "$(ls "$out" | wc -l)" -ge $n ]; then echo "$(date +%T) skip $P $3 (complete)"; return; fi
  if [ "$1" = lrs ] && [ ! -f "$2" ]; then echo "$(date +%T) MISSING model $2, skipping $3"; return; fi
  echo "$(date +%T) start $P $3 (M = $M)"
  D6_CLOSURE=$1 D6_MODEL=$2 D6_BLOCK=selection D6_MEMBERS=$M D6_NLEAD=400 D6_OUT="$out" \
    julia --startup-file=no --project=.. tools/run_d6.jl --all > "$L/${P}_$3.log" 2>&1
  echo "$(date +%T) done $P $3 rc=$? files=$(ls "$out" | wc -l)"
}
LRS=$R/exp_square_HIT/output/TO_LRS
run lstm diag/r2_lin_const_h2 lstm_r2_lin_const_h2
run lstm diag/r3_lin_sd_h2 lstm_r3_lin_sd_h2
run lrs "$LRS/LinReg1/LinReg.jld2" LinReg1 10 D6mini10
run lrs "$LRS/Splice1_E0x7_ar2/LinReg.jld2" Splice1_E0x7_ar2 10 D6mini10
for V in LinReg12 LinReg2 LinReg2_ar2 LinReg8_ar2 LinReg8; do
  run lrs "$LRS/$V/LinReg.jld2" "$V"
done
echo "$(date +%T) queue finished"
