#!/bin/bash
# Tail check (a): LinReg7_ar2 over the full 3 TU horizon on the 19 selection ICs D6_LinReg1/7 already cover (odd ordinals 89-125, M=10).
R=/export/scratch2/rik/time_series_TO/.claude/worktrees/overnight/lib/RikFlow
cd $R/exp_square_HIT || exit 1
export TMPDIR=/export/scratch2/rik/tmp JULIA_CUDA_SOFT_MEMORY_LIMIT=4GiB
D6_CLOSURE=lrs D6_MODEL=$R/exp_square_HIT/output/TO_LRS/${V:-LinReg7_ar2}/LinReg.jld2 \
D6_T_MIN=52.5 D6_T_MAX=70.75 D6_STRIDE=2 D6_MEMBERS=10 D6_OUT=$R/analysis/output/D6sel3_${V:-LinReg7_ar2} \
  julia --startup-file=no --project=.. tools/run_d6.jl --all
