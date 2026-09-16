#!/bin/bash
# D6 production run: TO+LRS with LinReg1 (h = 5, lambda = 0). K = 90, M = 10.
#
#     sbatch batch_scripts/run_d6_linreg1.sh          # from exp_square_HIT/
#
# One array task = one initial condition = M members run sequentially. $SLURM_ARRAY_TASK_ID is the
# **ordinal**, not the field index k; run_d6.jl maps one to the other through select_ics.
#
# 🔑 `--array=1-179:2` is K = 90: every second ordinal of the 180 packages. select_ics is strictly
# monotone, so that is exactly the half-density IC set -- spacing 0.97 TU, 1.79x the slowest level
# timescale. At the full K = 180 the spacing is 0.4832 TU, i.e. 0.89x, and adjacent ICs would be
# genuinely correlated. `--array=2-180:2` is the fill-in if the extra density is ever wanted.
# Add `%20` after the range if the queue prefers a concurrency cap.
#
# 🔴 This script is the one that owns the VALIDATION, because ordinal 0's oracle is R2's own
# LinReg1 replica 1. Run it first, as its own one-task submit, and check it before the array:
#
#     sbatch --array=0 batch_scripts/run_d6_linreg1.sh
#
# It writes `d6_valid_ic1_m*.jld2`, a different prefix from the scored `d6_online_*`, so it shares
# this directory without the scorer's glob seeing it. ⚠️ Running ordinal 0 under LinReg7 or the DDN
# validates nothing past the warm-up: the replayed dQ is model-independent and the gate still
# passes, but the post-warm-up comparison is then against a different model's trajectory.
#
# Sizing, walltime and the depot rationale are in batch_scripts/run_d6.sh and tools/RUNBOOK.md;
# they are not repeated here, so that three production scripts cannot drift apart from each other.

#SBATCH -J d6-lr1
#SBATCH -t 30:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --array=1-179:2

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# 🔴 Its own directory. The three closures MUST NOT share one: the scorer globs `d6_online_ic*_m*`
# and has no way to tell which model wrote a file.
export D6_OUT=$PWD/output/D6_LinReg1
mkdir -p "$D6_OUT"

export D6_CLOSURE=lrs
export D6_MODEL=$PWD/output/TO_LRS/LinReg1/LinReg.jld2
export D6_MEMBERS=10

# Optional; run_d6.jl falls back to output/d6_ics and then to the local analysis build directory.
# 🔴 The packages were rebuilt 2026-09-16 (N_LEAD 1200, N_WARM 100, k in [42, 388]). Anything still
# holding the pre-2026-09-16 set runs a length the scorer no longer expects.
# export D6_IC_DIR=$PWD/output/d6_ics

julia --project tools/run_d6.jl $SLURM_ARRAY_TASK_ID
