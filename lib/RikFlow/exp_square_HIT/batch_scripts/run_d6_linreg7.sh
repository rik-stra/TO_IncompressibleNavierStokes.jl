#!/bin/bash
# D6 production run: TO+LRS with LinReg7 (h = 5, lambda = 1). K = 90, M = 10.
#
#     sbatch batch_scripts/run_d6_linreg7.sh          # from exp_square_HIT/
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
# 🔴 **No validation here.** Ordinal 0's oracle is R2's own LinReg1 replica 1, so the validation
# belongs to run_d6_linreg1.sh. Running ordinal 0 under LinReg7 validates nothing past the warm-up:
# the replayed dQ is model-independent so the gate still passes, but the post-warm-up comparison is
# then against a different model's trajectory and reads as divergence that is really chaos plus a
# model difference. (That is exactly what the 2026-09-16 pilot did.) Run the LinReg1 validation,
# check it, then submit this array.
#
# Sizing, walltime and the depot rationale are in batch_scripts/run_d6.sh and tools/RUNBOOK.md;
# they are not repeated here, so that three production scripts cannot drift apart from each other.

#SBATCH -J d6-lr7
#SBATCH -t 30:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --array=1-179:2

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# 🔴 Its own directory. The three closures MUST NOT share one: the scorer globs `d6_online_ic*_m*`
# and has no way to tell which model wrote a file.
export D6_OUT=$PWD/output/D6_LinReg7
mkdir -p "$D6_OUT"

export D6_CLOSURE=lrs
export D6_MODEL=$PWD/output/TO_LRS/LinReg7/LinReg.jld2
export D6_MEMBERS=10

# Optional; run_d6.jl falls back to output/d6_ics and then to the local analysis build directory.
# 🔴 The packages were rebuilt 2026-09-16 (N_LEAD 1200, N_WARM 100, k in [42, 388]). Anything still
# holding the pre-2026-09-16 set runs a length the scorer no longer expects.
# export D6_IC_DIR=$PWD/output/d6_ics

julia --project tools/run_d6.jl $SLURM_ARRAY_TASK_ID
