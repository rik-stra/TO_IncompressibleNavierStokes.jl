#!/bin/bash
# D6 production run: the DDN (paper 1's data-driven noise model, the negative control). K = 90, M = 10.
#
#     sbatch batch_scripts/run_d6_ddn.sh              # from exp_square_HIT/
#
# One array task = one initial condition = M members run sequentially. $SLURM_ARRAY_TASK_ID is the
# **ordinal**, not the field index k; run_d6.jl maps one to the other through select_ics.
#
# 🔑 `--array=1-179:2` is K = 90: every second ordinal of the 180 packages, the same IC set the two
# LRS runs use. That is what makes D6 **paired** -- every closure forecasts from the same initial
# condition, so realisation variance cancels in the comparison. Do not change the range in one of
# the three scripts without changing it in all three.
#
# 🔴 **No D6_MODEL.** The DDN has no fitted file: `MVG_sampler` fits an MvNormal to a slice of the
# tracked dQ at construction. run_d6.jl reads that slice from `d6_ddn_traindata.jld2` in the IC
# directory (override with D6_DDN_DATA), which build_d6_ics.jl writes from DDN_TRAIN_RANGE = 400:4000.
#
# 🔴 **No validation here.** Ordinal 0's oracle is R2's own LinReg1 replica 1, so the validation
# belongs to run_d6_linreg1.sh and means nothing for a different closure past the warm-up.
#
# ⚠️ The DDN takes the same replayed warm-up as the LRS runs (`spinnup_data`, added 2026-09-16), so
# all three start their forecasts from the same state. Without it the DDN would have been sampling
# from step 1 while the LRS runs were still replaying, and the comparison would not have been paired.
# ⚠️ The DDN has no turbulence gate. That is a real defect on Taylor-Green and secondary on HIT,
# which starts from a spun-up field; see claude_memory.md gotcha #65.
#
# 🔑 **To rerun individual ordinals** (a diverged member, a lost task), override the array on
# the command line so this script's own D6_OUT and closure config are reused:
#
#     sbatch --array=<ordinals> batch_scripts/run_d6_ddn.sh
#
# `run_ic` skips members whose files exist, before the seed is derived, so only the missing
# ones run and every member keeps its seed. DDN had no divergences on 2026-09-16.
#
# Sizing, walltime and the depot rationale are in batch_scripts/run_d6.sh and tools/RUNBOOK.md;
# they are not repeated here, so that three production scripts cannot drift apart from each other.

#SBATCH -J d6-ddn
#SBATCH -t 30:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --array=1-179:2

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# 🔴 Its own directory. The three closures MUST NOT share one: the scorer globs `d6_online_ic*_m*`
# and has no way to tell which model wrote a file.
export D6_OUT=$PWD/output/D6_DDN
mkdir -p "$D6_OUT"

export D6_CLOSURE=ddn
export D6_MEMBERS=10

# Optional; run_d6.jl falls back to output/d6_ics and then to the local analysis build directory.
# 🔴 The packages were rebuilt 2026-09-16 (N_LEAD 1200, N_WARM 100, k in [42, 388]). Anything still
# holding the pre-2026-09-16 set runs a length the scorer no longer expects.
# export D6_IC_DIR=$PWD/output/d6_ics
# export D6_DDN_DATA=$PWD/output/d6_ics/d6_ddn_traindata.jld2

julia --project tools/run_d6.jl $SLURM_ARRAY_TASK_ID
