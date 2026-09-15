#!/bin/bash
# D6: the multi-IC ensemble. One array task = one initial condition = M members, run sequentially.
#
# Submit from exp_square_HIT/ , as the other scripts in this directory are:
#     sbatch batch_scripts/run_d6.sh
#
# $SLURM_ARRAY_TASK_ID is the **ordinal** (1..K) of the initial condition, NOT the field index k.
# run_d6.jl maps ordinal -> k through select_ics, so the pilot's 5 ICs are the first 5 of the same
# 180 the full run uses and scaling up renumbers nothing.
#
# 🔑 VALIDATION FIRST, and it is one task: `--array=0`, or just `julia --project tools/run_d6.jl 0`.
# Ordinal 0 is `fields[1]` of the 10 TU tracked record -- the initial condition every archived
# online run launched from -- so n_k = 0, ou_advance = 0 (the identity point of the replay) and the
# model seeds are the archive's own, Xoshiro(236 + member). Its `q` must then reproduce the archived
# LinReg1 replica's first 1309 columns; `compare_validation` in analysis/score_d6.jl checks it.
# That is the correctness check on the whole D6 path against a trajectory produced by different
# code years earlier, and it costs one task. Do it before the pilot.
#
# PILOT next: --array=1-5, ~17 MB of IC packages to copy. Read the wall time from the logs and
# write the measured s/TU into meta_files/handoff_p2c_d6.md section 2 -- the plan's two SBU figures
# differ by 10x and neither should be trusted. Then change to --array=1-180 (add %20 to cap
# concurrency if the queue prefers it) and change nothing else.
#
# ~19 s per member is the extrapolation from plan P2's one measured smoke run (1 TU, 400 steps,
# 5.8 s), so ~3.5 min of compute per task plus Julia startup and compilation. 30 minutes is
# generous; tighten it once the pilot has measured it.

#SBATCH -J d6
#SBATCH -t 30:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --array=1-5

# No -o/-e: on Snellius an array job already gets one slurm-%A_%a.out per task by default (Rik,
# 2026-09-14), so redirecting into logs/ bought a nicer name and a real trap -- SLURM opens those
# files before this script body runs, so the `mkdir` below was always too late for them, and logs/
# is not in the repository.

# Depot with the trailing colon, as every other script in this repository has it. ⚠️ The handoff
# said /scratch-shared/$USER/.julia_a100: ; the repository's own scripts use $HOME/julia/julia_<gpu>
# and that is what is followed here.
export JULIA_DEPOT_PATH=$HOME/julia/julia_a1003:

# ⚠️ The depot above does not follow this directory's convention: every other script uses
# $HOME/julia/julia_h100, and JULIA_CPU_TARGET multiversioning means one depot serves both
# partitions. `julia_a1003` is an artifact and is kept only because the 2026-09-11 D6 run populated
# it; consolidating is a one-line change whenever someone is willing to pay one precompile.
mkdir -p output/D6

# Optional; run_d6.jl falls back to output/d6_ics and then to the local analysis build directory.
# export D6_IC_DIR=$PWD/output/d6_ics
# export D6_MODEL=$PWD/output/TO_LRS/LinReg1/LinReg.jld2

julia --project tools/run_d6.jl $SLURM_ARRAY_TASK_ID
