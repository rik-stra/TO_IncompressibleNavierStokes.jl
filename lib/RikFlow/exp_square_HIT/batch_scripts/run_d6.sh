#!/bin/bash
# D6 SHAKEDOWN: validation and pilot only. One array task = one IC = M members, run sequentially.
#
# 🔴 **This is NOT the production script any more (2026-09-16).** Production is three scripts, one
# per closure, each with its own output directory, all at `--array=1-179:2` (K = 90, M = 10):
#
#     batch_scripts/run_d6_linreg1.sh   -> output/D6_LinReg1   (and it owns the validation)
#     batch_scripts/run_d6_linreg7.sh   -> output/D6_LinReg7
#     batch_scripts/run_d6_ddn.sh       -> output/D6_DDN
#
# This file stays because `output/D6` holds the 2026-09-16 LinReg7 pilot and because the sizing,
# walltime and depot rationale below is the single copy the three production scripts point at.
# ⚠️ Its D6_MODEL is LinReg7 and its D6_OUT is `output/D6`, which is the pilot's directory, not a
# production one. Do not submit a K = 90 array from here — it would land on top of the pilot.
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
# differ by 10x and neither should be trusted. Then submit the three production scripts above;
# do NOT widen this script's own array.
#
#
# Walltime, re-derived 2026-09-16 for N_LEAD = 1200 (was 2172, and 1208 before that). Paper 2's
# Appendix G gives ~4.1 s/TU per member wall including setup and the write; at nt = 1420 steps =
# 3.55 TU that is ~15 s/member, so ~2.5 min of stepping for M = 10, plus roughly 6 min of
# first-member GPU compilation -- about 9 min per task. Compilation now dominates, which is why
# 30 minutes is kept unchanged rather than scaled down with the run: it was ~3x margin at 2172 and
# is more now. Tighten it once the pilot has measured the real rate, and note the pilot's own first
# task carries the compilation for nothing else.

#SBATCH -J d6
#SBATCH -t 30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=0-5

# No -o/-e: on Snellius an array job already gets one slurm-%A_%a.out per task by default (Rik,
# 2026-09-14), so redirecting into logs/ bought a nicer name and a real trap -- SLURM opens those
# files before this script body runs, so the `mkdir` below was always too late for them, and logs/
# is not in the repository.

# Depot with the trailing colon, as every other script in this repository has it.
#
# 🔴 Moved to gpu_h100 + julia_h100 on 2026-09-16 (Rik). This was the only script in the directory
# on gpu_a100 and the only one on the julia_a1003 depot, while RUNBOOK.md tells the operator to
# export julia_h100 -- so a task could land on a depot nobody had warmed and precompile from
# scratch inside the walltime. JULIA_CPU_TARGET multiversioning means one depot serves both
# partitions, so there was never a reason for the split.
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:

# 🔴 Set explicitly, and it is not decoration: the h100 depot's caches were built by the online
# scripts WITH this target, and Julia validates a cache against the target it was compiled for. A
# task that omitted it could reject those caches and recompile from scratch inside the walltime --
# the same failure the depot move above exists to prevent. Identical string to
# run_online_array.sh and run_train_lrs.sh; keep them in step.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# (Historical: the handoff said /scratch-shared/$USER/.julia_a100: ; the repository's own scripts
# use $HOME/julia/julia_<gpu>. `julia_a1003` was this script's own artifact, kept only because the
# 2026-09-11 D6 run populated it; that run was against the archive and is superseded.)
mkdir -p output/D6

# Optional; run_d6.jl falls back to output/d6_ics and then to the local analysis build directory.
# export D6_IC_DIR=$PWD/output/d6_ics
export D6_MODEL=$PWD/output/TO_LRS/LinReg7/LinReg.jld2

julia --project tools/run_d6.jl $SLURM_ARRAY_TASK_ID
