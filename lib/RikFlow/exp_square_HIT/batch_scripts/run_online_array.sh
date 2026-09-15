#!/bin/bash
#SBATCH -J lrs_online
#SBATCH -t 03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=1-5

# One TO-LRS ensemble member per array task: 64^3 LF, Float64, LMWray3, 100 TU (40,000 steps).
# `SLURM_ARRAY_TASK_ID` IS the replica index.
#
# Usage:
#     sbatch --dependency=afterok:<trainjob> batch_scripts/run_online_array.sh 5
#     sbatch --array=1-5 batch_scripts/run_online_array.sh 5     # override the member list
#
# Normally submitted through `submit_lrs.sh`, which fits the model first and sets the dependency.
#
# 🔴 THE MODEL MUST ALREADY BE FITTED. This script does NOT train -- `run_online.sh` did both in one
# job, which cannot become an array without five tasks writing the same `LinReg.jld2` at once. If
# `output/TO_LRS/LinReg<n>/LinReg.jld2` is absent the tasks fail immediately, and if it is STALE
# they all silently deploy the previous fit. `parameters.jld2` beside it now records `lambda`,
# `ridge_solver` and `penalize_intercept`, so check that rather than the timestamp.
#
# 🔑 **An array member is bit-identical to the same member of the serial loop.** The only
# per-replica state is `Xoshiro(seeds.to + i + 2)`; the OU force cache is rebuilt per call and
# `solve_unsteady` deep-copies `ustart`. So task 3 writes exactly the `..._replica3.jld2` that
# `6_online_TO_LRS.jl <n>` would have written third, and an array and a serial run may be mixed in
# one ensemble. See the comment at `6_online_TO_LRS.jl:22`.
#
# ⚠️ No `-o`/`-e`: on Snellius an array job already gets one `slurm-%A_%a.out` per task by default
# (Rik, 2026-09-14), and SLURM opens those files before this script body runs, so a `mkdir` here
# would always be too late for them.
#
# ⚠️ WALL TIME IS AN ESTIMATE. Paper 2's Appendix G gives 5.8 s/TU for TO LRS, so 100 TU is roughly
# ten minutes -- on a different machine, in Float32, under RK44. Three hours is generous on purpose
# and SLURM bills time used. R2's serial five-replica job is the real yardstick; record the rate.

set -u

INDEX=${1:-}
if ! [[ "$INDEX" =~ ^[0-9]+$ ]]; then
    echo "run_online_array.sh: first argument must be a model index, e.g. 5" >&2
    exit 1
fi

REPLICA=${SLURM_ARRAY_TASK_ID:-}
if [ -z "$REPLICA" ]; then
    echo "run_online_array.sh: no SLURM_ARRAY_TASK_ID -- submit with sbatch --array, not bash" >&2
    exit 1
fi

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

if [ -f 3_track_ref.jl ]; then
    EXP=.
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=exp_square_HIT
else
    echo "run_online_array.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi

# Every task instantiates. Harmless once the depot is warm, and the alternative -- assuming the
# train job warmed it -- breaks whenever the array is submitted on its own.
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

echo "== LinReg$INDEX, replica $REPLICA of array $SLURM_ARRAY_JOB_ID, 100 TU"
julia --project "$EXP/6_online_TO_LRS.jl" "$INDEX" "$REPLICA"
