#!/bin/bash
#SBATCH -J m4_train
#SBATCH -t 02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
# 🔑 Explicit, though it is also SLURM's default: the job inherits the SUBMITTING environment.
# That is what makes `RIKFLOW_M4_UPDATES=5 sbatch ...` and `M4_DEVICE=cpu sbatch ...` work, and it
# is what the other scripts here have always relied on -- none of them sets `--export`, and they
# all need an inherited `PATH` just to find `julia`. Stating it protects against a site default of
# `NONE`, which would strip both the budget AND the PATH.
# 🔴 **Do NOT write `--export=VAR=value` on the command line: that REPLACES `ALL`**, so the job
# would lose `PATH` and die before Julia starts. The additive form is `--export=ALL,VAR=value`.
#SBATCH --export=ALL

# Fit ONE M4 configuration -- all of its seeds, or one seed if given.
#
# Usage:
#     sbatch batch_scripts/run_train_lstm.sh 4          # fit StochLSTM4, all n_seeds
#     sbatch batch_scripts/run_train_lstm.sh 4 3        # fit StochLSTM4, seed 3 only
#
# S6 asks for five training seeds per neural cell with the spread reported and the median-seed fit
# deployed. The second argument exists so an array can put the seeds on separate tasks; the
# median is computed by whichever run sees all five, so an array must be followed by one
# `11_train_StochLSTM.jl` pass or the summary will be missing and `12_online_StochLSTM.jl` will
# fall back to seed 1.
#
# ⚠️ **Runs under `--project=training`, not `--project`.** Training needs the Lux extension, which
# is triggered by Lux + Optimisers + Zygote together. The deployed closure needs none of them.
#
# 🔴 **It now TRAINS ON THE GPU: `M4_DEVICE=cuda` by default since 2026-09-18.** Until then it
# asked for a GPU and ran on the node's CPU, because nothing in the training path moved an array to
# a device. `train_stochlstm` gained a `device` keyword and `RikFlow.m4_device` resolves
# `M4_DEVICE`; override per submission with
#
#     M4_DEVICE=cpu sbatch batch_scripts/run_train_lstm.sh 19
#
# 🔴 **The GPU path has never actually run on a GPU.** It is verified against `JLArrays`, which
# refuses scalar indexing exactly as `CuArray` does (V54: all four architectures agree with the CPU
# fit to ~1e-7), but that cannot see a CUDA compilation failure -- the class this repository has
# been bitten by twice (`claude_memory.md` #56, #57), both times only on the cluster.
# 🔑 **Make the first submission a test:** `RIKFLOW_M4_EPOCHS=5 sbatch ... 19` runs in about a
# minute and exercises the whole path including the write. `m4_device` refuses `cuda` when no
# device is functional rather than falling back to the host, so a mis-scheduled job fails at load
# instead of reporting CPU time as GPU time.
# ⚠️ Expect it to be slower than the CPU at the current geometry -- `results_LSTMS.md` §5 measures
# the fit as overhead-bound, running at ~1% of one CPU core's arithmetic over a strictly sequential
# recurrence. The case where it could pay is a short `stride` with `batch = 32` (§6.2).
#
# 🔴 Run `10_setup_lstm.jl` first if the table has changed. It needs no GPU:
#     julia --project exp_square_HIT/10_setup_lstm.jl

set -u

INDEX=${1:-}
SEED=${2:-}
if ! [[ "$INDEX" =~ ^[0-9]+$ ]]; then
    echo "run_train_lstm.sh: first argument must be a model index, e.g. 4" >&2
    exit 1
fi

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# Train on the device unless the submitting shell says otherwise. SLURM's default `--export=ALL`
# carries `M4_DEVICE=cpu sbatch ...` through, so this is a default, not an override.
export M4_DEVICE=${M4_DEVICE:-cuda}
# The batches are assembled on the host whichever device trains, and on 64 x 23 matrices a full
# BLAS pool contends rather than helps.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

if [ -f 3_track_ref.jl ]; then
    EXP=.
    ROOT=..
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=exp_square_HIT
    ROOT=.
else
    echo "run_train_lstm.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi

julia --project="$ROOT/training" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

echo "== training StochLSTM$INDEX ${SEED:+(seed $SEED)} on device $M4_DEVICE"
julia --project="$ROOT/training" "$EXP/11_train_StochLSTM.jl" "$INDEX" $SEED
