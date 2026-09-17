#!/bin/bash
#SBATCH -J m4_train
#SBATCH -t 02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

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
# ⚠️ It asks for a GPU and does not currently use one: the model is ~10^4 parameters on 6 QoIs and
# trains on CPU in minutes. The request is here for the same reason `run_train_lrs.sh` has one --
# `using RikFlow` pulls CUDA in as a hard dependency and a CPU partition is an untested failure
# mode. Move it once someone has checked that import works there.
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

echo "== training StochLSTM$INDEX ${SEED:+(seed $SEED)}"
julia --project="$ROOT/training" "$EXP/11_train_StochLSTM.jl" "$INDEX" $SEED
