#!/bin/bash
#SBATCH -J lrs_train
#SBATCH -t 00:20:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# Fit ONE TO-LRS configuration. Separated from the online run so the ensemble can go out as an
# array (`run_online_array.sh`) without five tasks racing to write the same `LinReg.jld2`.
#
# Usage:
#     sbatch batch_scripts/run_train_lrs.sh 5      # fit LinReg5
#
# Normally submitted through `submit_lrs.sh`, which chains this and the array with the right
# dependency. Use this on its own only to refit.
#
# ⚠️ It asks for a GPU and does not need one -- the fit is a 3595 x 67 least-squares solve that
# takes under a second. `using RikFlow` pulls CUDA in as a hard dependency, and a CPU partition
# would be a new failure mode (CUDA.jl initialising with no device) for a job that costs twenty
# minutes of walltime at most. Move it to a CPU partition once someone has checked that import
# actually works there; the saving is real but it is not worth a round-trip to find out.
#
# 🔴 Run `4_setup_search.jl` first if the table has changed. It writes
# output/TO_LRS/inputs_example.jld2 and needs no GPU:
#     julia --project exp_square_HIT/4_setup_search.jl

set -u

INDEX=${1:-}
if ! [[ "$INDEX" =~ ^[0-9]+$ ]]; then
    echo "run_train_lrs.sh: first argument must be a model index, e.g. 5" >&2
    exit 1
fi

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

if [ -f 3_track_ref.jl ]; then
    EXP=.
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=exp_square_HIT
else
    echo "run_train_lrs.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

echo "== training LinReg$INDEX"
julia --project "$EXP/5_train_LinReg.jl" "$INDEX"
