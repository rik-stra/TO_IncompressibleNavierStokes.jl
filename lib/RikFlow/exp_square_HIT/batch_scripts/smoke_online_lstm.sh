#!/bin/bash
# The long-run SMOKE of network-path closures (WORKFLOW.md Stage 3 / GATE 3; 2026-10-23, step 3a): one
# replica of 12_online_StochLSTM.jl per fit dir at 0.5 TU -- the 100-step warm-up plus 100 forecast
# steps -- all at once on the job's one GPU, then tools/smoke_online_check.jl.
#
#     sbatch batch_scripts/smoke_online_lstm.sh <fit dir> [<fit dir> ...]   # from exp_square_HIT/
#
# e.g.  sbatch batch_scripts/smoke_online_lstm.sh step3/closures/CNN_rmse_mvg step3/closures/CNN_rmse_pl
#
# `<fit dir>` is under output/TO_LSTM, holding StochLSTM_seed1.jld2, as for run_d6_train_lstm.sh. It is
# deployed under model_index 2 (the StochLSTM2 row: replica seeds, as the production long runs).
# Output, the log and the replica (moved out of the fit dir by the check), in output/smoke_online/<name>/;
# delete it before production (#48). The hindcast half of GATE 3 is the training hindcast
# (run_d6_train_lstm.sh): the same run_d6.jl, with its own warm-up check, on the production device.
#
# 🔴 Settings are set HERE, never from the submitting environment (claude_memory.md #78), and printed.

#SBATCH -J smoke-online-lstm
#SBATCH -t 00:20:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
set -u
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

[ "$#" -ge 1 ] || { echo "usage: sbatch batch_scripts/smoke_online_lstm.sh <fit dir> [<fit dir> ...]" >&2; exit 1; }
[ -f 12_online_StochLSTM.jl ] || { echo "smoke_online_lstm.sh: submit from exp_square_HIT" >&2; exit 1; }
for FIT in "$@"; do
    [ -f "$PWD/output/TO_LSTM/$FIT/StochLSTM_seed1.jld2" ] ||
        { echo "smoke_online_lstm.sh: no StochLSTM_seed1.jld2 under output/TO_LSTM/$FIT" >&2; exit 1; }
done

TSIM=0.5
INDEX=2
unset RIKFLOW_ONLINE_IC RIKFLOW_ONLINE_DQ RIKFLOW_ONLINE_DEVICE RIKFLOW_TRACK_FILE RIKFLOW_M4_NWARM \
      RIKFLOW_PERT RIKFLOW_M4_UNTIED RIKFLOW_DEPLOY_SEED
echo "smoke_online_lstm.sh: tsim $TSIM TU, model_index $INDEX, replica 1, fits $*"
echo "smoke_online_lstm.sh: code $(git -C "$PWD" rev-parse --short HEAD) in $PWD"

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the runs will compile in-process (slower start)" >&2

pids=()
for FIT in "$@"; do
    NAME=$(basename "$FIT")
    mkdir -p "$PWD/output/smoke_online/$NAME"
    RIKFLOW_ONLINE_TSIM=$TSIM RIKFLOW_ONLINE_TAG=smoke RIKFLOW_M4_MODEL_DIR=$PWD/output/TO_LSTM/$FIT \
        julia --project 12_online_StochLSTM.jl $INDEX 1 > "$PWD/output/smoke_online/$NAME/run.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

for FIT in "$@"; do
    NAME=$(basename "$FIT")
    echo "== $NAME"
    grep -A4 -E "Deploying|Emission noise|ERROR" "$PWD/output/smoke_online/$NAME/run.log" | grep -v "^--$"
done
julia --project tools/smoke_online_check.jl $TSIM "$@"
