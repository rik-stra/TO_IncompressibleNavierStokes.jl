#!/bin/bash
# The TRAINING hindcast of a NETWORK-path closure (2026-10-23, step 3a: the CNN closures of
# tools/cnn_closure_export.jl): run_d6_train.sh with D6_CLOSURE=lstm -- 5 ICs inside the fit window x 10
# members, one array task per closure.
#
#     sbatch --array=1-<n> batch_scripts/run_d6_train_lstm.sh <fit dir> [<fit dir> ...]   # from exp_square_HIT/
#
# e.g.  sbatch --array=1 batch_scripts/run_d6_train_lstm.sh step3/closures/CNN_rmse_mvg
#
# Task i runs closure `${!i}`: a fit directory under output/TO_LSTM holding StochLSTM_seed1.jld2 (seed 1,
# the paper's rule; `resolve_lstm_model`), all 5 training ICs (`run_d6.jl 1,2,3,4,5`) in one process, into
# output/D6train_<basename>. IC packages and member seeds as run_d6_train.sh.
#
# 🔴 Settings are set HERE, from the arguments, never from the submitting environment
# (claude_memory.md #78), and printed in the log.

#SBATCH -J d6-train-lstm
#SBATCH -t 01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
set -u
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

TASK=${SLURM_ARRAY_TASK_ID:?run as an array job: sbatch --array=1-<n> ... <fit dirs>}
[ "$#" -ge "$TASK" ] || { echo "run_d6_train_lstm.sh: task $TASK but only $# fit dirs given" >&2; exit 1; }
FIT=${!TASK}
[ -f "$PWD/output/TO_LSTM/$FIT/StochLSTM_seed1.jld2" ] ||
    { echo "run_d6_train_lstm.sh: no StochLSTM_seed1.jld2 under output/TO_LSTM/$FIT" >&2; exit 1; }
NAME=$(basename "$FIT")

export D6_IC_DIR=$PWD/output/d6_ics_train
export D6_OUT=$PWD/output/D6train_$NAME
export D6_CLOSURE=lstm
export D6_MODEL=$FIT
export D6_MEMBERS=10
unset D6_NLEAD D6_BLOCK D6_T_MIN D6_T_MAX D6_STRIDE RIKFLOW_GATE RIKFLOW_DEPLOY_SEED
mkdir -p "$D6_OUT"
echo "run_d6_train_lstm.sh: task $TASK closure $NAME model $D6_MODEL (lstm) ICs $D6_IC_DIR out $D6_OUT members $D6_MEMBERS"
echo "run_d6_train_lstm.sh: code $(git -C "$PWD" rev-parse --short HEAD) in $PWD"

julia --project tools/run_d6.jl 1,2,3,4,5
