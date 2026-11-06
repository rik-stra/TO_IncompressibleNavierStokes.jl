#!/bin/bash
#SBATCH -J cnn_tend
# Too little data, or nothing to learn? (Rik, 2026-11-03: "I do want to add train on 1-50 TU ... But I suggest we
# train on even more data 1-80 TU".) The plain step-3 CNN (no skip, no input noise) fitted on 1-TEND TU
# (RIKFLOW_MLP_TEND, step3_data.jl; stage 1 on its first 80 %, early stopping on the rest, refit on all of it), the
# weight-decay sweep of every loss as the 1-10 TU sweep (grid_cap: wd {0, .01, .03, .1, .3, 1} x seeds 1-3, cap 200k),
# 54 fits, 16 at a time on the CPU. Scored afterwards on 80-100 TU next to the 1-10 TU fits and least squares on each
# window. MODE=smoke: one RMSE fit, 300 updates, refit.
#SBATCH -t 03:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#
# Usage, from lib/RikFlow:
#     TEND=80 sbatch exp_square_HIT/batch_scripts/run_cnn_tend.sh
#     TEND=80 MODE=smoke sbatch -t 00:30:00 exp_square_HIT/batch_scripts/run_cnn_tend.sh
# Outputs under exp_square_HIT/output/TO_LSTM/step3/: tend<TEND>_<loss>_wd<wd>_s<seed> (and tend<TEND>_smoke), a .log
# beside each. CNN_SCRIPT names another copy of the fit tool (in exp_square_HIT/tools).

set -u

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export OPENBLAS_NUM_THREADS=1

if [ ! -f exp_square_HIT/3_track_ref.jl ]; then
    echo "run_cnn_tend.sh: run from lib/RikFlow" >&2
    exit 1
fi
TEND=${TEND:?set TEND, e.g. 80}
MODE=${MODE:-all}
SCRIPT=${CNN_SCRIPT:-exp_square_HIT/tools/cnn_mean_fit.jl}
LOGD=exp_square_HIT/output/TO_LSTM/step3
mkdir -p "$LOGD"

julia --project=training -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

JOBS=$(mktemp)
if [ "$MODE" = smoke ]; then
    echo "tend${TEND}_smoke RIKFLOW_CNN_LOSS=rmse RIKFLOW_CNN_WDS=0.3 RIKFLOW_CNN_SEEDS=1 RIKFLOW_CNN_MAXUPD=300 RIKFLOW_CNN_REFIT=1" >> "$JOBS"
else
    for l in nll crps rmse; do
        for w in 0 0.01 0.03 0.1 0.3 1; do
            for s in 1 2 3; do
                echo "tend${TEND}_${l}_wd${w}_s$s RIKFLOW_CNN_LOSS=$l RIKFLOW_CNN_WDS=$w RIKFLOW_CNN_SEEDS=$s RIKFLOW_CNN_MAXUPD=200000 RIKFLOW_CNN_REFIT=1" >> "$JOBS"
            done
        done
    done
fi
echo "== $SCRIPT: training window 1-$TEND TU, mode $MODE, $(wc -l < "$JOBS") fits, 16 at a time, CPU"
export SCRIPT TEND LOGD
xargs -P 16 -L 1 bash -c 'tag=$0; env "$@" RIKFLOW_MLP_TEND=$TEND RIKFLOW_CNN_DEVICE=cpu RIKFLOW_CNN_OUT=step3/$tag \
    julia --startup-file=no --project=training "$SCRIPT" > "$LOGD/$tag.log" 2>&1 || { echo "FAILED $tag"; exit 1; }' < "$JOBS"
STATUS=$?                                       # xargs: 123 when any fit failed, so an afterok dependent does not start
rm -f "$JOBS"
grep -h "training window\|upd, best\|refit on 1-10 TU\|ERROR" "$LOGD"/tend${TEND}_*.log | cut -c1-220 | head -80
exit $STATUS
