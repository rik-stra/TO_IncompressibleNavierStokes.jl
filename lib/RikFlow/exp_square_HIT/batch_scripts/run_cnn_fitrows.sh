#!/bin/bash
#SBATCH -J cnn_fitrows
# Does the step-3 CNN fit its training data better than the linear mean model once nothing holds it back?
# (Rik, 2026-11-02: "Without weight decay do the CNNs fit better on 1-10 TU?") The CNN of the sweep, RMSE
# loss, at wd 0, never stopped early (RIKFLOW_CNN_NOSTOP = 1: the learning rate still decays on the
# validation plateaus, to its floor of 1e-4), to MAXUPD updates. cnn_mean_fit.jl prints the error on the fit
# rows (1-8.2 TU) of the best and the last iterate against least squares on the same rows, and the training
# curve at its checkpoints. Three seeds in parallel, one process each, sharing the GPU (Rik, 2026-11-02: "make
# sure training uses the GPU on snellius"; RIKFLOW_CNN_DEVICE = gpu, Lux's Conv through cuDNN; the tool
# errors when no device is functional, never falls back to the CPU silently). CNN_DEVICE=cpu runs it on the
# CPU instead (the sweep's fits: 4-6 ms per update). The training environment needs cuDNN in its Manifest.
#SBATCH -t 01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#
# Usage, from lib/RikFlow:
#     sbatch exp_square_HIT/batch_scripts/run_cnn_fitrows.sh
#     MAXUPD=2000 sbatch -t 00:20:00 exp_square_HIT/batch_scripts/run_cnn_fitrows.sh    # a smoke
# CNN_SCRIPT names another copy of the fit tool (it must sit in exp_square_HIT/tools: it includes its
# neighbours by @__DIR__). Outputs: exp_square_HIT/output/TO_LSTM/step3/fitrows_rmse_wd0_s<seed>/fits.jld2
# and fitrows_rmse_wd0_s<seed>.log beside them.
# REFIT=1 adds the protocol's second training on all of 1-10 TU, for the whole stage-1 run (REFIT_UPD = last,
# its schedule replayed), and pred.jld2 (Table tab: net-loss's wd-0 row; Rik, 2026-11-02); outputs then
# fitrows_<loss>_wd0_refit_s<seed>. LOSS = rmse (default) | nll | crps (Rik, 2026-11-02: "I also want the other
# losses added without weightdecay").

set -u

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export OPENBLAS_NUM_THREADS=1

if [ ! -f exp_square_HIT/3_track_ref.jl ]; then
    echo "run_cnn_fitrows.sh: run from lib/RikFlow" >&2
    exit 1
fi
SCRIPT=${CNN_SCRIPT:-exp_square_HIT/tools/cnn_mean_fit.jl}
MAXUPD=${MAXUPD:-100000}
DEV=${CNN_DEVICE:-gpu}
REFIT=${REFIT:-0}
LOSS=${LOSS:-rmse}
# INNOISE=fixed INNOISE_SIGMA=<run> (Rik, 2026-11-03: the same table "for these models fitted with noise on the
# inputs"): trained on histories carrying draws of that run's Σ, as the noise-training sweep (results_LSTMS.md
# 36c: step3/cnn_nt_rmse_wd0.3_s1_r5, the round-5 fixed point); then the fit tool's SCORE pass on the CPU (the
# refitted networks along one noise path of the Σ they deploy, step3/innoise_scores/<run>.jld2).
INNOISE=${INNOISE:-none}
INNOISE_SIGMA=${INNOISE_SIGMA:-ls}
TAG=fitrows_$([ "$INNOISE" != none ] && echo nt_)${LOSS}_wd0$([ "$REFIT" = 1 ] && echo _refit)
LOGD=exp_square_HIT/output/TO_LSTM/step3
mkdir -p "$LOGD"

julia --project=training -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2
[ "$DEV" = gpu ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== $SCRIPT: loss $LOSS, wd 0, NOSTOP, $MAXUPD updates, seeds 1-3, device $DEV, refit $REFIT, input noise $INNOISE ($INNOISE_SIGMA)"
for s in 1 2 3; do
    RIKFLOW_CNN_DEVICE=$DEV RIKFLOW_CNN_LOSS=$LOSS RIKFLOW_CNN_WDS=0 RIKFLOW_CNN_NOSTOP=1 RIKFLOW_CNN_MAXUPD=$MAXUPD \
    RIKFLOW_CNN_INNOISE=$INNOISE RIKFLOW_CNN_INNOISE_SIGMA=$INNOISE_SIGMA \
    RIKFLOW_CNN_REFIT=$REFIT RIKFLOW_CNN_REFIT_UPD=last RIKFLOW_CNN_SEEDS=$s RIKFLOW_CNN_OUT=step3/${TAG}_s$s \
        julia --startup-file=no --project=training "$SCRIPT" > "$LOGD/${TAG}_s$s.log" 2>&1 &
done
wait
grep -h "upd, best\|fit rows\|last iterate\|refit\|input noise" "$LOGD"/${TAG}_s?.log
if [ "$INNOISE" != none ] && [ "$REFIT" = 1 ]; then
    RIKFLOW_CNN_DEVICE=cpu RIKFLOW_CNN_LOSS=$LOSS RIKFLOW_CNN_INNOISE=$INNOISE RIKFLOW_CNN_INNOISE_SIGMA=$INNOISE_SIGMA \
    RIKFLOW_CNN_SCORE=step3/${TAG}_s1,step3/${TAG}_s2,step3/${TAG}_s3 \
        julia --startup-file=no --project=training "$SCRIPT" > "$LOGD/${TAG}_score.log" 2>&1
    grep -h "^score\|ERROR" "$LOGD/${TAG}_score.log"
fi
