#!/bin/bash
#SBATCH -J cnn_skip
# Step 4 offline: the step-3 CNN as a correction on a FROZEN LinReg (RIKFLOW_MLP_SKIP, step3_data.jl; Rik,
# 2026-11-03: "frozen but immediatly do both baselines so try both linreg1 and linreg7"). For one base, all on
# the CPU (Rik: "run on cpu if that is faster"; 3.5 ms per update against 4.9-6.5 on the GPU), 16 fits at a time:
#   MODE=all (default): the weight-decay sweep of every loss, as the clean sweep (grid_cap: wd {0, .01, .03, .1,
#     .3, 1} x seeds 1-3, cap 200k, refit on 1-10 TU for the best iterate's updates), and wd 0 without early
#     stopping (100k updates, refit for all of them), 63 fits;
#   MODE=smoke: one RMSE fit, 300 updates, refit (checks the skip end to end); with INNOISE also one of 1 update,
#     whose best iterate is update 0, the skip alone: its S_deploy / injected sd must be ~1 for a self-consistent base.
# INNOISE=fixed INNOISE_SIGMA=<run | lrs:<LinReg>> (Rik, 2026-11-03: "LinReg1_nt skip"): trained on histories that
# carry draws of that Σ, the skip reading them too (step3_data.jl `skip_dy`); tags skipnt_<base>_...
# INNOISE=head HEAD=inj (likelihood and CRPS only; Rik, 2026-11-03: "please switch to head mode"): the noise follows the
#   network's learned head (stop-gradient), the head starting at INNOISE_SIGMA's Σ; no outer loop. Tags skipnth_<base>_...
# LOSSES="nll crps" runs a subset of the losses (default all three); the smoke uses the first.
#   MODE=round ROUND_FROM=<tag stem without _s<seed>> LOSS=<loss> WD=<wd>: the outer loop's next round at one setting,
#     each seed injecting the S_deploy of its own run <stem>_s<seed>; tag <stem>_r<k> with ROUND=<k>.
#SBATCH -t 02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#
# Usage, from lib/RikFlow:
#     BASE=LinReg1 sbatch exp_square_HIT/batch_scripts/run_cnn_skip.sh
#     BASE=LinReg7 MODE=smoke sbatch -t 00:30:00 exp_square_HIT/batch_scripts/run_cnn_skip.sh
# CNN_SCRIPT names another copy of the fit tool (in exp_square_HIT/tools). Outputs under
# exp_square_HIT/output/TO_LSTM/step3/: skip_<base>_<loss>_wd<wd>_s<seed> (sweep), skip_<base>_<loss>_wd0_nostop_s<seed>,
# skip_<base>_smoke; a .log beside each.

set -u

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export OPENBLAS_NUM_THREADS=1

if [ ! -f exp_square_HIT/3_track_ref.jl ]; then
    echo "run_cnn_skip.sh: run from lib/RikFlow" >&2
    exit 1
fi
BASE=${BASE:?set BASE=LinReg1 or LinReg7}
MODE=${MODE:-all}
INNOISE=${INNOISE:-none}
INNOISE_SIGMA=${INNOISE_SIGMA:-ls}
HEAD=${HEAD:-identity}
LOSSES=${LOSSES:-nll crps rmse}
case "$INNOISE" in none) PFX=skip_${BASE};; head) PFX=skipnth_${BASE};; *) PFX=skipnt_${BASE};; esac
L1=${LOSSES%% *}
SCRIPT=${CNN_SCRIPT:-exp_square_HIT/tools/cnn_mean_fit.jl}
LOGD=exp_square_HIT/output/TO_LSTM/step3
mkdir -p "$LOGD"

julia --project=training -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

# one fit per line: "<out tag> <env assignments...>"
JOBS=$(mktemp)
NZ="RIKFLOW_CNN_INNOISE=$INNOISE RIKFLOW_CNN_INNOISE_SIGMA=$INNOISE_SIGMA RIKFLOW_CNN_HEAD=$HEAD"
if [ "$MODE" = smoke ]; then
    echo "${PFX}_smoke $NZ RIKFLOW_CNN_LOSS=$L1 RIKFLOW_CNN_WDS=0.3 RIKFLOW_CNN_SEEDS=1 RIKFLOW_CNN_MAXUPD=300 RIKFLOW_CNN_REFIT=1" >> "$JOBS"
    [ "$INNOISE" != none ] &&
        echo "${PFX}_smoke0 $NZ RIKFLOW_CNN_LOSS=$L1 RIKFLOW_CNN_WDS=0.3 RIKFLOW_CNN_SEEDS=1 RIKFLOW_CNN_MAXUPD=1 RIKFLOW_CNN_REFIT=1" >> "$JOBS"
elif [ "$MODE" = round ]; then
    for s in 1 2 3; do
        echo "${ROUND_FROM}_r${ROUND}_s$s RIKFLOW_CNN_INNOISE=$INNOISE RIKFLOW_CNN_INNOISE_SIGMA=step3/${ROUND_FROM}_s$s RIKFLOW_CNN_LOSS=$LOSS RIKFLOW_CNN_WDS=$WD RIKFLOW_CNN_SEEDS=$s RIKFLOW_CNN_MAXUPD=200000 RIKFLOW_CNN_REFIT=1" >> "$JOBS"
    done
else
    for l in $LOSSES; do
        for s in 1 2 3; do
            echo "${PFX}_${l}_wd0_nostop_s$s $NZ RIKFLOW_CNN_LOSS=$l RIKFLOW_CNN_WDS=0 RIKFLOW_CNN_SEEDS=$s RIKFLOW_CNN_NOSTOP=1 RIKFLOW_CNN_MAXUPD=100000 RIKFLOW_CNN_REFIT=1 RIKFLOW_CNN_REFIT_UPD=last" >> "$JOBS"
        done
    done
    for l in $LOSSES; do
        for w in 0 0.01 0.03 0.1 0.3 1; do
            for s in 1 2 3; do
                echo "${PFX}_${l}_wd${w}_s$s $NZ RIKFLOW_CNN_LOSS=$l RIKFLOW_CNN_WDS=$w RIKFLOW_CNN_SEEDS=$s RIKFLOW_CNN_MAXUPD=200000 RIKFLOW_CNN_REFIT=1" >> "$JOBS"
            done
        done
    done
fi
echo "== $SCRIPT: skip $BASE, mode $MODE, input noise $INNOISE ($INNOISE_SIGMA), $(wc -l < "$JOBS") fits, 16 at a time, CPU"
export SCRIPT BASE LOGD
xargs -P 16 -L 1 bash -c 'tag=$0; env "$@" RIKFLOW_MLP_SKIP=$BASE RIKFLOW_CNN_DEVICE=cpu RIKFLOW_CNN_OUT=step3/$tag \
    julia --startup-file=no --project=training "$SCRIPT" > "$LOGD/$tag.log" 2>&1 || { echo "FAILED $tag"; exit 1; }' < "$JOBS"
STATUS=$?                                       # xargs: 123 when any fit failed, so an afterok dependent does not start
rm -f "$JOBS"
grep -h "frozen linear skip\|SKIP: \|rms = \|upd, best\|refit on 1-10 TU\|S_deploy sd\|ERROR" "$LOGD"/${PFX}_*.log | cut -c1-220 | head -80
exit $STATUS
