#!/bin/bash
#
# Submit one TO-LRS configuration end to end: fit it, then run the ensemble as an array with one
# replica per GPU. Run this on the login node -- it is two `sbatch` calls, not a job itself.
#
# Usage (from exp_square_HIT or lib/RikFlow):
#     ./batch_scripts/submit_lrs.sh 5            # LinReg5, replicas 1-5
#     ./batch_scripts/submit_lrs.sh 6 1-5        # LinReg6, explicit member list
#     ./batch_scripts/submit_lrs.sh 5 3          # refit and rerun only replica 3
#
# 🔴 The dependency is `afterok`, so the array never starts on a failed fit. Without it the five
# tasks would race the trainer and deploy whatever `LinReg.jld2` happened to be on disk -- which,
# for a rerun at a new lambda, is the PREVIOUS lambda's model and would look like a null result.
#
# 🔑 P2r's lambda probe, 2026-09-15:
#     ./batch_scripts/submit_lrs.sh 5     # h = 5, lambda = 1e-5
#     ./batch_scripts/submit_lrs.sh 6     # h = 5, lambda = 1e-4
# against LinReg1 (lambda = 0), which R2 already ran. Run `4_setup_search.jl` once first if
# `output/TO_LRS/inputs_example.jld2` predates those cells -- it is instant and needs no GPU.

set -eu

INDEX=${1:-}
ARRAY=${2:-1-5}

if ! [[ "$INDEX" =~ ^[0-9]+$ ]]; then
    echo "usage: submit_lrs.sh <model_index> [array_spec]" >&2
    echo "  e.g. submit_lrs.sh 5          # LinReg5, replicas 1-5" >&2
    echo "       submit_lrs.sh 5 1-5:2    # replicas 1,3,5" >&2
    exit 1
fi

if [ -f batch_scripts/run_train_lrs.sh ]; then
    BS=batch_scripts
elif [ -f exp_square_HIT/batch_scripts/run_train_lrs.sh ]; then
    BS=exp_square_HIT/batch_scripts
else
    echo "submit_lrs.sh: cannot find batch_scripts/ from $(pwd)" >&2
    echo "  submit from exp_square_HIT or from lib/RikFlow" >&2
    exit 1
fi

TRAIN=$(sbatch --parsable "$BS/run_train_lrs.sh" "$INDEX")
echo "fit      LinReg$INDEX          job $TRAIN"

ONLINE=$(sbatch --parsable --dependency=afterok:"$TRAIN" --array="$ARRAY" \
                "$BS/run_online_array.sh" "$INDEX")
echo "ensemble LinReg$INDEX [$ARRAY]  job $ONLINE  (starts after $TRAIN succeeds)"
echo
echo "watch:   squeue -j $TRAIN,$ONLINE"
echo "cancel:  scancel $TRAIN $ONLINE"
