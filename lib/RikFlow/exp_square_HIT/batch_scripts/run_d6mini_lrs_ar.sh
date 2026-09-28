#!/bin/bash
#SBATCH -J d6mini_ar
# Stage F of the Snellius grid (plan step 1+, 2026-09-28): the ridge + colour cell (Agent D's
# "M0ᶜ-ridge", results_LSTMS.md §10h/§12). Mini-D6 on the selection block, M = 5, N_LEAD = 400, of a
# base LinReg AND its AR-residual variants, so every pair is on the same ICs and -- same family, same
# member seeds -- common random numbers apply.
#
#     sbatch --array=1-4 batch_scripts/run_d6mini_lrs_ar.sh LinReg7            # base + LinReg7_ar2
#     sbatch --array=1-6 batch_scripts/run_d6mini_lrs_ar.sh LinReg2 ar2 ar1    # base + _ar2 + _ar1
#
# Arguments: the BASE LinReg name, then the variant suffixes (default `ar2`). Models, in order:
# BASE, BASE_<suffix>... ; task t -> model (t-1)/NCHUNK + 1, chunk (t-1)%NCHUNK + 1, so
# --array=1-<NCHUNK x (1 + #suffixes)>.
# 🔴 The variants are Agent D's, built by tools/lrs_ar_variant.jl into output/TO_LRS/<BASE>_ar<p>/ --
# this script never builds one; it refuses a model that is not there. Copy them to Snellius first.
#
# D6_OUT = output/D6mini_<model>, one per model (score_d6 refuses mixed dirs). Sizing: an LRS member is
# ~15-21 s per 3.25 TU on the H100 (handoff_p2c_d6.md) = ~6.5 s/TU, so a model is
# 46 x 5 x 1.25 TU x 6.5 s = ~31 min; NCHUNK = 2 -> ~16 min + ~1.5 min compile per task.
#SBATCH -t 01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL

set -u
BASE=${1:-}
[[ "$BASE" =~ ^LinReg[0-9]+$ ]] || { echo "run_d6mini_lrs_ar.sh: first argument must be a base LinReg name, e.g. LinReg7" >&2; exit 1; }
shift
SUFFIXES=("$@"); [ ${#SUFFIXES[@]} -eq 0 ] && SUFFIXES=(ar2)
MODELS=("$BASE"); for s in "${SUFFIXES[@]}"; do MODELS+=("${BASE}_$s"); done

if [ -f 3_track_ref.jl ]; then
    EXP=$PWD
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=$PWD/exp_square_HIT
else
    echo "run_d6mini_lrs_ar.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi
NCHUNK=${P4GRID_NCHUNK:-2}
TASK=${P4GRID_TASK:-${SLURM_ARRAY_TASK_ID:-}}
[ -n "$TASK" ] || { echo "run_d6mini_lrs_ar.sh: no SLURM_ARRAY_TASK_ID (or P4GRID_TASK)" >&2; exit 1; }
M=$(( (TASK - 1) / NCHUNK )); CHUNK=$(( (TASK - 1) % NCHUNK + 1 ))
[ $M -lt ${#MODELS[@]} ] || { echo "task $TASK: past the ${#MODELS[@]} models ${MODELS[*]} -- nothing to do"; exit 0; }
NAME=${MODELS[$M]}
MODEL=$EXP/output/TO_LRS/$NAME/LinReg.jld2
[ -f "$MODEL" ] || { echo "task $TASK: no $MODEL -- build it with tools/lrs_ar_variant.jl (Agent D) and copy it" >&2; exit 1; }

if [ "${P4GRID_LOCAL:-0}" != "1" ]; then
    export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
    export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
fi
export D6_CLOSURE=lrs
export D6_MODEL=$MODEL
export D6_BLOCK=${D6_BLOCK:-selection}
export D6_MEMBERS=${D6_MEMBERS:-5}
export D6_NLEAD=${D6_NLEAD:-400}
export D6_OUT=${P4GRID_D6_ROOT:-$EXP/output}/D6mini_$NAME
mkdir -p "$D6_OUT"

echo "== mini-D6 $NAME (models ${MODELS[*]}): chunk $CHUNK/$NCHUNK, block $D6_BLOCK, M $D6_MEMBERS, N_LEAD $D6_NLEAD -> $D6_OUT [$(date +%T)]"
cd "$EXP" && julia --startup-file=no --project=.. tools/p4grid_d6_chunk.jl "$CHUNK" "$NCHUNK"
rc=$?
echo "== rc=$rc [$(date +%T)]"
exit $rc
