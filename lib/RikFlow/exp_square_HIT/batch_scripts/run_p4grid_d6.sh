#!/bin/bash
#SBATCH -J p4grid_d6
# Stage D of the Snellius grid (plan step 1+, 2026-09-28): the paired mini-D6 on the SELECTION block
# (46 ICs, ordinals 88-133, t in [52, 74] TU), M = 5, N_LEAD = 400, for every closure in a list --
# the <= 6 selected configs x fit seeds 1-3, plus their matched M0s, as written by
# `analysis/p4grid_select.jl --after-smoke` (d6_list.txt, copied here).
#
#     sbatch --array=1-<2 x lines>%16 batch_scripts/run_p4grid_d6.sh           # from exp_square_HIT/
#
# One array task = (closure, IC chunk): task t -> line (t-1)/NCHUNK + 1, chunk (t-1)%NCHUNK + 1, run by
# tools/p4grid_d6_chunk.jl in ONE Julia process (one compile per chunk). Each closure gets its own
# D6_OUT = output/D6mini_p4grid/<fit tag> -- one directory per closure, model, block and N_LEAD, which
# run_d6.jl and score_d6.jl both enforce -- and the matched M0 runs through the same StochLSTM path,
# same ICs, so the pairing (by IC; not common random numbers across samplers) is exact.
#
# 🔑 Rerun a lost task with the same --array index: run_ic skips members whose files exist, before the
# seed is derived, so only the missing members run and each keeps its seed.
#
# Sizing (per closure): 46 ICs x 5 members x (100 warm + 400 lead) steps = 46 x 5 x 1.25 TU = 288 TU.
# At 12 s/TU budgeted for an LSTM closure on the H100 (unmeasured there; LRS measured ~6.5 s/TU on the
# H100, LSTM 7.5-12 s/TU on the desktop 3090) that is ~58 min, so NCHUNK = 2 gives ~30 min + ~1.5 min
# compile per task. Walltime 1:30 = 3x.
#SBATCH -t 01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#
# Environment: P4GRID_D6_LIST (default batch_scripts/p4grid/d6_list.txt), P4GRID_NCHUNK (2),
# D6_MEMBERS (5), D6_NLEAD (400), D6_BLOCK (selection), P4GRID_D6_ROOT (default output/D6mini_p4grid),
# P4GRID_TASK (run one task without SLURM), P4GRID_LOCAL=1 (desktop: keep the local depot).

set -u
if [ -f 3_track_ref.jl ]; then
    EXP=$PWD
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=$PWD/exp_square_HIT
else
    echo "run_p4grid_d6.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi
LIST=${P4GRID_D6_LIST:-$EXP/batch_scripts/p4grid/d6_list.txt}
NCHUNK=${P4GRID_NCHUNK:-2}
TASK=${P4GRID_TASK:-${SLURM_ARRAY_TASK_ID:-}}
[ -n "$TASK" ] || { echo "run_p4grid_d6.sh: no SLURM_ARRAY_TASK_ID (or P4GRID_TASK)" >&2; exit 1; }
[ -f "$LIST" ] || { echo "run_p4grid_d6.sh: no list $LIST" >&2; exit 1; }
LINE=$(( (TASK - 1) / NCHUNK + 1 )); CHUNK=$(( (TASK - 1) % NCHUNK + 1 ))
DIR=$(grep -v '^\s*\(#\|$\)' "$LIST" | sed -n "${LINE}p")
[ -n "$DIR" ] || { echo "task $TASK: line $LINE is past $LIST -- nothing to do"; exit 0; }
case "$DIR" in /*) MD=$DIR ;; *) MD=$EXP/output/TO_LSTM/$DIR ;; esac
[ -f "$MD/StochLSTM_seed1.jld2" ] || { echo "task $TASK: no fit in $MD" >&2; exit 1; }

if [ "${P4GRID_LOCAL:-0}" != "1" ]; then
    export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
    export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
fi
export D6_CLOSURE=lstm
export D6_MODEL=$MD
export D6_BLOCK=${D6_BLOCK:-selection}
export D6_MEMBERS=${D6_MEMBERS:-5}
export D6_NLEAD=${D6_NLEAD:-400}
export D6_OUT=${P4GRID_D6_ROOT:-$EXP/output/D6mini_p4grid}/$(basename "$MD")
mkdir -p "$D6_OUT"
# Optional; run_d6.jl falls back to output/d6_ics and then to the local analysis build directory.
# export D6_IC_DIR=$EXP/output/d6_ics

echo "== mini-D6 $(basename "$MD"): chunk $CHUNK/$NCHUNK, block $D6_BLOCK, M $D6_MEMBERS, N_LEAD $D6_NLEAD -> $D6_OUT [$(date +%T)]"
cd "$EXP" && julia --startup-file=no --project=.. tools/p4grid_d6_chunk.jl "$CHUNK" "$NCHUNK"
rc=$?
echo "== rc=$rc [$(date +%T)]"
exit $rc
