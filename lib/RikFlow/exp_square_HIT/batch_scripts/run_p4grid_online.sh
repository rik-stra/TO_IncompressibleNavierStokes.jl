#!/bin/bash
#SBATCH -J p4grid_online
# Stages C and E of the Snellius grid (plan step 1+, 2026-09-28): free-running online replicas of a
# LIST of fit directories through 12_online_StochLSTM.jl. One array task = (list line, replica):
# task t -> line (t-1)/NREP + 1, replica (t-1)%NREP + 1.
#
#   stage C, 20 TU smoke (list = analysis/p4grid_select.jl's smoke_list.txt, copied here):
#     sbatch -t 00:45:00 --array=1-<3 x lines>%12 batch_scripts/run_p4grid_online.sh
#   stage E, 100 TU long runs (list = analysis/p4grid_collect.jl's long_list.txt):
#     P4GRID_ONLINE_LIST=batch_scripts/p4grid/long_list.txt P4GRID_TSIM=100 \
#       sbatch --array=1-<3 x lines>%12 batch_scripts/run_p4grid_online.sh
#
# Each line is a fit dir relative to output/TO_LSTM (or absolute). The replicas are written BESIDE the
# model, `data_online_tsim<T>_replica<i>.jld2`, as every M4 online run is -- which is where
# m4_screen.jl (M4_SCREEN_SUBDIR=p4grid), m4_online_moments.jl, m4_screen_long.jl and
# p4grid_select.jl --after-smoke read them. A replica whose file exists is skipped.
#
# Model index 2 (the StochLSTM2 row of inputs_lstm.jld2): every p4grid fit is built on that row, and
# 12_online_StochLSTM.jl checks the fit's cell against it. The IC is the ~7 MB extract
# (tools/m4_extract_ic.jl), not the 2.7 GB record -- same objects, same run.
#
# Walltime: unmeasured for an M4 closure on the H100. Desktop RTX 3090, shared: 7.5-12 s/TU in D6
# (LSTM closure); LRS on the H100 ~6.5 s/TU (handoff_p2c_d6.md, 21 s per 3.25 TU member). Budget
# 12 s/TU: 20 TU = 4 min, 100 TU = 20 min, + ~2 min compile. The header's 1 h covers 100 TU at 2.5x;
# pass -t 00:45:00 for the smoke.
#SBATCH -t 01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL

set -u
if [ -f 3_track_ref.jl ]; then
    EXP=$PWD
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=$PWD/exp_square_HIT
else
    echo "run_p4grid_online.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi
LIST=${P4GRID_ONLINE_LIST:-$EXP/batch_scripts/p4grid/smoke_list.txt}
TSIM=${P4GRID_TSIM:-20}
NREP=${P4GRID_NREP:-3}
TASK=${P4GRID_TASK:-${SLURM_ARRAY_TASK_ID:-}}
[ -n "$TASK" ] || { echo "run_p4grid_online.sh: no SLURM_ARRAY_TASK_ID (or P4GRID_TASK)" >&2; exit 1; }
[ -f "$LIST" ] || { echo "run_p4grid_online.sh: no list $LIST" >&2; exit 1; }
LINE=$(( (TASK - 1) / NREP + 1 )); REP=$(( (TASK - 1) % NREP + 1 ))
DIR=$(grep -v '^\s*\(#\|$\)' "$LIST" | sed -n "${LINE}p")
[ -n "$DIR" ] || { echo "task $TASK: line $LINE is past $LIST -- nothing to do"; exit 0; }
case "$DIR" in /*) MD=$DIR ;; *) MD=$EXP/output/TO_LSTM/$DIR ;; esac
[ -f "$MD/StochLSTM_seed1.jld2" ] || { echo "task $TASK: no fit in $MD" >&2; exit 1; }
OUTF=$MD/data_online_tsim$(printf '%.1f' "$TSIM")_replica$REP.jld2
if [ -f "$OUTF" ] && [ "${P4GRID_FORCE:-0}" != "1" ]; then
    echo "== $DIR replica $REP: $OUTF exists -- skipped"; exit 0
fi

if [ "${P4GRID_LOCAL:-0}" != "1" ]; then
    export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
    export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
fi
export RIKFLOW_ONLINE_IC=${RIKFLOW_ONLINE_IC:-$EXP/output/online_ic_data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2}
export RIKFLOW_ONLINE_TSIM=$TSIM
export RIKFLOW_M4_MODEL_DIR=$MD
[ -f "$RIKFLOW_ONLINE_IC" ] || { echo "no IC extract $RIKFLOW_ONLINE_IC (copy it, RUNBOOK)" >&2; exit 1; }

[ -n "${P4GRID_NO_INSTANTIATE:-}" ] || julia --project="$EXP/.." -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed -- continuing; the run will compile in-process" >&2

echo "== $DIR, replica $REP, $TSIM TU (task $TASK, line $LINE of $LIST) [$(date +%T)]"
julia --startup-file=no --project="$EXP/.." "$EXP/12_online_StochLSTM.jl" 2 "$REP"
rc=$?
echo "== rc=$rc [$(date +%T)]"
exit $rc
