#!/bin/bash
#SBATCH -J p4grid_fit
# Stage A of the Snellius grid (plan step 1+, 2026-09-28): ONE fit per array task, the row of
# batch_scripts/p4grid/fits.csv whose `row` is $SLURM_ARRAY_TASK_ID. The CSV is generated from
# p4grid/spec.toml by tools/p4grid_generate.jl, which prints the exact --array to use:
#
#     sbatch --array=1-72%20 batch_scripts/run_p4grid_fit.sh               # from exp_square_HIT/
#     P4GRID_PACK=8 sbatch --array=1-9 batch_scripts/run_p4grid_fit.sh     # 8 fits per task, in parallel
#
# 🔑 No --array in this header on purpose: the row count comes from the spec, so the command line
# (printed by the generator) is the only place it is written.
#
# Walltime: measured on the desktop, one M3ᶠ fit = 0.7-1.5 min of training (Agent C's h = 2 fits,
# 2026-09-28; `wall` in their curve.jld2 = 84 s), plus ~1-2 min of Julia load + first-call compile;
# an M0@50 row is seconds; an M0ᵛ row (m4_diag_fit, L = 500, 1-50 TU) was not timed at 1-50 TU --
# r3_lin_sd_tu50 stopped at 228 updates. h = 5 has twice h = 2's inputs. 30 min is >= 5x margin for
# one row; with P4GRID_PACK=8 the rows run concurrently (2 threads each on the 16 cores a 1-GPU
# gpu_h100 share gets), so 30 min still holds.
#SBATCH -t 00:30:00
# 🔒 gpu_h100 because the allocation has GPU nodes only (Rik, 2026-09-23; run_m4_sweeps.sh). The fit
# runs on the CPU (M4_DEVICE=cpu, measured faster, results_LSTMS.md §5); the GPU idles. That is why
# P4GRID_PACK exists: 72 one-row tasks bill ~72 x 5 min of GPU, 9 packed tasks ~9 x 10 min.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
# Inherit the submitting environment (P4GRID_* and the PATH to julia); never `--export=VAR=...`,
# which REPLACES ALL (see run_m4_sweeps.sh).
#SBATCH --export=ALL
#
# Environment:
#   P4GRID_CSV       the table (default batch_scripts/p4grid/fits.csv)
#   P4GRID_PACK      rows per task (default 1): task t runs rows (t-1)*PACK+1 .. t*PACK in parallel
#   P4GRID_ROW       run this row instead of $SLURM_ARRAY_TASK_ID (local use: bash run_p4grid_fit.sh)
#   P4GRID_FORCE=1   refit a row whose output already exists (default: skip it)
#   P4GRID_EXTRA     extra KEY=VALUE assignments appended AFTER the row's own (they win), e.g. the
#                    dry-run budget cut `RIKFLOW_W_EPOCHS=1 RIKFLOW_D_EPOCHS=1`
#   P4GRID_LOCAL=1   desktop: do not set the cluster depot / JULIA_CPU_TARGET
#   P4GRID_NO_INSTANTIATE=1  skip the per-task Pkg.instantiate/precompile
#   P4GRID_OUTROOT   DRY RUNS ONLY: an absolute directory that replaces output/TO_LSTM/p4grid for
#                    every row (the tools accept an absolute OUTSUB / TAG), so a local test never
#                    writes into the real output tree
#
# 🔒 Every row trains on 1-50 TU and scores 52-74 TU (TRAIN_TU / SCORE_TU in the row). m4_window_fit.jl
# and m4_linear_eta.jl LOAD the whole QoI cache (0-100 TU) and build the history over it, but score
# only 52-74 TU and fit only 1-50 TU; they print the windows ("windows (TU)", "held-out score window")
# in every log -- check them.

set -u
set -f          # no globbing: the env values (FREEZE=Ws,V1, ...) are word-split, never glob-expanded

if [ -f 3_track_ref.jl ]; then
    EXP=$PWD; ROOT=$(cd .. && pwd)
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=$PWD/exp_square_HIT; ROOT=$PWD
else
    echo "run_p4grid_fit.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi
CSV=${P4GRID_CSV:-$EXP/batch_scripts/p4grid/fits.csv}
[ -f "$CSV" ] || { echo "run_p4grid_fit.sh: no table $CSV -- run tools/p4grid_generate.jl" >&2; exit 1; }
PACK=${P4GRID_PACK:-1}
TASK=${P4GRID_ROW:-${SLURM_ARRAY_TASK_ID:-}}
[ -n "$TASK" ] || { echo "run_p4grid_fit.sh: no SLURM_ARRAY_TASK_ID (or P4GRID_ROW)" >&2; exit 1; }
NROWS=$(( $(wc -l < "$CSV") - 1 ))

# P4GRID_LOCAL=1 (desktop dry runs): keep the local depot and CPU target -- setting the cluster's
# JULIA_CPU_TARGET here would invalidate every local compile cache.
if [ "${P4GRID_LOCAL:-0}" != "1" ]; then
    export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
    export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
fi
export M4_DEVICE=${M4_DEVICE:-cpu}
export OPENBLAS_NUM_THREADS=2
export JULIA_NUM_THREADS=2
TO=$EXP/output/TO_LSTM
LOGDIR=${P4GRID_OUTROOT:-$TO/p4grid}/_logs
mkdir -p "$LOGDIR"

# one row: parse, skip if done, run
run_row() {
    local r=$1 line cell h lam wd nh seed tag outdir m0 tool status envs done_file log
    line=$(awk -F, -v r="$r" 'NR > 1 && $1 == r { print; exit }' "$CSV")
    [ -n "$line" ] || { echo "row $r: not in $CSV ($NROWS rows)" >&2; return 1; }
    IFS=, read -r _ cell h lam wd nh seed tag outdir m0 tool status _ <<< "$line"
    envs=${line#*,\"}; envs=${envs%\"}
    if [ "$status" != "ok" ]; then
        echo "== row $r $tag: $status -- not run"
        return 0
    fi
    if [ -n "${P4GRID_OUTROOT:-}" ]; then
        # absolute output root for dry runs: the tools' joinpath(TO, <absolute>) returns <absolute>
        envs=$(echo "$envs" | sed -e "s#RIKFLOW_W_OUTSUB=p4grid#RIKFLOW_W_OUTSUB=$P4GRID_OUTROOT#" \
                                  -e "s#\bOUTSUB=p4grid#OUTSUB=$P4GRID_OUTROOT#" \
                                  -e "s#RIKFLOW_D_TAG=../p4grid/#RIKFLOW_D_TAG=$P4GRID_OUTROOT/#")
        done_file=$P4GRID_OUTROOT/$tag/StochLSTM_seed1.jld2
    else
        done_file=$TO/$outdir/StochLSTM_seed1.jld2
    fi
    if [ -f "$done_file" ] && [ "${P4GRID_FORCE:-0}" != "1" ]; then
        echo "== row $r $tag: exists ($done_file) -- skipped (P4GRID_FORCE=1 to refit)"
        return 0
    fi
    log=$LOGDIR/$tag.log
    echo "== row $r/$NROWS $cell h=$h lambda=$lam wd=$wd nh=$nh seed=$seed -> $outdir  [$(date +%T)] log $log"
    echo "   env: $envs ${P4GRID_EXTRA:-}"
    ( cd "$ROOT" && env $envs ${P4GRID_EXTRA:-} \
        julia --startup-file=no --project="$ROOT/training" "$EXP/tools/$tool.jl" ) > "$log" 2>&1
    local rc=$?
    echo "== row $r $tag: rc=$rc [$(date +%T)]"
    tail -n 3 "$log" | cut -c1-400
    [ -f "$done_file" ] || { echo "row $r $tag: no $done_file after the run" >&2; return 1; }
    return $rc
}

# Instantiate once per task (harmless when warm; warm the depot on the login node first, see RUNBOOK).
if [ -z "${P4GRID_NO_INSTANTIATE:-}" ]; then
    julia --project="$ROOT/training" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
        echo "precompile step failed -- continuing; the run will compile in-process" >&2
fi

first=$(( (TASK - 1) * PACK + 1 )); last=$(( TASK * PACK ))
[ $last -gt $NROWS ] && last=$NROWS
[ $first -le $NROWS ] || { echo "task $TASK: rows $first.. are past the table ($NROWS rows)"; exit 0; }
echo "== p4grid task $TASK: rows $first-$last of $CSV (pack $PACK, device $M4_DEVICE)"
fail=0
if [ "$PACK" = "1" ]; then
    run_row "$first" || fail=1
else
    pids=()
    for r in $(seq "$first" "$last"); do
        run_row "$r" & pids+=($!)
        sleep 2
    done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
fi
exit $fail
