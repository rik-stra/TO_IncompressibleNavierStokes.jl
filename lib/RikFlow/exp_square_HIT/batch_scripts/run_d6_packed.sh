#!/bin/bash
# Full D6 hindcast, PACKED: several initial conditions per array task (Rik, 2026-10-06).
#
#     D6_OUT=$PWD/output/D6_<closure> D6_CLOSURE=<lrs|lstm|ddn> D6_MODEL=<abs path> \
#         sbatch batch_scripts/run_d6_packed.sh                     # from exp_square_HIT/
#
# 🔑 Why: one IC per task (`run_d6_linreg1.sh`, `--array=1-179:2`) pays Julia's start-up and the
# solver's compile ~90 times per hindcast, ~1-2 min each, at 192 SBU per job-hour on gpu_h100. Here
# each task runs D6_PACK ICs in one process (`run_d6.jl o1,o2,...`), so the hindcast pays it ~10
# times. Members, seeds, warm-up check and output are exactly those of the one-IC tasks: the same
# `run_ic`, called per ordinal.
#
# The IC set is the paper's K = 90: ordinals 1, 3, ..., 179 (every second of the 180 packages, as
# `--array=1-179:2`). Task t runs the t-th slice of D6_PACK of them; the default D6_PACK = 9 gives
# `--array=1-10`. With another D6_PACK, pass `--array=1-<ceil(90 / D6_PACK)>` on the sbatch line.
#
# 🔑 Reruns are cheap: `run_ic` skips a member whose file exists (before deriving its seed), so
# resubmitting a task (`sbatch --array=<t> ...`) runs only what is missing.
#
# 🔴 D6_OUT, D6_CLOSURE and D6_MODEL are REQUIRED here, on the sbatch line (never `--export=VAR=`,
# claude_memory.md #69) -- this script has no default model, so it cannot write one closure's runs
# into another's directory. `run_d6.jl` also refuses a directory holding another model's members.
# For lstm, D6_MODEL must be the seed-1 FILE (`.../StochLSTM_seed1.jld2`), as WORKFLOW.md §4 says.
#
# Wall time: 9 ICs x 10 members x ~15-21 s (q99 ~95 s on the A100, divergences included) + start-up
# ~ 25-45 min; 1:30 h is the margin. SLURM bills elapsed time, not the limit.

#SBATCH -J d6-packed
#SBATCH -t 01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=1-10
#SBATCH --export=ALL

set -u
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

for v in D6_OUT D6_CLOSURE D6_MODEL; do
    if [ -z "${!v:-}" ]; then
        echo "run_d6_packed.sh: $v is not set; put D6_OUT, D6_CLOSURE and D6_MODEL on the sbatch line" >&2
        exit 1
    fi
done
export D6_MEMBERS=${D6_MEMBERS:-10}
PACK=${D6_PACK:-9}
NIC=90                                   # ordinals 1, 3, ..., 179
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}

FIRST=$(( (TASK - 1) * PACK + 1 ))
LAST=$(( TASK * PACK < NIC ? TASK * PACK : NIC ))
if [ "$FIRST" -gt "$NIC" ]; then
    echo "task $TASK: no ICs (D6_PACK = $PACK covers $NIC ICs in $(( (NIC + PACK - 1) / PACK )) tasks)"
    exit 0
fi
ORDS=$(seq -s, $(( 2 * FIRST - 1 )) 2 $(( 2 * LAST - 1 )))

mkdir -p "$D6_OUT"
echo "== task $TASK: ordinals $ORDS | closure $D6_CLOSURE | model $D6_MODEL | M = $D6_MEMBERS | out $D6_OUT"
julia --project tools/run_d6.jl "$ORDS"
