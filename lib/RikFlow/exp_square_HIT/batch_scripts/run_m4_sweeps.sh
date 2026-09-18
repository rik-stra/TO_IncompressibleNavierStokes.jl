#!/bin/bash
#SBATCH -J m4_sweep
#SBATCH -t 02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# M4's two training sweeps: the learning-rate scan (§6.1) and the stride/batch scan (§6.2).
#
# Usage:
#     sbatch batch_scripts/run_m4_sweeps.sh lr              # the lr scan, cell 2, seed 1
#     sbatch batch_scripts/run_m4_sweeps.sh stride          # the stride scan, cell 2, seed 1
#     sbatch batch_scripts/run_m4_sweeps.sh lr 5 3          # cell 5, seed 3
#
# Budgets are the scans' own defaults and are overridable from the submitting shell:
#     RIKFLOW_M4_LR_EPOCHS   epochs per lr point       (default 1500 in the script, 1000 used so far)
#     RIKFLOW_M4_LRS         comma-separated rates
#     RIKFLOW_M4_UPDATES     optimiser steps per stride point (default 3000)
#
# ---------------------------------------------------------------------------------------------
# 🔴 IT TRAINS ON THE GPU (`M4_DEVICE=cuda` by default since 2026-09-18)
# ---------------------------------------------------------------------------------------------
#
# `M4_DEVICE` decides where the fit runs and **defaults to `cuda` here**. Override per submission:
#
#     M4_DEVICE=cpu sbatch batch_scripts/run_m4_sweeps.sh stride     # the node's CPU instead
#
# 🔴 **THE GPU PATH HAS NEVER RUN ON A GPU.** It is verified only against `JLArrays`, which
# enforces the same no-scalar-indexing semantics on the host (V54), and all four architectures
# agree with the CPU fit to 6.6e-8-2.1e-7 there. What that does NOT establish is that CUDA.jl
# compiles these kernels -- and this repository has been bitten twice by exactly the class of
# defect a CPU test cannot see (`claude_memory.md` #56 defeated constant propagation, #57
# non-isbits kernel arguments), both found only on the cluster.
# 🔑 **So make the FIRST submission a test, not a measurement:**
#
#     RIKFLOW_M4_UPDATES=5 sbatch batch_scripts/run_m4_sweeps.sh stride
#
# It runs every point at a trivial budget in about a minute and exercises the whole path including
# the write. `m4_device` refuses `cuda` when no device is functional rather than falling back to
# the host, so a mis-scheduled job fails at load instead of quietly reporting CPU time as GPU time.
#
# ⚠️ **Expect it to be SLOWER at the current geometry, and that is a measurement rather than a
# guess about the hardware.** One gradient step is ~75 MFLOP at ~0.34 GFLOP/s -- about 1% of one
# CPU core -- over a recurrence whose 500 timesteps are strictly SEQUENTIAL, i.e. ~1000+ dependent
# kernel launches per step on matrices of 64 x 23. Launch latency alone is then several ms per
# step, against ~200 ms currently spent almost entirely in Zygote's HOST-side tracing, which a
# device does not remove. **The configuration where it could pay is exactly what §6.2 sweeps:** a
# short `stride` with `batch = 32` makes each launch 32 segments wide instead of 7.
#
# ✅ A CPU partition also works if the GPU turns out not to pay: `m4_lr_scan.jl` -- `using RikFlow`
# plus the Lux extension -- ran six fits to completion on a machine with no GPU at all
# (2026-09-18), so CUDA.jl loading without a device is not the failure mode `run_train_lrs.sh`
# feared. `--partition=rome` or `genoa` are covered by the depot's `JULIA_CPU_TARGET` and need no
# new depot; pair either with `M4_DEVICE=cpu`.
#
# ⚠️ `OPENBLAS_NUM_THREADS=1` is set below. It still matters on the GPU path: the batches are
# assembled on the host, and on `64 x 23` matrices a full BLAS pool contends rather than helps.
#
# 🔴 Run `10_setup_lstm.jl` first if the configuration table has changed. It needs no GPU:
#     julia --project exp_square_HIT/10_setup_lstm.jl

set -u

WHICH=${1:-}
CELL=${2:-2}
SEED=${3:-1}
case "$WHICH" in
    lr)     SCAN=m4_lr_scan.jl ;;
    stride) SCAN=m4_stride_scan.jl ;;
    *) echo "run_m4_sweeps.sh: first argument must be 'lr' or 'stride'; got '${WHICH}'" >&2
       exit 1 ;;
esac
if ! [[ "$CELL" =~ ^[0-9]+$ ]] || ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "run_m4_sweeps.sh: cell and seed must be integers; got '$CELL' '$SEED'" >&2
    exit 1
fi

# Reuse the existing depot rather than building another; the CPU target multiversions the
# precompiled code, so one depot serves the GPU and CPU partitions alike.
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export OPENBLAS_NUM_THREADS=1

# Train on the device unless the submitting shell says otherwise. SLURM's default `--export=ALL`
# carries `M4_DEVICE=cpu sbatch ...` through, so this is a default and not an override.
export M4_DEVICE=${M4_DEVICE:-cuda}
echo "== M4_DEVICE=$M4_DEVICE"

# Find the drivers from whichever directory the job started in, and say so if it is neither.
if [ -f 3_track_ref.jl ]; then
    EXP=.
    ROOT=..
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=exp_square_HIT
    ROOT=.
else
    echo "run_m4_sweeps.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    exit 1
fi

# 🔑 Prefer the extracted QoI cache: it is ~7 MB against the tracking record's 2.7 GB, and the
# scans read nothing else from it. `analysis/data` is gitignored, so on a fresh checkout the cache
# is absent and the scan falls back to the record -- correct either way, just slower to start.
if [ -z "${RIKFLOW_QOI_CACHE:-}" ]; then
    CACHE="$ROOT/analysis/data/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"
    if [ -f "$CACHE" ]; then
        export RIKFLOW_QOI_CACHE=$CACHE
        echo "== using the QoI cache: $CACHE"
    else
        echo "== no QoI cache at $CACHE; the scan will read the 2.7 GB tracking record" >&2
    fi
fi

julia --project="$ROOT/training" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

echo "== M4 $WHICH scan, cell $CELL, seed $SEED (device $M4_DEVICE)"
julia --project="$ROOT/training" "$EXP/tools/$SCAN" "$CELL" "$SEED"
