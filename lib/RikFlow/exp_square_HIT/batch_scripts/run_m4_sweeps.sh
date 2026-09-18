#!/bin/bash
#SBATCH -J m4_sweep
#SBATCH -t 02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
# 🔑 Explicit, though it is also SLURM's default: the job inherits the SUBMITTING environment.
# That is what makes `RIKFLOW_M4_UPDATES=5 sbatch ...` and `M4_DEVICE=cpu sbatch ...` work, and it
# is what the other scripts here have always relied on -- none of them sets `--export`, and they
# all need an inherited `PATH` just to find `julia`. Stating it protects against a site default of
# `NONE`, which would strip both the budget AND the PATH.
# 🔴 **Do NOT write `--export=VAR=value` on the command line: that REPLACES `ALL`**, so the job
# would lose `PATH` and die before Julia starts. The additive form is `--export=ALL,VAR=value`.
#SBATCH --export=ALL

# M4's two training sweeps: the learning-rate scan (§6.1) and the stride/batch scan (§6.2).
#
# Usage:
#     sbatch batch_scripts/run_m4_sweeps.sh smoke           # 🔴 RUN THIS FIRST -- ~3 min, finishes
#     sbatch batch_scripts/run_m4_sweeps.sh stride          # the stride scan, cell 2, seed 1
#     sbatch batch_scripts/run_m4_sweeps.sh lr 5 3          # cell 5, seed 3
#
# 🔴 **`smoke` is the first job to run on any new device.** A scan is a bad first job: one point is
# thousands of updates and prints nothing until it ends, so a slow device and a hung one look
# identical -- which is how a 20-minute job came to be cancelled blind on 2026-09-18. The smoke
# runs a few seconds of arithmetic, prints a timestamped line at every phase, and exits 0 with
# `M4 SMOKE PASS` or non-zero naming the phase that failed. It needs no `inputs_lstm.jld2` and
# writes only to a temporary directory.
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
    smoke)  SCAN=m4_smoke.jl ;;
    lr)     SCAN=m4_lr_scan.jl ;;
    stride) SCAN=m4_stride_scan.jl ;;
    *) echo "run_m4_sweeps.sh: first argument must be 'smoke', 'lr' or 'stride'; got '${WHICH}'" >&2
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
# 🔑 Echo the budgets that actually arrived. Whether an env var survives submission is the
# kind of thing that is easy to assume and expensive to assume wrongly, so the log answers
# it rather than the reader inferring it. The Julia driver prints them again from its own
# side (`@info "M4 <which> scan" ... updates=/epochs=`), so the two together show the value
# crossed the shell/Julia boundary too.
echo "== budget: RIKFLOW_M4_UPDATES=${RIKFLOW_M4_UPDATES:-<default 3000>}"\n     " RIKFLOW_M4_LR_EPOCHS=${RIKFLOW_M4_LR_EPOCHS:-<default 1500>}"\n     " RIKFLOW_M4_EPOCHS=${RIKFLOW_M4_EPOCHS:-<unset>}"

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

# 🔑 **QoI resolution lives in Julia, in `tools/m4_data.jl`, not here.** All three M4 drivers
# include it, so the cache pattern and the "which record is this?" rule have ONE definition --
# and `_f64_lmwray3` in that pattern is load-bearing: `analysis/data/` also holds caches of paper
# 2's archived records, which are a different dynamical system (`claude_memory.md` #45, #46).
# `RIKFLOW_QOI_CACHE` still wins if it is set. If no cache is found the driver warns and reads the
# 2.7 GB tracking record, which works but is slow and heavy when jobs overlap -- extract it once,
# on the login node, before submitting the grid:
#
#   julia --project=analysis analysis/extract_qois.jl #       exp_square_HIT/output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2

julia --project="$ROOT/training" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

if [ "$WHICH" = "smoke" ]; then
    echo "== M4 smoke (device $M4_DEVICE)"
    julia --project="$ROOT/training" "$EXP/tools/$SCAN"
else
    echo "== M4 $WHICH scan, cell $CELL, seed $SEED (device $M4_DEVICE)"
    julia --project="$ROOT/training" "$EXP/tools/$SCAN" "$CELL" "$SEED"
fi
