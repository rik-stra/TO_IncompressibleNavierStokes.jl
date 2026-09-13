#!/bin/bash
#SBATCH -J hf_timing
#SBATCH -t 01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# Cost the 512^3 HF reference run before launching it.
#
# Submit from either exp_square_HIT or lib/RikFlow:
#     sbatch batch_scripts/run_hf_timing_probe.sh
#     sbatch exp_square_HIT/batch_scripts/run_hf_timing_probe.sh
#
# Prints the per-step cost at the production configuration, a projection to tsim = 100, and the
# disk the production run will need. Read the PROJECTION block at the end.
#
# Needs exp_square_HIT/output/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2 — the same
# initial condition 2_HF_ref.jl and cfl_probe.jl use. Without it the probe stops and prints the
# path; HFT_SYNTHETIC=1 substitutes a synthetic field, which is acceptable *here* because the
# per-step cost is set by the grid rather than the values (it is not acceptable for cfl_probe.jl,
# where max|u| sets the answer).
#
# One hour is generous: the probe sizes its measured block to HFT_BUDGET_S (default 900 s) from
# the warm-up rate, so the run is roughly module load + 200 warm-up steps + 15 minutes + one
# checkpoint write. It does not scale up if the node is fast, only down if it is slow.
#
# Knobs, all optional:
#   HFT_BUDGET_S=1800     longer measured block, tighter projection
#   HFT_MEASURE=5000      fix the block size outright instead of sizing it
#   HFT_PLOTFREQ=2000     cost a different field-saving policy (must match 2_HF_ref.jl to transfer)
#   HFT_N_CHECKPOINTS=3   cost a different checkpoint policy
#   HFT_WALL_HOURS=120    the wall limit the projection is judged against
#   HFT_SYNTHETIC=1       run without the archived spin-up

# Reuse the existing depot rather than building another. The CPU target multiversions the
# precompiled code across Zen2, Zen4 and Icelake-server, so one depot serves the a100 and h100
# partitions without recompiling per architecture — which is what makes sharing it safe.
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# Find the probe from whichever directory the job started in, and say so if it is neither.
if [ -f hf_timing_probe.jl ]; then
    SCRIPT=hf_timing_probe.jl
elif [ -f exp_square_HIT/hf_timing_probe.jl ]; then
    SCRIPT=exp_square_HIT/hf_timing_probe.jl
else
    echo "run_hf_timing_probe.sh: cannot find hf_timing_probe.jl from $(pwd)" >&2
    echo "  submit from exp_square_HIT or from lib/RikFlow" >&2
    exit 1
fi

# Bring the depot's caches in line with this invocation before running.
#
# 🔴 Needed because of JULIA_CPU_TARGET above. Multiversioning changes the *content* of the
# precompiled images, so a depot populated without it — or with a different target — holds caches
# this run will not accept, and julia reports that as
#
#     Precompiled image ... "Adapt" not available with flags CacheFlags(...)
#
# rather than rebuilding silently. Doing it here rebuilds once, in the same environment the run
# uses, instead of failing mid-load.
#
# ⚠️ A failure here is deliberately not fatal, and that is load-bearing rather than sloppy. Measured
# 2026-09-13 under Julia 1.13: a stale `Zstd_jll` image in the depot made this step error out while
# the probe itself then ran to completion, compiling what it needed in-process. The script has no
# `set -e`, so it continues either way; the `||` makes that a decision instead of an accident, and
# puts a line in the log so a slow start is explicable.
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

julia --project "$SCRIPT"
