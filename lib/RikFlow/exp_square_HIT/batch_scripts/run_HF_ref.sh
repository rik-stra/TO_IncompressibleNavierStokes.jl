#!/bin/bash
#SBATCH -J HF_ref
#SBATCH -t 24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# The 100 TU high-fidelity reference regeneration: 512^3, Float64, LMWray3, 400,000 steps.
#
# Submit from either exp_square_HIT or lib/RikFlow:
#     sbatch batch_scripts/run_HF_ref.sh
#     sbatch exp_square_HIT/batch_scripts/run_HF_ref.sh
#
# 🔴 READ THIS BEFORE SUBMITTING. The wall limit is 24 h and the run may not fit in it.
#
# The archived Float32/RK44 reference recorded comptime = 69,329 s = 19.3 h for these same 400,000
# steps. This run is Float64, which roughly doubles the memory traffic this solver is bound by, and
# LMWray3, which removes one of RK44's four stages. Those pull in opposite directions and the net
# is not known — `hf_timing_probe.jl` exists to measure it and, as of 2026-09-13, has not been run
# at 512^3. The unmeasured estimate is 25–35 h, i.e. **over this limit**.
#
# And there is no resume. `create_ref_data` writes its output only at the very end; its checkpoints
# are written but nothing reads them back. A job killed at 24 h therefore loses the entire run, not
# the last hour of it.
#
# So one of these first:
#   1. `sbatch batch_scripts/run_hf_timing_probe.sh` (1 h) and read its projection. If it says
#      under ~19 h, this fits with margin and nothing else is needed.
#   2. Lower `tsim` in 2_HF_ref.jl to what does fit, and accept a shorter reference.
#   3. Write the restart. It is feasible — `OU_advance!` replays the chain from (seed, n, Δt)
#      alone (gotcha #33) and a checkpoint already holds `u_cpu` and `results` — but it does not
#      exist yet, and without it 24 h is a hard cap on a single job.
#
# Disk: the run refuses to start unless ~6.9 GiB is free (2.58 GiB final file, one 4.33 GiB
# checkpoint, plus a 15% margin). It prints the full table before doing anything expensive.
#
# Needs exp_square_HIT/output/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2 — the same
# archived spin-up cfl_probe.jl and hf_timing_probe.jl use.
#
# Output: output/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0_f64_lmwray3.jld2. The
# `_f64_lmwray3` suffix is deliberate: this must never be confusable with the archived Float32/RK44
# file, which is otherwise identically named.
#
# HF_REF_SMOKE=1 runs the 128^3 / 0.5 TU variant instead — but note it then wants a 128^3 spin-up
# initial condition, which the archive does not provide.

# Reuse the existing depot rather than building another. The CPU target multiversions the
# precompiled code across Zen2, Zen4 and Icelake-server, so one depot serves the a100 and h100
# partitions without recompiling per architecture.
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# Find the driver from whichever directory the job started in, and say so if it is neither.
if [ -f 2_HF_ref.jl ]; then
    SCRIPT=2_HF_ref.jl
elif [ -f exp_square_HIT/2_HF_ref.jl ]; then
    SCRIPT=exp_square_HIT/2_HF_ref.jl
else
    echo "run_HF_ref.sh: cannot find 2_HF_ref.jl from $(pwd)" >&2
    echo "  submit from exp_square_HIT or from lib/RikFlow" >&2
    exit 1
fi

# Bring the depot's caches in line with this invocation before running. Needed because of
# JULIA_CPU_TARGET: multiversioning changes the content of the precompiled images, so a depot
# populated without it holds caches this run will not accept, and julia errors rather than
# rebuilding silently.
#
# ⚠️ A failure here is deliberately not fatal. Measured 2026-09-13 under Julia 1.13: a stale
# Zstd_jll image made this step error out while the run itself completed, compiling in-process.
# The script has no `set -e`, so it continues either way; the `||` makes that a decision.
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

julia --project "$SCRIPT"
