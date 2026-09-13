#!/bin/bash
#SBATCH -J HF_ref
#SBATCH -t 30:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# The 100 TU high-fidelity reference regeneration: 512^3, Float64, LMWray3, 400,000 steps.
#
# Submit from either exp_square_HIT or lib/RikFlow:
#     sbatch batch_scripts/run_HF_ref.sh
#     sbatch exp_square_HIT/batch_scripts/run_HF_ref.sh
#
# Wall limit: 30 h against a measured 20.9 h. Measured 2026-09-13 by `hf_timing_probe.jl` on an
# H100 over 5000 steps at this exact configuration: 0.175 s per plain step, two independent
# projections agreeing to 1.1% (20.65 h and 20.87 h). The archived Float32/RK44 run took 19.3 h, so
# Float64 plus LMWray3 costs about 8% more.
#
# 🔑 The 9 h of headroom is not padding for its own sake. `create_ref_data` writes its output only
# at the very end, and its checkpoints are written but nothing reads them back — so a job that hits
# the limit loses the entire run rather than its last hour. At 24 h the margin was 15%, which one
# slow node eats. SLURM bills time used, not time requested, so the extra costs nothing.
#
# If this ever does need splitting, the restart is feasible but unwritten: `OU_advance!` replays the
# chain from (seed, n, Δt) alone (gotcha #33) and a checkpoint already holds `u_cpu` and `results`.
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
