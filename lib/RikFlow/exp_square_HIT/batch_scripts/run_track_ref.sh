#!/bin/bash
#SBATCH -J track_ref
#SBATCH -t 02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

# P2r/R1 -- the tracking run on the REGENERATED HF reference. 64^3 LF, Float64, LMWray3,
# 40,000 steps, 100 TU.
#
# Submit from either exp_square_HIT or lib/RikFlow:
#     sbatch batch_scripts/run_track_ref.sh
#     sbatch exp_square_HIT/batch_scripts/run_track_ref.sh
#
# 🔴 Needs output/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0_f64_lmwray3.jld2 -- the
# regenerated reference, NOT the archive under output/paper_data_HIT/. Override with RIKFLOW_HF_REF.
#
# Output: output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2, ~2.5 GiB
# (401 LF fields at 0.25 TU + the q/q_star/dQ/tau series). The driver refuses to start unless that
# much is free: track_ref accumulates every field in host memory and writes at the end, so running
# out of space costs the whole run -- the same failure mode 2_HF_ref.jl's preflight exists for.
#
# 🔑 tsim = 100, not 10. One run carries the 1-10 TU fit window AND the 401 fields D6 draws its
# initial conditions from, and it removes the archive's two-record dQ decorrelation trap
# (claude_memory.md #39) by construction. train_range = (400, 4000) selects t in [1, 10] out of
# whatever record it is given, so "fit to 10 TU" needs no change anywhere downstream.
#
# ⚠️ WALL TIME IS AN ESTIMATE, NOT A MEASUREMENT. Paper 2's Appendix G gives 5.0 s/TU for TO
# tracking, i.e. ~8 min for 100 TU -- but that is Float32/RK44 on a different machine, and a rate
# taken from a short Julia run is a compilation measurement wearing a throughput label (#44). Two
# hours is generous on purpose; SLURM bills time used, not time requested. Read the real rate out of
# the log and record it in analysis/results.md.
#
# After it finishes, check the tracking-error table the driver prints. The gate is deliberately
# loose (1e-1, a blow-up detector); the tight threshold gets set from that table, once, and
# RIKFLOW_TRACK_GATE overrides it meanwhile. Do not tighten it by guessing -- a threshold in the
# same range as the error it must detect is not a check (#45).

# Reuse the existing depot rather than building another. `JULIA_CPU_TARGET` multiversions the
# precompiled code across Zen2, Zen4 and Icelake-server, so **one depot serves the a100 and h100
# partitions** without recompiling per architecture -- which is why the name says h100 while this job
# asks for a100. That is deliberate: it is the populated depot every other script here uses, and
# splitting it per partition only means precompiling twice.
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# Find the driver from whichever directory the job started in, and say so if it is neither.
if [ -f 3_track_ref.jl ]; then
    SCRIPT=3_track_ref.jl
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    SCRIPT=exp_square_HIT/3_track_ref.jl
else
    echo "run_track_ref.sh: cannot find 3_track_ref.jl from $(pwd)" >&2
    echo "  submit from exp_square_HIT or from lib/RikFlow" >&2
    exit 1
fi

# Bring the depot's caches in line with this invocation before running. Needed because of
# JULIA_CPU_TARGET: multiversioning changes the content of the precompiled images, so a depot
# populated without it holds caches this run will not accept.
# ⚠️ Deliberately non-fatal -- a stale image can make this step error while the run itself completes,
# compiling in-process. No `set -e`; the `||` makes that a decision rather than an accident.
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

julia --project "$SCRIPT"
