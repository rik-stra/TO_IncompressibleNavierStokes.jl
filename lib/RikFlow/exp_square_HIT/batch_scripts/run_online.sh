#!/bin/bash
#SBATCH -J online
#SBATCH -t 06:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

# P2r/R2 -- fit a TO-LRS configuration and run it online, or run one of the other closures.
# Every case is 64^3 LF, Float64, LMWray3, 100 TU (40,000 steps).
#
# Usage (submit from exp_square_HIT or lib/RikFlow):
#     sbatch batch_scripts/run_online.sh lrs 1     # train LinReg<n>, then 5 online replicas
#     sbatch batch_scripts/run_online.sh ddn       # 5 DDN replicas
#     sbatch batch_scripts/run_online.sh smag      # Smagorinsky, c_s = 0.07
#     sbatch batch_scripts/run_online.sh nomodel   # no SGS model
#
# 🔴 Run `julia --project 4_setup_search.jl` ONCE first (it is instant and needs no GPU) to write
# output/TO_LRS/inputs_example.jld2. `lrs 1` is h = 5, lambda = 0 -- paper 2's headline
# configuration. It is NOT "the best": the paper says "one of the best performing models" and
# "near-optimal accuracy across a wide range of history lengths", and results.md finds LinReg1 and
# LinReg64 inside the 0.1848 KS noise floor of each other.
#
# 🔴 All four cases need R1's tracking record:
# output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2 (RIKFLOW_TRACK_FILE overrides).
# They take the initial field, the parameters and the OU forcing from it, and the two TO drivers
# assert the inherited ou_bodyforce.freeze == 1 -- which is what keeps the LF forcing aligned with
# the reference's freeze = 10 at a ten-times-smaller step (#33).
#
# ⚠️ WALL TIME IS AN ESTIMATE. Paper 2's Appendix G gives 5.8 s/TU for TO LRS and 2.8 s/TU with no
# model, so five 100 TU replicas is roughly 50 min -- on a different machine, in Float32, under
# RK44. Six hours is generous on purpose; SLURM bills time used. Record the real rate.
#
# 🔑 Smoke first. One replica at 1 TU costs seconds and catches a bad path, a missing file or a
# precision mismatch before five 100 TU runs do:
#     julia --project -e 'ENV["RIKFLOW_TRACK_FILE"]="..."; include("6_online_TO_LRS.jl")'
# with tsim edited down, or just watch the first timelogger lines of the real job.

set -u

CASE=${1:-}
INDEX=${2:-1}

case "$CASE" in
    lrs|ddn|smag|nomodel) ;;
    *)
        echo "run_online.sh: first argument must be one of lrs | ddn | smag | nomodel" >&2
        echo "  e.g. sbatch batch_scripts/run_online.sh lrs 1" >&2
        exit 1
        ;;
esac

# Reuse the existing depot rather than building another. `JULIA_CPU_TARGET` multiversions the
# precompiled code across Zen2, Zen4 and Icelake-server, so **one depot serves the a100 and h100
# partitions** without recompiling per architecture -- which is why the name says h100 while this job
# asks for a100. That is deliberate: it is the populated depot every other script here uses, and
# splitting it per partition only means precompiling twice.
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# Find the experiment directory from wherever the job started.
if [ -f 3_track_ref.jl ]; then
    EXP=.
elif [ -f exp_square_HIT/3_track_ref.jl ]; then
    EXP=exp_square_HIT
else
    echo "run_online.sh: cannot find the exp_square_HIT drivers from $(pwd)" >&2
    echo "  submit from exp_square_HIT or from lib/RikFlow" >&2
    exit 1
fi

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' ||
    echo "precompile step failed — continuing; the run will compile in-process (slower start)" >&2

case "$CASE" in
    lrs)
        # Train, then deploy the SAME index. 🔴 These two used to be able to disagree:
        # 6_online_TO_LRS.jl parsed ARGS[1] and then overwrote it with a hard-coded 2 (#55).
        echo "== training LinReg$INDEX"
        julia --project "$EXP/5_train_LinReg.jl" "$INDEX" || exit 1
        echo "== running LinReg$INDEX online, 5 replicas x 100 TU"
        julia --project "$EXP/6_online_TO_LRS.jl" "$INDEX"
        ;;
    ddn)
        # 🔴 New measurement, not reproduction: the DDN online runs have never existed in either
        # archive root. Only a precomputed ks_dists_DDN_smag_lf.jld2 survives, so there is no
        # archived trajectory to check these against.
        echo "== running DDN online, 5 replicas x 100 TU"
        julia --project "$EXP/7_online_DDN.jl"
        ;;
    smag)
        # c_s = 0.07 (Rik, 2026-09-14), not paper 2's tuned 0.071 -- and on upstream's Smagorinsky
        # kernels rather than the fork's, so this baseline is neither paper 2's constant nor paper
        # 2's code. Its summed KS must not be compared with the published 0.705 without saying so.
        echo "== running Smagorinsky online, c_s = 0.07, 100 TU"
        julia --project "$EXP/8_smag_online.jl"
        ;;
    nomodel)
        echo "== running no-SGS online, 100 TU"
        julia --project "$EXP/9_no_sgs.jl"
        ;;
esac
