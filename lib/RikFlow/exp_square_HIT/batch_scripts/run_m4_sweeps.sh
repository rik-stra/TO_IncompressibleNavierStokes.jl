#!/bin/bash
#SBATCH -J m4_sweep
# 🔴 **2 h, and the old 18 min measurement NO LONGER APPLIES.** It was taken when each point
# ran to a fixed 3000-UPDATE budget; since 2026-09-18 every point runs to a 3000-EPOCH cap with
# early stopping, which is ~1.8x the updates before any point converges out. The number has not
# been re-measured on this node.
# 🔑 **Re-measure it before trusting it:** `sbatch batch_scripts/run_m4_sweeps.sh smoke` now
# projects under the new policy and prints a suggested walltime. Until then this is a guess with
# margin, which is the one thing a walltime may not silently be.
# ⚠️ The only number in hand is **119 min on the Windows workstation** (its stage 9, 2026-09-18),
# which suggested `-t 04:00:00` THERE. That machine has run ~4x slower than this node before, but
# a cross-machine ratio is not a measurement -- that mistake already inverted the GPU verdict once
# (`claude_memory.md`). So: 2 h here as a bounded guess, and the node's own smoke settles it.
# ⚠️ `M4_DEVICE=cuda` was ~7x slower than this node's CPU -- raise to `-t 08:00:00` if used.
# ⚠️ The `smoke` mode itself takes ~3 minutes; this is the cap for the scans.
#SBATCH -t 02:00:00
# 🔒 Stays on gpu_h100: the allocation has access to GPU nodes only (Rik, 2026-09-23), so a CPU
# partition is not an option. Only the DEVICE is the CPU (`M4_DEVICE=cpu`), and that is also the
# configuration that was measured -- every CPU timing in `results_LSTMS.md` §5 is this node's.
# ⚠️ The GPU is requested and, at `M4_DEVICE=cpu`, sits idle. That is a known cost, accepted.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
# 🔑 Explicit, though it is also SLURM's default: the job inherits the SUBMITTING environment.
# That is what makes `RIKFLOW_M4_EPOCHS=5 sbatch ...` and `M4_DEVICE=cpu sbatch ...` work, and it
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
#     RIKFLOW_M4_INIT=StochLSTM2_s100b32_points3_cap10000 sbatch batch_scripts/run_m4_sweeps.sh rollout
#                                                          # rollout fine-tune of an exported fit;
#                                                          # RIKFLOW_M4_ROLLOUT / _L / _BURN / _LR / _CLIP
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
#     RIKFLOW_M4_EPOCHS      epoch CAP per stride point   (default 3000; early stopping ends
#                            each point once it has converged, so this bounds rather than sets it)
#     RIKFLOW_M4_STOP_PATIENCE  validations past the best before stopping (default 100)
#     RIKFLOW_M4_VAL_EVERY   optimiser updates between validations (default 2). 🔴 Everything
#                            paced by it -- the plateau rule and the early stop -- is then paced
#                            in UPDATES, identically at every point, rather than by an epoch that
#                            is 1 update at the tiling stride and 3 at stride 20.
#     RIKFLOW_M4_STOP_WINDOW / RIKFLOW_M4_STOP_REL  stop once the best improved by < REL over
#                            the last WINDOW updates (default 500 / 0.005; WINDOW=0 disables)
#     RIKFLOW_M4_POINTS      stride scan only: comma-separated 1-based rows to run (default
#                            all five). A subset writes its OWN file, never the canonical one:
#     RIKFLOW_M4_POINTS=2,5 RIKFLOW_M4_EPOCHS=10000 sbatch batch_scripts/run_m4_sweeps.sh stride
#
# ---------------------------------------------------------------------------------------------
# 🔒 IT TRAINS ON THE CPU (`M4_DEVICE=cpu`), because the GPU was measured ~7x SLOWER
# ---------------------------------------------------------------------------------------------
#
# `M4_DEVICE` decides where the fit runs and **defaults to `cpu`**. Override per submission:
#
#     M4_DEVICE=cuda sbatch -t 08:00:00 batch_scripts/run_m4_sweeps.sh stride    # ~7x slower
#
# ✅ **The GPU path WORKS** -- `M4 SMOKE PASS on device=cuda` on `gpu_h100`, 2026-09-18, every
# stage including the save/load round-trip -- and it LOSES: 96 min against 14 min for the whole
# old stride scan on the same node, after both cheap optimisations (fused gate activations, zero
# per-update transfers). The loop is bound by the length of its dependent kernel-launch chain,
# set by `L = 500` sequential timesteps, and no differentiable fused RNN exists in Julia
# (`results_LSTMS.md` §5). Do not spend more on the GPU for M4.
#
# ⚠️ `OPENBLAS_NUM_THREADS=1` is set below: on `64 x 23` matrices a full BLAS pool contends rather
# than helps.
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
    rollout) SCAN=m4_rollout_train.jl ;;
    *) echo "run_m4_sweeps.sh: first argument must be 'smoke', 'lr', 'stride' or 'rollout'; got '${WHICH}'" >&2
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

# Train on the CPU unless the submitting shell says otherwise. SLURM's default `--export=ALL`
# carries `M4_DEVICE=cuda sbatch ...` through, so this is a default and not an override.
# 🔑 `M4_DEVICE=cuda` works here as-is -- the partition still provides the GPU; it is only the
# default that changed. Give it `-t 08:00:00`, because it is ~7x slower.
export M4_DEVICE=${M4_DEVICE:-cpu}
echo "== M4_DEVICE=$M4_DEVICE"
# 🔑 Echo the budgets that actually arrived. Whether an env var survives submission is the
# kind of thing that is easy to assume and expensive to assume wrongly, so the log answers
# it rather than the reader inferring it. The Julia driver prints them again from its own
# side (`@info "M4 <which> scan" ... updates=/epochs=`), so the two together show the value
# crossed the shell/Julia boundary too.
# 🔴 One echo, one line. The earlier `"..."\n     "..."` form put a literal `n` into the log
# (`<default 3000>n`): outside quotes `\n` is an escaped 'n', not a newline and not a line
# continuation. Same class as the `\r` that once broke the reproduce block in results_LSTMS.md.
echo "== budget: RIKFLOW_M4_EPOCHS=${RIKFLOW_M4_EPOCHS:-<default 3000>} RIKFLOW_M4_STOP_PATIENCE=${RIKFLOW_M4_STOP_PATIENCE:-<default 100>} RIKFLOW_M4_VAL_EVERY=${RIKFLOW_M4_VAL_EVERY:-<default 2>} RIKFLOW_M4_LR_EPOCHS=${RIKFLOW_M4_LR_EPOCHS:-<default 1500>} M4_DEVICE_RNG=${M4_DEVICE_RNG:-0} RIKFLOW_M4_POINTS=${RIKFLOW_M4_POINTS:-<all>} RIKFLOW_M4_STOP_WINDOW=${RIKFLOW_M4_STOP_WINDOW:-<default 500>} RIKFLOW_M4_STOP_REL=${RIKFLOW_M4_STOP_REL:-<default 0.005>}"

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

if [ "$WHICH" = "rollout" ]; then
    echo "== M4 rollout fine-tune from ${RIKFLOW_M4_INIT:-<RIKFLOW_M4_INIT NOT SET>} (K ${RIKFLOW_M4_ROLLOUT:-<L-burn>}, L ${RIKFLOW_M4_L:-<fit>}, burn ${RIKFLOW_M4_BURN:-<fit>}, lr ${RIKFLOW_M4_LR:-<1e-4>}, clip ${RIKFLOW_M4_CLIP:-<off>})"
    julia --project="$ROOT/training" "$EXP/tools/$SCAN"
elif [ "$WHICH" = "smoke" ]; then
    echo "== M4 smoke (device $M4_DEVICE)"
    julia --project="$ROOT/training" "$EXP/tools/$SCAN"
else
    echo "== M4 $WHICH scan, cell $CELL, seed $SEED (device $M4_DEVICE)"
    julia --project="$ROOT/training" "$EXP/tools/$SCAN" "$CELL" "$SEED"
fi
