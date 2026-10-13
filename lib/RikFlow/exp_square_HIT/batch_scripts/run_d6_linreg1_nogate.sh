#!/bin/bash
# LinReg1's D6 hindcast without the turbulence gate: ONLY the 35 members that fired (2026-10-13,
# handoff_gate_off_2026-10-13 Step 3, Rik chose option (a)).
#
#     sbatch batch_scripts/run_d6_linreg1_nogate.sh                 # from exp_square_HIT/
#
# 🔑 Why 35 of 900: the census (`analysis/gate_census.jl`) finds firings in 35 members of 27 ICs,
# none before lead 302. A member that never fired is its own gate-off run (same seed, same
# arithmetic: `abs(x) < 0` never holds), so only these 35 are rerun; the other 865 are copied into
# the scored directory afterwards.
#
# 🔴 The pairing check. `D6_PAIR_WITH` points at the gated originals: every member's seed must equal
# the original's (checked before the GPU time), and each output records `pair_first_diff`, the first
# `dQ` column where rerun and original differ, against `pair_expected` (the original's first
# firing). `pair_ok = false` on any member means this machine/software does not reproduce the
# September run, and the 35 must then also be rerun WITH the gate (same script, `RIKFLOW_GATE=on`,
# D6_OUT=$PWD/output/D6_LinReg1_rerun35) so the pair is clean.
#
# 🔑 gpu_a100: the originals came from `run_d6_linreg1.sh`, which was on gpu_a100 on 2026-09-16
# (42e9b1b0). Wall: 35 members x ~20-75 s (members that fire run longer) + ~2 min start-up.

#SBATCH -J d6-l1-nogate
#SBATCH -t 01:30:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL

set -u
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

# The 35 (k:member) pairs that fired in output/D6_LinReg1 (census 2026-10-13).
FIRED="42:3,42:10,46:2,57:3,61:1,85:3,143:3,150:7,166:8,170:1,170:7,193:3,197:3,197:7,224:2,224:3,224:5,228:3,231:8,239:4,243:10,251:8,255:1,255:10,258:8,282:8,313:6,316:2,316:4,324:5,328:4,336:1,336:6,340:7,359:5"

export RIKFLOW_GATE=${RIKFLOW_GATE:-off}
export D6_ONLY=${D6_ONLY:-$FIRED}
export D6_CLOSURE=lrs
export D6_MODEL=${D6_MODEL:-$PWD/output/TO_LRS/LinReg1/LinReg.jld2}
export D6_PAIR_WITH=${D6_PAIR_WITH:-$PWD/output/D6_LinReg1}
export D6_OUT=${D6_OUT:-$PWD/output/D6_LinReg1_nogate}
export D6_MEMBERS=10

[ -f "$D6_MODEL" ] || { echo "no model at $D6_MODEL" >&2; exit 1; }
[ -d "$D6_PAIR_WITH" ] || { echo "no originals at $D6_PAIR_WITH" >&2; exit 1; }
mkdir -p "$D6_OUT"
echo "== gate $RIKFLOW_GATE | model $D6_MODEL | paired with $D6_PAIR_WITH | out $D6_OUT"
echo "== members $D6_ONLY"
julia --project tools/run_d6.jl --only
