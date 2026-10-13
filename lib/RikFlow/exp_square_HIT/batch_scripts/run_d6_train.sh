#!/bin/bash
# The TRAINING hindcast (Rik, 2026-10-13): 5 ICs inside the fit window x 10 members, one array task
# per closure, a diagnostic of the fits in-sample (analysis/cards/round1.md, amendment 7).
#
#     sbatch --array=1-<n> batch_scripts/run_d6_train.sh <LinReg dir> [<LinReg dir> ...]   # from exp_square_HIT/
#
# e.g.  sbatch --array=1-3 batch_scripts/run_d6_train.sh LinReg1 LinReg6 LinReg7
#
# Task i runs closure `${!i}`: the prebuilt LinReg-path model output/TO_LRS/<name>/LinReg.jld2, all 5
# training ICs (`run_d6.jl 1,2,3,4,5`) in one process, into output/D6train_<name>. The IC packages
# are output/d6_ics_train (analysis/build_d6_ics.jl training; the manifest says set = "training", and
# run_d6.jl checks it against `select_train_ics`). Member seeds are member_seed(k, m), as in the
# held-out hindcast, so closures share their random numbers here too.
#
# 🔴 Settings are set HERE, from the arguments, never from the submitting environment
# (claude_memory.md #78), and printed in the log.
#
# Cost: 50 members x ~20 s + ~2 min start-up ~ 20 min per closure on gpu_a100 (~45 SBU).

#SBATCH -J d6-train
#SBATCH -t 01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
set -u
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

TASK=${SLURM_ARRAY_TASK_ID:?run as an array job: sbatch --array=1-<n> ... <names>}
[ "$#" -ge "$TASK" ] || { echo "run_d6_train.sh: task $TASK but only $# closure names given" >&2; exit 1; }
NAME=${!TASK}
MODEL=$PWD/output/TO_LRS/$NAME/LinReg.jld2
[ -f "$MODEL" ] || { echo "run_d6_train.sh: no model at $MODEL" >&2; exit 1; }

export D6_IC_DIR=$PWD/output/d6_ics_train
export D6_OUT=$PWD/output/D6train_$NAME
export D6_CLOSURE=lrs
export D6_MODEL=$MODEL
export D6_MEMBERS=10
unset D6_NLEAD D6_BLOCK D6_T_MIN D6_T_MAX D6_STRIDE RIKFLOW_GATE
mkdir -p "$D6_OUT"
echo "run_d6_train.sh: task $TASK closure $NAME model $D6_MODEL ICs $D6_IC_DIR out $D6_OUT members $D6_MEMBERS"

julia --project tools/run_d6.jl 1,2,3,4,5
