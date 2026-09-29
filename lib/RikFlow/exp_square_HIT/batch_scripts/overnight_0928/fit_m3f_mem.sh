#!/bin/bash
# M3f with memory: recurrent :lstm over a W = 10 window of the h = 2 regressor, frozen lambda = 0 skip,
# 1-50 TU, scored 52-74 TU (results_LSTMS §11e's open item). Matched floors: 0.2098 (h = 2, one step),
# 0.2074 (stacked W = 10 window, least squares).
cd /export/scratch2/rik/time_series_TO/.claude/worktrees/overnight/lib/RikFlow || exit 1
export TMPDIR=/export/scratch2/rik/tmp
T=${1:-m3fmem_lstm_l0_wd1e-2_s1}
RIKFLOW_W_TAG=$T RIKFLOW_W_OUTSUB=p4 RIKFLOW_W_ARCH=${ARCH:-lstm} RIKFLOW_W_NZ=${NZ:-4} RIKFLOW_W_NH=16 \
RIKFLOW_W_W=10 RIKFLOW_W_H=2 RIKFLOW_W_HIST_VAR=q_star_q RIKFLOW_W_EMISSION=constant RIKFLOW_W_SEEDHEAD=1 \
RIKFLOW_W_LAMBDA=0 RIKFLOW_W_WD=${WD:-1e-2} RIKFLOW_W_WD_EXCLUDE=bd,Araw RIKFLOW_W_SEED=${SEED:-1} \
RIKFLOW_W_TRAIN_TU=50 RIKFLOW_W_SCORE_TU=52,74 RIKFLOW_W_LR=${LR:-1e-2} RIKFLOW_W_BATCH=256 RIKFLOW_W_VAL_EVERY=100 \
RIKFLOW_W_PATIENCE=30 RIKFLOW_W_STOP_PATIENCE=80 RIKFLOW_W_STOP_WINDOW=20000 RIKFLOW_W_EPOCHS=400 \
  julia -t 8 --startup-file=no --project=training exp_square_HIT/tools/m4_window_fit.jl
