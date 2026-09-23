# Rollout fine-tuning of a teacher-forced M4 fit: the model's own output fed back into its level lags.
#
#     RIKFLOW_M4_INIT=<fit dir> julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_rollout_train.jl
#
# `<fit dir>` is a deployable fit directory under `output/TO_LSTM/` (a name or a path) -- e.g. the
# stride-100 export `StochLSTM2_s100b32_points3_cap10000`. Environment:
#
#   RIKFLOW_M4_INIT      the fit to warm-start from (required)
#   RIKFLOW_M4_ROLLOUT   horizon K (default L - burn: one free run per segment, deployment's shape;
#                        1 would be teacher forcing, i.e. no change)
#   RIKFLOW_M4_L / _BURN segment length / warm-up (default: the fit's own, 500 / 100). 🔴 Rik,
#                        2026-09-23: a 400-step free run may explode the gradient -- L = 200,
#                        burn = 100 (100 free steps) is the fallback, and at that L the fit's stride
#                        of 100 is exactly the tiling stride.
#   RIKFLOW_M4_STRIDE    training stride (default: the fit's; must be <= L - burn)
#   RIKFLOW_M4_LR        start rate (default 1e-4 -- a FINE-TUNE. Measured 2026-09-23 on the
#                        stride-100 fit: 1e-3 made both losses worse within 10 updates
#                        (rollout 1.01e-3 -> 1.99e-3); 1e-4 improved both over 40.)
#   RIKFLOW_M4_CLIP      gradient-norm clip (default 0 = off; `history.gmax` says if it is needed)
#   RIKFLOW_M4_EPOCHS    epoch cap (default 2000); the windowed stop ends it earlier
#
# Writes a DEPLOYABLE fit to `output/TO_LSTM/<init>_roll<K>_L<L>/StochLSTM_seed<seed>.jld2`, so
# `RIKFLOW_M4_MODEL_DIR` runs it online with no export step.
#
# 🔑 **What it reports, and why both columns.** The starting model and the fine-tuned one are each
# scored on the validation block twice: teacher-forced (what every earlier number in
# `results_LSTMS.md` is) and under the rollout being trained. The gap between the two columns for
# the STARTING model is the exposure the online plateau came from; what rollout training has to
# do is close it without giving the teacher-forced number away.
# 🔴 The scores come from `train_stochlstm` itself at `lr = 0` -- Adam does not move at a zero
# rate, so the one validation it runs IS the model's -- rather than from a second copy of the
# segmentation and the validation noise here, which could drift from the one the fit uses.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Printf

const RF = RikFlow

Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
init_arg = strip(get(ENV, "RIKFLOW_M4_INIT", ""))
isempty(init_arg) && error("set RIKFLOW_M4_INIT to the fit directory to warm-start from")
init_dir = isdir(init_arg) ? init_arg : joinpath(TO_folder, init_arg)
isdir(init_dir) || error("RIKFLOW_M4_INIT = $init_arg is not a directory (looked in $TO_folder)")
init_file = joinpath(init_dir, "StochLSTM_seed1.jld2")
isfile(init_file) || error("no StochLSTM_seed1.jld2 in $init_dir")

include(joinpath(@__DIR__, "m4_data.jl"))

fit = RF.load_stochlstm(init_file)
x = fit.extras
hasproperty(x, :ps) || error("$init_file stores no training parameters `ps` -- cannot warm-start")
cfg, spec, seed = x.cfg, fit.spec, x.seed
RF.emission_noise(spec) && error("rollout training needs emission = :none; this fit has $(spec.emission)")

L = parse(Int, get(ENV, "RIKFLOW_M4_L", string(cfg.L)))
burn = parse(Int, get(ENV, "RIKFLOW_M4_BURN", string(cfg.burn)))
stride = parse(Int, get(ENV, "RIKFLOW_M4_STRIDE", string(get(x, :stride, L - burn))))
K = parse(Int, get(ENV, "RIKFLOW_M4_ROLLOUT", string(L - burn)))
lr = parse(Float64, get(ENV, "RIKFLOW_M4_LR", "1e-4"))
clip = parse(Float64, get(ENV, "RIKFLOW_M4_CLIP", "0"))
epochs = parse(Int, get(ENV, "RIKFLOW_M4_EPOCHS", "2000"))
batch = get(x, :batch, cfg.batch)
stride <= L - burn || error("stride $stride > L - burn = $(L - burn): pass RIKFLOW_M4_STRIDE")
@info "M4 rollout fine-tune" init = basename(init_dir) cell = cfg.name beta = cfg.beta L burn stride batch rollout = K lr clip epochs

# --- exactly the training drivers' data preparation, and the fit's scaling must match it --------
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(TO_folder, "..", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
rec = load_m4_qois(; track_file)
a, b = cfg.train_range
_, in_scaling = RF._normalise(rec.q[:, a:(b - 1)]; normalization = cfg.normalization)
in_scaling.mu == fit.scaling.in_scaling.mu && in_scaling.sigma == fit.scaling.in_scaling.sigma ||
    error("the recomputed scaling differs from the fit's -- wrong record or train_range?")
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)
qs = RF.scale_input(rec.q[:, a:b], in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b - 1)], in_scaling)
X, Y, steps = RF.build_history(spec.hist, qss, qs)
Xc, Yc = permutedims(X), permutedims(Y)

common = (; L, burn, stride, batch, cfg.beta, cfg.val_frac, seed, stop_window = 500,
          stop_rel = 0.005)
"The model's validation loss under rollout horizon `k` (k = 1: teacher forcing)."
score(p, k) = RF.train_stochlstm(spec, Xc, Yc, steps; common..., epochs = 1, lr = 0.0,
                                 rollout = k, init_ps = p, verbose = false)[2].val[1]

p_init = x.ps
init_tf, init_ro = score(p_init, 1), score(p_init, K)
@printf("\nstarting model: validation %.4g teacher-forced, %.4g under rollout K = %d (%.2fx)\n",
        init_tf, init_ro, K, init_ro / init_tf)

t0 = time()
ps, h = RF.train_stochlstm(spec, Xc, Yc, steps; common..., epochs, lr, rollout = K,
                           init_ps = p_init, clip, verbose = true)
wall = time() - t0
fin_tf, fin_ro = score(ps, 1), score(ps, K)

println()
@printf("%-16s %14s %14s\n", "", "teacher-forced", "rollout K=$K")
@printf("%-16s %14.4g %14.4g\n", "starting model", init_tf, init_ro)
@printf("%-16s %14.4g %14.4g\n", "after rollout", fin_tf, fin_ro)
@printf("\n%d updates in %.1f min, %s; best at update %d; largest gradient norm %.3g (clip %s)\n",
        h.updates, wall / 60, h.stopped_early ? "stopped early ($(h.stop_reason))" : "hit the cap",
        h.best_update, maximum(h.gmax), clip > 0 ? string(clip) : "off")
any(!isfinite, h.gmax) && println("🔴 non-finite gradient norms -- the rollout exploded; lower L or K, or set RIKFLOW_M4_CLIP")

out_dir = joinpath(TO_folder, "$(basename(normpath(init_dir)))_roll$(K)_L$(L)")
path = joinpath(out_dir, "StochLSTM_seed$(seed).jld2")
RF.save_stochlstm(path, spec, RF.LSTMWeights(ps, spec), scaling;
                  cfg, seed, train_range = cfg.train_range, track_file, qoi_source = rec.source,
                  losses = h, ps, stride, batch, rollout = K, L, burn, lr, clip,
                  init_from = basename(normpath(init_dir)),
                  val = (; init_tf, init_ro, fin_tf, fin_ro))
fit2 = RF.load_stochlstm(path)                        # the round-trip the online driver does
println("wrote $path")
println("deploy with:  RIKFLOW_M4_MODEL_DIR=$(out_dir) sbatch batch_scripts/run_online.sh lstm " *
        "$(findfirst(c -> c.name == cfg.name, load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")))")
