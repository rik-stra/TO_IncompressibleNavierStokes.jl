# One exploratory M4 fit, written as a deployable model -- for screening variants online.
#
#     RIKFLOW_X_TAG=<name> [overrides...] julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_explore_fit.jl
#
# Starts from a configuration-table cell (`RIKFLOW_X_CELL`, default 2 = StochLSTM2) and overrides:
#
#   RIKFLOW_X_ARCH      :storn | :vrnn | :vaernn        (where the latent noise enters)
#   RIKFLOW_X_BETA      KL weight
#   RIKFLOW_X_NLATENT   latent dimension
#   RIKFLOW_X_NHIDDEN   LSTM width (capacity)
#   RIKFLOW_X_WD        decoupled weight decay per update (default 0)
#   RIKFLOW_X_TRAINRANGE  "a,b" record columns to train on (default the cell's 400,4000 -- a protocol change)
#   RIKFLOW_X_HELDOUT   "a,b" held-out window for the R^2 score (default 4000,7600, or just past a
#                       longer training range)
#   RIKFLOW_X_EMISSION  none | constant | state_dependent  (an output noise channel)
#   RIKFLOW_X_TARGET    q | dQ | logr   (the level, the additive correction, or log1p(dQ/q*))
#   RIKFLOW_X_L / _BURN / _STRIDE   segmentation (default 500 / 100 / 100)
#   RIKFLOW_X_EPOCHS    epoch cap (default 3000); the windowed/floor stops end it earlier
#   RIKFLOW_X_LR        start rate (default the cell's, 1e-2; 1e-4 when warm-starting)
#   RIKFLOW_X_ROLLOUT   rollout horizon K (default 1 = teacher forcing)
#   RIKFLOW_X_INIT      a fit directory to warm-start from (its spec must match)
#
# Writes `output/TO_LSTM/explore/<tag>/StochLSTM_seed1.jld2`, deployable with
# `RIKFLOW_M4_MODEL_DIR=<that dir> julia ... 12_online_StochLSTM.jl <cell> <replica>`: the fit keeps
# the BASE cell's name, so the online driver's cell check passes with the base index, and the
# overrides are recorded in `extras.cfg` and `extras.overrides`.
#
# 🔴 Exploration, not results: one seed, a modest budget, screened on short online runs. Anything that
# survives is refitted properly before it is quoted.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Printf
using Statistics

const RF = RikFlow
Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
include(joinpath(@__DIR__, "m4_data.jl"))

envs(k, d) = (v = strip(get(ENV, "RIKFLOW_X_" * k, "")); isempty(v) ? d : v)
tag = envs("TAG", "")
isempty(tag) && error("set RIKFLOW_X_TAG to name this fit")
cell = parse(Int, envs("CELL", "2"))
base = load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")[cell]

init_dir = envs("INIT", "")
warm = !isempty(init_dir)
init_fit = warm ? RF.load_stochlstm(joinpath(isdir(init_dir) ? init_dir : joinpath(TO_folder, init_dir),
                                               "StochLSTM_seed1.jld2")) : nothing

ov = (; arch = Symbol(envs("ARCH", string(base.arch))),
      beta = parse(Float64, envs("BETA", string(base.beta))),
      n_latent = parse(Int, envs("NLATENT", string(base.n_latent))),
      n_hidden = parse(Int, envs("NHIDDEN", string(base.n_hidden))),
      # ⚠️ a longer training range is a PROTOCOL change (the project's partition is 1-10 TU train)
      train_range = Tuple(parse.(Int, split(envs("TRAINRANGE", join(base.train_range, ",")), ","))),
      L = parse(Int, envs("L", string(base.L))),
      burn = parse(Int, envs("BURN", string(base.burn))),
      # 🔑 emission noise: :none (latent only), :constant or :state_dependent -- a white noise
      # channel at the output, the analogue of the linear cells' eta (2026-09-23, wave 2)
      emission = Symbol(envs("EMISSION", string(get(base, :emission, :none)))))
target = Symbol(envs("TARGET", warm ? string(RF._lstm_target(init_fit.scaling)) : "q"))
stride = parse(Int, envs("STRIDE", "100"))
epochs = parse(Int, envs("EPOCHS", "3000"))
lr = parse(Float64, envs("LR", warm ? "1e-4" : string(base.lr)))
K = parse(Int, envs("ROLLOUT", "1"))
wd = parse(Float64, envs("WD", "0"))
stride <= ov.L - ov.burn || error("stride $stride > L - burn = $(ov.L - ov.burn)")
cfg = merge(base, ov)                    # keeps the base NAME, so the online cell check passes

rec = load_m4_qois(; track_file = joinpath(TO_folder, "..",
                                           "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
hist = RF.HistorySpec(; h = cfg.h, n_qoi = size(rec.q, 1), cfg.hist_var, cfg.include_predictor)
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, cfg.arch, cfg.uclip,
                   emission = get(cfg, :emission, :none))
dat = m4_training_data(rec, cfg, hist; target)
if warm
    init_fit.spec.arch == spec.arch && init_fit.spec.n_latent == spec.n_latent ||
        error("RIKFLOW_X_INIT has arch $(init_fit.spec.arch) / n_latent $(init_fit.spec.n_latent), " *
              "this fit $(spec.arch) / $(spec.n_latent)")
    RF._lstm_target(init_fit.scaling) == target || error("RIKFLOW_X_INIT has another target")
end
# the rollout's level lag for the :dQ target: scale_in(q* + dQ) = qstar_lag + a .* y .+ b
# (`:logr` has no linear lag map -- its level is q* exp(r) -- so it is teacher-forced only; the `dQ`
# fits showed the feedback costs nothing offline once the level comes from q*.)
target === :logr && K > 1 && error(":logr supports teacher forcing only (RIKFLOW_X_ROLLOUT = 1)")
lagmap = target !== :dQ ? nothing :
         (; a = Float32.(vec(dat.scaling.out_scaling.sigma) ./ vec(dat.scaling.in_scaling.sigma)),
          b = Float32.(vec(dat.scaling.out_scaling.mu) ./ vec(dat.scaling.in_scaling.sigma)))

@info "M4 explore" tag cell = cfg.name ov... target stride epochs lr rollout = K warm
common = (; cfg.L, cfg.burn, stride, cfg.batch, cfg.beta, cfg.val_frac, seed = 1, lagmap)
t0 = time()
ps, h = RF.train_stochlstm(spec, dat.Xc, dat.Yc, dat.steps; common..., epochs, lr, rollout = K,
                           init_ps = warm ? init_fit.extras.ps : nothing, verbose = false,
                           weight_decay = wd)
wall = time() - t0
# the exposure, measured the same way for every variant: its own output fed back for 100 steps
score(p, k) = RF.train_stochlstm(spec, dat.Xc, dat.Yc, dat.steps; common..., epochs = 1, lr = 0.0,
                                 rollout = k, init_ps = p, verbose = false)[2].val[1]
Kx = min(100, cfg.L - cfg.burn)
# the rollout score needs emission = :none (the feedback would be a draw); NaN for an emission head
tf = score(ps, 1)
# held-out skill on the CORRECTION, disjoint from training and from the stopping set
hw = let e = envs("HELDOUT", "")
    isempty(e) ? (cfg.train_range[2] <= 4000 ? (4000, 7600) :
                  (cfg.train_range[2] + 400, cfg.train_range[2] + 4000)) : Tuple(parse.(Int, split(e, ",")))
end
hs = m4_heldout_skill(spec, ps, rec, cfg, dat.scaling; window = hw, burn = cfg.burn)
ro = (RF.emission_noise(spec) || target === :logr) ? NaN : score(ps, Kx)

out_dir = joinpath(TO_folder, "explore", tag)
path = joinpath(out_dir, "StochLSTM_seed1.jld2")
RF.save_stochlstm(path, spec, RF.LSTMWeights(ps, spec), dat.scaling;
                  cfg, seed = 1, train_range = cfg.train_range, qoi_source = rec.source, losses = h,
                  ps, stride, batch = cfg.batch, rollout = K, target, overrides = ov, lr,
                  init_from = warm ? init_dir : "", val = (; tf, ro, Kx), weight_decay = wd,
                  heldout = hs)
RF.load_stochlstm(path)
@printf("%s: %s em %s beta %.0e nz %d L %d target %s K %d | %d updates %.1f min (%s) | val TF %.4g, rollout-%d %.4g (%.2fx) | gmax %.3g\n",
        tag, cfg.arch, cfg.emission, cfg.beta, cfg.n_latent, cfg.L, target, K, h.updates, wall / 60, h.stop_reason,
        tf, Kx, ro, ro / tf, maximum(h.gmax))
@printf("   best at update %d; train at end %.4g; held-out R2 (dQ, window %s): mean %.3f | %s
",
        h.best_update, h.train[end], string(hw), hs.r2mean, join((@sprintf("%.3f", r) for r in hs.r2), " "))
println("wrote $path")
