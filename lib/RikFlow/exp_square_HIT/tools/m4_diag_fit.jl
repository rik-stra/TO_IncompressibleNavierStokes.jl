# One M4 fit with its checkpoints scored on a HELD-OUT window -- for diagnosing overfitting.
#
#     RIKFLOW_D_TAG=<name> [overrides...] julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_diag_fit.jl
#
# Overrides (defaults = the explore fits of `m4_explore_fit.jl`: cell 2, `:dQ` target):
#
#   RIKFLOW_D_ARCH / _NH / _NZ / _BETA / _EMISSION   architecture, hidden, latent, KL weight, emission
#   RIKFLOW_D_H          level/predictor lags in the input (default the cell's, 1)
#   RIKFLOW_D_TARGET     dQ | logr | q
#   RIKFLOW_D_L / _BURN / _STRIDE / _BATCH           segmentation and batch
#   RIKFLOW_D_LR / _WD / _CLIP / _EPOCHS / _SEED     optimiser
#   RIKFLOW_D_TRAIN_TU   end of the training range in TU (default 10 = the project's 1-10 TU)
#   RIKFLOW_D_EVAL_EVERY updates between held-out scorings (default 10)
#   RIKFLOW_D_SKIP       1 = linear skip `y += Ws x`, seeded with the least-squares map and `V1 = 0`,
#                        so update 0 IS the linear model and the recurrence learns its residual
#   RIKFLOW_D_SEEDHEAD   1 = (with SKIP and an emission head) seed the head at the linear map's
#                        training-residual covariance -- `bd = log sd`, `A'A = R^{-1}` -- so update 0
#                        is the linear mean with M0's eta; otherwise it starts at Sigma = I
#   RIKFLOW_D_FREEZE     1 = hold `Ws` at the least-squares map (default 0: trained jointly), or a
#                        comma list of parameter names, e.g. `Ws,V1,Wx,Wh,b` (with SKIP: V1 stays 0,
#                        so the LSTM is switched off -- the linear mean + emission head baseline)
#
# 🔑 **The held-out window is 50-100 TU for every fit**, whatever the training range, so fits trained
# on 1-10, 1-25 and 1-50 TU are scored on the same rows. Scored as ONE teacher-forced pass with the
# latent at its mean (`eps = 0`) -- the network is never reset online either -- after a 100-step
# warm-up. The loss is `elbo`'s reconstruction term: 0.5 x SSE per step, summed over the six QoIs, in
# the standardised target, so the inner validation curve and the held-out curve are in one unit.
# A least-squares fit on the same regressor is scored the same way as the floor to beat.
#
# 🔴 The saved model is the one the PROTOCOL picks -- best inner validation, the library's return
# value. The held-out curve only diagnoses; the update it would have picked is recorded, not used.
#
# Writes `output/TO_LSTM/diag/<tag>/StochLSTM_seed1.jld2` (deployable like an explore fit, with
# `RIKFLOW_M4_MODEL_DIR`) and `curve.jld2` (inner val, held-out and linear-baseline numbers).

using RikFlow
using Lux, Optimisers, Zygote
using JLD2, Printf, Statistics, LinearAlgebra, Random

const RF = RikFlow
Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
include(joinpath(@__DIR__, "m4_data.jl"))

envs(k, d) = (v = strip(get(ENV, "RIKFLOW_D_" * k, "")); isempty(v) ? d : v)
tag = envs("TAG", "")
isempty(tag) && error("set RIKFLOW_D_TAG to name this fit")
base = load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")[2]

const DT = 2.5e-3
train_tu = parse(Float64, envs("TRAIN_TU", "10"))
btrain = round(Int, train_tu / DT)                   # train_range = (400, btrain), as the cells
ov = (; arch = Symbol(envs("ARCH", "vrnn")), n_hidden = parse(Int, envs("NH", string(base.n_hidden))),
      n_latent = parse(Int, envs("NZ", string(base.n_latent))), beta = parse(Float64, envs("BETA", "1e-4")),
      emission = Symbol(envs("EMISSION", "none")), h = parse(Int, envs("H", string(base.h))),
      L = parse(Int, envs("L", "500")), burn = parse(Int, envs("BURN", "100")),
      batch = parse(Int, envs("BATCH", "32")), lr = parse(Float64, envs("LR", "1e-2")),
      train_range = (base.train_range[1], btrain))
target = Symbol(envs("TARGET", "dQ"))
stride = parse(Int, envs("STRIDE", "100"))
wd = parse(Float64, envs("WD", "0"))
clip = parse(Float64, envs("CLIP", "0"))
epochs = parse(Int, envs("EPOCHS", "3000"))
seed = parse(Int, envs("SEED", "1"))
eval_every = parse(Int, envs("EVAL_EVERY", "10"))
cfg = merge(base, ov)

rec = load_m4_qois(; track_file = joinpath(TO_folder, "..",
                                           "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
hist = RF.HistorySpec(; h = cfg.h, n_qoi = size(rec.q, 1), cfg.hist_var, cfg.include_predictor)
skip = envs("SKIP", "0") == "1"
frz = envs("FREEZE", "0")
freeze = frz == "0" ? () : frz == "1" ? (:Ws,) : Tuple(Symbol.(strip.(split(frz, ","))))
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, cfg.arch, cfg.uclip, cfg.emission, skip)

# --- training data: exactly what every M4 driver fits on ------------------------------------------
dat = m4_training_data(rec, cfg, hist; target)

# --- the whole record in the SAME scaling, for the held-out window --------------------------------
function full_record(rec, scaling, hist, a, target)
    b = size(rec.q_star, 2)
    qs = RF.scale_input(rec.q[:, a:(b + 1)], scaling.in_scaling)
    qss = RF.scale_input(rec.q_star[:, a:b], scaling.in_scaling)
    X, _, steps = RF.build_history(hist, qss, qs)
    cols = (a - 1) .+ steps
    D = rec.dQ[:, cols]
    target === :logr && (D = log1p.(D ./ rec.q_star[:, cols]))
    Y = target === :q ? permutedims(RF.build_history(hist, qss, qs)[2]) : RF.scale_input(D, scaling.out_scaling)
    return permutedims(X), Y, cols
end
Xf, Yf, colsf = full_record(rec, dat.scaling, hist, cfg.train_range[1], target)
const HELD = findall(c -> 20_000 <= c, colsf)         # 50-100 TU
const WARM = 100
Xh, Yh = Float32.(Xf[:, HELD]), Float32.(Yf[:, HELD])

# held-out NLL: `elbo` at beta = 0 over the whole window as one segment -- the Gaussian NLL per step
# for an emission head, 0.5 x SSE for `:none` (no density: then it equals the first number)
heldnll(ps) = RF.elbo(spec, ps, reshape(Xh, size(Xh, 1), :, 1), reshape(Yh, size(Yh, 1), :, 1),
                      (WARM + 1):size(Xh, 2), zeros(Float32, spec.n_latent, size(Xh, 2), 1); beta = 0)
function heldout(ps)
    o = RF.lstm_forward(spec, ps, Xh, zeros(Float32, spec.n_latent, size(Xh, 2)))
    r = Yh[:, (WARM + 1):end] .- o.Y[:, (WARM + 1):end]
    return 0.5 * sum(abs2, r) / size(r, 2), vec(1 .- sum(abs2, r; dims = 2) ./ sum(abs2, Yh[:, (WARM + 1):end] .- mean(Yh[:, (WARM + 1):end]; dims = 2); dims = 2))
end

# --- the linear floor: least squares on the same regressor, the same rows -----------------------
# 🔴 Fitted on the TRAINING rows only (the inner split's first 80%), so the linear seed of a skip
# fit has seen exactly what the network trains on and nothing of its early-stopping block.
ntr = floor(Int, (1 - cfg.val_frac) * size(dat.Xc, 2))
Clin = Float64.(dat.Xc[:, 1:ntr])' \ Float64.(dat.Yc[:, 1:ntr])'     # (n_in x N_Q)
rl = Yh[:, (WARM + 1):end] .- (Clin' * Xh)[:, (WARM + 1):end]
lin = 0.5 * sum(abs2, rl) / size(rl, 2)
# the linear map's Gaussian NLL with the training residual's full covariance -- M0's eta, in this unit
Sig = cov(permutedims(Float64.(dat.Yc[:, 1:ntr]) .- Clin' * Float64.(dat.Xc[:, 1:ntr])))
linnll = 0.5 * (6 * log(2pi) + logdet(Sig) + mean(sum(rl .* (Sig \ rl); dims = 1)))

# --- fit, scoring checkpoints -------------------------------------------------------------------
curve = (; update = Int[], val = Float64[], held = Float64[])
cb(upd, ps, vl) = (upd % eval_every == 0) && (push!(curve.update, upd); push!(curve.val, vl);
                                              push!(curve.held, heldout(ps)[1]))
lagmap = target !== :dQ ? nothing :
         (; a = Float32.(vec(dat.scaling.out_scaling.sigma) ./ vec(dat.scaling.in_scaling.sigma)),
          b = Float32.(vec(dat.scaling.out_scaling.mu) ./ vec(dat.scaling.in_scaling.sigma)))
@info "M4 diag" tag ov... target stride wd clip epochs seed ntrain = size(dat.Xc, 2) held = length(HELD) linear_floor = lin
init_ps = nothing
if skip
    p0 = RF.init_lstm_params(Xoshiro(seed), spec)
    init_ps = merge(p0, (; Ws = Float32.(permutedims(Clin)), V1 = zero(p0.V1)))
    if envs("SEEDHEAD", "0") == "1"
        RF.emission_noise(spec) || error("RIKFLOW_D_SEEDHEAD needs an emission head")
        Rtr = Float64.(dat.Yc[:, 1:ntr]) .- Clin' * Float64.(dat.Xc[:, 1:ntr])
        sd = vec(std(Rtr; dims = 2))
        P = inv(Symmetric(cor(permutedims(Rtr))))
        J = reverse(Matrix{Float64}(I, 6, 6); dims = 1)
        A = J * cholesky(Symmetric(J * P * J)).U * J          # lower triangular, A'A = R^{-1}
        Araw = tril(A, -1) + Diagonal(log.(diag(A)))
        init_ps = merge(init_ps, (; bd = Float32.(log.(sd)), Araw = Float32.(Araw)))
    end
end
t0 = time()
ps, h = RF.train_stochlstm(spec, dat.Xc, dat.Yc, dat.steps; cfg.L, cfg.burn, stride, cfg.batch, cfg.beta,
                           cfg.val_frac, seed, lagmap, epochs, cfg.lr, weight_decay = wd, clip,
                           verbose = false, callback = cb, init_ps, freeze)
wall = time() - t0
hbest, r2 = heldout(ps)
hnll = heldnll(ps)
isempty(curve.held) && (push!(curve.update, h.updates); push!(curve.val, h.best_val); push!(curve.held, hbest))
ib = argmin(curve.held)

out_dir = joinpath(TO_folder, "diag", tag)
mkpath(out_dir)
RF.save_stochlstm(joinpath(out_dir, "StochLSTM_seed1.jld2"), spec, RF.LSTMWeights(ps, spec), dat.scaling;
                  cfg, seed, train_range = cfg.train_range, qoi_source = rec.source, losses = h, ps, stride,
                  batch = cfg.batch, rollout = 1, target, overrides = ov, lr = cfg.lr,
                  init_from = "", val = (; tf = h.best_val, ro = NaN, Kx = 0),
                  diag = (; held = hbest, r2, linear_floor = lin, held_nll = hnll, linear_nll = linnll, wd, skip, freeze))
jldsave(joinpath(out_dir, "curve.jld2"); curve, lin, linnll, hnll, hbest, r2, best_update = h.best_update, ov, target, stride, wd, skip, freeze)
@printf("%-26s %-6s %-5s sk %d%s nh %2d nz %d h %d L %4d str %3d wd %.0e TU %3.0f | %4d upd, best@%4d (%s) %.1f min | val %.3f | held %.3f (lin %.3f) held-best %.3f@%d | NLL %.3f (lin %.3f) | R2 %s\n",
        tag, cfg.arch, string(cfg.emission)[1:min(5, end)], skip, isempty(freeze) ? " " : "f", cfg.n_hidden, cfg.n_latent, cfg.h, cfg.L, stride, wd, train_tu, h.updates, h.best_update,
        h.stop_reason, wall / 60, h.best_val, hbest, lin, curve.held[ib], curve.update[ib], hnll, linnll,
        join((@sprintf("%.2f", x) for x in r2), " "))
