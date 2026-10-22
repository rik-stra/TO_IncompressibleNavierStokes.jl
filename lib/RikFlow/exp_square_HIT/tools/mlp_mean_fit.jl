# Step 3a, MLP + MVG: the network mean of paper Sec. 7 (closures.tex, `sec:lever-net`), fitted by the
# protocol of `sec:closures-training`, and its predictions on the whole record for the offline residual
# analysis of paper Sec. 6 (`analysis/mlp_residual.jl`).
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/mlp_mean_fit.jl
#
# 🔑 The model (closures.tex, Eq. MLP mean): dQ^n = f(P x^n) + eta, with x^n the regressor of LinReg1
# (h = 5, `[q*_n; q_{n-1}, q*_{n-1}; ...; q_{n-5}, q*_{n-5}; 1]`, 67 entries) and P the fixed map to
# the dQ history plus the predictor increments, `[q*_n; dQ_{n-1}, q*_{n-1}; ...; dQ_{n-5}, q*_{n-5}; 1;
# Dq*_{n-1}; ...; Dq*_{n-5}]`, 97 entries (Rik, 2026-10-22; see FEAT below), f a tanh MLP with NL = 4
# hidden layers of a common width (Rik: "depth 4 with {16,64,128}")
# and no linear skip (`arch = :dense`, `n_layers = 4`, `n_latent = 0`, window 1), eta a constant
# correlated Gaussian. The target is dQ^n = q^n - q^{n*} from the record's levels, standardised (not the
# recorded correction; see the 🔴 note at the target).
#
# The protocol, stage by stage:
#   1. For every width in NHS, weight decay in WDS and seed in SEEDS: train on 1-8.2 TU (the first
#      80 % of the 1-10 TU rows) on the one-step Gaussian likelihood with a constant head trained
#      jointly, early-stopped on 8.2-10 TU (the library's best-validation iterate). Weight decay acts
#      on the network only (`decay_exclude = (:bd, :Araw)`).
#   2. The (width, weight decay) with the lowest validation loss, averaged over the seeds, is chosen.
#   3. Each seed is trained AGAIN on all of 1-10 TU at the chosen setting, for as many updates as its
#      stage-1 best iterate took, at the learning rate stage 1 used at each update (its plateau
#      schedule, replayed; no validation, the last iterate is kept).
#   The covariance Sigma of the closure (refitted to the stage-3 residuals) is NOT made here (Rik,
#   2026-10-22): the offline residual analysis does not use it, and whether it is taken about zero
#   or about the residual's mean is open. It is fitted when the step-3a closure is built for coupled
#   runs, so this tool writes no deployable file.
#
# The offline check of `sec:closures-training`: the stage-1 network's one-step NLL on 8.2-10 TU next to
# least squares on the same regressor fitted on 1-8.2 TU, with its training residual's covariance
# (same rows, same unit: the Gaussian NLL per step of the standardised dQ, summed over the QoIs).
#
# Settings: the M3ᶠ fits' (results_LSTMS.md §11: batch 256, lr 1e-2, a validation every 100 updates,
# patience 30, stop patience 80, stop window 20 000 updates), with the cap at 20 000 updates.
# Weight decay {0, 1e-2, 1e-1} (Rik, 2026-10-22: "do include 0 weight decay").
# Overrides: RIKFLOW_MLP_NHS (widths), _NL (depth), _WDS, _SEEDS, _H, _FEAT (dq | raw), _OUT;
# _MAXUPD for a smoke only.
#
# Writes to `output/TO_LSTM/<OUT>/`: `fits.jld2` (every stage-1 fit: curves, validation, the LS
# floor; the choice; the stage-3 parameters) and `pred.jld2` (the predicted dQ of the chosen setting's
# stage-1 and stage-3 fits on every step of the record, raw units, with the record columns), which
# the analysis reads without Lux.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2, Printf, Statistics, LinearAlgebra, Random

const RF = RikFlow
Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

include(joinpath(@__DIR__, "step3_data.jl"))        # record, target, input map, checks, LS floor

const NHS = envv("NHS", "16,64,128", Int)
const NL = parse(Int, envs("NL", "4"))
const WDS = envv("WDS", "0,1e-2,1e-1", Float64)
const SEEDS = envv("SEEDS", "1,2,3", Int)
# Probes (Rik, 2026-10-22, after the grid missed LinReg1): RIKFLOW_MLP_LR (default 1e-2), _LOSS = nll
# (default, the protocol) | mse (the mean trained on squared error: the head is frozen at Sigma = I, so
# the one-step likelihood is 0.5 SSE + const; Sigma is refitted afterwards in any case), _STAGE3 = 0
# skips the 1-10 TU refit and pred.jld2 (stage 1 and its scores only).
# Second probes (Rik, 2026-10-22): _LOSS = rmse, the per-QoI RMSE summed over the QoIs (each QoI's error
# relative to its own size; a custom loop, `train_rmse`, with the library's stopping rules), and
# _SEEDHEAD = 1, the likelihood head started at least squares' residual covariance (`LS_HEAD`) instead of
# Sigma = I, so the likelihood weights the QoIs as the achievable error does from the first update.
const LR = parse(Float64, envs("LR", "1e-2"))
const LOSS = Symbol(envs("LOSS", "nll"))
LOSS in (:nll, :mse, :rmse) || error("RIKFLOW_MLP_LOSS must be nll, mse or rmse; got $LOSS")
const FREEZE = LOSS === :nll ? () : (:bd, :Araw)            # the head is unused / fixed by mse and rmse
const SEEDHEAD = envs("SEEDHEAD", "0") == "1"
SEEDHEAD && LOSS !== :nll && error("RIKFLOW_MLP_SEEDHEAD seeds the likelihood head: it needs LOSS = nll")
const STAGE3 = envs("STAGE3", "1") == "1"
STAGE3 && LOSS === :rmse && error("the 1-10 TU refit is not implemented for LOSS = rmse; set RIKFLOW_MLP_STAGE3 = 0")
const ST = (; batch = 256, lr = LR, val_every = 100, patience = 30, stop_patience = 80,
            stop_window = 20_000, max_updates = parse(Int, envs("MAXUPD", "20000")))
const OUTSUB = envs("OUT", "step3/MLP_h5_d$(NL)_$(FEAT)")
const MAXEPOCHS = cld(ST.max_updates, fld(ntr, ST.batch))

# the MLP's settings in `cfg` (saved with the fits)
cfg = merge(cfg, (; L = 1, burn = 0, batch = ST.batch, lr = ST.lr))
mkspec(H) = RF.LSTMSpec(; hist, n_hidden = H, cfg.n_latent, cfg.n_encoder, cfg.arch, cfg.uclip,
                        cfg.emission, skip = false, window = 1, n_layers = NL,
                        n_in = nin == nreg ? 0 : nin)

@info "MLP + MVG, step 3a" nin NL NHS WDS SEEDS rows = ncol fit = (tu(1), tu(ntr)) validation = (tu(ntr + 1), tu(ncol))

# --- the network's output on columns of a regressor matrix, as the deployed forward pass ----------
predict(spec, ps, X) = Float64.(RF.lstm_forward(spec, ps, reshape(T.(X), nin, 1, :),
                                                zeros(T, 0, 1, size(X, 2))).Y[:, 1, :])
nparams(ps) = sum(length(v) for v in (ps.Wx, ps.Wh, ps.b, ps.V1, ps.cdec))
"Fresh parameters for `spec` at `seed`, with the head at least squares' residual covariance if SEEDHEAD."
init_params(spec, seed) = (p = RF.init_lstm_params(Xoshiro(seed), spec; T); SEEDHEAD ? merge(p, LS_HEAD) : p)

"""
LOSS = rmse: the MLP trained on the per-QoI RMSE summed over the QoIs (`rmse_sum`), minibatches of the
1-8.2 TU rows, validated on 8.2-10 TU every `val_every` updates; the best iterate is kept, the learning
rate x0.3 after `patience` validations without a new best (floor 1e-4), stopped at the floor after
`stop_patience` more, capped at `max_updates`. Returns `(ps, h)` with the fields the grid reads.
"""
function train_rmse(sp, seed, wd)
    rng = Xoshiro(seed)
    ps = RF.init_lstm_params(rng, sp; T)
    rule = wd > 0 ? Optimisers.AdamW(T(ST.lr), (T(0.9), T(0.999)), T(wd); couple = true) :
           Optimisers.Adam(T(ST.lr))
    opt = Optimisers.setup(rule, ps)
    for k in FREEZE                                         # the head: unused by this loss
        Optimisers.freeze!(getfield(opt, k))
    end
    f(p, X) = RF.lstm_forward(sp, p, X, zeros(T, 0, 1, size(X, 3))).Y[:, 1, :]
    X3, Y3 = reshape(Xc[:, 1:ntr], nin, 1, ntr), T.(Ytr)
    Xv3, Yv3 = reshape(T.(Xva), nin, 1, :), T.(Yva)
    vloss(p) = Float64(rmse_sum(f(p, Xv3), Yv3))
    best = (; val = vloss(ps), ps = deepcopy(ps), update = 0)
    hist_u, hist_lr, hist_v = Int[], Float64[], Float64[]
    cur, since_best, since_decay, upd, stop = T(ST.lr), 0, 0, 0, :cap
    nb = fld(ntr, ST.batch)
    while upd < ST.max_updates && stop === :cap
        idxs = shuffle(rng, 1:ntr)
        for k in 1:nb
            c = idxs[((k - 1) * ST.batch + 1):(k * ST.batch)]
            Xb, Yb = X3[:, :, c], Y3[:, c]
            g = Zygote.gradient(p -> rmse_sum(f(p, Xb), Yb), ps)[1]
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
            upd % ST.val_every == 0 || continue
            v = vloss(ps)
            push!(hist_u, upd); push!(hist_lr, cur); push!(hist_v, v)
            if v < best.val
                best = (; val = v, ps = deepcopy(ps), update = upd)
                since_best, since_decay = 0, 0
            else
                since_best += 1
                since_decay += 1
            end
            if since_decay >= ST.patience && cur > 1e-4
                cur = max(T(1e-4), cur * T(0.3))
                Optimisers.adjust!(opt, cur)
                since_decay = 0
            end
            if cur <= 1e-4 && since_best >= ST.stop_patience
                stop = :floor
                break
            end
            upd >= ST.max_updates && break
        end
    end
    return best.ps, (; update = hist_u, lr = hist_lr, val = hist_v, best_val = best.val,
                     best_update = best.update, updates = upd, stop_reason = stop)
end


# --- stage 1: the grid ------------------------------------------------------------------------------
stage1 = Dict{Tuple{Int,Float64,Int},Any}()
for H in NHS, wd in WDS, seed in SEEDS
    sp = mkspec(H)                                          # not `spec`: a top-level binding (V36)
    t0 = time()
    ps, h = if LOSS === :rmse
        train_rmse(sp, seed, wd)
    else
        RF.train_stochlstm(sp, Xc, dat.Yc, dat.steps; L = 1, burn = 0, stride = 1,
                           batch = ST.batch, cfg.beta, cfg.val_frac, seed, epochs = MAXEPOCHS,
                           lr = ST.lr, val_every = ST.val_every, patience = ST.patience,
                           stop_patience = ST.stop_patience, stop_window = ST.stop_window,
                           weight_decay = wd, decay_exclude = (:bd, :Araw), verbose = false,
                           kl_mode = :per_step, freeze = FREEZE,
                           init_ps = SEEDHEAD ? init_params(sp, seed) : nothing)
    end
    # the validation loss recomputed here, so its unit is checked against the LS floor's
    rva = Yva .- predict(sp, ps, Xva)
    if LOSS !== :rmse
        nll = gnll(rva, headcov(RF.LSTMWeights(ps, sp)))
        abs(nll - h.best_val) < 1e-3 * max(1, abs(nll)) ||
            error("validation NLL $nll recomputed, $(h.best_val) from the library: the units differ")
    end
    # the validation NLL with Sigma refitted to the fit rows' residuals, as the LS floor's is (for
    # LOSS = mse the trained head is I, so only this one compares with the floor)
    rtr = Ytr .- predict(sp, ps, Float64.(Xc[:, 1:ntr]))
    nll_refit = gnll(rva, cov(permutedims(rtr)))
    # held out, 10-100 TU: the rms one-step error per QoI over least squares fitted on the same rows
    ho_ratio = rmsq(Yf[:, HO] .- predict(sp, ps, Xf[:, HO])) ./ ls_ho
    stage1[(H, wd, seed)] = (; ps, h, val = h.best_val, half_sse = 0.5 * sum(abs2, rva) / size(rva, 2),
                             nll_refit, ho_ratio, best_update = h.best_update, wall = time() - t0)
    @printf("stage 1  %d x %3d (%6d weights) wd %.0e seed %d lr %.0e %s%s | %5d upd, best @ %5d (%s) | val loss %.4f, NLL Sigma refit %.4f (LS %.4f) | 0.5 SSE %.4f (LS %.4f) | 10-100 TU rms / LS %s | %.1f min\n",
            NL, H, nparams(ps), wd, seed, LR, LOSS, SEEDHEAD ? " seeded" : "", h.updates, h.best_update, h.stop_reason, h.best_val,
            nll_refit, ls_val, stage1[(H, wd, seed)].half_sse, ls_sse,
            join((@sprintf("%.2f", x) for x in ho_ratio), " "), (time() - t0) / 60)
    flush(stdout)
end
meanval = Dict((H, wd) => mean(stage1[(H, wd, s)].val for s in SEEDS) for H in NHS, wd in WDS)
H_star, wd_star = argmin(meanval)
@info "setting chosen (lowest mean validation NLL over the seeds)" meanval H_star wd_star
spec = mkspec(H_star)

# --- stage 3: retrain on all of 1-10 TU, the stage-1 schedule replayed --------------------------------
"The learning rate stage 1 used at update `u`: the rate recorded at the first validation at or after it."
lr_at(h, u) = h.lr[searchsortedfirst(h.update, u)]

function retrain(spec, seed, wd, h1)
    nupd = h1.best_update
    rng = Xoshiro(seed)
    ps = RF.init_lstm_params(rng, spec; T)
    SEEDHEAD && (ps = merge(ps, LS_HEAD))
    adam = wd > 0 ? Optimisers.AdamW(T(ST.lr), (T(0.9), T(0.999)), T(wd); couple = true) :
           Optimisers.Adam(T(ST.lr))
    opt = Optimisers.setup(adam, ps)
    if wd > 0
        for k in (:bd, :Araw)
            Optimisers.adjust!(getfield(opt, k); lambda = zero(T))
        end
    end
    for k in FREEZE
        Optimisers.freeze!(getfield(opt, k))
    end
    X3 = reshape(Xc, nin, 1, ncol)
    Y3 = reshape(dat.Yc, nq, 1, ncol)
    nb = fld(ncol, ST.batch)
    upd, cur = 0, T(ST.lr)
    while upd < nupd
        idxs = shuffle(rng, 1:ncol)
        for k in 1:nb
            upd >= nupd && break
            lr = T(lr_at(h1, upd + 1))
            lr == cur || (Optimisers.adjust!(opt, lr); cur = lr)
            c = view(idxs, ((k - 1) * ST.batch + 1):(k * ST.batch))
            Xb, Yb = X3[:, :, c], Y3[:, :, c]
            gs = Zygote.gradient(p -> RF.elbo(spec, p, Xb, Yb, 1:1, zeros(T, 0, 1, ST.batch); cfg.beta,
                                              kl_mode = :per_step), ps)
            opt, ps = Optimisers.update(opt, ps, gs[1])
            upd += 1
        end
    end
    return ps
end

out_dir = joinpath(TO_folder, OUTSUB)
mkpath(out_dir)
Xall = Float64.(Xc)
final = Dict{Int,Any}()
for seed in (STAGE3 ? SEEDS : Int[])
    h1 = stage1[(H_star, wd_star, seed)].h
    ps3 = retrain(spec, seed, wd_star, h1)
    r = Float64.(dat.Yc) .- predict(spec, ps3, Xall)
    final[seed] = (; ps = ps3, resid_mean = vec(mean(r; dims = 2)), resid_sd = vec(std(r; dims = 2)),
                   half_sse = 0.5 * sum(abs2, r) / ncol)
    @printf("stage 3  seed %d | %d x %d, wd %.0e, %5d updates on 1-10 TU | train 0.5 SSE %.4f | residual mean %s (sd units)\n",
            seed, NL, H_star, wd_star, h1.best_update, final[seed].half_sse,
            join((@sprintf("%+.3f", m / s) for (m, s) in zip(final[seed].resid_mean, final[seed].resid_sd)), " "))
end

# --- predictions on the whole record, raw dQ, for the analysis ------------------------------------
raw(y) = y .* so .+ mo
if STAGE3
    pred1 = Dict(s => raw(predict(spec, stage1[(H_star, wd_star, s)].ps, Xf)) for s in SEEDS)
    pred3 = Dict(s => raw(predict(spec, final[s].ps, Xf)) for s in SEEDS)
    # the same regressor's least squares on all of 1-10 TU, in the same raw units (the linear comparator)
    Cls10 = Float64.(dat.Xc)' \ Float64.(dat.Yc)'
    predls = raw(Cls10' * Xr64)
    jldsave(joinpath(out_dir, "pred.jld2"); cols, pred1, pred3, predls, seeds = SEEDS, wd = wd_star,
            n_hidden = H_star, n_layers = NL, feat = FEAT, scaling)
end
jldsave(joinpath(out_dir, "fits.jld2");
        stage1 = Dict(k => (; v.h, v.val, v.half_sse, v.nll_refit, v.ho_ratio, v.best_update, v.wall,
                           nparams = nparams(v.ps), v.ps)
                      for (k, v) in stage1),
        stage3 = Dict(s => (; final[s].ps, final[s].resid_mean, final[s].resid_sd, final[s].half_sse) for s in keys(final)),
        meanval, H_star, wd_star, ls_val, ls_sse, ls_ho, lr = LR, loss = LOSS, seedhead = SEEDHEAD, settings = ST, nhs = NHS, nl = NL, wds = WDS,
        seeds = SEEDS, cfg, feat = FEAT, input_map = P, spec = (; n_hidden = H_star, n_layers = NL, n_in = nin, h = cfg.h, arch = cfg.arch,
                                    emission = cfg.emission, skip = false, window = 1))
@info "written" out_dir
