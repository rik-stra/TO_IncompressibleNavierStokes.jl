# M4 in WINDOW MODE: a small stochastic LSTM on short q*-only windows, reset for every prediction.
#
#     RIKFLOW_W_TAG=<name> [overrides...] julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_window_fit.jl
#
# 🔑 The model (Rik, 2026-09-24): the input is the predictor alone, `x_t = [q*_t; 1]` (h = 0), the
# network sees the last `W` of them, and the recurrent state is RESET for every prediction -- the
# deployed closure (`StochLSTM`, `spec.window = W`) replays the last `W` inputs from `h = c = 0`,
# each step with its own stored latent draw. So a training segment IS a deployed window: `L = W`,
# `stride = 1`, and every row of the record that can end a full window is one training sample.
# `L = 500` came from the level's ACF; an SGS correction does not need that memory.
#
# Overrides:
#
#   RIKFLOW_W_W          window length = segment length (default 10)
#   RIKFLOW_W_SCORE      last (default) | all -- `last` scores only the window's final output, i.e.
#                        exactly the deployed prediction (`burn = W - 1`); `all` also scores the
#                        shorter-context outputs inside each window (`burn = 0`)
#   RIKFLOW_W_ARCH / _NH / _NZ / _NENC / _BETA / _EMISSION
#                        architecture (default vrnn), hidden (16), latent (4), encoder width (0),
#                        KL weight (1e-4), emission head (none)
#   RIKFLOW_W_KL         reference (default) | per_step -- `reference` is the source's objective
#                        (`vae_loss_2D`): mean SSE plus BETA x the KL summed over batch, window
#                        steps and latent dims; `per_step` is every earlier fit's (0.5 SSE + BETA KL)
#                        per scored step. BETA keeps its meaning as the source's `lam` (1e-4).
#   RIKFLOW_W_H / _HIST_VAR   level history: e.g. H=1 HIST_VAR=q_star_q (default H=0, q_star)
#   RIKFLOW_W_LAMBDA     ridge penalty for the skip's seed (default 0 = least squares)
#   RIKFLOW_W_SUMK       k > 1: fit the skip's seed on k-step moving sums (integrated regression)
#   RIKFLOW_W_POST       x (default, the source's encoder q(z | x)) | xy (conditional VAE: q(z | x, dQ)
#                        in training, prior N(0, I) online) -- with SCORE=all, an emission head, BETA=1
#   RIKFLOW_W_PRIOR      standard (default, N(0, I)) | learned (VRNN prior p(z_t | h_{t-1}), with POST=xy)
#   RIKFLOW_W_TARGET     dQ (default) | logr | q -- held out is always scored as the standardised dQ
#   RIKFLOW_W_FEAT       raw (default, x_t = [q*_t; 1]) | diff (x_t = [q*_t; standardised increment; 1])
#   RIKFLOW_W_SKIP       1 (default) = linear skip `y += Ws x_t` -- an unbounded output path --
#                        seeded with the least-squares map on the current input and `V1 = V2 = 0`,
#                        frozen (FREEZE default `Ws`); 0 = none. RIKFLOW_W_FREEZE = comma list, empty = joint
#   RIKFLOW_W_BATCH / _LR / _EPOCHS / _SEED / _VAL_EVERY / _PATIENCE / _STOP_PATIENCE / _STOP_WINDOW
#   RIKFLOW_W_VAL        tail (default: the trailing 20%) | blocked (chunks 3 and 8 of 10 validate,
#                        EMBARGO rows dropped around each -- selects worse fits here, see below)
#   RIKFLOW_W_TRAIN_TU   end of the training range in TU (default 10 = the project's 1-10 TU)
#   RIKFLOW_W_SCORE_TU   `a,b`: the held-out window, TU (default: 50 TU to the end of the record,
#                        the pre-2026-09-28 convention). 🔴 Plan step 1 (1-50 TU fits): `52,74`, the
#                        selection block -- never score past 74 TU (76-97 is the confirmation block)
#   RIKFLOW_W_OUTSUB     subdirectory of output/TO_LSTM the fit is written to (default `window`)
#   RIKFLOW_W_NZ = 0     with ARCH=dense: a deterministic residual MLP, no latent at all (M3ᶠ);
#                        needs an emission head (EMISSION=constant, SEEDHEAD=1 for M0's eta at update 0)
#   RIKFLOW_W_WD         decoupled (AdamW, coupled to the rate: shrink lr*WD per update) weight
#                        decay on the trained blocks (default 0); RIKFLOW_W_WD_EXCLUDE = bd,Araw
#                        exempts the noise head
#
# 🔑 **Two linear floors are printed beside every fit**, both least squares on the fits' own
# training rows and scored on the held-out window (SCORE_TU; default 50-100 TU) in the fit's loss unit (0.5 x SSE per
# step over the six standardised QoIs): on the current input `[q*_n; 1]` alone, and on the whole
# stacked window `[q*_{n-W+1}; ...; q*_n; 1]`. The second is what a linear model can get out of the
# same information the LSTM sees. A fit that does not beat it has not learned the linear part
# (`results_LSTMS.md` §7c) -- the check that caught every pre-2026-09-24 fit.
#
# 🔑 With a skip, also printed: **M0(lambda)** = the skip alone at update 0 (held-out loss, CRPS and
# spread/skill; with SEEDHEAD = 1 that is the linear map + M0's eta at the SAME lambda -- the matched
# floor for LAMBDA > 0), and the **nonlinearity used**, RMS of the network's output g(x) = y - Ws x
# over the skip's Ws x and over the skip's held-out residual.
#
# Writes `output/TO_LSTM/<OUTSUB>/<tag>/StochLSTM_seed1.jld2` -- deployable with
# `RIKFLOW_M4_MODEL_DIR` and model index 2, like the diag fits -- and `curve.jld2`.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2, Printf, Statistics, LinearAlgebra, Random

const RF = RikFlow
Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
include(joinpath(@__DIR__, "m4_data.jl"))

envs(k, d) = (v = strip(get(ENV, "RIKFLOW_W_" * k, "")); isempty(v) ? d : v)
envi(k, d) = parse(Int, envs(k, string(d)))
tag = envs("TAG", "")
isempty(tag) && error("set RIKFLOW_W_TAG to name this fit")
base = load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")[2]

const DT = 2.5e-3
W = envi("W", 10)
score = Symbol(envs("SCORE", "last"))
score in (:last, :all) || error("RIKFLOW_W_SCORE must be last or all; got $score")
burn = score === :last ? W - 1 : 0
train_tu = parse(Float64, envs("TRAIN_TU", "10"))
ov = (; arch = Symbol(envs("ARCH", "vrnn")), n_hidden = envi("NH", 16), n_latent = envi("NZ", 4),
      n_encoder = envi("NENC", 0), beta = parse(Float64, envs("BETA", "1e-4")),
      emission = Symbol(envs("EMISSION", "none")), h = 0, hist_var = :q_star,
      include_predictor = true, L = W, burn, batch = envi("BATCH", 32),
      lr = parse(Float64, envs("LR", "1e-2")),
      train_range = (base.train_range[1], round(Int, train_tu / DT)))
target = Symbol(envs("TARGET", "dQ"))
feat = Symbol(envs("FEAT", "raw"))
feat in (:raw, :diff) || error("RIKFLOW_W_FEAT must be raw or diff; got $feat")
kl_mode = Symbol(envs("KL", envs("EMISSION", "none") == "none" ? "reference" : "per_step"))
wd = parse(Float64, envs("WD", "0"))
# `WD_EXCLUDE = bd,Araw`: no decay on the noise head (the network alone is decayed); default none
wdx = (v = envs("WD_EXCLUDE", ""); isempty(v) ? () : Tuple(Symbol.(strip.(split(v, ",")))))
kl_mode in (:reference, :per_step) || error("RIKFLOW_W_KL must be reference or per_step; got $kl_mode")
epochs = envi("EPOCHS", 200)
seed = envi("SEED", 1)
val_every = envi("VAL_EVERY", 20)
patience = envi("PATIENCE", 10)
stop_patience = envi("STOP_PATIENCE", 50)
stop_window = envi("STOP_WINDOW", 2000)
cfg = merge(base, ov)

rec = load_m4_qois(; track_file = joinpath(TO_folder, "..",
                                           "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
nq = size(rec.q, 1)
# 🔑 `FEAT = diff` feeds the standardised INCREMENT of q* beside q* itself: x_t = [q*_t;
# (q*_t - q*_{t-1} - m) / s; 1], a fixed invertible map of [q*_t; q*_{t-1}; 1] (h = 1). Same
# information, but the least-squares map on it needs coefficients of O(1) instead of ~130 -- the
# cancellation a network does not learn (`results_LSTMS.md` §7f). Stored as `scaling.input_map`,
# which the deployed closure applies too.
# 🔑 PHASE B (plan): `H` / `HIST_VAR` add level history, e.g. `H=1 HIST_VAR=q_star_q` gives
# x_t = [q*_t; q_{t-1}; q*_{t-1}; 1]. Online the replayed `q_{t-1}` is the closure's own
# `q* + dQ` -- the closed-loop exposure of §6 returns with it.
hvar = Symbol(envs("HIST_VAR", "q_star"))
hh = feat === :diff ? 1 : envi("H", 0)
feat === :diff && hvar !== :q_star && error("FEAT=diff is defined for HIST_VAR=q_star only")
hist = RF.HistorySpec(; h = hh, n_qoi = nq, hist_var = hvar, include_predictor = true)
cfg = merge(cfg, (; h = hh, hist_var = hvar))
# 🔑 The linear skip is ON by default (Rik, 2026-09-24: "make sure all outputs can be predicted,
# not limited to the training range"). Without it `:storn`/`:lstm` are hard-bounded at
# `cdec +- sum|V1|` (`h = o tanh(c)` lies in (-1, 1)), and every architecture saturates for inputs
# past the training range; `Ws x` is an unbounded path, and on held-out steps whose dQ lies
# outside the training range it raises the prediction~truth slope from 0.35 to 0.54-0.61
# (results_LSTMS.md §7f). Seeded at least squares and frozen by default (see FREEZE).
skip = envs("SKIP", "1") == "1"
# `Ws` frozen by default: trained jointly it leaves the least-squares map within ~20 updates and
# ends worse held out (1.96-2.00 against 1.74-1.75 frozen, blocked validation, §7f)
frz = envs("FREEZE", skip ? "Ws" : "")
freeze = isempty(frz) ? () : Tuple(Symbol.(strip.(split(frz, ","))))
# 🔑 `POST = xy`: the conditional-VAE encoder q(z_t | x_t, dQ_t) in training, the prior N(0, I)
# at deployment (Rik, 2026-09-24). Use with SCORE=all (every z_t then has to explain its own step)
# and an emission head with BETA=1 (the proper ELBO).
posterior = Symbol(envs("POST", "x"))
# `PRIOR = learned`: the VRNN prior p(z_t | h_{t-1}) -- learned colour (§7j)
prior = Symbol(envs("PRIOR", "standard"))
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, cfg.arch, cfg.uclip,
                   cfg.emission, skip, window = W, posterior, prior)

# --- training data: the same loader every M4 driver uses -------------------------------------------
dat = m4_training_data(rec, cfg, hist; target)
datD = target === :dQ ? dat : m4_training_data(rec, cfg, hist; target = :dQ)   # the common unit
nin = RF.n_input(spec)
# 🔑 THE SPLIT. `VAL = blocked`: the 1-10 TU rows are cut into `VAL_CHUNKS` chunks and
# chunks 3 and 8 (of 10; generally every 5th, starting at the 3rd) validate, with `EMBARGO` rows
# dropped on each side of every validation chunk. The trailing 20% (`VAL = tail`, the library's
# default) sits off the stationary regime -- dQ mean -0.23..-0.34 sd in four QoIs, q* +0.25..+0.51
# -- and on it even the linear ranking reverses (W = 10 below W = 1), so it misled early stopping
# (review, 2026-09-24). Everything fitted from "training rows" below uses `tr`.
ncolt = size(dat.Xc, 2)
# ⚠️ Default `tail`, measured (2026-09-24): blocked validation shares the training regime, so it
# does not penalise the residual the LSTM grows on top of the skip, and returned held-out
# 1.95-2.02 (embargo 40/200/400) against the tail split's 1.736 -- and a long embargo costs the
# linear map itself (floor 1.975 at 400). The off-regime tail block is what agrees with 50-100 TU.
valmode = Symbol(envs("VAL", "tail"))
nchunk = envi("VAL_CHUNKS", 10)
embargo = envi("EMBARGO", 40)
if valmode === :tail
    ntr0 = floor(Int, (1 - cfg.val_frac) * ncolt)
    tr, va = collect(1:ntr0), collect((ntr0 + 1):ncolt)
elseif valmode === :blocked
    edges = round.(Int, range(0, ncolt; length = nchunk + 1))
    vch = [c for c in 1:nchunk if mod(c, 5) == 3]
    isval = falses(ncolt); isdrop = falses(ncolt)
    for c in vch
        r = (edges[c] + 1):edges[c + 1]
        isval[r] .= true
        isdrop[max(1, first(r) - embargo):min(ncolt, last(r) + embargo)] .= true
    end
    va = findall(isval)
    tr = findall(.!isdrop)
else
    error("RIKFLOW_W_VAL must be blocked or tail; got $valmode")
end
@info "split" valmode train_rows = length(tr) val_rows = length(va) dropped = ncolt - length(tr) - length(va)
# the windows in TU, printed so a fit's partition is in its log (`dat.steps` are 1-based within
# `train_range`, so row k sits at record column train_range[1] - 1 + steps[k])
tu(k) = (cfg.train_range[1] - 1 + dat.steps[k]) * DT
@info "windows (TU)" train_fit = (tu(first(tr)), tu(last(tr))) validation = isempty(va) ? () : (tu(first(va)), tu(last(va)))
"Window ends among `cols` whose W-1 predecessors are also in `cols` and contiguous."
function wends(cols, W)
    s = Set(cols)
    return [c for c in cols if c >= W && all(in(s), (c - W + 1):c)]
end
P = Matrix{Float64}(I, nin, nin)
if feat === :diff
    d = Float64.(dat.Xc[1:nq, tr] .- dat.Xc[(nq + 1):(2nq), tr])   # training rows only
    m, sd = vec(mean(d; dims = 2)), vec(std(d; dims = 2))
    r = (nq + 1):(2nq)
    P[r, r] .= -Diagonal(1 ./ sd)
    P[r, 1:nq] .= Diagonal(1 ./ sd)
    P[r, end] .= -m ./ sd
end
scaling = feat === :diff ? merge(dat.scaling, (; input_map = P)) : dat.scaling
Xc = Float32.(P * Float64.(dat.Xc))
# the training range of every (mapped) input -- a DIAGNOSTIC only (Rik, 2026-09-24: inputs are
# never clamped): how far held-out inputs leave it, and how much the linear floor rests on that
lo, hi = vec(minimum(Xc[:, tr]; dims = 2)), vec(maximum(Xc[:, tr]; dims = 2))

# --- the whole record in the SAME scaling, for the held-out window --------------------------------
a, b = cfg.train_range[1], size(rec.q_star, 2)
qs = RF.scale_input(rec.q[:, a:(b + 1)], dat.scaling.in_scaling)
qss = RF.scale_input(rec.q_star[:, a:b], dat.scaling.in_scaling)
Xr, _, stepsf = RF.build_history(hist, qss, qs)
colsf = (a - 1) .+ stepsf
Xf = P * permutedims(Xr)
# held-out targets are ALWAYS the standardised correction, whatever the fit predicts, so fits on
# different targets share one number
Ydf = RF.scale_input(rec.dQ[:, colsf], datD.scaling.out_scaling)
# 🔑 `SCORE_TU = a,b` (2026-09-28, plan step 1): the held-out window is explicit, window ends with
# a <= t <= b TU. Unset = the old convention, every step from 50 TU to the end of the record (so old
# runs reproduce). 🔴 Under the plan's partition (plan.md §7) scoring must stay inside the selection
# block 52-74 TU; 76-97 TU is the reserved confirmation block.
score_tu = envs("SCORE_TU", "")
score_lo, score_hi = isempty(score_tu) ? (50.0, Inf) : Tuple(parse.(Float64, split(score_tu, ",")))
ends = findall(c -> score_lo / DT <= c <= score_hi / DT, colsf)
isempty(ends) && error("RIKFLOW_W_SCORE_TU = $score_tu selects no held-out step")
first(ends) > W || error("held-out window starts too early for a $W-step window")
score_hi > 74 &&
    @warn "held-out window reaches past 74 TU -- under the 2026-09-28 partition 76-97 TU is the confirmation block" score_lo score_hi
colsf[ends[1]] > cfg.train_range[2] ||
    error("held-out window starts at step $(colsf[ends[1]]), inside the training range $(cfg.train_range)")
const WH = W

"`n_in x W x N` windows ending at the given columns of `X`."
function windows(X, ends)
    out = similar(X, size(X, 1), WH, length(ends))
    for (k, e) in enumerate(ends)
        out[:, :, k] = view(X, :, (e - WH + 1):e)
    end
    return out
end
Xh = Float32.(windows(Xf, ends))
Xh_raw = Xh
oor = vec(any((Xh_raw[:, end, :] .< lo) .| (Xh_raw[:, end, :] .> hi); dims = 1))
Yh = Ydf[:, ends]
@info "held-out score window (TU)" first = colsf[ends[1]] * DT last = colsf[ends[end]] * DT rows = length(ends)
qsh = rec.q_star[:, colsf[ends]]
nh = length(ends)

"The fit's output, in the fit's own target units, as the standardised correction."
function as_dQ(y)
    target === :dQ && return Float64.(y)
    out = RF.scale_output(Float64.(y), dat.scaling.out_scaling)
    dq = target === :q ? out .- qsh : qsh .* expm1.(out)
    return RF.scale_input(dq, datD.scaling.out_scaling)
end

function heldout(ps)
    o = RF.lstm_forward(spec, ps, Xh, zeros(Float32, spec.n_latent, WH, nh))
    r = Yh .- as_dQ(o.Y[:, end, :])
    return 0.5 * sum(abs2, r) / nh,
           vec(1 .- sum(abs2, r; dims = 2) ./ sum(abs2, Yh .- mean(Yh; dims = 2); dims = 2))
end

# --- the two linear floors, fitted on the training rows only, in dQ units -------------------------
Xtr, Ytr = Float64.(Xc[:, tr]), Float64.(datD.Yc[:, tr])
# (a) the current input only: [q*_n; 1], or [q*_n; dq*_n; 1] under FEAT = diff
C0d = Xtr' \ Ytr'
floor0 = 0.5 * sum(abs2, Yh .- C0d' * Float64.(Xh_raw[:, end, :])) / nh
# how much of the linear floor rests on extrapolating past the training range
floor0c = 0.5 * sum(abs2, Yh .- C0d' * Float64.(clamp.(Xh_raw[:, end, :], lo, hi))) / nh
@info "tails" held_rows_out_of_train_range = round(mean(oor); digits = 4) floor_current_raw = floor0 floor_current_clamped = floor0c sse_share_of_oor_rows = round(sum(abs2, (Yh .- C0d' * Float64.(Xh_raw[:, end, :]))[:, oor]) / sum(abs2, Yh .- C0d' * Float64.(Xh_raw[:, end, :])); digits = 3)
# (b) the whole window of q*, stacked: [q*_{n-W+1}; ...; q*_n; 1] (one bias)
stackwin(Xw) = vcat(reshape(Xw[1:nq, :, :], nq * size(Xw, 2), :), ones(1, size(Xw, 3)))
wtr = wends(tr, W)
Xtrw = stackwin(Float64.(windows(Xc, wtr)))
Ctrw = Xtrw' \ Float64.(datD.Yc[:, wtr])'
floorw = 0.5 * sum(abs2, Yh .- Ctrw' * stackwin(Float64.(Xh))) / nh
# (c) with level history (h > 0), the whole window of the FULL regressor stacked,
# [x_{n-W+1}; ...; x_n] (one bias) -- the linear model on exactly what a W > 1 network sees, so a
# window network's gain over the skip splits into "more lags" (this floor) and "nonlinearity"
stackfull(Xw) = vcat(reshape(Xw[1:(end - 1), :, :], (size(Xw, 1) - 1) * size(Xw, 2), :), ones(1, size(Xw, 3)))
floorwf = if hh > 0 && W > 1
    Xtrf = stackfull(Float64.(windows(Xc, wtr)))
    Cf = Xtrf' \ Float64.(datD.Yc[:, wtr])'
    0.5 * sum(abs2, Yh .- Cf' * stackfull(Float64.(Xh))) / nh
else
    floor0
end
@info "linear floors (held-out window, dQ units)" current_input = floor0 whole_window = floorw whole_window_full_regressor = floorwf
# the skip's seed is the same least-squares map in the fit's OWN target units
# `LAMBDA > 0` seeds the skip with the RIDGE map instead (plan B2; §7d: the unregularised map runs
# +1 sd high online and lambda ~ 1e-5 zeroes it). Penalty `lambda N` on every column but the bias.
lam = parse(Float64, envs("LAMBDA", "0"))
# 🔑 `SUMK = k` (2026-09-27): INTEGRATED regression -- fit the linear map on k-step moving SUMS of the
# regressors and targets over contiguous training rows. The LF solver passes a correction on ~1:1
# (§7h e: one-step kernel ~ I, step response ~ k to k = 10-25), so what reaches the LEVEL is the
# running sum of the correction error, not the one-step error; this weights the slow, correlated part
# of the residual that one-step least squares ignores. Still offline and teacher-forced.
sumk = envi("SUMK", 0)
function movsum(A, cols, k)
    s = Set(cols)
    ends = [c for c in cols if c >= k && all(in(s), (c - k + 1):c)]
    return reduce(hcat, [vec(sum(A[:, (c - k + 1):c]; dims = 2)) for c in ends])
end
C0 = if sumk > 1
    Xs_ = movsum(Float64.(Xc), tr, sumk); Ys_ = movsum(Float64.(dat.Yc), tr, sumk)
    Xs_' \ Ys_'
elseif lam == 0
    Xtr' \ Float64.(dat.Yc[:, tr])'
else
    Dp = Diagonal([fill(lam * size(Xtr, 2), nin - 1); 0.0])
    (Xtr * Xtr' + Dp) \ (Xtr * Float64.(dat.Yc[:, tr])')
end

# --- fit -------------------------------------------------------------------------------------------
curve = (; update = Int[], val = Float64[], held = Float64[])
cb(upd, ps, vl) = (push!(curve.update, upd); push!(curve.val, vl); push!(curve.held, heldout(ps)[1]))
init_ps = nothing
if skip
    p0 = RF.init_lstm_params(Xoshiro(seed), spec)
    init_ps = merge(p0, (; Ws = Float32.(permutedims(C0)), V1 = zero(p0.V1),
                         V2 = p0.V2 === nothing ? nothing : zero(p0.V2)))
    # `SEEDHEAD = 1`: the emission head starts at the linear map's training-residual covariance
    # (`bd = log sd`, `A'A = R^{-1}`), so update 0 is the linear mean with M0's eta -- as in
    # `m4_diag_fit.jl`, where this is the lead candidate's recipe
    if envs("SEEDHEAD", "0") == "1"
        RF.emission_noise(spec) || error("RIKFLOW_W_SEEDHEAD needs an emission head")
        Rtr = Float64.(dat.Yc[:, tr]) .- C0' * Float64.(Xc[:, tr])
        sdr = vec(std(Rtr; dims = 2))
        Pr = inv(Symmetric(cor(permutedims(Rtr))))
        J = reverse(Matrix{Float64}(I, nq, nq); dims = 1)
        A = J * cholesky(Symmetric(J * Pr * J)).U * J          # lower triangular, A'A = R^{-1}
        init_ps = merge(init_ps, (; bd = Float32.(log.(sdr)),
                                  Araw = Float32.(tril(A, -1) + Diagonal(log.(diag(A))))))
    end
end
# held-out Gaussian NLL per step, for an emission head -- in the fit's OWN target units (a density
# is not invariant to the change of units between targets), latent at its mean
dqh = rec.dQ[:, colsf[ends]]
Yth = target === :dQ ? Yh :
      RF.scale_input(target === :q ? qsh .+ dqh : log1p.(dqh ./ qsh), dat.scaling.out_scaling)
Y3h = zeros(Float32, nq, WH, nh)
Y3h[:, WH, :] .= Yth
# the baseline for it: the linear map on the current input + the training residual's full
# covariance (M0's eta), same units
Rlin = Float64.(dat.Yc[:, tr]) .- C0' * Float64.(Xc[:, tr])
Slin = cov(permutedims(Rlin))
rlh = Float64.(Yth) .- C0' * Float64.(Xh[:, end, :])
linnll = 0.5 * (nq * log(2pi) + logdet(Slin) + mean(sum(rlh .* (Slin \ rlh); dims = 1)))
# 🔴 not for `posterior = :xy`: `elbo` hands the encoder the true target, which is not a forecast
heldnll(ps) = RF.emission_noise(spec) && posterior === :x ?
              RF.elbo(spec, ps, Xh, Y3h, WH:WH, zeros(Float32, spec.n_latent, WH, nh); beta = 0) :
              NaN
@info "M4 window fit" tag ov... target feat posterior kl_mode hvar hh lam sumk epochs seed skip freeze ntrain = size(dat.Xc, 2) held = nh
t0 = time()
ps, h = RF.train_stochlstm(spec, Xc, dat.Yc, dat.steps; L = W, burn, stride = 1, cfg.batch,
                           split = (; train = tr, val = va),
                           cfg.beta, cfg.val_frac, seed, epochs, cfg.lr, val_every, patience,
                           stop_patience, stop_window, verbose = false, callback = cb, init_ps,
                           freeze, kl_mode, weight_decay = wd, decay_exclude = wdx)
wall = time() - t0
hbest, r2 = heldout(ps)
hnll = heldnll(ps)

# 🔑 Ensemble CRPS on the held-out window (plan A3): the probabilistic score that ranks latent
# models too. `K` independent draws per window -- the latent for every step of the window, plus
# the emission noise when there is a head -- which is the marginal of ONE deployed prediction
# (tying only correlates consecutive predictions). In dQ units, mean over QoIs and rows.
const KCRPS = envi("CRPS_K", 32)
function heldcrps(ps; K = KCRPS, rng = Xoshiro(2024))
    wdep = RF.LSTMWeights(ps, spec)
    s_log = Float64.(wdep.bd .- ps.bd)                 # log s: deployed minus trained log-scale
    ens = zeros(nq, nh, K)
    for k in 1:K
        epsz = randn(rng, Float32, spec.n_latent, WH, nh)
        o = RF.lstm_forward(spec, ps, Xh, epsz)
        y = Float64.(o.Y[:, end, :])
        if RF.emission_noise(spec)
            d = exp.(Float64.(o.LOGD[:, end, :]) .+ s_log)
            y = y .+ d .* (Float64.(wdep.LR) * randn(rng, nq, nh))
        end
        ens[:, :, k] = as_dQ(y)
    end
    c = mean(RF.crps_ensemble(view(ens, i, j, :), Yh[i, j]) for i in 1:nq, j in 1:nh)
    # spread vs skill per QoI: mean ensemble sd against the rmse of the ensemble mean
    spread = vec(mean(std(ens; dims = 3); dims = 2))
    skill = vec(sqrt.(mean(abs2, dropdims(mean(ens; dims = 3); dims = 3) .- Yh; dims = 2)))
    return c, spread ./ skill
end
hcrps, hss = heldcrps(ps)
@info "held-out spread/skill per QoI (1 = calibrated)" ratio = round.(hss; digits = 2)
# the same score for the linear map + M0's eta (constant Gaussian), in dQ units by sampling
function lincrps(; K = KCRPS, rng = Xoshiro(2024))
    Rd = Float64.(datD.Yc[:, tr]) .- (Xtr' \ Float64.(datD.Yc[:, tr])')' * Xtr
    Ld = cholesky(Symmetric(cov(permutedims(Rd)))).L
    mu = ((Xtr' \ Float64.(datD.Yc[:, tr])')' * Float64.(Xh[:, end, :]))
    return mean(RF.crps_ensemble([mu[i, j] + (Ld * randn(rng, nq))[i] for _ in 1:K], Yh[i, j])
                for i in 1:nq, j in 1:nh)
end
lcrps = lincrps()
ib = isempty(curve.held) ? 0 : argmin(curve.held)

# the skip alone = M0(lambda): update 0 of a skip fit (V1 = 0), with the seeded head when SEEDHEAD = 1
skip0 = init_ps === nothing ? (; held = NaN, crps = NaN, ss = fill(NaN, nq)) :
        (h0 = heldout(init_ps)[1]; (c0, s0) = heldcrps(init_ps); (; held = h0, crps = c0, ss = s0))
# how much NONLINEARITY the fit uses: the network's output g(x) = y - Ws x against the skip's Ws x,
# RMS over the held-out rows and QoIs (and against the skip's held-out residual)
nlr = if skip
    o = RF.lstm_forward(spec, ps, Xh, zeros(Float32, spec.n_latent, WH, nh))
    sx = Float64.(ps.Ws * Xh[:, end, :]); g = Float64.(o.Y[:, end, :]) .- sx
    (; net_over_skip = sqrt(sum(abs2, g) / sum(abs2, sx)),
     net_over_resid = sqrt(sum(abs2, g) / sum(abs2, Yh .- sx)),
     per_qoi = vec(sqrt.(sum(abs2, g; dims = 2) ./ sum(abs2, sx; dims = 2))))
else
    (; net_over_skip = NaN, net_over_resid = NaN, per_qoi = fill(NaN, nq))
end
@info "M0(lambda) = the skip at update 0, and the nonlinearity used" skip_held = skip0.held skip_crps = skip0.crps skip_ss = round.(skip0.ss; digits = 2) nlr.net_over_skip nlr.net_over_resid

out_dir = joinpath(TO_folder, envs("OUTSUB", "window"), tag)
mkpath(out_dir)
RF.save_stochlstm(joinpath(out_dir, "StochLSTM_seed1.jld2"), spec, RF.LSTMWeights(ps, spec),
                  scaling; cfg, seed, train_range = cfg.train_range, qoi_source = rec.source,
                  losses = h, ps, stride = 1, batch = cfg.batch, rollout = 1, target, overrides = ov,
                  lr = cfg.lr, init_from = "", val = (; tf = h.best_val, ro = NaN, Kx = 0),
                  diag = (; held = hbest, held_nll = hnll, held_crps = hcrps, spread_skill = hss, lin_crps = lcrps, r2, floor_current = floor0, floor_window = floorw, skip,
                          freeze, score, kl_mode, feat, valmode, embargo, posterior, prior, hvar,
                          h = hh, lambda = lam, sumk,
                          floor_current_clamped = floor0c, frac_out_of_range = mean(oor),
                          score_tu = (score_lo, score_hi), wd, skip0, nlr, floor_window_full = floorwf))
jldsave(joinpath(out_dir, "curve.jld2"); curve, floor0, floorw, hbest, hnll, linnll, hcrps, lcrps, r2, skip0, nlr,
        score_tu = (score_lo, score_hi),
        best_update = h.best_update, ov, target, skip, freeze, score, wall, kl_mode, feat)
@printf("%-24s %-6s p%-2s h%d%-1s l%.0e %-4s %-4s %-4s W %2d %-4s kl %-3s b %.0e nh %2d nz %d sk %d | %5d upd (%d/ep), best@%5d (%s) %.1f min | val %.4f | held %.3f crps %.4f (lin+eta %.4f) ss %s nll %.3f (lin+eta %.3f)  floors: q*_n %.3f, window %.3f | held-best %.3f@%d | R2 %s\n",
        tag, cfg.arch, string(posterior), hh, hvar === :q_star_q ? "q" : "", lam, string(target), string(feat), string(valmode)[1:4], W, score, string(kl_mode)[1:3], cfg.beta, cfg.n_hidden, cfg.n_latent, skip, h.updates, h.upd_per_epoch,
        h.best_update, h.stop_reason, wall / 60, h.best_val, hbest, hcrps, lcrps, join((@sprintf("%.2f", x) for x in hss), "/"), hnll, linnll, floor0, floorw,
        ib == 0 ? NaN : curve.held[ib], ib == 0 ? 0 : curve.update[ib],
        join((@sprintf("%.2f", x) for x in r2), " "))
@printf("%-24s vs M0(lambda = %.0e): held %.4f / %.4f (%+.2f%%; stacked-window LS %.4f) | crps %.5f / %.5f (%+.2f%%) | ss %s / %s | wd %.0e | |g|/|Ws x| %.4f, |g|/|resid| %.3f | score %g-%g TU\n",
        tag, lam, hbest, skip0.held, 100 * (hbest / skip0.held - 1), floorwf, hcrps, skip0.crps, 100 * (hcrps / skip0.crps - 1),
        join((@sprintf("%.2f", x) for x in hss), "/"), join((@sprintf("%.2f", x) for x in skip0.ss), "/"), wd,
        nlr.net_over_skip, nlr.net_over_resid, score_lo, score_hi)
