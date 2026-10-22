# Step 3, an OFFLINE PROBE: a 1-D convolutional network mean on the lag history (Rik, 2026-10-22: "also
# try an CNN architecture"), on the same data, target, input and comparators as the MLP of step 3a
# (`step3_data.jl`). Not deployable: the closure path (`StochLSTM`) has no convolutional arch.
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/cnn_mean_fit.jl
#
# 🔑 The input is the MLP's (FEAT = dqinc), rearranged as a sequence over the h = 5 history steps,
# oldest first: at step k the channels are [q^{n-k*}, dQ^{n-k}, Dq*^{n-k}] (3 N_Q = 18), each
# standardised as in the MLP's input map. The current predictor q^{n*} enters after the convolutions:
#
#     two Conv1D(kernel K, tanh) over the 5 steps -> flatten -> [.; q^{n*}] -> Dense(D, tanh) ->
#     Dense(D, tanh) -> Dense(N_Q)
#
# trained on squared error of the standardised target dQ^n = q^n - q^{n*}, as the MLP's mse probe (Sigma
# is refitted to the fit rows' residuals for the NLL that is reported next to the LS floor's). Adam(W),
# batch 256, a validation on
# 8.2-10 TU every 100 updates, the best iterate kept, the learning rate x0.3 after 30 validations
# without a new best (floor 1e-4), stopped at the floor after 80 more, capped at MAXUPD updates -- the
# MLP's rules (train_stochlstm), written out here.
#
# Second probes (Rik, 2026-10-22): RIKFLOW_CNN_LOSS = mse (default, above) | rmse (the per-QoI RMSE summed
# over the QoIs) | nll (the Gaussian likelihood with a constant correlated head, trained jointly and
# started at least squares' residual covariance, `LS_HEAD`); RIKFLOW_CNN_INPUT = full (default: q*, dQ
# and Dq* per step) | nodiff (q* and dQ per step: the kernel spans neighbouring steps and can form the
# predictor increments itself, Rik: "the CNN might not need the inputs that were subtracted").
#
# Third probes (Rik, 2026-10-22): RIKFLOW_CNN_SEQ = lags (default, above: the h history steps, q^{n*}
# joining after the convolutions) | full (q^{n*} IN the sequence, k = 0..h: "please include q*_n in the q*
# sequence"; dQ^{n-k} -- and Dq*^{n-k} with INPUT = full -- is unknown at k = 0, so it is set to 0, its
# standardised mean, and a MASK channel, 0 at k = 0 and 1 elsewhere, lets the shared kernels tell the
# missing value from a zero, Rik's choice; q^{n*} then no longer enters the dense layers separately).
# Kernel size via RIKFLOW_CNN_K (Rik: 3, stride 1).
#
# Grid (RIKFLOW_CNN_*): CHS channels {16, 32}, LRS {1e-3, 3e-3}, WDS {1e-1}, SEEDS {1}, K = 2, D = 64,
# MAXUPD 20000, OUT step3/probe_cnn. Prints, per fit, the validation 0.5 SSE and NLL against least
# squares on the same rows, and the rms one-step error on 10-100 TU per QoI over least squares'.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2, Printf, Statistics, LinearAlgebra, Random

const RF = RikFlow
include(joinpath(@__DIR__, "step3_data.jl"))        # record, target, input map, checks, LS floor
FEAT === :dqinc || error("the CNN reads the dqinc input (q*, dQ and Dq* per step); got FEAT = $FEAT")

envc(k, d) = (v = strip(get(ENV, "RIKFLOW_CNN_" * k, "")); isempty(v) ? d : v)
envcv(k, d, U) = parse.(U, strip.(split(envc(k, d), ",")))
const CHS = envcv("CHS", "16,32", Int)
const LRS = envcv("LRS", "1e-3,3e-3", Float64)
const CWDS = envcv("WDS", "1e-1", Float64)
const CSEEDS = envcv("SEEDS", "1", Int)
const K = parse(Int, envc("K", "2"))
const D = parse(Int, envc("D", "64"))
const CST = (; batch = 256, val_every = 100, patience = 30, stop_patience = 80, min_lr = 1e-4,
             lr_decay = 0.3, max_updates = parse(Int, envc("MAXUPD", "20000")))
const COUT = envc("OUT", "step3/probe_cnn")
const CLOSS = Symbol(envc("LOSS", "mse"))
CLOSS in (:mse, :rmse, :nll) || error("RIKFLOW_CNN_LOSS must be mse, rmse or nll; got $CLOSS")
const CINPUT = Symbol(envc("INPUT", "full"))
CINPUT in (:full, :nodiff) || error("RIKFLOW_CNN_INPUT must be full or nodiff; got $CINPUT")
const CSEQ = Symbol(envc("SEQ", "lags"))
CSEQ in (:lags, :full) || error("RIKFLOW_CNN_SEQ must be lags or full; got $CSEQ")
const NSTEP = CSEQ === :full ? cfg.h + 1 : cfg.h              # sequence length
const NCH = (CINPUT === :full ? 3nq : 2nq) + (CSEQ === :full ? 1 : 0)   # channels per step (+ mask)
NSTEP - 2(K - 1) >= 1 || error("kernel $K leaves no position after two convolutions of $NSTEP steps")

"""
The sequence (NSTEP x NCH x N, oldest step first) and q^{n*} (N_Q x N) from the mapped input `Z`.
SEQ = lags: steps k = h..1. SEQ = full: steps k = h..0, the last being the current step, where q^{n*}
is known and dQ^n (and Dq*^n) are not: they stay 0 and the mask channel (the last) is 0 there, 1 elsewhere.
"""
function lagtensor(Z)
    N, H = size(Z, 2), cfg.h
    A = zeros(T, NSTEP, NCH, N)
    ks = CSEQ === :full ? (H:-1:0) : (H:-1:1)
    for (t, k) in enumerate(ks)
        if k == 0
            A[t, 1:nq, :] = Z[1:nq, :]                       # q^{n*}; dQ^n, Dq*^n unknown: 0, mask 0
            continue
        end
        lr = lagrows(k)
        A[t, 1:nq, :] = Z[lr.s, :]                          # q^{n-k*}
        A[t, (nq + 1):(2nq), :] = Z[lr.q, :]                # dQ^{n-k}
        CINPUT === :full &&                                 # Dq*^{n-k}
            (A[t, (2nq + 1):(3nq), :] = Z[nreg .+ (((k - 1) * nq + 1):(k * nq)), :])
        CSEQ === :full && (A[t, NCH, :] .= one(T))          # the mask: known
    end
    return A, T.(Z[1:nq, :])
end

function mkmodel(C)
    conv = Chain(Conv((K,), NCH => C, tanh), Conv((K,), C => C, tanh), FlattenLayer())
    nflat = (NSTEP - 2(K - 1)) * C
    CSEQ === :full &&                                       # q^{n*} is in the sequence
        return Chain(conv, Dense(nflat => D, tanh), Dense(D => D, tanh), Dense(D => nq))
    return Chain(Parallel(vcat, conv, NoOpLayer()), Dense(nflat + nq => D, tanh), Dense(D => D, tanh),
                 Dense(D => nq))
end
"The model's input: the sequence alone (SEQ = full), or the sequence and q^{n*} (SEQ = lags)."
modelin(A, Q) = CSEQ === :full ? A : (A, Q)

Atr, Qtr = lagtensor(Float64.(Xc[:, 1:ntr]))
# the sequence is a rearrangement of the checked input (step3_data.jl); check the current step's layout
if CSEQ === :full
    Atr[end, 1:nq, :] == Qtr || error("the last step of the sequence is not q^{n*}")
    all(iszero, Atr[end, (nq + 1):end, :]) || error("dQ^n / Dq*^n / mask at the current step are not 0")
    all(isone, Atr[1:(end - 1), NCH, :]) || error("the mask is not 1 at the history steps")
    Atr[end - 1, 1:nq, :] == T.(Xc[lagrows(1).s, 1:ntr]) || error("step n-1 does not hold q^{n-1*}")
end
Ava, Qva = lagtensor(Float64.(Xc[:, (ntr + 1):end]))
Aho, Qho = lagtensor(Xf[:, HO])
Ytr32, Yva32 = T.(Ytr), T.(Yva)
@info "CNN probe" CHS LRS CWDS CSEEDS K D CLOSS CINPUT CSEQ NSTEP NCH CST.max_updates fit_rows = ntr val_rows = ncol - ntr

# the likelihood's head, as the library's: Sigma = D R D, D = diag(exp(bd)), R^{-1} = A'A with A lower
# triangular from Araw (strict lower part, exp of the diagonal); the strict-lower mask is a constant
const LMASK = T[i > j for i in 1:nq, j in 1:nq]
"Gaussian NLL per step of residuals `r` (nq x B) under the head `hd = (; bd, Araw)`."
function head_nll(hd, r)
    A = hd.Araw .* LMASK .+ Diagonal(exp.(diag(hd.Araw)))
    z = A * (r ./ exp.(hd.bd))
    return T(0.5) * (nq * T(log(2pi)) + 2sum(hd.bd) - 2sum(diag(hd.Araw)) + sum(abs2, z) / size(r, 2))
end

function fit_cnn(C, lr, wd, seed)
    rng = Xoshiro(seed)
    model = mkmodel(C)
    net, st = Lux.setup(rng, model)
    ps = (; net, head = (; bd = copy(LS_HEAD.bd), Araw = copy(LS_HEAD.Araw)))
    rule = wd > 0 ? Optimisers.AdamW(T(lr), (T(0.9), T(0.999)), T(wd); couple = true) : Optimisers.Adam(T(lr))
    opt = Optimisers.setup(rule, ps)
    # the head: no weight decay; trained only by the likelihood
    wd > 0 && Optimisers.adjust!(opt.head; lambda = zero(T))
    CLOSS === :nll || Optimisers.freeze!(opt.head)
    out(p, A, Q) = first(model(modelin(A, Q), p.net, st))
    loss(p, A, Q, Y) = CLOSS === :mse ? T(0.5) * sum(abs2, out(p, A, Q) .- Y) / size(Y, 2) :
                       CLOSS === :rmse ? rmse_sum(out(p, A, Q), Y) : head_nll(p.head, out(p, A, Q) .- Y)
    vloss(p) = Float64(loss(p, Ava, Qva, Yva32))
    best = (; val = vloss(ps), ps = deepcopy(ps), update = 0)
    cur, since_best, since_decay, upd, stop = T(lr), 0, 0, 0, :cap
    nb = fld(ntr, CST.batch)
    while upd < CST.max_updates
        idxs = shuffle(rng, 1:ntr)
        for k in 1:nb
            c = idxs[((k - 1) * CST.batch + 1):(k * CST.batch)]
            g = Zygote.gradient(p -> loss(p, Atr[:, :, c], Qtr[:, c], Ytr32[:, c]), ps)[1]
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
            upd % CST.val_every == 0 || continue
            v = vloss(ps)
            if v < best.val
                best = (; val = v, ps = deepcopy(ps), update = upd)
                since_best, since_decay = 0, 0
            else
                since_best += 1
                since_decay += 1
            end
            if since_decay >= CST.patience && cur > CST.min_lr
                cur = max(T(CST.min_lr), cur * T(CST.lr_decay))
                Optimisers.adjust!(opt, cur)
                since_decay = 0
            end
            if cur <= CST.min_lr && since_best >= CST.stop_patience
                stop = :floor
                break
            end
            upd >= CST.max_updates && break
        end
        stop === :floor && break
    end
    pb = best.ps
    pred(A, Q) = Float64.(out(pb, A, Q))
    rtr, rva = Ytr .- pred(Atr, Qtr), Yva .- pred(Ava, Qva)
    ho_ratio = rmsq(Yf[:, HO] .- pred(Aho, Qho)) ./ ls_ho
    return (; ps = pb, st, val_loss = best.val, val_half_sse = 0.5 * sum(abs2, rva) / size(rva, 2),
            nll_refit = gnll(rva, cov(permutedims(rtr))), ho_ratio, best_update = best.update, updates = upd,
            stop, nparams = Lux.parameterlength(model), loss = CLOSS, input = CINPUT, seq = CSEQ)
end

res = Dict{Tuple{Int,Float64,Float64,Int},Any}()
for C in CHS, lr in LRS, wd in CWDS, seed in CSEEDS
    t0 = time()
    r = fit_cnn(C, lr, wd, seed)
    res[(C, lr, wd, seed)] = merge(r, (; wall = time() - t0))
    @printf("cnn  C %3d K %d D %d (%6d weights) lr %.0e wd %.0e seed %d %s %s seq %s | %5d upd, best @ %5d (%s) | val loss %.4f, NLL Sigma refit %.4f (LS %.4f) | 0.5 SSE %.4f (LS %.4f) | 10-100 TU rms / LS %s | %.1f min\n",
            C, K, D, r.nparams, lr, wd, seed, CLOSS, CINPUT, CSEQ, r.updates, r.best_update, r.stop, r.val_loss, r.nll_refit, ls_val,
            r.val_half_sse, ls_sse, join((@sprintf("%.2f", x) for x in r.ho_ratio), " "), (time() - t0) / 60)
    flush(stdout)
end
out_dir = joinpath(TO_folder, COUT)
mkpath(out_dir)
jldsave(joinpath(out_dir, "fits.jld2"); res, ls_val, ls_sse, ls_ho, settings = CST, K, D, feat = FEAT, cfg,
        loss = CLOSS, input = CINPUT, seq = CSEQ)
@info "written" out_dir
