# Step 3: a 1-D convolutional network mean on the lag history (Rik, 2026-10-22: "also try an CNN
# architecture"), on the same data, target, input and comparators as the MLP of step 3a
# (`step3_data.jl`). Not deployable yet: the closure path (`StochLSTM`) has no convolutional arch.
#
# 🔑 THE STEP-3 NETWORK (Rik, 2026-10-22: "make HEAD=identity the CNN tool's default and treat the
# plain-likelihood CNN as the step-3 network") is this tool's DEFAULT: SEQ = lags, K = 2, INPUT = full,
# 16 channels, D = 64, tanh, Glorot, the plain Gaussian likelihood with the head starting at Sigma = I,
# Adam lr 1e-3, weight decay 1e-1 (network only), batch 256, early stopping on 8.2-10 TU, cap 40 000
# updates, seeds 1-3. On the desktop (3 seeds): validation 0.5 SSE 0.2023-0.2030 (LS 0.2050), NLL -6.69
# to -6.77 (LS -6.95); 10-100 TU rms / LS 1.01 1.00 1.15-1.19 1.23-1.28 1.36-1.42 1.28-1.31, within 0-10 %
# of LS on states inside the training range, the rest is extrapolation. Every other option below was a
# probe and lost (mse / rmse losses, the seeded head (equal), SEQ = full / fullq, kernel 3, 32 channels,
# no weight decay, relu / He); the probes' numbers are in TO_LSTM/step3/*/fit.log.
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
# started at least squares' residual covariance, `LS_HEAD`) | crps (Rik, 2026-10-23: one-step CRPS against
# the likelihood; the closed-form Gaussian CRPS of each QoI, summed over the QoIs, under a head of
# marginal sds exp(bd) trained with the network; `Araw` stays frozen at A = I, since the marginal CRPS does
# not see the correlations). Every fit, whatever its loss, also prints the validation CRPS with the
# diagonal of Sigma refitted to the fit rows' residuals (least squares' beside it) and its own head's
# marginal sd over the validation rms per QoI (1 = calibrated one step ahead; mse/rmse have no head).
# RIKFLOW_CNN_INPUT = full (default: q*, dQ
# and Dq* per step) | nodiff (q* and dQ per step: the kernel spans neighbouring steps and can form the
# predictor increments itself, Rik: "the CNN might not need the inputs that were subtracted").
#
# Third probes (Rik, 2026-10-22): RIKFLOW_CNN_SEQ = lags (default, above: the h history steps, q^{n*}
# joining after the convolutions) | full (q^{n*} IN the sequence, k = 0..h: "please include q*_n in the q*
# sequence"; dQ^{n-k} -- and Dq*^{n-k} with INPUT = full -- is unknown at k = 0, so it is set to 0, its
# standardised mean, and a MASK channel, 0 at k = 0 and 1 elsewhere, lets the shared kernels tell the
# missing value from a zero, Rik's choice; q^{n*} then no longer enters the dense layers separately).
# Kernel size via RIKFLOW_CNN_K (Rik: 3, stride 1). SEQ = fullq: q^{n*} both in the sequence (as full)
# and appended to the dense layers (as lags) (Rik, 2026-10-22: "q* inside the sequence and added at the end").
#
# Device (2026-10-22, Rik: "lets go for cuDNN", for bigger kernels and histories): RIKFLOW_CNN_DEVICE = cpu
# (default) | gpu. On the GPU the data and the parameters live on the device (CuArray, RikFlow's own CUDA
# through its Lux extension) and Lux's Conv runs through cuDNN (`cuDNN` in training/Project.toml, loaded
# only for DEVICE = gpu); the scores are computed on the host. Random draws stay on the host.
#
# Grid (RIKFLOW_CNN_*): CHS channels {16}, LRS {1e-3}, WDS {1e-1}, SEEDS {1, 2, 3}, K = 2, D = 64,
# MAXUPD 40000, OUT step3/CNN_step3 (defaults = the step-3 network; each a comma list for a grid). Prints,
# per fit, the validation 0.5 SSE and NLL against least
# squares on the same rows, and the rms one-step error on 10-100 TU per QoI over least squares'.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2, Printf, Statistics, LinearAlgebra, Random

const RF = RikFlow
include(joinpath(@__DIR__, "step3_data.jl"))        # record, target, input map, checks, LS floor
FEAT === :dqinc || error("the CNN reads the dqinc input (q*, dQ and Dq* per step); got FEAT = $FEAT")

envc(k, d) = (v = strip(get(ENV, "RIKFLOW_CNN_" * k, "")); isempty(v) ? d : v)
envcv(k, d, U) = parse.(U, strip.(split(envc(k, d), ",")))
const CHS = envcv("CHS", "16", Int)
const LRS = envcv("LRS", "1e-3", Float64)
const CWDS = envcv("WDS", "1e-1", Float64)
const CSEEDS = envcv("SEEDS", "1,2,3", Int)
const K = parse(Int, envc("K", "2"))
const D = parse(Int, envc("D", "64"))
# RIKFLOW_CNN_NOSTOP = 1: never stop early, train to MAXUPD however flat the validation loss (Rik,
# 2026-10-22: "keep continuing despite flat validation loss to see if we get an improvement later");
# RIKFLOW_CNN_NODECAY = 1: also keep the learning rate constant. The validation curve is kept either way,
# and the LAST iterate is scored next to the best one.
const CST = (; batch = 256, val_every = 100, patience = 30, stop_patience = 80, min_lr = 1e-4,
             lr_decay = 0.3, max_updates = parse(Int, envc("MAXUPD", "40000")),
             nostop = envc("NOSTOP", "0") == "1", nodecay = envc("NODECAY", "0") == "1")
const COUT = envc("OUT", "step3/CNN_step3")
const CDEV = Symbol(envc("DEVICE", "cpu"))
CDEV in (:cpu, :gpu) || error("RIKFLOW_CNN_DEVICE must be cpu or gpu; got $CDEV")
CDEV === :gpu && @eval using cuDNN                          # NNlib's cuDNN extension: Lux's Conv on CuArrays
"To the device: `CuArray` on the GPU (errors if none is functional, never a silent CPU fallback), else identity."
const DEVF = CDEV === :gpu ? RF.m4_device("cuda") : identity
todev(x) = Lux.fmap(DEVF, x)
const CLOSS = Symbol(envc("LOSS", "nll"))
CLOSS in (:mse, :rmse, :nll, :crps) || error("RIKFLOW_CNN_LOSS must be mse, rmse, nll or crps; got $CLOSS")
CLOSS === :crps && CDEV === :gpu && error("LOSS = crps is written for the CPU (norm_cdf is a scalar series)")
const CINPUT = Symbol(envc("INPUT", "full"))
CINPUT in (:full, :nodiff, :rel) || error("RIKFLOW_CNN_INPUT must be full, nodiff or rel; got $CINPUT")
# INPUT = rel (Rik, 2026-10-22): per step [(q^{n-k*} - q^{n*}) / s_k, dQ^{n-k}] -- the predictor RELATIVE TO
# NOW, each lag standardised by its own sd on the fit rows, so the within-window trajectory is on a usable
# scale without step-to-step differencing; the level q^{n*} itself enters after the convolutions (SEQ = lags).
# History length: RIKFLOW_MLP_H (step3_data.jl; default 5). Every fit is also scored against LinReg1's least
# squares (h = 5) on the same columns, so fits at different h compare against one reference.
CINPUT === :rel && envc("SEQ", "lags") != "lags" && error("INPUT = rel is defined for SEQ = lags")
# RIKFLOW_CNN_ACT = tanh (default) | relu: the activation of both convolutions and the two hidden dense
# layers (Rik, 2026-10-22: "try using relu instead of tanh"; relu is unbounded, so it extrapolates linearly
# where tanh saturates). Lux's default (Glorot) initialisation either way.
const CACT = Symbol(envc("ACT", "tanh"))
CACT in (:tanh, :relu) || error("RIKFLOW_CNN_ACT must be tanh or relu; got $CACT")
const ACTF = CACT === :relu ? relu : tanh
# RIKFLOW_CNN_INIT = glorot (default, Lux's) | he: Kaiming-normal weights (gain sqrt 2) for both
# convolutions and the two hidden dense layers, the usual initialisation for relu (Rik, 2026-10-22: "please
# try tuning a relu network"); the linear output layer keeps Glorot.
const CINIT = Symbol(envc("INIT", "glorot"))
CINIT in (:glorot, :he) || error("RIKFLOW_CNN_INIT must be glorot or he; got $CINIT")
const WINIT = CINIT === :he ? kaiming_normal : glorot_uniform
# RIKFLOW_CNN_HEAD = identity (default: the head starts at Sigma = I, the plain likelihood; Rik,
# 2026-10-22: "I am not sure if I like the preset likelihood loss", then made it the default) | ls (it
# starts at least squares' residual covariance, `LS_HEAD`; within seed noise of identity, 3 seeds each).
# Either way the head is trained with the network.
const CHEAD = Symbol(envc("HEAD", "identity"))
CHEAD in (:ls, :identity) || error("RIKFLOW_CNN_HEAD must be ls or identity; got $CHEAD")
const HEAD0 = CHEAD === :ls ? (; bd = copy(LS_HEAD.bd), Araw = copy(LS_HEAD.Araw)) :
              (; bd = zeros(T, nq), Araw = zeros(T, nq, nq))   # A = I (exp of a zero diagonal), Sigma = I
const CSEQ = Symbol(envc("SEQ", "lags"))
const INSEQ = CSEQ in (:full, :fullq)                    # q^{n*} inside the sequence (with the mask)
CSEQ in (:lags, :full, :fullq) || error("RIKFLOW_CNN_SEQ must be lags, full or fullq; got $CSEQ")
const NSTEP = INSEQ ? cfg.h + 1 : cfg.h              # sequence length
const NCH = (CINPUT === :full ? 3nq : 2nq) + (INSEQ ? 1 : 0)   # channels per step (+ mask)
"sd of (q^{n-k*} - q^{n*}) per QoI and lag on the fit rows, in the input's standardised units (INPUT = rel)"
const REL_S = let Z = Float64.(Xc[:, 1:ntr])
    reduce(hcat, [vec(std(Z[lagrows(k).s, :] .- Z[1:nq, :]; dims = 2)) for k in 1:cfg.h])
end
NSTEP - 2(K - 1) >= 1 || error("kernel $K leaves no position after two convolutions of $NSTEP steps")

"""
The sequence (NSTEP x NCH x N, oldest step first) and q^{n*} (N_Q x N) from the mapped input `Z`.
SEQ = lags: steps k = h..1. SEQ = full: steps k = h..0, the last being the current step, where q^{n*}
is known and dQ^n (and Dq*^n) are not: they stay 0 and the mask channel (the last) is 0 there, 1 elsewhere.
"""
function lagtensor(Z)
    N, H = size(Z, 2), cfg.h
    A = zeros(T, NSTEP, NCH, N)
    ks = INSEQ ? (H:-1:0) : (H:-1:1)
    for (t, k) in enumerate(ks)
        if k == 0
            A[t, 1:nq, :] = Z[1:nq, :]                       # q^{n*}; dQ^n, Dq*^n unknown: 0, mask 0
            continue
        end
        lr = lagrows(k)
        A[t, 1:nq, :] = CINPUT === :rel ? (Z[lr.s, :] .- Z[1:nq, :]) ./ REL_S[:, k] :   # q^{n-k*} - q^{n*}
                        Z[lr.s, :]                                                     # q^{n-k*}
        A[t, (nq + 1):(2nq), :] = Z[lr.q, :]                # dQ^{n-k}
        CINPUT === :full &&                                 # Dq*^{n-k}
            (A[t, (2nq + 1):(3nq), :] = Z[nreg .+ (((k - 1) * nq + 1):(k * nq)), :])
        INSEQ && (A[t, NCH, :] .= one(T))          # the mask: known
    end
    return A, T.(Z[1:nq, :])
end

function mkmodel(C)
    conv = Chain(Conv((K,), NCH => C, ACTF; init_weight = WINIT), Conv((K,), C => C, ACTF; init_weight = WINIT),
                 FlattenLayer())
    nflat = (NSTEP - 2(K - 1)) * C
    CSEQ === :full &&                                       # q^{n*} is in the sequence
        return Chain(conv, Dense(nflat => D, ACTF; init_weight = WINIT), Dense(D => D, ACTF; init_weight = WINIT),
                     Dense(D => nq))
    return Chain(Parallel(vcat, conv, NoOpLayer()), Dense(nflat + nq => D, ACTF; init_weight = WINIT),
                 Dense(D => D, ACTF; init_weight = WINIT), Dense(D => nq))
end
"The model's input: the sequence alone (SEQ = full), or the sequence and q^{n*} (SEQ = lags, fullq)."
modelin(A, Q) = CSEQ === :full ? A : (A, Q)

Atr, Qtr = lagtensor(Float64.(Xc[:, 1:ntr]))
# the sequence is a rearrangement of the checked input (step3_data.jl); check the current step's layout
if INSEQ
    Atr[end, 1:nq, :] == Qtr || error("the last step of the sequence is not q^{n*}")
    all(iszero, Atr[end, (nq + 1):end, :]) || error("dQ^n / Dq*^n / mask at the current step are not 0")
    all(isone, Atr[1:(end - 1), NCH, :]) || error("the mask is not 1 at the history steps")
    Atr[end - 1, 1:nq, :] == T.(Xc[lagrows(1).s, 1:ntr]) || error("step n-1 does not hold q^{n-1*}")
end
Ava, Qva = lagtensor(Float64.(Xc[:, (ntr + 1):end]))
Aho, Qho = lagtensor(Xf[:, HO])
Ytr32, Yva32 = T.(Ytr), T.(Yva)
@info "CNN probe" CHS LRS CWDS CSEEDS K D CLOSS CHEAD CINPUT CSEQ CACT CINIT NSTEP NCH CST.max_updates fit_rows = ntr val_rows = ncol - ntr

# the likelihood's head, as the library's: Sigma = D R D, D = diag(exp(bd)), R^{-1} = A'A with A lower
# triangular from Araw (strict lower part, exp of the diagonal); the strict-lower mask is a constant
# (elementwise with constant masks, no `Diagonal` / `diag`, so the same code runs on a device)
const LMASK = DEVF(T[i > j for i in 1:nq, j in 1:nq])
const IMASK = DEVF(T[i == j for i in 1:nq, j in 1:nq])
"Gaussian NLL per step of residuals `r` (nq x B) under the head `hd = (; bd, Araw)`."
function head_nll(hd, r)
    A = hd.Araw .* LMASK .+ IMASK .* exp.(hd.Araw .* IMASK)      # exp only of the diagonal: no Inf * 0
    z = A * (r ./ exp.(hd.bd))
    return T(0.5) * (nq * T(log(2pi)) + 2sum(hd.bd) - 2sum(hd.Araw .* IMASK) + sum(abs2, z) / size(r, 2))
end
"""
Gaussian CRPS per step, summed over the QoIs, of residuals `r` (nq x B) under marginal sds `exp.(bd)`:
s [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)], z = r / s. The gradient is written out, since `norm_cdf` is
a series that Zygote would otherwise differentiate term by term: d/dr = 2 Phi(z) - 1, d/ds = 2 phi(z) -
1/sqrt(pi) (the z terms cancel), d/dbd = s d/ds.
"""
function head_crps(bd, r)
    s = exp.(bd)
    z = r ./ s
    Phi, phi = T.(RF.norm_cdf.(z)), T.(RF.norm_pdf.(z))
    return sum(s .* (z .* (2 .* Phi .- 1) .+ 2 .* phi .- 1 / sqrt(T(pi)))) / size(r, 2)
end
Zygote.@adjoint function head_crps(bd, r)
    s = exp.(bd)
    z = r ./ s
    Phi, phi = T.(RF.norm_cdf.(z)), T.(RF.norm_pdf.(z))
    B = size(r, 2)
    y = sum(s .* (z .* (2 .* Phi .- 1) .+ 2 .* phi .- 1 / sqrt(T(pi)))) / B
    back(dy) = (dy .* vec(sum(s .* (2 .* phi .- 1 / sqrt(T(pi))); dims = 2)) ./ B, dy .* (2 .* Phi .- 1) ./ B)
    return y, back
end
"Validation CRPS per step, summed over the QoIs, of residuals `r` (nq x N) under marginal sds `s` (Float64)."
crps_eval(r, s) = mean(sum(RF.crps_gaussian.(0.0, s, r); dims = 1))
"The head's marginal sds: Sigma = D (A'A)^{-1} D, D = diag(exp(bd)) (A = I unless the head is a likelihood's)."
function head_sd(hd)
    Ar = Float64.(Array(hd.Araw))
    A = tril(Ar, -1) + Diagonal(exp.(diag(Ar)))
    return exp.(Float64.(Array(hd.bd))) .* sqrt.(diag(inv(A' * A)))
end
const LS_CRPS = crps_eval(Yva .- Cls' * Rva_, vec(std(Rls; dims = 2)))
# the arrays the fits read, on the device
const Atr_d, Qtr_d, Ytr_d = DEVF(Atr), DEVF(Qtr), DEVF(Ytr32)
const Ava_d, Qva_d, Yva_d = DEVF(Ava), DEVF(Qva), DEVF(Yva32)
const Aho_d, Qho_d = DEVF(Aho), DEVF(Qho)

function fit_cnn(C, lr, wd, seed)
    rng = Xoshiro(seed)
    model = mkmodel(C)
    net, st = Lux.setup(rng, model)
    ps = todev((; net, head = (; bd = copy(HEAD0.bd), Araw = copy(HEAD0.Araw))))
    st = todev(st)
    rule = wd > 0 ? Optimisers.AdamW(T(lr), (T(0.9), T(0.999)), T(wd); couple = true) : Optimisers.Adam(T(lr))
    opt = Optimisers.setup(rule, ps)
    # the head: no weight decay; trained only by the likelihood, or (its sds only) by the CRPS
    wd > 0 && Optimisers.adjust!(opt.head; lambda = zero(T))
    CLOSS in (:mse, :rmse) && Optimisers.freeze!(opt.head)
    CLOSS === :crps && Optimisers.freeze!(opt.head.Araw)
    out(p, A, Q) = first(model(modelin(A, Q), p.net, st))
    loss(p, A, Q, Y) = CLOSS === :mse ? T(0.5) * sum(abs2, out(p, A, Q) .- Y) / size(Y, 2) :
                       CLOSS === :rmse ? rmse_sum(out(p, A, Q), Y) :
                       CLOSS === :crps ? head_crps(p.head.bd, out(p, A, Q) .- Y) : head_nll(p.head, out(p, A, Q) .- Y)
    vloss(p) = Float64(loss(p, Ava_d, Qva_d, Yva_d))
    best = (; val = vloss(ps), ps = deepcopy(ps), update = 0)
    hist = (; update = Int[], val = Float64[], lr = Float64[])
    cur, since_best, since_decay, upd, stop = T(lr), 0, 0, 0, :cap
    nb = fld(ntr, CST.batch)
    while upd < CST.max_updates
        idxs = shuffle(rng, 1:ntr)
        for k in 1:nb
            c = idxs[((k - 1) * CST.batch + 1):(k * CST.batch)]
            g = Zygote.gradient(p -> loss(p, Atr_d[:, :, c], Qtr_d[:, c], Ytr_d[:, c]), ps)[1]
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
            upd % CST.val_every == 0 || continue
            v = vloss(ps)
            push!(hist.update, upd); push!(hist.val, v); push!(hist.lr, cur)
            if v < best.val
                best = (; val = v, ps = deepcopy(ps), update = upd)
                since_best, since_decay = 0, 0
            else
                since_best += 1
                since_decay += 1
            end
            if !CST.nodecay && since_decay >= CST.patience && cur > CST.min_lr
                cur = max(T(CST.min_lr), cur * T(CST.lr_decay))
                Optimisers.adjust!(opt, cur)
                since_decay = 0
            end
            if !CST.nostop && cur <= CST.min_lr && since_best >= CST.stop_patience
                stop = :floor
                break
            end
            upd >= CST.max_updates && break
        end
        stop === :floor && break
    end
    pb = best.ps
    pred(A, Q; p = pb) = Float64.(Array(out(p, A, Q)))
    rtr, rva = Ytr .- pred(Atr_d, Qtr_d), Yva .- pred(Ava_d, Qva_d)
    ho_ratio = rmsq(Yf[:, HO] .- pred(Aho_d, Qho_d)) ./ ls_ho
    ho_ratio1 = rmsq(Yf[:, HO] .- pred(Aho_d, Qho_d)) ./ ls1_ho     # against LinReg1 (h = 5)
    # the last iterate too: with NOSTOP the best and the last can be far apart
    rva_last = Yva .- pred(Ava_d, Qva_d; p = ps)
    last = (; val = vloss(ps), half_sse = 0.5 * sum(abs2, rva_last) / size(rva_last, 2),
            ho_ratio = rmsq(Yf[:, HO] .- pred(Aho_d, Qho_d; p = ps)) ./ ls_ho)
    crps_refit = crps_eval(rva, vec(std(rtr; dims = 2)))
    head_ratio = CLOSS in (:nll, :crps) ? head_sd(pb.head) ./ rmsq(rva) : fill(NaN, nq)
    return (; ps = Lux.fmap(Array, pb), st = Lux.fmap(Array, st), device = CDEV, ho_ratio1, h = cfg.h, val_loss = best.val, val_half_sse = 0.5 * sum(abs2, rva) / size(rva, 2),
            nll_refit = gnll(rva, cov(permutedims(rtr))), crps_refit, head_ratio, ho_ratio, best_update = best.update, updates = upd,
            stop, nparams = Lux.parameterlength(model), loss = CLOSS, head = CHEAD, input = CINPUT, seq = CSEQ, act = CACT,
            init = CINIT,
            hist, last)
end

res = Dict{Tuple{Int,Float64,Float64,Int},Any}()
for C in CHS, lr in LRS, wd in CWDS, seed in CSEEDS
    t0 = time()
    r = fit_cnn(C, lr, wd, seed)
    res[(C, lr, wd, seed)] = merge(r, (; wall = time() - t0))
    @printf("cnn  C %3d K %d D %d (%6d weights) lr %.0e wd %.0e seed %d %s %s seq %s %s %s | %5d upd, best @ %5d (%s) | val loss %.4f, NLL Sigma refit %.4f (LS %.4f) | 0.5 SSE %.4f (LS %.4f) | 10-100 TU rms / LS %s | %.1f min\n",
            C, K, D, r.nparams, lr, wd, seed, CLOSS === :nll ? string("nll/", CHEAD) : string(CLOSS), CINPUT, CSEQ,
            string(CACT, "/", CINIT), CDEV, r.updates, r.best_update, r.stop, r.val_loss, r.nll_refit, ls_val,
            r.val_half_sse, ls_sse, join((@sprintf("%.2f", x) for x in r.ho_ratio), " "), (time() - t0) / 60)
    @printf("     h %d | against LinReg1 (h = 5): val 0.5 SSE %.4f (LinReg1 %.4f) | 10-100 TU rms / LinReg1 %s\n", r.h,
            r.val_half_sse, ls1_sse, join((@sprintf("%.2f", x) for x in r.ho_ratio1), " "))
    @printf("     val CRPS, diagonal Sigma refit %.4f (LS %.4f) | own head's sd / val rms %s\n", r.crps_refit, LS_CRPS,
            join((@sprintf("%.3f", x) for x in r.head_ratio), " "))
    @printf("     %.2f ms per update (including validations and scoring)\n", 1e3 * (time() - t0) / r.updates)
    # the validation curve at checkpoints, and the last iterate
    cps = [u for u in (1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 150_000, 200_000, 300_000) if u <= r.updates]
    @printf("     val loss at %s\n", join((@sprintf("%dk: %.3f (lr %.0e)", u ÷ 1000, r.hist.val[findfirst(>=(u), r.hist.update)],
                                                    r.hist.lr[findfirst(>=(u), r.hist.update)]) for u in cps), ", "))
    @printf("     last iterate (update %d): val loss %.4f, 0.5 SSE %.4f | 10-100 TU rms / LS %s\n", r.updates,
            r.last.val, r.last.half_sse, join((@sprintf("%.2f", x) for x in r.last.ho_ratio), " "))
    flush(stdout)
end
out_dir = joinpath(TO_folder, COUT)
mkpath(out_dir)
jldsave(joinpath(out_dir, "fits.jld2"); res, ls_val, ls_sse, ls_ho, ls_crps = LS_CRPS, settings = CST, K, D, feat = FEAT, cfg,
        loss = CLOSS, input = CINPUT, seq = CSEQ, act = CACT)
@info "written" out_dir
