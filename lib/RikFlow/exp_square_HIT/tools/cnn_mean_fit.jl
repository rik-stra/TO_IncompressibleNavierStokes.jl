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
include(joinpath(@__DIR__, "step3_innoise.jl"))     # the closure's noise in the training window (INNOISE, PROBE)

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
# RIKFLOW_CNN_REFIT = 1 (2026-10-23, Rik: "take both losses forward"): after each stage-1 fit, the
# protocol's second training (closures.tex `sec:closures-training`): the same network trained again on
# ALL of 1-10 TU, from the same initialisation, for as many updates as its stage-1 best iterate took, at
# the learning rate stage 1 used at each update (its plateau schedule replayed; no validation, the last
# iterate is kept); same loss, head rules and weight decay. Then `pred.jld2`: the stage-1 and refitted
# predictions of every seed on every step of the record (raw dQ), and least squares on 1-10 TU (at h = 5,
# LinReg1), in the layout `analysis/net_residual.jl` reads. One (C, lr, wd) setting per run.
const CREFIT = envc("REFIT", "0") == "1"
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
# starts at least squares' residual covariance, `LS_HEAD`; within seed noise of identity, 3 seeds each) |
# inj (2026-10-26, input-noise training: it starts at the injected Σ of RIKFLOW_CNN_INNOISE_SIGMA, its full
# covariance for the likelihood, its sds alone for the CRPS, so that INNOISE = fixed and head start from the
# same noise; `head_init`). Either way the head is trained with the network.
const CHEAD = Symbol(envc("HEAD", "identity"))
CHEAD in (:ls, :identity, :inj) || error("RIKFLOW_CNN_HEAD must be ls, identity or inj; got $CHEAD")
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
# REFIT: the 1-10 TU rows in record order (the fit rows, then the validation rows), on the device
const Aall_d, Qall_d, Yall_d = DEVF(cat(Atr, Ava; dims = 3)), DEVF(hcat(Qtr, Qva)), DEVF(hcat(Ytr32, Yva32))

# RIKFLOW_CNN_PROBE = <run dir under TO_LSTM> (2026-10-26, Rik: input-noise training): does a network pass
# the closure's own noise into its mean more than least squares does? The run's refitted networks and LinReg1
# (least squares on all 1-10 TU, h = 5), on 1-10 and 10-100 TU, clean and with the window noise of a closure
# drawing from each model's own MVG (its residual on 1-10 TU, about zero), the same standard normals for
# both. Per QoI: the rms one-step error clean and noisy, and the gain, rms(noisy - clean prediction) / sd(η).
# Also the LINEAR ANALOGUE of input-noise training: least squares on the same rows with the window noise of
# LinReg1's own MVG, in closed form -- E|y - C'(x + dx)|^2 = |y - C'x|^2 + tr(C' S_dx C), so
# C = (X X' + N S_dx) \ X Y', S_dx = E[dx dx'] exact from the linear map eta -> dx: a ridge with the closure's
# noise as its penalty (the bias is not penalised). Scores only; no fit.
const CPROBE = envc("PROBE", "")
if !isempty(CPROBE)
    F = load(joinpath(TO_folder, CPROBE, "fits.jld2"))
    (F["K"], F["D"], F["input"], F["seq"], F["act"]) == (K, D, CINPUT, CSEQ, CACT) ||
        error("PROBE: $CPROBE was fitted with K, D, input, seq, act = $((F["K"], F["D"], F["input"], F["seq"], F["act"]))")
    cfg.h == 5 || error("PROBE compares against LinReg1 (h = 5)")
    rows10 = findall(n -> cfg.train_range[1] <= n <= cfg.train_range[2], cols)
    X10, Y10 = Float64.(dat.Xc), Float64.(dat.Yc)
    Cls10 = X10' \ Y10'                                          # LinReg1: least squares on all 1-10 TU
    "E[dx dx'] of the window noise for corrections of covariance L L' (standardised target units)."
    function window_cov(L)
        M = zeros(nreg, nq * cfg.h)
        for j in 1:cfg.h, i in 1:nq
            e = zeros(nq, cfg.h, 1)
            e[i, j, 1] = 1
            M[:, (j - 1) * nq + i] = window_noise(e)[:, 1]          # dx per unit raw eta
        end
        Sraw = Diagonal(so) * (L * L') * Diagonal(so)
        return M * kron(Matrix(1.0I, cfg.h, cfg.h), Sraw) * M'
    end
    lsmodel(Cm) = (rows, dx) -> Cm' * (Xr64[:, rows] .+ dx)
    res10(f) = Yf[:, rows10] .- f(rows10, zeros(nreg, length(rows10)))   # residual on the 1-10 TU columns
    Lls = cov_factor(mom0(res10(lsmodel(Cls10))))
    Cnt = (X10 * X10' .+ size(X10, 2) .* window_cov(Lls)) \ (X10 * Y10')   # LinReg1 trained with its own window noise
    Lnt = cov_factor(mom0(res10(lsmodel(Cnt))))
    nrep = parse(Int, envc("PROBE_REPS", "3"))
    fmtp(v) = join((@sprintf("%.3f", x) for x in v), " ")
    for ((C, lr, wd, seed), r) in sort(collect(F["res"]); by = first)
        haskey(r, :refit) || continue
        model = mkmodel(C)
        nnmean(Z) = (AQ = lagtensor(Z); Float64.(first(model(modelin(AQ...), r.refit.ps.net, r.refit.st))))
        nn = (rows, dx) -> nnmean(Xf[:, rows] .+ P * dx)
        Lnn = cov_factor(mom0(res10(nn)))
        models = (("CNN", nn, Lnn), ("LinReg1", lsmodel(Cls10), Lls), ("LinReg1 + noise", lsmodel(Cnt), Lnt))
        @printf("\nPROBE %s: C %d lr %.0e wd %.0e seed %d, the refitted network; each model with the noise of its own MVG\n",
                CPROBE, C, lr, wd, seed)
        for (nm, _, L) in models
            @printf("  sd(eta) / sd(target) %-16s %s\n", nm, fmtp(sqrt.(diag(L * L'))))
        end
        for (wl, rows) in (("1-10 TU", rows10), ("10-100 TU", HO))
            N = length(rows)
            Es = [randn(Xoshiro(1000seed + rep), nq, cfg.h * N) for rep in 1:nrep]   # common to every model
            @printf("  %s: rms clean | rms noisy | noisy / clean | gain = rms(noisy - clean mean) / sd(eta)\n", wl)
            for (nm, f, L) in models
                yc = f(rows, zeros(nreg, N))
                ec = rmsq(Yf[:, rows] .- yc)
                mn, g = zeros(nq), zeros(nq)
                for E in Es
                    yn = f(rows, window_noise(eta_from(L, E)))
                    mn .+= rmsq(Yf[:, rows] .- yn) .^ 2 ./ nrep
                    g .+= rmsq(yn .- yc) .^ 2 ./ nrep
                end
                @printf("    %-16s %s | %s | %s | %s\n", nm, fmtp(ec), fmtp(sqrt.(mn)), fmtp(sqrt.(mn) ./ ec),
                        fmtp(sqrt.(g) ./ sqrt.(diag(L * L'))))
            end
        end
    end
    exit()
end

# RIKFLOW_CNN_INNOISE = none (default) | fixed (2026-10-26, Rik: "the noise-trained CNN (RMSE, fixed Σ with the
# outer loop)"): every training window carries the noise of a closure drawing from a FIXED Σ (tools/
# step3_innoise.jl: one η per correction, the levels the running sum anchored at the prediction step), fresh
# draws every update from the noise's own stream (the shuffle stays the clean run's). Σ, in the standardised
# target units, from RIKFLOW_CNN_INNOISE_SIGMA = ls (least squares' residual on the fit rows, about zero) |
# <run dir under TO_LSTM>: the Σ that run deploys -- `S_deploy` in its pred.jld2 (a noise-trained run: the
# residual of its refitted network on 1-10 TU under its own injected noise, about zero, 4 draws), else the
# second moment of its refitted network's clean residual on 1-10 TU (a clean run: its MVG, as the export's
# `mvg`). The OUTER LOOP is run by hand: run 1 from a clean run's Σ, run 2 from run 1's S_deploy, until
# S_deploy / Σ stops moving (printed). The validation loss (early stopping) and the noisy scores use one fixed
# draw; the clean scores are printed beside them.
# INNOISE = head (2026-10-26, Rik: the noise "could dynamically move with the learned MVG head"; likelihood and
# CRPS only): the injected Σ is the head's CURRENT Σ, read before each gradient (so outside it: a stop-gradient,
# without which the head would shrink to clean its own inputs) -- the full covariance for the likelihood, the
# sds alone for the CRPS, whose head has no correlations (inject what is deployed). The validation noise is one
# fixed draw of standard normals times the current head's factor; the refit follows its own head the same way.
# No outer loop: the head is fitted to the residual under its own noise as it trains. `S_deploy` is the refitted
# head's Σ; `S_noisy` (pred.jld2) the refitted network's residual under that head's noise, for the check.
# INNOISE_SIGMA then only sets the head's start (with HEAD = inj).
const CINN = Symbol(envc("INNOISE", "none"))
CINN in (:none, :fixed, :head) || error("RIKFLOW_CNN_INNOISE must be none, fixed or head; got $CINN")
CINN === :head && CLOSS ∉ (:nll, :crps) && error("INNOISE = head needs a head: LOSS nll or crps, got $CLOSS")
CHEAD === :inj && CINN === :none && error("HEAD = inj starts the head at the injected Σ: set INNOISE")
const CINN_SRC = envc("INNOISE_SIGMA", "ls")
"The Σ (standardised target units) a run under TO_LSTM deploys, for seed `seed` (see INNOISE above)."
function run_sigma(run, seed)
    pr = load(joinpath(TO_folder, run, "pred.jld2"))
    pick(dct) = haskey(dct, seed) ? dct[seed] : only(values(dct))
    haskey(pr, "S_deploy") && return pick(pr["S_deploy"]), "S_deploy of $run"
    pr["cols"] == cols || error("$run's pred.jld2 is on other record columns")
    rows = findall(n -> cfg.train_range[1] <= n <= cfg.train_range[2], cols)
    r = (rec.q[:, cols[rows] .+ 1] .- rec.q_star[:, cols[rows]] .- pick(pr["pred3"])[:, rows]) ./ so
    return mom0(r), "the clean residual of $(run)'s refitted network on 1-10 TU, about zero"
end
"The injected Σ for seed `seed` and where it came from; `nothing` without input noise (INNOISE = head: the start)."
inj_sigma(seed) = CINN === :none ? (nothing, "none") :
                  CINN_SRC == "ls" ? (mom0(Rls), "least squares' residual on the fit rows, about zero") :
                  run_sigma(CINN_SRC, seed)
"""
The head's start for seed `seed`: `HEAD0`, or (HEAD = inj) the head of the injected Σ in the library's
parametrisation (as `head_from_residuals`): its sds, and for the likelihood its correlation; the CRPS head has
none (A = I, frozen).
"""
function head_init(seed)
    CHEAD === :inj || return (; bd = copy(HEAD0.bd), Araw = copy(HEAD0.Araw))
    S = first(inj_sigma(seed))
    sd = sqrt.(diag(S))
    CLOSS === :crps && return (; bd = T.(log.(sd)), Araw = zeros(T, nq, nq))
    Pm = inv(Symmetric(S ./ (sd * sd')))
    J = reverse(Matrix{Float64}(I, nq, nq); dims = 1)
    A = J * cholesky(Symmetric(J * Pm * J)).U * J          # lower triangular, A'A = R^{-1}
    return (; bd = T.(log.(sd)), Araw = T.(tril(A, -1) + Diagonal(log.(diag(A)))))
end
"The factor of the noise injected at parameters `p`: the head's (INNOISE = head; host, no gradient), else `Lfix`."
inj_factor(p, Lfix) = CINN === :head ? head_factor(p.head) : Lfix
"The network's input (A, Q) for mapped columns `Z` (nin x B) with the window noise `eta` added (host)."
noisy_tensor(Z, eta) = lagtensor(Float64.(Z) .+ P * window_noise(eta))
const LLS_INJ = cov_factor(mom0(Rls))                   # least squares' own noise, for its noisy comparator
CINN === :none || @info "input noise" CINN CINN_SRC CHEAD
# the head's start reproduces the injected Σ (HEAD = inj): its factor's covariance against Σ (the CRPS: diagonal)
if CHEAD === :inj
    let S = first(inj_sigma(first(CSEEDS))), L = head_factor(head_init(first(CSEEDS)))
        Sh, St = L * L', CLOSS === :crps ? Diagonal(diag(S)) : S
        (e = maximum(abs, Sh .- St) / maximum(diag(St))) < 1e-5 || error("the head's start differs from the injected Σ by $e")
        @info "head starts at the injected Σ" rel_err = e
    end
end

# RIKFLOW_CNN_SCORE = <run>[,<run>...] (dirs under TO_LSTM, each written with REFIT; 2026-10-26, the offline basis of
# input-noise training, results_LSTMS.md Sec. 36): scores only, no fit. Per seed of each run, writes
# `TO_LSTM/<RIKFLOW_CNN_SCORE_OUT, default step3/innoise_scores>/<basename of run>.jld2`:
#  - the stage-1 network on the validation rows (8.2-10 TU), clean and with the window noise of its own Σ (the
#    injected Σ of a noise-trained run, `noisy.Sinj`; a clean run's: its fit-row residual about zero), the draws
#    of `fit_cnn` (so the noisy 0.5 SSE reproduces a noise-trained run's own): 0.5 SSE, NLL with Σ refitted to the
#    fit-row residuals (noisy for noisy), CRPS with the sds refitted; and the same under one COMMON Σ for every run
#    (RIKFLOW_CNN_SCORE_COMMON = <run>: its seed-1 deployed Σ; default the run's own);
#  - the refitted network on the whole record along ONE noise path: η^m = L E^m drawn per record step m from the
#    Σ the run deploys (`run_sigma`), every window reading the path's η^{n-1} ... η^{n-h} as a closure's history
#    would carry its own draws (`pred3n`, raw dQ; the path's standard normals `E` and first step `m0`, shared by
#    every run of the same seed). With pred.jld2's clean `pred3` this gives the residual under the deployed noise,
#    its colour, and the gain (`analysis/net_noise_residual.jl`).
const CSCORE = envc("SCORE", "")
const CSCORE_COMMON = envc("SCORE_COMMON", "")
if !isempty(CSCORE)
    sout = joinpath(TO_folder, envc("SCORE_OUT", "step3/innoise_scores"))
    mkpath(sout)
    m0 = cols[1] - cfg.h                                    # the path's first record step
    raw(y) = y .* so .+ mo                                  # standardised -> raw dQ
    for run in split(CSCORE, ",")
        F = load(joinpath(TO_folder, run, "fits.jld2"))
        Pr = load(joinpath(TO_folder, run, "pred.jld2"))
        (F["K"], F["D"], F["input"], F["seq"], F["act"]) == (K, D, CINPUT, CSEQ, CACT) ||
            error("SCORE: $run was fitted with K, D, input, seq, act = $((F["K"], F["D"], F["input"], F["seq"], F["act"]))")
        Pr["cols"] == cols || error("SCORE: $run's pred.jld2 is on other record columns")
        out = Dict{Int,Any}()
        for ((C, lr, wd, seed), r) in sort(collect(F["res"]); by = first)
            model = mkmodel(C)
            prd(Z, ps, st) = Float64.(first(model(modelin(lagtensor(Z)...), ps.net, st)))
            # stage 1 on the validation rows: clean, and with the window noise of its own Σ
            yt, yv = prd(Float64.(Xc[:, 1:ntr]), r.ps, r.st), prd(Float64.(Xc[:, (ntr + 1):end]), r.ps, r.st)
            rtr, rva = Ytr .- yt, Yva .- yv
            nt = get(r, :noisy, nothing) !== nothing
            S1 = nt ? r.noisy.Sinj : mom0(rtr)
            L1 = cov_factor(S1)
            Eva = randn(Xoshiro(seed + 104729), nq, cfg.h * (ncol - ntr))
            Et = randn(Xoshiro(seed + 61), nq, cfg.h * ntr)
            rva_n = Yva .- prd(Float64.(Xc[:, (ntr + 1):end]) .+ P * window_noise(eta_from(L1, Eva)), r.ps, r.st)
            rtr_n = Ytr .- prd(Float64.(Xc[:, 1:ntr]) .+ P * window_noise(eta_from(L1, Et)), r.ps, r.st)
            hs(x) = 0.5 * sum(abs2, x) / size(x, 2)
            # and under one COMMON Σ for every run (RIKFLOW_CNN_SCORE_COMMON = <run>, its seed-1 deployed Σ; the same
            # draws), so that networks trained under different noise compare under the same noise
            Sc = isempty(CSCORE_COMMON) ? S1 : first(run_sigma(CSCORE_COMMON, 1))
            Lc = cov_factor(Sc)
            rva_c = Yva .- prd(Float64.(Xc[:, (ntr + 1):end]) .+ P * window_noise(eta_from(Lc, Eva)), r.ps, r.st)
            rtr_c = Ytr .- prd(Float64.(Xc[:, 1:ntr]) .+ P * window_noise(eta_from(Lc, Et)), r.ps, r.st)
            val = (; half_sse = hs(rva), nll = gnll(rva, cov(permutedims(rtr))), crps = crps_eval(rva, vec(std(rtr; dims = 2))),
                   half_sse_n = hs(rva_n), nll_n = gnll(rva_n, cov(permutedims(rtr_n))),
                   crps_n = crps_eval(rva_n, vec(std(rtr_n; dims = 2))), Sigma1 = S1, val_loss = r.val_loss,
                   half_sse_c = hs(rva_c), nll_c = gnll(rva_c, cov(permutedims(rtr_c))),
                   crps_c = crps_eval(rva_c, vec(std(rtr_c; dims = 2))), Sigma_c = Sc)
            nt && abs(val.half_sse_n - r.noisy.val_half_sse) > 1e-4 * r.noisy.val_half_sse &&
                error("SCORE $run seed $seed: noisy val 0.5 SSE $(val.half_sse_n) does not reproduce the run's $(r.noisy.val_half_sse)")
            # the refitted network along one noise path of the Σ it deploys
            Sd, srcd = run_sigma(run, seed)
            L = cov_factor(Sd)
            E = randn(Xoshiro(20261026 + seed), nq, cols[end] - m0 + 1)
            path = so .* (L * E)                             # raw η^m, m = m0, m0 + 1, ...
            eta = zeros(nq, cfg.h, length(cols))
            for (b, n) in enumerate(cols), j in 1:cfg.h
                eta[:, j, b] = path[:, n - j - m0 + 1]
            end
            y3 = prd(Xf, r.refit.ps, r.refit.st)
            y3n = prd(Xf .+ P * window_noise(eta), r.refit.ps, r.refit.st)
            maximum(abs, raw(y3) .- Pr["pred3"][seed]) < 1e-5 * maximum(abs, Pr["pred3"][seed]) ||
                error("SCORE $run seed $seed: the refitted network does not reproduce pred.jld2's pred3")
            out[seed] = (; val, Sigma = Sd, sigma_src = srcd, E, pred3n = raw(y3n), wd, lr, C)
            @printf("score %s seed %d wd %g: val clean 0.5 SSE %.4f NLL %.3f CRPS %.4f | noisy 0.5 SSE %.4f NLL %.3f CRPS %.4f | common 0.5 SSE %.4f NLL %.3f CRPS %.4f | own val loss %.4f\n",
                    run, seed, wd, val.half_sse, val.nll, val.crps, val.half_sse_n, val.nll_n, val.crps_n, val.half_sse_c,
                    val.nll_c, val.crps_c, r.val_loss)
            flush(stdout)
        end
        jldsave(joinpath(sout, basename(run) * ".jld2"); run = String(run), cols, m0, out, so, mo, h = cfg.h,
                loss = F["loss"], innoise = get(F, "innoise", :none), innoise_sigma = get(F, "innoise_sigma", "none"))
    end
    exit()
end

function fit_cnn(C, lr, wd, seed)
    rng = Xoshiro(seed)
    model = mkmodel(C)
    net, st = Lux.setup(rng, model)
    ps = todev((; net, head = head_init(seed)))
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
    # INNOISE: the injected Σ's factor (INNOISE = head: the head's, read at every use), the noise's own stream,
    # and one fixed draw for the validation rows
    Sinj, src = inj_sigma(seed)
    CINN === :head && (src = "the learned head (stop-gradient), started at " * (CHEAD === :inj ? src : string(CHEAD)))
    Linj = Sinj === nothing ? nothing : cov_factor(Sinj)
    nrng = Xoshiro(seed + 7919)
    Eva = randn(Xoshiro(seed + 104729), nq, cfg.h * (ncol - ntr))
    val_in(p) = CINN === :none ? (Ava_d, Qva_d) : DEVF.(noisy_tensor(Xc[:, (ntr + 1):end], eta_from(inj_factor(p, Linj), Eva)))
    vloss(p) = Float64(loss(p, val_in(p)..., Yva_d))
    best = (; val = vloss(ps), ps = deepcopy(ps), update = 0)
    hist = (; update = Int[], val = Float64[], lr = Float64[])
    cur, since_best, since_decay, upd, stop = T(lr), 0, 0, 0, :cap
    nb = fld(ntr, CST.batch)
    while upd < CST.max_updates
        idxs = shuffle(rng, 1:ntr)
        for k in 1:nb
            c = idxs[((k - 1) * CST.batch + 1):(k * CST.batch)]
            g = if CINN === :none
                Zygote.gradient(p -> loss(p, Atr_d[:, :, c], Qtr_d[:, c], Ytr_d[:, c]), ps)[1]
            else
                # the noisy batch is built before the gradient: a constant to it (stop-gradient for INNOISE = head)
                Ab, Qb = DEVF.(noisy_tensor(Xc[:, c], draw_window_eta(nrng, inj_factor(ps, Linj), length(c))))
                Zygote.gradient(p -> loss(p, Ab, Qb, Ytr_d[:, c]), ps)[1]
            end
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
    # INNOISE: the same scores with the window noise, the network under its injected Σ (INNOISE = head: its best
    # iterate's head) and least squares under its own (common standard normals); and the network's noisy residual
    # on the fit rows over the injected Σ, the outer loop's fixed-point check (1 = self-consistent). Also the
    # validation NLL and CRPS under noise, with the constant Gaussian refitted to the noisy fit-row residuals.
    noisy = if CINN === :none
        nothing
    else
        Lb = inj_factor(pb, Linj)
        Sb = CINN === :head ? Lb * Lb' : Sinj               # fixed: Σ as given, so the refit draws as before
        E = randn(Xoshiro(seed + 31), nq, cfg.h * length(HO))
        yh = pred(DEVF.(noisy_tensor(Xf[:, HO], eta_from(Lb, E)))...)
        lh = Cls' * (Xr64[:, HO] .+ window_noise(eta_from(LLS_INJ, E)))
        lv = Cls' * (Rva_ .+ window_noise(eta_from(LLS_INJ, Eva)))
        Et = randn(Xoshiro(seed + 61), nq, cfg.h * ntr)
        rtr_n = Ytr .- pred(DEVF.(noisy_tensor(Xc[:, 1:ntr], eta_from(Lb, Et)))...)
        rva_n = Yva .- pred(val_in(pb)...)
        (; src, Sinj = Sb, val_half_sse = 0.5 * sum(abs2, rva_n) / size(Yva, 2),
         ls_val_half_sse = 0.5 * sum(abs2, Yva .- lv) / size(Yva, 2),
         nll_refit = gnll(rva_n, cov(permutedims(rtr_n))), crps_refit = crps_eval(rva_n, vec(std(rtr_n; dims = 2))),
         ho_ratio = rmsq(Yf[:, HO] .- yh) ./ rmsq(Yf[:, HO] .- lh), sd_fit_over_inj = sqrt.(diag(mom0(rtr_n)) ./ diag(Sb)))
    end
    return (; ps = Lux.fmap(Array, pb), st = Lux.fmap(Array, st), device = CDEV, ho_ratio1, h = cfg.h, val_loss = best.val, val_half_sse = 0.5 * sum(abs2, rva) / size(rva, 2),
            nll_refit = gnll(rva, cov(permutedims(rtr))), crps_refit, head_ratio, ho_ratio, best_update = best.update, updates = upd,
            stop, nparams = Lux.parameterlength(model), loss = CLOSS, head = CHEAD, input = CINPUT, seq = CSEQ, act = CACT,
            init = CINIT, innoise = CINN, noisy,
            hist, last)
end

"""
The second training (RIKFLOW_CNN_REFIT = 1): set up exactly as `fit_cnn` (same seed, so the same
initialisation; same loss, head rules and weight decay), trained on all 1-10 TU rows for `nupd` updates
at the learning rate the stage-1 history `hist` used at each update; the last iterate, on the host. With
INNOISE, the same window noise (factor `Linj`, the noise's own stream; INNOISE = head: the refit's own current
head) on every batch.
"""
function refit_cnn(C, lr, wd, seed, hist, nupd; Linj = nothing)
    rng = Xoshiro(seed)
    nrng = Xoshiro(seed + 7919)
    model = mkmodel(C)
    net, st = Lux.setup(rng, model)
    ps = todev((; net, head = head_init(seed)))
    st = todev(st)
    rule = wd > 0 ? Optimisers.AdamW(T(lr), (T(0.9), T(0.999)), T(wd); couple = true) : Optimisers.Adam(T(lr))
    opt = Optimisers.setup(rule, ps)
    wd > 0 && Optimisers.adjust!(opt.head; lambda = zero(T))
    CLOSS in (:mse, :rmse) && Optimisers.freeze!(opt.head)
    CLOSS === :crps && Optimisers.freeze!(opt.head.Araw)
    out(p, A, Q) = first(model(modelin(A, Q), p.net, st))
    loss(p, A, Q, Y) = CLOSS === :mse ? T(0.5) * sum(abs2, out(p, A, Q) .- Y) / size(Y, 2) :
                       CLOSS === :rmse ? rmse_sum(out(p, A, Q), Y) :
                       CLOSS === :crps ? head_crps(p.head.bd, out(p, A, Q) .- Y) : head_nll(p.head, out(p, A, Q) .- Y)
    # the rate stage 1 used at update u: recorded at the first validation at or after u (before any decay
    # that validation triggered), as the MLP tool's `lr_at`
    lr_at(u) = T(hist.lr[searchsortedfirst(hist.update, u)])
    N = size(Yall_d, 2)
    nb = fld(N, CST.batch)
    upd, cur = 0, T(lr)
    while upd < nupd
        idxs = shuffle(rng, 1:N)
        for k in 1:nb
            upd >= nupd && break
            l = lr_at(upd + 1)
            l == cur || (Optimisers.adjust!(opt, l); cur = l)
            c = idxs[((k - 1) * CST.batch + 1):(k * CST.batch)]
            g = if CINN === :none
                Zygote.gradient(p -> loss(p, Aall_d[:, :, c], Qall_d[:, c], Yall_d[:, c]), ps)[1]
            else
                Ab, Qb = DEVF.(noisy_tensor(Xc[:, c], draw_window_eta(nrng, inj_factor(ps, Linj), length(c))))
                Zygote.gradient(p -> loss(p, Ab, Qb, Yall_d[:, c]), ps)[1]
            end
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
        end
    end
    return Lux.fmap(Array, ps), Lux.fmap(Array, st), model
end

if CREFIT
    length(CHS) == length(LRS) == length(CWDS) == 1 || error("RIKFLOW_CNN_REFIT = 1 takes one (C, lr, wd) setting per run")
end
# the whole record as the network reads it (host), and the standardised -> raw map of the target
const Afull, Qfull = CREFIT ? lagtensor(Xf) : (zeros(T, 0, 0, 0), zeros(T, 0, 0))
rawq(y) = y .* so .+ mo
"The network's standardised prediction on the whole record, on the host."
predfull(model, ps, st) = Float64.(first(model(modelin(Afull, Qfull), ps.net, st)))

res = Dict{Tuple{Int,Float64,Float64,Int},Any}()
pred1, pred3 = Dict{Int,Matrix{Float64}}(), Dict{Int,Matrix{Float64}}()
Sdep, Sinjd, Snoisy = Dict{Int,Matrix{Float64}}(), Dict{Int,Matrix{Float64}}(), Dict{Int,Matrix{Float64}}()   # INNOISE with REFIT
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
    if r.noisy !== nothing
        @printf("     input noise (Σ: %s; sd / LS residual sd %s)\n", r.noisy.src,
                join((@sprintf("%.3f", x) for x in sqrt.(diag(r.noisy.Sinj) ./ diag(mom0(Rls)))), " "))
        @printf("       with the window noise: val 0.5 SSE %.4f (LS %.4f) | 10-100 TU rms / LS %s | fit rows: noisy residual sd / injected sd %s\n",
                r.noisy.val_half_sse, r.noisy.ls_val_half_sse, join((@sprintf("%.2f", x) for x in r.noisy.ho_ratio), " "),
                join((@sprintf("%.3f", x) for x in r.noisy.sd_fit_over_inj), " "))
        @printf("       with the window noise: val NLL, Sigma refit to the noisy fit rows %.4f | val CRPS, sds refit %.4f\n",
                r.noisy.nll_refit, r.noisy.crps_refit)
    end
    if CREFIT
        t1 = time()
        ps3, st3, model3 = refit_cnn(C, lr, wd, seed, r.hist, r.best_update;
                                     Linj = r.noisy === nothing ? nothing : cov_factor(r.noisy.Sinj))
        y1, y3 = predfull(model3, r.ps, r.st), predfull(model3, ps3, st3)
        pred1[seed], pred3[seed] = rawq(y1), rawq(y3)
        # the refitted network on its own training window, and on 10-100 TU against LinReg1
        rows10 = findall(n -> cfg.train_range[1] <= n <= cfg.train_range[2], cols)
        sse10 = 0.5 * sum(abs2, Yf[:, rows10] .- y3[:, rows10]) / length(rows10)
        ho3 = rmsq(Yf[:, HO] .- y3[:, HO]) ./ ls1_ho
        res[(C, lr, wd, seed)] = merge(res[(C, lr, wd, seed)], (; refit = (; ps = ps3, st = st3, updates = r.best_update,
                                                                          half_sse_1_10 = sse10, ho_ratio1 = ho3)))
        @printf("     refit on 1-10 TU (%d updates, %.1f min): 0.5 SSE on 1-10 TU %.4f | 10-100 TU rms / LinReg1 %s\n",
                r.best_update, (time() - t1) / 60, sse10, join((@sprintf("%.2f", x) for x in ho3), " "))
        if r.noisy !== nothing
            # the Σ this network deploys: its residual on all 1-10 TU rows under the injected noise, about zero
            # (4 draws); S_deploy / Σ_inj -> 1 is the outer loop's fixed point. INNOISE = head: the refitted
            # head's Σ is deployed and injected; the residual under it (S_noisy) is the check
            L = CINN === :head ? head_factor(ps3.head) : cov_factor(r.noisy.Sinj)
            Sd = sum(1:4) do rep
                An, Qn = noisy_tensor(Xc, eta_from(L, randn(Xoshiro(seed + 500 + rep), nq, cfg.h * ncol)))
                mom0(Float64.(dat.Yc) .- Float64.(first(model3(modelin(An, Qn), ps3.net, st3)))) ./ 4
            end
            if CINN === :head
                Sdep[seed], Sinjd[seed], Snoisy[seed] = L * L', L * L', Sd
                @printf("     refit: head sd / LS residual sd %s | residual under its noise, sd / head sd %s (1 = self-consistent)\n",
                        join((@sprintf("%.3f", x) for x in sqrt.(diag(L * L') ./ diag(mom0(Rls)))), " "),
                        join((@sprintf("%.3f", x) for x in sqrt.(diag(Sd) ./ diag(L * L'))), " "))
            else
                Sdep[seed], Sinjd[seed] = Sd, r.noisy.Sinj
                @printf("     refit: S_deploy sd / injected sd %s (1 = the outer loop's fixed point)\n",
                        join((@sprintf("%.3f", x) for x in sqrt.(diag(Sd) ./ diag(r.noisy.Sinj))), " "))
            end
        end
    end
    flush(stdout)
end
out_dir = joinpath(TO_folder, COUT)
mkpath(out_dir)
jldsave(joinpath(out_dir, "fits.jld2"); res, ls_val, ls_sse, ls_ho, ls_crps = LS_CRPS, settings = CST, K, D, feat = FEAT, cfg,
        loss = CLOSS, input = CINPUT, seq = CSEQ, act = CACT, innoise = CINN, innoise_sigma = CINN_SRC)
if CREFIT
    # least squares on the same regressor and all of 1-10 TU (at h = 5: LinReg1), in the same raw units
    Cls10 = Float64.(dat.Xc)' \ Float64.(dat.Yc)'
    jldsave(joinpath(out_dir, "pred.jld2"); cols, pred1, pred3, predls = rawq(Cls10' * Xr64), seeds = CSEEDS,
            wd = only(CWDS), n_channels = only(CHS), loss = CLOSS, head = CHEAD, h = cfg.h, K, D, feat = FEAT,
            input = CINPUT, seq = CSEQ, act = CACT, scaling,
            # INNOISE: the Σ each seed's network deploys (standardised target units; `run_sigma` and the export
            # read it) and the Σ it was trained with; INNOISE = head: both the refitted head's, and S_noisy the residual
            # under it
            innoise = CINN, innoise_sigma = CINN_SRC, (CINN === :none ? (;) : (; S_deploy = Sdep, Sigma_inj = Sinjd))...,
            (CINN === :head ? (; S_noisy = Snoisy) : (;))...)
end
@info "written" out_dir
