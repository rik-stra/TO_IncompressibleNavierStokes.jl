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
# STEP 5 (2026-11-06): RIKFLOW_CNN_LOSS = crps_gen, the stochastic skip network (paper Eq. stochastic net, card
# analysis/cards/step5.md); see the block "STEP 5" below `refit_cnn`.
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
# RIKFLOW_CNN_REFIT_UPD = best (default, the protocol) | last: refit for as many updates as stage 1 ran in all,
# its whole schedule replayed (with NOSTOP: the network trained to MAXUPD without early stopping, on 1-10 TU;
# Rik, 2026-11-02, the wd-0 row of Table tab: net-loss).
const CREFIT_UPD = Symbol(envc("REFIT_UPD", "best"))
CREFIT_UPD in (:best, :last) || error("RIKFLOW_CNN_REFIT_UPD must be best or last; got $CREFIT_UPD")
const CDEV = Symbol(envc("DEVICE", "cpu"))
CDEV in (:cpu, :gpu) || error("RIKFLOW_CNN_DEVICE must be cpu or gpu; got $CDEV")
CDEV === :gpu && @eval using cuDNN                          # NNlib's cuDNN extension: Lux's Conv on CuArrays
"To the device: `CuArray` on the GPU (errors if none is functional, never a silent CPU fallback), else identity."
const DEVF = CDEV === :gpu ? RF.m4_device("cuda") : identity
todev(x) = Lux.fmap(DEVF, x)
const CLOSS = Symbol(envc("LOSS", "nll"))
CLOSS in (:mse, :rmse, :nll, :crps, :crps_gen) ||
    error("RIKFLOW_CNN_LOSS must be mse, rmse, nll, crps or crps_gen; got $CLOSS")
CLOSS === :crps && CDEV === :gpu && error("LOSS = crps is written for the CPU (norm_cdf is a scalar series)")
CLOSS === :crps_gen && CDEV === :gpu && error("LOSS = crps_gen (step 5) is written for the CPU")
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
# SKIP (step3_data.jl): the output layer's WEIGHTS start at zero (OUTINIT) and, after `Lux.setup`, its BIAS too
# (`skip_start!`), so a fit starts exactly at the LinReg. 🔴 Lux 1.x's `Dense` bias default is PyTorch's uniform
# ±1/sqrt(fan_in), not zero: every skip fit before 2026-11-06 started at the LinReg plus a random constant of up to
# 0.125 target sd per QoI (found by the step-5 session; seed 1: up to 0.110). The bias is zeroed AFTER setup, not by
# `init_bias = zeros32`, so that setup draws the same random numbers and the batch shuffle that follows is the old
# one: a rerun differs from the old fit by the start bias alone. RIKFLOW_CNN_SKIPBIAS = zero (default) | lux (the
# pre-2026-11-06 start, to reproduce those fits).
const OUTINIT = isempty(SKIP) ? glorot_uniform : zeros32
const CSKIPB = Symbol(envc("SKIPBIAS", "zero"))
CSKIPB in (:zero, :lux) || error("RIKFLOW_CNN_SKIPBIAS must be zero or lux; got $CSKIPB")
"With SKIP (and SKIPBIAS = zero), zero the output layer's bias (`layer_4` of `mkmodel`'s chain): the fit starts at the LinReg."
function skip_start!(net)
    (isempty(SKIP) || CSKIPB === :lux) && return net
    all(iszero, net.layer_4.weight) || error("skip_start!: layer_4 is not the zero-initialised output layer")
    fill!(net.layer_4.bias, 0)
    return net
end
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
                     Dense(D => nq; init_weight = OUTINIT))
    return Chain(Parallel(vcat, conv, NoOpLayer()), Dense(nflat + nq => D, ACTF; init_weight = WINIT),
                 Dense(D => D, ACTF; init_weight = WINIT), Dense(D => nq; init_weight = OUTINIT))
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
@info "CNN probe" SKIP CHS LRS CWDS CSEEDS K D CLOSS CHEAD CINPUT CSEQ CACT CINIT NSTEP NCH CST.max_updates fit_rows = ntr val_rows = ncol - ntr

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
# The LS floor's rms on its own fit rows (1-8.2 TU), per QoI: below it, the network fits its training data
# better than any linear mean model of the regressor (Rik, 2026-11-02: "Without weight decay do the CNNs fit
# better on 1-10 TU?"; run with WDS = 0 and NOSTOP = 1).
const ls_fit = rmsq(Ytr .- Cls' * Rtr_)
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
    # `lrs:<name>`: a deployed LinReg's own Σ (stoch_distr, its scaled level units) in this run's standardised target
    # (SKIP = LinReg1_nt with INNOISE_SIGMA = lrs:LinReg1_nt starts at that closure's own fixed point)
    if startswith(run, "lrs:")
        m = load(joinpath(TO_folder, "..", "TO_LRS", run[5:end], "LinReg.jld2"), "scaling", "stoch_distr")
        sgl = vec(m[1].in_scaling.sigma)
        Sl = Matrix{Float64}(hasproperty(m[2].Σ, :mat) ? m[2].Σ.mat : m[2].Σ)
        return Diagonal(sgl ./ so) * Sl * Diagonal(sgl ./ so), "the deployed covariance of $(run[5:end])"
    end
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
    raw(y) = y .* so .+ mo .+ skipq(cols)                   # standardised -> raw dQ (with SKIP: + the skip)
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
            # SKIP: minus the skip's response to the same window noise (skip_dy); pred3n below stays the network's alone
            rva_n = Yva .- prd(Float64.(Xc[:, (ntr + 1):end]) .+ P * window_noise(eta_from(L1, Eva)), r.ps, r.st) .-
                    skip_dy(window_noise(eta_from(L1, Eva)))
            rtr_n = Ytr .- prd(Float64.(Xc[:, 1:ntr]) .+ P * window_noise(eta_from(L1, Et)), r.ps, r.st) .-
                    skip_dy(window_noise(eta_from(L1, Et)))
            hs(x) = 0.5 * sum(abs2, x) / size(x, 2)
            # and under one COMMON Σ for every run (RIKFLOW_CNN_SCORE_COMMON = <run>, its seed-1 deployed Σ; the same
            # draws), so that networks trained under different noise compare under the same noise
            Sc = isempty(CSCORE_COMMON) ? S1 : first(run_sigma(CSCORE_COMMON, 1))
            Lc = cov_factor(Sc)
            rva_c = Yva .- prd(Float64.(Xc[:, (ntr + 1):end]) .+ P * window_noise(eta_from(Lc, Eva)), r.ps, r.st) .-
                    skip_dy(window_noise(eta_from(Lc, Eva)))
            rtr_c = Ytr .- prd(Float64.(Xc[:, 1:ntr]) .+ P * window_noise(eta_from(Lc, Et)), r.ps, r.st) .-
                    skip_dy(window_noise(eta_from(Lc, Et)))
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
            y3n = prd(Xf .+ P * window_noise(eta), r.refit.ps, r.refit.st)   # SKIP: the skip's response NOT included
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
    skip_start!(net)
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
    val_eta(p) = eta_from(inj_factor(p, Linj), Eva)
    val_in(p) = CINN === :none ? (Ava_d, Qva_d) : DEVF.(noisy_tensor(Xc[:, (ntr + 1):end], val_eta(p)))
    # SKIP under noise: the target minus the skip's response to the same window noise
    val_y(p) = CINN === :none || SKIPK === nothing ? Yva_d : DEVF(T.(Yva .- skip_dy(window_noise(val_eta(p)))))
    vloss(p) = Float64(loss(p, val_in(p)..., val_y(p)))
    best = (; val = vloss(ps), ps = deepcopy(ps), update = 0)
    # fit: 0.5 SSE on the clean fit rows at each validation, the training curve next to the validation one
    hist = (; update = Int[], val = Float64[], lr = Float64[], fit = Float64[])
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
                eb = draw_window_eta(nrng, inj_factor(ps, Linj), length(c))
                Ab, Qb = DEVF.(noisy_tensor(Xc[:, c], eb))
                Yb = SKIPK === nothing ? Ytr_d[:, c] : DEVF(T.(Ytr[:, c] .- skip_dy(window_noise(eb))))
                Zygote.gradient(p -> loss(p, Ab, Qb, Yb), ps)[1]
            end
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
            upd % CST.val_every == 0 || continue
            v = vloss(ps)
            push!(hist.update, upd); push!(hist.val, v); push!(hist.lr, cur)
            push!(hist.fit, 0.5 * sum(abs2, Ytr .- Float64.(Array(out(ps, Atr_d, Qtr_d)))) / ntr)
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
            ho_ratio = rmsq(Yf[:, HO] .- pred(Aho_d, Qho_d; p = ps)) ./ ls_ho,
            fit_ratio = rmsq(Ytr .- pred(Atr_d, Qtr_d; p = ps)) ./ ls_fit)
    fit_ratio = rmsq(rtr) ./ ls_fit                                   # the best iterate on its fit rows
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
        # SKIP: every noisy prediction plus the skip's response to its window noise (least squares' too: the full
        # least-squares model is the skip plus Cls)
        yh = pred(DEVF.(noisy_tensor(Xf[:, HO], eta_from(Lb, E)))...) .+ skip_dy(window_noise(eta_from(Lb, E)))
        lh = Cls' * (Xr64[:, HO] .+ window_noise(eta_from(LLS_INJ, E))) .+ skip_dy(window_noise(eta_from(LLS_INJ, E)))
        lv = Cls' * (Rva_ .+ window_noise(eta_from(LLS_INJ, Eva))) .+ skip_dy(window_noise(eta_from(LLS_INJ, Eva)))
        Et = randn(Xoshiro(seed + 61), nq, cfg.h * ntr)
        rtr_n = Ytr .- pred(DEVF.(noisy_tensor(Xc[:, 1:ntr], eta_from(Lb, Et)))...) .- skip_dy(window_noise(eta_from(Lb, Et)))
        rva_n = (SKIPK === nothing ? Yva : Yva .- skip_dy(window_noise(val_eta(pb)))) .- pred(val_in(pb)...)
        (; src, Sinj = Sb, val_half_sse = 0.5 * sum(abs2, rva_n) / size(Yva, 2),
         ls_val_half_sse = 0.5 * sum(abs2, Yva .- lv) / size(Yva, 2),
         nll_refit = gnll(rva_n, cov(permutedims(rtr_n))), crps_refit = crps_eval(rva_n, vec(std(rtr_n; dims = 2))),
         ho_ratio = rmsq(Yf[:, HO] .- yh) ./ rmsq(Yf[:, HO] .- lh), sd_fit_over_inj = sqrt.(diag(mom0(rtr_n)) ./ diag(Sb)))
    end
    return (; ps = Lux.fmap(Array, pb), st = Lux.fmap(Array, st), device = CDEV, ho_ratio1, h = cfg.h, val_loss = best.val, val_half_sse = 0.5 * sum(abs2, rva) / size(rva, 2),
            nll_refit = gnll(rva, cov(permutedims(rtr))), crps_refit, head_ratio, ho_ratio, best_update = best.update, updates = upd,
            stop, nparams = Lux.parameterlength(model), loss = CLOSS, head = CHEAD, input = CINPUT, seq = CSEQ, act = CACT,
            init = CINIT, innoise = CINN, noisy, fit_ratio,
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
    skip_start!(net)
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
                eb = draw_window_eta(nrng, inj_factor(ps, Linj), length(c))
                Ab, Qb = DEVF.(noisy_tensor(Xc[:, c], eb))
                Yb = SKIPK === nothing ? Yall_d[:, c] : DEVF(T.(Float64.(dat.Yc[:, c]) .- skip_dy(window_noise(eb))))
                Zygote.gradient(p -> loss(p, Ab, Qb, Yb), ps)[1]
            end
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
        end
    end
    return Lux.fmap(Array, ps), Lux.fmap(Array, st), model
end

# ==================================================================================================
# 🔑 STEP 5 (2026-11-06; paper Eq. stochastic net, closures.tex sec:lever-stoch; card analysis/cards/step5.md):
# RIKFLOW_CNN_LOSS = crps_gen, the STOCHASTIC SKIP NETWORK
#
#     y = g_θ(z, q*, ε) + L ε,   ε ~ N(0, I_6) fresh per sample,   in the target of SKIP (what the LinReg misses)
#
# g_θ is step 3a's CNN with ε joining q^{n*} in the first dense layer (Parallel(vcat, conv, NoOp, NoOp), 6 x D
# more weights; the deployed `cnn_window!` with emission = :generator reads the same order), its output layer at
# zero (OUTINIT with SKIP); L is lower triangular, starts at the Cholesky factor of the skip LinReg's Σ in the
# target units (diag(σ_out / so) chol Σ), FULLY learnable, no weight decay (Rik: "keep it and make L fully
# learnable"). With RIKFLOW_MLP_SKIP_CENTER = mvg (required) the fit starts as that LinReg + MVG exactly.
# Loss (RIKFLOW_CNN_GEN_SCORE): energy (default; Rik at GATE 1, 2026-11-06: "Energy score, rerun") = the fair
# energy score of RIKFLOW_CNN_GEN_M (default 4) samples of the six-QoI vector per window (`RF.energy_fair_samples`),
# which sees the dependence between the QoIs | crps = the fair CRPS per QoI summed over the QoIs
# (`RF.crps_fair_samples`; card (b) as first decided: its grid, 2026-11-06, coupled the bands); fresh draws per update from the noise's own stream (the shuffle stays the
# clean run's); the validation loss uses ONE fixed set of RIKFLOW_CNN_GEN_MVAL (default 16) draws per validation
# row. Weight decay on the network (the noise weights included), not on L. The schedule, early stopping,
# weight-decay grid and the refit (REFIT = 1, one setting per run) are the protocol's (fit_cnn / refit_cnn,
# written out again here so that the other modes stay untouched). Nothing is fitted after training.
#
# Printed per fit (stage 1, 8.2-10 TU, its best iterate) -- the card's offline check and what is reported beside
# it: the CRPS of RIKFLOW_CNN_GEN_MCHK (default 256) fixed draws per row against LinReg1 + MVG on the same rows (the
# PASS reference, Rik at GATE 0), estimated with the SAME draws through the start model (g = 0, L = L0, which IS
# LinReg1 + MVG): a paired comparison free of Monte Carlo noise; its closed form beside it, which differs from any
# sample estimate by that noise (0.5 % at M = 64 on the 720 rows in the smoke), and
# against least squares refitted on 1-8.2 TU with its sds (`LS_CRPS`); per QoI the samples' sd over the error of
# their mean, the error of their mean over LinReg1's, the share of the sample variance from L ε and from the
# network's response to ε, and the samples' correlation against Σ's (and L L''s). With REFIT, `pred.jld2` holds per
# seed the stage-1 and refitted networks' sample mean (raw dQ, skip included), sample sd and the sd of the
# network's part alone (target units) on every record step from RIKFLOW_CNN_GEN_MREC (default 32) fixed draws, the
# pooled correlations, and a translation check set for the export (300 steps of 10-100 TU, one fixed ε each).
# ==================================================================================================
const GEN = CLOSS === :crps_gen
if GEN
    isempty(SKIP) && error("LOSS = crps_gen needs RIKFLOW_MLP_SKIP (the frozen LinReg; card (a): LinReg1)")
    # RIKFLOW_CNN_GEN_NETSKIP = <a step-4 run dir under TO_LSTM, REFIT = 1> (B; card step5_skips.md, Rik 2026-11-11:
    # "A2, both skips; and B"): the skip is that run's FROZEN CNN mean on top of its LinReg (RIKFLOW_MLP_SKIP = the
    # same LinReg), in its target frame (SKIP_CENTER = data, as step 4 trained); stage 1 on its stage-1 network,
    # the refit on its refitted one; L starts at the Cholesky factor of its deployed headcorrnt Σ (the refitted
    # head's sds with the correlation of S_noisy), so the fit starts as that step-4 closure (k1a) exactly.
    const GEN_NETSKIP = envc("GEN_NETSKIP", "")
    const GEN_NET = !isempty(GEN_NETSKIP)
    strip(get(ENV, "RIKFLOW_MLP_SKIP_CENTER", "data")) == (GEN_NET ? "data" : "mvg") ||
        error(GEN_NET ? "LOSS = crps_gen with GEN_NETSKIP: set RIKFLOW_MLP_SKIP_CENTER = data (the step-4 run's frame)" :
              "LOSS = crps_gen: set RIKFLOW_MLP_SKIP_CENTER = mvg, so that the fit starts as the LinReg + MVG")
    CINN === :none || error("LOSS = crps_gen sets its history noise with RIKFLOW_CNN_GEN_HIST, not INNOISE")
    CSEQ === :lags || error("LOSS = crps_gen is defined for SEQ = lags")
    # RIKFLOW_CNN_GEN_HIST = recorded (default: the recorded history, step 5) | own (A2; card step5_skips.md, Rik
    # 2026-11-11: "A2, both skips; and B"): every training window carries the closure's own noise, as step 4's
    # INNOISE = head -- η = the CURRENT noise skip L times standard normals, read before each gradient (a stop-gradient),
    # one η per correction of the window, the levels the running sum anchored at the prediction step (`window_noise`);
    # the frozen skip reads the same noisy history (`skip_dy`; stage 1 the refitted skip's own response), so the target
    # is what the noisy skip misses. Only the CONSTANT part of the closure's noise (L ε) is injected, not the network's
    # state-dependent part. The validation history is one fixed set of standard normals times the current L; the
    # check is reported on the recorded history and under each model's own noise (the A2 PASS rule: the latter).
    const GEN_HIST = Symbol(envc("GEN_HIST", "recorded"))
    # | owntot (A2c; Rik 2026-11-11: "A2c, check B too"): as own, but the injected factor is that of the generator's
    # AVERAGE TOTAL sample covariance -- the pooled covariance of its samples about each window's mean, the network's
    # response to ε included -- on 512 fixed training rows with 8 fixed draws, refreshed at every validation (stage 1)
    # or every val_every updates (the refit), outside the gradient. Under own the optimiser can move the noise from L
    # into the network and so escape the injection (s5r: L collapsed to 1.4 % of L0); here it cannot. At the start
    # (g = 0) the two coincide.
    GEN_HIST in (:recorded, :own, :owntot) || error("RIKFLOW_CNN_GEN_HIST must be recorded, own or owntot; got $GEN_HIST")
    const GEN_OWN = GEN_HIST in (:own, :owntot)
    const GEN_TOT = GEN_HIST === :owntot
    # Card step5_reach_k1a.md (Rik 2026-11-13: "let the step 5 agent try to at least reach the performance of step 4"):
    #   RIKFLOW_CNN_GEN_LFIX = 1: L is frozen at L0 (its gradient dropped; with GEN_NETSKIP: k1a's Σ);
    #   RIKFLOW_CNN_GEN_CENTRE = 1: the network's output is CENTRED on ε, g(z, q*, ε) - g(z, q*, 0), so at ε = 0 the closure is
    #     its skip (it cannot move the mean there; deployed with `Cv.centred`, `RF.with_centred`);
    #   RIKFLOW_CNN_GEN_SDCAP = c (> 0): a penalty GEN_SDCAP_W (default 100) x Σ_i max(sd_i / sd0_i - c, 0)^2 on the drawn sd per
    #     QoI (the samples' pooled within-window sd over the batch) against the skip's √diag Σ (B: k1a's), in training and in
    #     the validation loss.
    const GEN_LFIX = envc("GEN_LFIX", "0") == "1"
    const GEN_CENTRE = envc("GEN_CENTRE", "0") == "1"
    const GEN_SDCAP = parse(Float64, envc("GEN_SDCAP", "0"))
    const GEN_SDCAP_W = parse(Float64, envc("GEN_SDCAP_W", "100"))
    const GEN_M = parse(Int, envc("GEN_M", "4"))
    const GEN_MVAL = parse(Int, envc("GEN_MVAL", "16"))
    const GEN_MCHK = parse(Int, envc("GEN_MCHK", "256"))
    const GEN_MREC = parse(Int, envc("GEN_MREC", "32"))
    const GEN_SCORE = Symbol(envc("GEN_SCORE", "energy"))
    GEN_SCORE in (:energy, :crps) || error("RIKFLOW_CNN_GEN_SCORE must be energy or crps; got $GEN_SCORE")
    "The generator's training and validation score of samples `X` (N_Q x B x M) against `Y`."
    gen_score(X, Y) = GEN_SCORE === :energy ? RF.energy_fair_samples(X, Y) : RF.crps_fair_samples(X, Y)
    GEN_M >= 2 && GEN_MVAL >= 2 && GEN_MCHK >= 2 && GEN_MREC >= 2 || error("GEN_M, GEN_MVAL, GEN_MCHK, GEN_MREC must be >= 2")
    # Σ of the skip LinReg in the network's standardised target (its draw is σ_out ⊙ (μ + chol(Σ) ε) raw, and
    # the target is (raw - mo) / so with mo = σ_out ⊙ μ)
    # B (GEN_NETSKIP): the step-4 run, its networks (Lux, frozen) and its predictions, checked against its pred.jld2
    const GEN_K = !GEN_NET ? nothing : let
        Fk = load(joinpath(TO_folder, GEN_NETSKIP, "fits.jld2"))
        Pk = load(joinpath(TO_folder, GEN_NETSKIP, "pred.jld2"))
        (Fk["K"], Fk["D"], Fk["input"], Fk["seq"], Fk["act"]) == (K, D, CINPUT, CSEQ, CACT) ||
            error("GEN_NETSKIP: $GEN_NETSKIP has K, D, input, seq, act = $((Fk["K"], Fk["D"], Fk["input"], Fk["seq"], Fk["act"]))")
        Pk["cols"] == cols || error("GEN_NETSKIP: $(GEN_NETSKIP)'s pred.jld2 is on other record columns")
        get(Pk, "innoise", :none) === :head && Pk["loss"] === :crps ||
            error("GEN_NETSKIP: a step-4 CRPS run in head mode (k1 / k2) is expected")
        sck = Pk["scaling"]
        maximum(abs, Float64.(sck.input_map) .- Float64.(scaling.input_map)) < 1e-12 &&
            maximum(abs, vec(sck.out_scaling.mu) .- mo) < 1e-12 * maximum(so) &&
            maximum(abs, vec(sck.out_scaling.sigma) .- so) < 1e-12 * maximum(so) ||
            error("GEN_NETSKIP: $(GEN_NETSKIP)'s scaling is not this run's (SKIP = $SKIP, SKIP_CENTER = data?)")
        key, rk = only(Fk["res"])
        haskey(rk, :refit) || error("GEN_NETSKIP: $GEN_NETSKIP has no refit")
        (; model = mkmodel(key[1]), ps1 = rk.ps.net, st1 = rk.st, ps3 = rk.refit.ps.net, st3 = rk.refit.st,
         Pk, seed = key[4], bd = Float64.(vec(rk.refit.ps.head.bd)), key)
    end
    "B: the frozen step-4 network (stage 1 or 3 = the refit) on windows (A, Q), standardised target, Float64."
    knet(A, Q, stage) = Float64.(first(GEN_K.model(modelin(A, Q), stage == 1 ? GEN_K.ps1 : GEN_K.ps3,
                                                   stage == 1 ? GEN_K.st1 : GEN_K.st3)))
    all(diff(cols) .== 1) || error("the record columns are not contiguous")
    "Index of record columns `c` in `cols` (the whole-record arrays)."
    atc(c) = c .- cols[1] .+ 1
    # B: its stage-1 and refitted networks on the recorded history, every step of `cols`; checked against pred.jld2
    const GEN_YK1, GEN_YK3 = !GEN_NET ? (nothing, nothing) : let AQ = lagtensor(Xf)
        y1, y3 = knet(AQ[1], AQ[2], 1), knet(AQ[1], AQ[2], 3)
        sk = GEN_K.seed
        for (y, p, lab) in ((y1, GEN_K.Pk["pred1"][sk], "stage 1"), (y3, GEN_K.Pk["pred3"][sk], "refit"))
            e = maximum(abs, (y .* so .+ mo .+ skipq(cols) .- p) ./ so)
            e < 1e-4 || error("GEN_NETSKIP: the $lab network differs from $(GEN_NETSKIP)'s pred.jld2 by $e target sd")
            @info "B: the frozen step-4 network reproduces its pred.jld2" GEN_NETSKIP lab max_diff_target_sd = e
        end
        y1, y3
    end
    "B: the frozen network's response to noisy windows (A, Q) at record columns `c` (stage 1 or 3); 0 otherwise."
    knet_dy(A, Q, c, stage) = !GEN_NET ? 0.0 : knet(A, Q, stage) .- (stage == 1 ? GEN_YK1 : GEN_YK3)[:, atc(c)]
    "B: the frozen network's clean prediction at record columns `c` (stage 1 or 3), standardised; 0 otherwise."
    knet_off(c, stage) = !GEN_NET ? zeros(nq, length(c)) : (stage == 1 ? GEN_YK1 : GEN_YK3)[:, atc(c)]
    const GEN_SIGMA = GEN_NET ? let Sd = GEN_K.Pk["S_deploy"][GEN_K.seed], Sn = GEN_K.Pk["S_noisy"][GEN_K.seed]
        # k1a's deployed Σ (cnn_closure_export.jl headcorrnt): the refitted head's sds, the correlation of S_noisy
        sd = exp.(GEN_K.bd)
        maximum(abs, sd ./ sqrt.(diag(Sd)) .- 1) < 1e-5 || error("GEN_NETSKIP: the refitted head's sds are not S_deploy's")
        Diagonal(sd) * (Sn ./ sqrt.(diag(Sn) * diag(Sn)')) * Diagonal(sd)
    end : let ds = load(joinpath(TO_folder, "..", "TO_LRS", SKIP, "LinReg.jld2"), "stoch_distr", "scaling")
        d, sc_ = ds
        sgo = vec(sc_.out_scaling.sigma)
        S = Matrix{Float64}(hasproperty(d.Σ, :mat) ? d.Σ.mat : d.Σ)
        Diagonal(sgo ./ so) * S * Diagonal(sgo ./ so)
    end
    const GEN_L0 = T.(Matrix(cholesky(Symmetric(GEN_SIGMA)).L))
    const LTRI = T[i >= j for i in 1:nq, j in 1:nq]
    # RIKFLOW_CNN_GEN_STAGE1 = skip (default: stage 1 on the frozen skip, as above) | heldout (Rik at GATE 1,
    # 2026-11-06: "please try this: Held-out early stop"): the skip LinReg is fitted on all of 1-10 TU, so the
    # early-stopping rows (8.2-10 TU) lie in its training window and any mean the network learns on 1-8.2 TU scores
    # worse there (least squares on the fit rows: CRPS +2.6 % on 8.2-10 TU). With heldout, STAGE 1 (settings, early
    # stop, the offline check) uses as its skip the skip LinReg PLUS least squares of its residual on the fit rows
    # (`Cls`: together, least squares on 1-8.2 TU) -- its target is that model's residual, and L starts at the Cholesky
    # factor of its residual's covariance on the fit rows (about zero) -- so stage 1 starts as least squares on 1-8.2
    # TU + its MVG and the early stop is held out. The REFIT (stage 2) is unchanged: the skip LinReg, its target, L0.
    # | heldout_skip (card step5_skips.md, 2026-11-11: step 5 on LinReg1_nt and LinReg7): for any linear skip on this
    # regressor, heldout's "skip + least squares of its residual" is least squares on 1-8.2 TU, so stage 1 would not
    # depend on the skip. Here stage 1's skip is the skip's own RECIPE refitted on the fit rows (ridge with its λ; the
    # noise-trained closed form with its self-consistent Σ), `TO_LRS/<SKIP>_s1` from `tools/skip_heldout.jl`: its
    # target is that model's residual about its MVG mean, L starts at the Cholesky factor of its Σ, so stage 1 starts
    # as the refitted skip + its MVG. For LinReg1 it is heldout to round-off (asserted). The refit is unchanged.
    const GEN_S1MODE = envc("GEN_STAGE1", "skip")
    GEN_S1MODE in ("skip", "heldout", "heldout_skip") || error("RIKFLOW_CNN_GEN_STAGE1 must be skip, heldout or heldout_skip")
    const GEN_HELD = GEN_S1MODE != "skip"                             # stage 1 held out (either construction)
    const GEN_HSKIP = GEN_S1MODE == "heldout_skip"
    GEN_NET && GEN_S1MODE != "skip" && error("GEN_NETSKIP: stage 1 is the step-4 run's own stage-1 network (GEN_STAGE1 = skip)")
    """
    A white LinReg's (`TO_LRS/<name>/LinReg.jld2`) predicted correction (raw) on the record columns of the skip, and
    its MVG mean (raw) and Σ (its scaled units), as step3_data.jl's SKIPQ computes the skip's.
    """
    function gen_linreg_pred(name)
        d1 = load(joinpath(TO_folder, "..", "TO_LRS", name, "LinReg.jld2"))
        s1, hl1 = d1["scaling"].in_scaling, d1["hist_len"]
        hl1 == SKIPQ.hl || error("$name: history $hl1, the skip's $(SKIPQ.hl)")
        hs = RF.HistorySpec(; h = hl1, n_qoi = nq, hist_var = :q_star_q, include_predictor = true)
        a0, b0 = cfg.train_range[1], size(rec.q_star, 2)
        X, _, st = RF.build_history(hs, RF.scale_input(rec.q_star[:, a0:b0], s1), RF.scale_input(rec.q[:, a0:(b0 + 1)], s1))
        X = Float64.(X)
        cs = (a0 - 1) .+ st
        cs[1] == SKIPQ.first && cs[end] == SKIPQ.last || error("$name: steps $(cs[1])-$(cs[end]), the skip's $(SKIPQ.first)-$(SKIPQ.last)")
        sg1 = vec(s1.sigma)
        pred = permutedims((X * permutedims(Matrix{Float64}(d1["c"])) .- X[:, 1:nq]) .* sg1')
        sgo1 = vec(d1["scaling"].out_scaling.sigma)
        # its linear response to a regressor perturbation dx in `dat`'s scaling (raw), as step3_data.jl's SKIPK
        fsd = vcat(repeat(vec(dat.scaling.in_scaling.sigma), 2cfg.h + 1), 1.0)
        K1 = Diagonal(sg1) * Matrix{Float64}(d1["c"]) * Diagonal(fsd ./ vcat(repeat(sg1, 2cfg.h + 1), 1.0))
        return (; d = d1, pred, mu_raw = sgo1 .* Vector{Float64}(mean(d1["stoch_distr"])),
                Sigma = Matrix{Float64}(cov(d1["stoch_distr"])), sgo = sgo1, K = K1)
    end
    const GEN_S1 = !GEN_HSKIP ? nothing : let
        own = gen_linreg_pred(SKIP)                                   # the function, checked on the skip itself
        maximum(abs, own.pred .- SKIPQ.pred) <= 1e-12 * maximum(abs, SKIPQ.pred) ||
            error("gen_linreg_pred does not reproduce the skip's own predictions")
        maximum(abs, own.K .- SKIPK) <= 1e-12 * maximum(abs, SKIPK) || error("gen_linreg_pred's response is not SKIPK")
        s = gen_linreg_pred(SKIP * "_s1")
        s.d["s1_fit_rows"] == (cols_tr[1], cols_tr[ntr]) ||
            error("$(SKIP)_s1 was fitted on rows $(s.d["s1_fit_rows"]), stage 1 fits $((cols_tr[1], cols_tr[ntr]))")
        # in the network's standardised target, at record column n: (stage-1 skip + its MVG mean - skip - mo) / so
        off = (s.pred .+ s.mu_raw .- SKIPQ.pred .- mo) ./ so
        # its response checked against its own predictions on pairs of record steps (as step3_data.jl's SKIPK)
        let ra = 1:200, rb = 1001:1200, ca = cols_tr[ra], cb = cols_tr[rb], at(c) = c .- SKIPQ.first .+ 1
            lhs = s.pred[:, at(ca)] .- s.pred[:, at(cb)]
            rhs = s.K * (Float64.(dat.Xc[:, ra]) .- Float64.(dat.Xc[:, rb])) .- (rec.q_star[:, ca] .- rec.q_star[:, cb])
            (e = maximum(abs, lhs .- rhs) / maximum(abs, lhs)) < 1e-8 || error("$(SKIP)_s1: linear response off by $e")
        end
        (; off, Sigma = Diagonal(s.sgo ./ so) * s.Sigma * Diagonal(s.sgo ./ so), prov = s.d["s1_provenance"], K = s.K)
    end
    "Stage 1's skip beyond the skip LinReg (standardised target) at record columns `c` (heldout_skip)."
    gen_s1off(c) = GEN_S1.off[:, c .- SKIPQ.first .+ 1]
    const GEN_SIGMA_S = GEN_HSKIP ? GEN_S1.Sigma : GEN_HELD ? mom0(Rls) : GEN_SIGMA   # stage 1's start Σ (target units)
    const GEN_L0S = T.(Matrix(cholesky(Symmetric(GEN_SIGMA_S)).L))
    const Ytr_g = GEN_HSKIP ? Ytr .- gen_s1off(cols_tr[1:ntr]) :       # stage 1's target (B: minus its stage-1 network)
                  GEN_HELD ? Ytr .- Cls' * Rtr_ : Ytr .- knet_off(cols_tr[1:ntr], 1)
    const Ytr_g32 = T.(Ytr_g)
    """
    Stage 1's skip's response to a regressor perturbation `dx` (`dat`'s scaling), standardised target: the refitted
    skip's (heldout_skip), least squares' on the fit rows (heldout: the skip's + Cls'), else the skip's (`skip_dy`).
    """
    s1_dy(dx) = GEN_HSKIP ? (GEN_S1.K * dx) ./ so : GEN_HELD ? skip_dy(dx) .+ Cls' * dx : skip_dy(dx)
    const Yva_g = GEN_HSKIP ? Yva .- gen_s1off(cols_tr[(ntr + 1):end]) : GEN_HELD ? Yva .- Cls' * Rva_ :
                  Yva .- knet_off(cols_tr[(ntr + 1):end], 1)
    "The deployed skip's residual on the validation rows (B: minus the refitted step-4 network), standardised."
    const Yva_3 = Yva .- knet_off(cols_tr[(ntr + 1):end], 3)
    "The refit's target on all 1-10 TU rows (B: minus the refitted step-4 network), standardised."
    const Yall_g = Float64.(dat.Yc) .- knet_off(cols_tr, 3)
    "The deployed skip's residual on every step of `cols` (B: minus the refitted step-4 network), standardised."
    const Yf_g = Yf .- knet_off(cols, 3)
    const Yva_g32 = T.(Yva_g)
    "Stage 1's skip beyond the skip LinReg, on the whole record (standardised target): Cls' x, the refitted skip's, or 0."
    gen_s1_offset() = GEN_HSKIP ? gen_s1off(cols) : GEN_HELD ? Cls' * Xr64 : knet_off(cols, 1)
    "The refit's skip beyond the skip LinReg on the whole record: B's refitted step-4 network, or 0."
    gen_s3_offset() = knet_off(cols, 3)
    GEN_S1MODE == "heldout" && @info "stage 1 held out: skip + least squares on the fit rows" ls_sd_over_skip_sd = sqrt.(diag(GEN_SIGMA_S) ./ diag(GEN_SIGMA))
    if GEN_HSKIP
        # against heldout's construction (least squares on the fit rows): the same for LinReg1, the refitted
        # skip's deficit to least squares otherwise (rms over the rows, standardised target)
        dtr = rmsq((Ytr .- Cls' * Rtr_) .- Float64.(Ytr_g32))
        dva = rmsq((Yva .- Cls' * Rva_) .- Yva_g)
        @info "stage 1 held out: the skip's recipe refitted on the fit rows" SKIP recipe = GEN_S1.prov.recipe rows = GEN_S1.prov.rows sd_over_skip_sd = sqrt.(diag(GEN_SIGMA_S) ./ diag(GEN_SIGMA)) vs_ls_fit_rows = dtr vs_ls_val_rows = dva clean_rms_val_over_ls = rmsq(Yva_g) ./ rmsq(Yva .- Cls' * Rva_)
        SKIP == "LinReg1" && (maximum(dva) < 1e-6 && maximum(abs, GEN_SIGMA_S .- mom0(Rls)) < 1e-8 ||
                              error("heldout_skip on LinReg1 differs from heldout (LS on the fit rows): $(maximum(dva))"))
    end
    # RIKFLOW_CNN_GEN_OUTINIT = zero (default: the output layer's weights and bias start at zero, so the fit starts at
    # the LinReg + MVG) | random (Rik, 2026-11-06: "If that does not work; try using random intialization instead of
    # 0"): Glorot-uniform output weights (the bias stays zero), so the network starts away from the LinReg and the
    # gradient reaches its hidden layers from the first update; the start check is then skipped
    const GEN_OUTRAND = envc("GEN_OUTINIT", "zero") == "random"
    envc("GEN_OUTINIT", "zero") in ("zero", "random") || error("RIKFLOW_CNN_GEN_OUTINIT must be zero or random")
    @info "step 5 generator" SKIP GEN_NETSKIP GEN_HIST GEN_LFIX GEN_CENTRE GEN_SDCAP GEN_M GEN_MVAL GEN_MCHK GEN_MREC sd_L0 = sqrt.(diag(GEN_SIGMA)) sd_val_target = vec(std(Yva; dims = 2))
end

function mkmodel_gen(C)
    conv = Chain(Conv((K,), NCH => C, ACTF; init_weight = WINIT), Conv((K,), C => C, ACTF; init_weight = WINIT),
                 FlattenLayer())
    nflat = (NSTEP - 2(K - 1)) * C
    # 🔴 the output BIAS starts at zero too: Lux's Dense default (`init_bias = nothing`) is PyTorch's uniform
    # ±1/sqrt(fan_in), so OUTINIT alone left a constant offset of up to 0.125 target sd (smoke, 2026-11-06)
    return Chain(Parallel(vcat, conv, NoOpLayer(), NoOpLayer()), Dense(nflat + 2nq => D, ACTF; init_weight = WINIT),
                 Dense(D => D, ACTF; init_weight = WINIT),
                 Dense(D => nq; init_weight = GEN_OUTRAND ? glorot_uniform : OUTINIT, init_bias = zeros32))
end
"The noise skip L (lower triangular) of the parameters `p`."
gen_L(p) = p.L .* LTRI
"""
`M = size(E, 2) / B` samples for each of the `B` windows (A, Q): the network on [A; Q; ε] with column `b + (m - 1) B`
of `E` the deviates of sample `m` of window `b`, plus `L ε`; returns `N_Q x B x M`.
"""
function gen_samples(model, p, st, A, Q, E)
    B = size(Q, 2)
    M = size(E, 2) ÷ B
    M * B == size(E, 2) || error("gen_samples: E has $(size(E, 2)) columns for $B windows")
    G = first(model((repeat(A, 1, 1, M), repeat(Q, 1, M), E), p.net, st))
    if GEN_CENTRE                                                   # g(z, q*, ε) - g(z, q*, 0)
        G = G .- repeat(first(model((A, Q, zero(E[:, 1:B])), p.net, st)), 1, M)
    end
    Y = G .+ gen_L(p) * E
    return reshape(Y, nq, B, M)
end
"""
Sample statistics of a generator on windows (A, Q) from `M` draws of `rng`, chunked by draw: the sample mean and
sd and the sd of the network's part alone, without L ε (all `N_Q x N`, target units).
"""
function gen_stats(model, p, st, A, Q, M, rng)
    N = size(Q, 2)
    s1, s2, g1, g2 = zeros(nq, N), zeros(nq, N), zeros(nq, N), zeros(nq, N)
    for _ in 1:M
        E = randn(rng, T, nq, N)
        g = Float64.(first(model((A, Q, E), p.net, st)))
        GEN_CENTRE && (g = g .- Float64.(first(model((A, Q, zero(E)), p.net, st))))
        y = g .+ Float64.(gen_L(p) * E)
        s1 .+= y; s2 .+= y .^ 2; g1 .+= g; g2 .+= g .^ 2
    end
    mu = s1 ./ M
    sd = sqrt.(max.(s2 ./ M .- mu .^ 2, 0) .* (M / (M - 1)))
    sdg = sqrt.(max.(g2 ./ M .- (g1 ./ M) .^ 2, 0) .* (M / (M - 1)))
    return (; mu, sd, sdg)
end
"Pooled correlation of the samples about their window's mean, from `M` draws (N_Q x N_Q)."
function gen_corr(model, p, st, A, Q, M, rng)
    N = size(Q, 2)
    E = randn(rng, T, nq, N * M)
    X = Float64.(gen_samples(model, p, st, A, Q, E))
    R = X .- mean(X; dims = 3)
    Cm = reshape(R, nq, :) * reshape(R, nq, :)'
    return Cm ./ sqrt.(diag(Cm) * diag(Cm)')
end
corrof(S) = S ./ sqrt.(diag(S) * diag(S)')
"The sd-cap penalty (GEN_SDCAP > 0) on samples X (N_Q x B x M): the drawn sd per QoI over the skip's, above the cap."
function gen_cap(X)
    GEN_SDCAP > 0 || return zero(eltype(X))
    D = X .- mean(X; dims = 3)
    sd = sqrt.(vec(sum(abs2, D; dims = (2, 3))) ./ (size(X, 2) * (size(X, 3) - 1)))
    return T(GEN_SDCAP_W) * sum(abs2, max.(sd ./ T.(sqrt.(diag(GEN_SIGMA))) .- T(GEN_SDCAP), zero(T)))
end
"""
A2c: the lower factor of the generator's average conditional sample covariance on windows (A, Q) from `M` draws of a
FIXED stream `seed` (so that it changes with the parameters only): the samples' pooled covariance about each window's
mean, the network's response to ε included (host, outside any gradient).
"""
function gen_tot_factor(model, p, st, A, Q, M, seed)
    B = size(Q, 2)
    X = Float64.(gen_samples(model, p, st, A, Q, randn(Xoshiro(seed), T, nq, B * M)))
    R = reshape(X .- mean(X; dims = 3), nq, :)
    return Matrix(cholesky(Symmetric(R * R' ./ (B * (M - 1)))).L)
end

function fit_gen(C, lr, wd, seed)
    rng = Xoshiro(seed)
    model = mkmodel_gen(C)
    net, st = Lux.setup(rng, model)
    ps = (; net, L = copy(GEN_L0S))                                 # GEN_STAGE1 = heldout: least squares' Σ
    rule = wd > 0 ? Optimisers.AdamW(T(lr), (T(0.9), T(0.999)), T(wd); couple = true) : Optimisers.Adam(T(lr))
    opt = Optimisers.setup(rule, ps)
    wd > 0 && Optimisers.adjust!(opt.L; lambda = zero(T))          # no weight decay on L
    loss(p, A, Q, E, Y) = (X = gen_samples(model, p, st, A, Q, E); gen_score(X, Y) + gen_cap(X))
    nrng = Xoshiro(seed + 7919)                                     # the samples' own stream
    Eva = randn(Xoshiro(seed + 104729), T, nq, size(Yva32, 2) * GEN_MVAL)   # ONE fixed set of validation draws
    # GEN_HIST = own: the history noise's own stream, and ONE fixed set of validation history normals (times the
    # current L at every validation, as step 4's head mode)
    hrng = Xoshiro(seed + 15485863)
    Evh = randn(Xoshiro(seed + 104723), nq, cfg.h * size(Yva, 2))
    # the injected factor: own = the current L; owntot (A2c) = the generator's average total sample covariance on 512
    # fixed fit rows, refreshed at every validation
    injrows = GEN_TOT ? shuffle(Xoshiro(seed + 98), 1:ntr)[1:min(512, ntr)] : Int[]
    injf(p) = GEN_TOT ? gen_tot_factor(model, p, st, Atr[:, :, injrows], Qtr[:, injrows], 8, seed + 99) : Float64.(gen_L(p))
    Linj = Ref(GEN_OWN ? injf(ps) : zeros(nq, nq))
    "The validation rows' (A, Q, Y): the recorded history, or (own) with the window noise of the current L."
    function val_data(p)
        GEN_OWN || return (Ava, Qva, Yva_g32)
        eta = eta_from(Linj[], Evh)                                  # refreshed for p before each validation
        A, Q = noisy_tensor(Xc[:, (ntr + 1):end], eta)
        return (A, Q, T.(Yva_g .- s1_dy(window_noise(eta)) .- knet_dy(A, Q, cols_tr[(ntr + 1):end], 1)))
    end
    vloss(p) = (d = val_data(p); Float64(loss(p, d[1], d[2], Eva, d[3])))
    best = (; val = vloss(ps), ps = deepcopy(ps), update = 0)
    val0 = best.val
    # the start IS LinReg1 + MVG (heldout: least squares on 1-8.2 TU + its MVG): the network's output is zero, the
    # samples are L0 ε (checked on the validation rows)
    GEN_OUTRAND || let E0 = randn(Xoshiro(1), T, nq, size(Yva32, 2) * 2), X0 = gen_samples(model, ps, st, Ava, Qva, E0)
        e = maximum(abs, X0 .- reshape(GEN_L0S * E0, nq, size(Yva32, 2), :))
        e == 0 || error("the generator's start is not L0 ε: max |diff| $e")
    end
    hist = (; update = Int[], val = Float64[], lr = Float64[])
    cur, since_best, since_decay, upd, stop = T(lr), 0, 0, 0, :cap
    nb = fld(ntr, CST.batch)
    while upd < CST.max_updates
        idxs = shuffle(rng, 1:ntr)
        for k in 1:nb
            c = idxs[((k - 1) * CST.batch + 1):(k * CST.batch)]
            E = randn(nrng, T, nq, length(c) * GEN_M)
            if GEN_OWN                                              # η = the current L ε, outside the gradient
                GEN_TOT || (Linj[] = Float64.(gen_L(ps)))
                eb = eta_from(Linj[], randn(hrng, nq, cfg.h * length(c)))
                Ab, Qb = noisy_tensor(Xc[:, c], eb)
                Yb = T.(Ytr_g[:, c] .- s1_dy(window_noise(eb)) .- knet_dy(Ab, Qb, cols_tr[c], 1))
            else
                Ab, Qb, Yb = Atr[:, :, c], Qtr[:, c], Ytr_g32[:, c]
            end
            g = Zygote.gradient(p -> loss(p, Ab, Qb, E, Yb), ps)[1]
            GEN_LFIX && (g = merge(g, (; L = nothing)))                # L frozen at L0
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
            upd % CST.val_every == 0 || continue
            GEN_OWN && (Linj[] = injf(ps))                          # owntot: refresh the total factor
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
    # --- the offline check (8.2-10 TU) and what is reported beside it ---
    Ec = randn(Xoshiro(seed + 271), T, nq, size(Yva, 2) * GEN_MCHK)
    Xc_ = Float64.(gen_samples(model, pb, st, Ava, Qva, Ec))
    Xs_ = Float64.(reshape(GEN_L0S * Ec, nq, size(Yva, 2), :))                # the start: g = 0, L = L0
    # in stage 1's frame (heldout: against its own skip); the scores equal those of the full prediction against
    # the truth, so they compare directly with LinReg1 + MVG's closed form in LinReg1's frame
    crps_chk = RF.crps_fair_samples(Xc_, Yva_g)
    crps_start = RF.crps_fair_samples(Xs_, Yva_g)
    es_chk, es_start = RF.energy_fair_samples(Xc_, Yva_g), RF.energy_fair_samples(Xs_, Yva_g)   # the joint score too
    best.update == 0 && @info "best iterate = the start" max_abs_sample_diff = maximum(abs, Xc_ .- Xs_) crps_chk crps_start
    # GEN_HIST = own: the same check with each model under ITS OWN history noise -- the fit's window noise from its L,
    # the start's from L0, the same standard normals (paired); the start's samples are L0 ε whatever its input, its
    # target the stage-1 skip's noisy residual. The A2 PASS rule.
    own = !GEN_OWN ? nothing : let Eh = randn(Xoshiro(seed + 274), nq, cfg.h * size(Yva, 2))
        Lb_inj = injf(pb)                                           # its injected factor (own: L; owntot: total)
        eb, e0 = eta_from(Lb_inj, Eh), eta_from(Float64.(GEN_L0S), Eh)
        An, Qn = noisy_tensor(Xc[:, (ntr + 1):end], eb)
        cva = cols_tr[(ntr + 1):end]
        Yb_ = Yva_g .- s1_dy(window_noise(eb)) .- knet_dy(An, Qn, cva, 1)
        Y0_ = Yva_g .- s1_dy(window_noise(e0)) .- (GEN_NET ? knet_dy(noisy_tensor(Xc[:, (ntr + 1):end], e0)..., cva, 1) : 0.0)
        Xn = Float64.(gen_samples(model, pb, st, An, Qn, Ec))
        sn = gen_stats(model, pb, st, An, Qn, GEN_MCHK, Xoshiro(seed + 275))
        (; crps_chk = RF.crps_fair_samples(Xn, Yb_), crps_start = RF.crps_fair_samples(Xs_, Y0_),
         es_chk = RF.energy_fair_samples(Xn, Yb_), es_start = RF.energy_fair_samples(Xs_, Y0_),
         err_over_start = rmsq(Yb_ .- sn.mu) ./ rmsq(Y0_), spread_over_err = sqrt.(vec(mean(sn.sd .^ 2; dims = 2))) ./ rmsq(Yb_ .- sn.mu),
         # the escape measure: the injected noise's sd over the sd of the noise the closure draws (1 = all injected)
         inj_over_drawn = sqrt.(diag(Lb_inj * Lb_inj')) ./ sqrt.(vec(mean(sn.sd .^ 2; dims = 2))),
         # step5_reach_k1a gate 2 (stage 1): the drawn sd under its own noise over the skip's (B: k1a's) sd
         drawn_over_ref = sqrt.(vec(mean(sn.sd .^ 2; dims = 2))) ./ sqrt.(diag(GEN_SIGMA)))
    end
    crps_mvg = crps_eval(Yva_3, sqrt.(diag(GEN_SIGMA)))            # the skip + MVG, closed form (B: k1a), in sample
    s = gen_stats(model, pb, st, Ava, Qva, GEN_MCHK, Xoshiro(seed + 272))
    err = rmsq(Yva_g .- s.mu)                                       # = the full prediction's error
    Lb = Float64.(gen_L(pb))
    vtot = vec(mean(s.sd .^ 2; dims = 2))
    Rs = gen_corr(model, pb, st, Ava, Qva, GEN_MCHK, Xoshiro(seed + 273))
    return (; ps = pb, st, val_loss = best.val, val0, best_update = best.update, updates = upd, stop, hist,
            nparams = Lux.parameterlength(model), crps_chk, crps_start, crps_mvg, crps_ls = LS_CRPS,
            spread_over_err = sqrt.(vtot) ./ err, err_over_lr1 = err ./ rmsq(Yva_3),
            share_L = diag(Lb * Lb') ./ vtot, share_net = vec(mean(s.sdg .^ 2; dims = 2)) ./ vtot,
            corr_samples = Rs, corr_LLt = corrof(Lb * Lb'), corr_Sigma = corrof(GEN_SIGMA), L = Lb,
            es_chk, es_start, score = GEN_SCORE, stage1 = Symbol(GEN_S1MODE),
            crps_start_closed = crps_eval(Yva_g, sqrt.(diag(GEN_SIGMA_S))), own, hist_noise = GEN_HIST,
            loss = CLOSS, input = CINPUT, seq = CSEQ, act = CACT, init = CINIT, M = GEN_M, Mval = GEN_MVAL)
end

"""
The second training of a generator (REFIT = 1): set up exactly as `fit_gen` (same seed, same initialisation, the
same noise stream), on all 1-10 TU rows for `nupd` updates at the learning rate stage 1 used at each update.
"""
function refit_gen(C, lr, wd, seed, hist, nupd)
    rng = Xoshiro(seed)
    model = mkmodel_gen(C)
    net, st = Lux.setup(rng, model)
    ps = (; net, L = copy(GEN_L0))
    rule = wd > 0 ? Optimisers.AdamW(T(lr), (T(0.9), T(0.999)), T(wd); couple = true) : Optimisers.Adam(T(lr))
    opt = Optimisers.setup(rule, ps)
    wd > 0 && Optimisers.adjust!(opt.L; lambda = zero(T))
    loss(p, A, Q, E, Y) = (X = gen_samples(model, p, st, A, Q, E); gen_score(X, Y) + gen_cap(X))
    nrng = Xoshiro(seed + 7919)
    hrng = Xoshiro(seed + 15485863)                                 # GEN_HIST = own: as fit_gen
    injrows = GEN_TOT ? shuffle(Xoshiro(seed + 98), 1:ntr)[1:min(512, ntr)] : Int[]   # owntot: as fit_gen
    injf(p) = GEN_TOT ? gen_tot_factor(model, p, st, Atr[:, :, injrows], Qtr[:, injrows], 8, seed + 99) : Float64.(gen_L(p))
    Linj = Ref(GEN_OWN ? injf(ps) : zeros(nq, nq))
    lr_at(u) = T(hist.lr[searchsortedfirst(hist.update, u)])
    Aall, Qall, Yall = cat(Atr, Ava; dims = 3), hcat(Qtr, Qva), T.(Yall_g)   # = hcat(Ytr32, Yva32) without B
    Yall64 = Yall_g
    N = size(Yall, 2)
    nb = fld(N, CST.batch)
    upd, cur = 0, T(lr)
    while upd < nupd
        idxs = shuffle(rng, 1:N)
        for k in 1:nb
            upd >= nupd && break
            l = lr_at(upd + 1)
            l == cur || (Optimisers.adjust!(opt, l); cur = l)
            c = idxs[((k - 1) * CST.batch + 1):(k * CST.batch)]
            E = randn(nrng, T, nq, length(c) * GEN_M)
            if GEN_OWN                                              # the skip's own response (refit: the skip LinReg)
                Linj[] = (GEN_TOT && upd % CST.val_every != 0) ? Linj[] : injf(ps)   # owntot: every val_every updates
                eb = eta_from(Linj[], randn(hrng, nq, cfg.h * length(c)))
                Ab, Qb = noisy_tensor(Xc[:, c], eb)
                Yb = T.(Yall64[:, c] .- skip_dy(window_noise(eb)) .- knet_dy(Ab, Qb, cols_tr[c], 3))
            else
                Ab, Qb, Yb = Aall[:, :, c], Qall[:, c], Yall[:, c]
            end
            g = Zygote.gradient(p -> loss(p, Ab, Qb, E, Yb), ps)[1]
            GEN_LFIX && (g = merge(g, (; L = nothing)))                # L frozen at L0
            opt, ps = Optimisers.update(opt, ps, g)
            upd += 1
        end
    end
    return ps, st, model
end

# RIKFLOW_CNN_GEN_TRAINLOG = <N> (Rik, 2026-11-06: "show me the training loss of the network fitted on 1-10 TU vs the
# loss of the linreg on this data"): no grid. For each (C, lr, wd, seed), the REFIT's setup (LinReg1's skip and
# target, L0, the same initialisation, shuffle and noise streams) trained on all 1-10 TU rows for N updates at the
# constant rate lr -- which IS the refit for as long as its replayed schedule stayed at lr -- logging every 200 updates
# the energy score and the summed CRPS of 16 fixed draws per row on all 1-10 TU rows (the training data) and of 4 on
# 10-100 TU; update 0 is LinReg1 + MVG on the same draws. Lines start with "TRAINLOG"; nothing is saved.
const GEN_TRAINLOG = GEN ? parse(Int, envc("GEN_TRAINLOG", "0")) : 0
if GEN && GEN_TRAINLOG > 0
    Aall_, Qall_, Yall_ = cat(Atr, Ava; dims = 3), hcat(Qtr, Qva), hcat(Ytr32, Yva32)
    Yall64 = Float64.(dat.Yc)
    Aho_, Qho_ = Aho, Qho
    for C in CHS, lr in LRS, wd in CWDS, seed in CSEEDS
        rng = Xoshiro(seed)
        model = mkmodel_gen(C)
        net, st = Lux.setup(rng, model)
        ps = (; net, L = copy(GEN_L0))
        rule = wd > 0 ? Optimisers.AdamW(T(lr), (T(0.9), T(0.999)), T(wd); couple = true) : Optimisers.Adam(T(lr))
        opt = Optimisers.setup(rule, ps)
        wd > 0 && Optimisers.adjust!(opt.L; lambda = zero(T))
        loss(p, A, Q, E, Y) = (X = gen_samples(model, p, st, A, Q, E); gen_score(X, Y) + gen_cap(X))
        nrng = Xoshiro(seed + 7919)
        Etr = randn(Xoshiro(seed + 555), T, nq, size(Yall_, 2) * 16)
        Eho = randn(Xoshiro(seed + 556), T, nq, length(HO) * 4)
        function logline(p, upd)
            Xt = Float64.(gen_samples(model, p, st, Aall_, Qall_, Etr))
            Xh = Float64.(gen_samples(model, p, st, Aho_, Qho_, Eho))
            @printf("TRAINLOG wd %g seed %d upd %6d | 1-10 TU: ES %.5f CRPS %.5f | 10-100 TU: ES %.5f CRPS %.5f\n", wd, seed,
                    upd, RF.energy_fair_samples(Xt, Yall64), RF.crps_fair_samples(Xt, Yall64),
                    RF.energy_fair_samples(Xh, Yf[:, HO]), RF.crps_fair_samples(Xh, Yf[:, HO]))
            flush(stdout)
        end
        logline(ps, 0)
        N = size(Yall_, 2)
        nb = fld(N, CST.batch)
        upd = 0
        while upd < GEN_TRAINLOG
            idxs = shuffle(rng, 1:N)
            for k in 1:nb
                upd >= GEN_TRAINLOG && break
                c = idxs[((k - 1) * CST.batch + 1):(k * CST.batch)]
                E = randn(nrng, T, nq, length(c) * GEN_M)
                g = Zygote.gradient(p -> loss(p, Aall_[:, :, c], Qall_[:, c], E, Yall_[:, c]), ps)[1]
                GEN_LFIX && (g = merge(g, (; L = nothing)))                # L frozen at L0
                opt, ps = Optimisers.update(opt, ps, g)
                upd += 1
                upd % 200 == 0 && logline(ps, upd)
            end
        end
    end
    exit()
end

if GEN
    fmt(v) = join((@sprintf("%.3f", x) for x in v), " ")
    "Standardised target -> raw dQ at record columns `c` (the skip's correction included)."
    rawg(y, c) = y .* so .+ mo .+ skipq(c)
    resg = Dict{Tuple{Int,Float64,Float64,Int},Any}()
    gstat = Dict{Int,Any}()
    Arec, Qrec = CREFIT ? lagtensor(Xf) : (zeros(T, 0, 0, 0), zeros(T, 0, 0))
    CREFIT && (length(CHS) == length(LRS) == length(CWDS) == 1 ||
               error("RIKFLOW_CNN_REFIT = 1 takes one (C, lr, wd) setting per run"))
    for C in CHS, lr in LRS, wd in CWDS, seed in CSEEDS
        t0 = time()
        r = fit_gen(C, lr, wd, seed)
        resg[(C, lr, wd, seed)] = merge(r, (; wall = time() - t0))
        @printf("gen  C %3d K %d D %d (%6d weights) lr %.0e wd %.0e seed %d | %5d upd, best @ %5d (%s) | val fair %s (M %d fixed) %.4f (start %.4f) | %.1f min\n",
                C, K, D, r.nparams, lr, wd, seed, r.updates, r.best_update, r.stop,
                GEN_SCORE === :energy ? "energy score" : "CRPS", GEN_MVAL, r.val_loss, r.val0,
                (time() - t0) / 60)
        @printf("     CHECK val CRPS (M %d fixed draws) %.4f | the start (%s), same draws %.4f -> %s | closed forms: the start %.4f, %s + MVG %.4f (%s) | LS refit 1-8.2 TU %.4f\n",
                GEN_MCHK, r.crps_chk, GEN_HSKIP ? "$(SKIP) refitted on 1-8.2 TU + its MVG" : GEN_HELD ? "LS on 1-8.2 TU + its MVG" : "$(SKIP) + MVG", r.crps_start,
                r.crps_chk <= r.crps_start ? "PASS" : "FAIL", r.crps_start_closed, SKIP, r.crps_mvg,
                r.crps_chk <= r.crps_mvg ? "below" : "above", r.crps_ls)
        @printf("     val energy score (same %d draws) %.4f | the start %.4f (%+.2f %%)\n", GEN_MCHK, r.es_chk, r.es_start,
                100 * (r.es_chk / r.es_start - 1))
        r.own === nothing ||
            @printf("     OWNCHECK (each under its own history noise, paired) val CRPS %.4f | the start %.4f (%+.2f %%) -> %s | energy %.4f | the start %.4f (%+.2f %%) | error of the mean / the start's %s | sd / error %s\n",
                    r.own.crps_chk, r.own.crps_start, 100 * (r.own.crps_chk / r.own.crps_start - 1),
                    r.own.crps_chk <= r.own.crps_start ? "PASS" : "FAIL", r.own.es_chk, r.own.es_start,
                    100 * (r.own.es_chk / r.own.es_start - 1), fmt(r.own.err_over_start), fmt(r.own.spread_over_err))
        r.own === nothing || @printf("     ESCAPE CHECK (%s): injected noise sd / drawn noise sd %s | L sd / L0 sd %s | the network's share %s
",
                                     GEN_HIST, fmt(r.own.inj_over_drawn), fmt(sqrt.(diag(r.L * r.L')) ./ sqrt.(diag(GEN_SIGMA_S))),
                                     fmt(r.share_net))
        r.own === nothing || @printf("     GATE 2 (stage 1): drawn sd under its own noise / the skip's sd %s | LFIX %s CENTRE %s SDCAP %g
",
                                     fmt(r.own.drawn_over_ref), GEN_LFIX, GEN_CENTRE, GEN_SDCAP)
        @printf("     val per QoI: samples' sd / error of their mean %s | error of the mean / the skip's (%s, in sample there) %s\n",
                fmt(r.spread_over_err), SKIP, fmt(r.err_over_lr1))
        @printf("     share of the sample variance: L eps %s | the network's part %s (rest: their covariance)\n",
                fmt(r.share_L), fmt(r.share_net))
        @printf("     correlation: max |samples - Sigma| %.3f, max |L L' - Sigma| %.3f | L sd / L0 sd %s\n",
                maximum(abs, r.corr_samples .- r.corr_Sigma), maximum(abs, r.corr_LLt .- r.corr_Sigma),
                fmt(sqrt.(diag(r.L * r.L')) ./ sqrt.(diag(GEN_SIGMA))))
        @printf("     %.2f ms per update (including validations and scoring)\n", 1e3 * (time() - t0) / max(r.updates, 1))
        cps = [u for u in (1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000) if u <= r.updates]
        isempty(cps) || @printf("     val loss at %s\n", join((@sprintf("%dk: %.4f (lr %.0e)", u ÷ 1000,
                                 r.hist.val[findfirst(>=(u), r.hist.update)], r.hist.lr[findfirst(>=(u), r.hist.update)]) for u in cps), ", "))
        if CREFIT
            t1 = time()
            ps3, st3, model3 = refit_gen(C, lr, wd, seed, r.hist, r.best_update)
            # the stage-1 and refitted networks on every record step, fixed draws (common to both)
            st1 = gen_stats(model3, r.ps, st3, Arec, Qrec, GEN_MREC, Xoshiro(20261106 + seed))
            st3s = gen_stats(model3, ps3, st3, Arec, Qrec, GEN_MREC, Xoshiro(20261106 + seed))
            rows10 = findall(n -> cfg.train_range[1] <= n <= cfg.train_range[2], cols)
            corr10 = gen_corr(model3, ps3, st3, Arec[:, :, rows10], Qrec[:, rows10], 8, Xoshiro(seed + 911))
            corrho = gen_corr(model3, ps3, st3, Arec[:, :, HO], Qrec[:, HO], 8, Xoshiro(seed + 912))
            # the export's translation check: 300 steps of 10-100 TU, one fixed draw each, the refitted network
            ich = HO[1:300]
            Ech = randn(Xoshiro(seed + 2026), T, nq, length(ich))
            ych = Float64.(dropdims(gen_samples(model3, ps3, st3, Arec[:, :, ich], Qrec[:, ich], Ech); dims = 3))
            gstat[seed] = (; mean1 = rawg(st1.mu .+ gen_s1_offset(), cols), sd1 = st1.sd, sdg1 = st1.sdg, mean3 = rawg(st3s.mu .+ gen_s3_offset(), cols), sd3 = st3s.sd,
                           sdg3 = st3s.sdg, corr10, corrho, L3 = Float64.(gen_L(ps3)),
                           chk = (; cols = cols[ich], E = Float64.(Ech), y = rawg(ych .+ gen_s3_offset()[:, ich], cols[ich])))
            resg[(C, lr, wd, seed)] = merge(resg[(C, lr, wd, seed)], (; refit = (; ps = ps3, st = st3, updates = r.best_update)))
            ho = rmsq(Yf_g[:, HO] .- st3s.mu[:, HO])
            @printf("     refit on 1-10 TU (%d updates, %.1f min): 10-100 TU error of the sample mean / %s's %s | sd / error %s | corr max |.-Sigma| 1-10 %.3f, 10-100 %.3f\n",
                    r.best_update, (time() - t1) / 60, SKIP, fmt(ho ./ rmsq(Yf_g[:, HO])),
                    fmt(sqrt.(vec(mean(st3s.sd[:, HO] .^ 2; dims = 2))) ./ ho),
                    maximum(abs, corr10 .- corrof(GEN_SIGMA)), maximum(abs, corrho .- corrof(GEN_SIGMA)))
        end
        flush(stdout)
    end
    out_dir = joinpath(TO_folder, COUT)
    mkpath(out_dir)
    jldsave(joinpath(out_dir, "fits.jld2"); res = resg, ls_crps = LS_CRPS, settings = CST, K, D, feat = FEAT, cfg,
            loss = CLOSS, input = CINPUT, seq = CSEQ, act = CACT, innoise = CINN, innoise_sigma = CINN_SRC,
            skip = SKIP, Sigma = GEN_SIGMA, L0 = Float64.(GEN_L0), M = GEN_M, Mval = GEN_MVAL, score = GEN_SCORE,
            stage1 = Symbol(GEN_S1MODE), Sigma_stage1 = GEN_SIGMA_S, outinit = GEN_OUTRAND ? :random : :zero, hist_noise = GEN_HIST, netskip = GEN_NETSKIP,
            lfix = GEN_LFIX, centre = GEN_CENTRE, sdcap = GEN_SDCAP, sdcap_w = GEN_SDCAP_W)
    if CREFIT
        jldsave(joinpath(out_dir, "pred.jld2"); cols, gen = gstat, seeds = CSEEDS, wd = only(CWDS), n_channels = only(CHS),
                loss = CLOSS, h = cfg.h, K, D, feat = FEAT, input = CINPUT, seq = CSEQ, act = CACT, scaling,
                skip = SKIP, Sigma = GEN_SIGMA, Mrec = GEN_MREC, innoise = CINN, stage1 = Symbol(GEN_S1MODE), hist_noise = GEN_HIST,
                centre = GEN_CENTRE, lfix = GEN_LFIX, sdcap = GEN_SDCAP,
                # the target on every step of `cols`: the skip LinReg's residual, centred at its MVG mean, / so
                ytarget = Yf_g, netskip = GEN_NETSKIP)
    end
    @info "written" out_dir
    exit()
end

if CREFIT
    length(CHS) == length(LRS) == length(CWDS) == 1 || error("RIKFLOW_CNN_REFIT = 1 takes one (C, lr, wd) setting per run")
end
# the whole record as the network reads it (host), and the standardised -> raw map of the target
const Afull, Qfull = CREFIT ? lagtensor(Xf) : (zeros(T, 0, 0, 0), zeros(T, 0, 0))
rawq(y) = y .* so .+ mo .+ skipq(cols)                     # with SKIP: + the skip's correction
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
    @printf("     fit rows (1-8.2 TU) 0.5 SSE at %s (LS %.4f)\n", join((@sprintf("%dk: %.4f", u ÷ 1000,
            r.hist.fit[findfirst(>=(u), r.hist.update)]) for u in cps), ", "), 0.5 * sum(abs2, ls_fit))
    @printf("     fit rows (1-8.2 TU) rms / LS: best iterate %s | last iterate %s\n",
            join((@sprintf("%.3f", x) for x in r.fit_ratio), " "), join((@sprintf("%.3f", x) for x in r.last.fit_ratio), " "))
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
        ps3, st3, model3 = refit_cnn(C, lr, wd, seed, r.hist, CREFIT_UPD === :last ? r.updates : r.best_update;
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
                CREFIT_UPD === :last ? r.updates : r.best_update, (time() - t1) / 60, sse10, join((@sprintf("%.2f", x) for x in ho3), " "))
        if r.noisy !== nothing
            # the Σ this network deploys: its residual on all 1-10 TU rows under the injected noise, about zero
            # (4 draws); S_deploy / Σ_inj -> 1 is the outer loop's fixed point. INNOISE = head: the refitted
            # head's Σ is deployed and injected; the residual under it (S_noisy) is the check
            L = CINN === :head ? head_factor(ps3.head) : cov_factor(r.noisy.Sinj)
            Sd = sum(1:4) do rep
                en = eta_from(L, randn(Xoshiro(seed + 500 + rep), nq, cfg.h * ncol))
                An, Qn = noisy_tensor(Xc, en)
                mom0(Float64.(dat.Yc) .- Float64.(first(model3(modelin(An, Qn), ps3.net, st3))) .- skip_dy(window_noise(en))) ./ 4
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
