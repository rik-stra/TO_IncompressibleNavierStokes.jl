# Step 3a closures: a CNN of paper Sec. 6 deployed on the network path (2026-10-23, Rik: "build it on the
# network path ... the AR(p) and power-law code shared with LinReg ... the export from cnn_mean_fit.jl, the
# translation check"). Writes `output/TO_LSTM/step3/closures/<name>/StochLSTM_seed<s>.jld2`, the file
# `d6_lib.jl` resolves for D6_MODEL = step3/closures/<name>, and checks it.
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/cnn_closure_export.jl \
#           <run> <noise> <name> [skip=<LinReg>] [meannet=<step-4 closure>]
#
# `meannet=<step-4 closure under step3/closures>` (step 5 variant B, 2026-11-11, card analysis/cards/step5_skips.md;
# noise gen and skip= only): the run was trained by cnn_mean_fit.jl with RIKFLOW_CNN_GEN_NETSKIP = that closure's fit
# dir, on its FROZEN CNN mean. The generator then carries that closure's network as `Cv.mean_net` (`RF.mean_net_from`,
# `RF.with_mean_net`): its fit dir must be the run's `netskip`, its Ws and scaling this run's; `frozen_mean + Ws z` is
# checked against that closure's own deployed mean on 300 record steps, and the translation check covers the sum.
#
# `skip=<LinReg>` (step 4, 2026-11-06; paper Eq. skip net): the run was trained on a FROZEN LinReg under
# output/TO_LRS (`cnn_mean_fit.jl` with RIKFLOW_MLP_SKIP=<LinReg>; the fit files do not record which, so it is
# given here, and the translation check fails on a wrong one). The closure deploys that LinReg's mean model as
# the linear skip `Ws` (`RF.linreg_skip_map`, checked against the LinReg's own predictions on the record) plus
# the network, and is exported in FLOAT64: the least-squares skip's coefficients reach ~1e4 target sd, and
# StochLSTM's Float32 lag window alone would put it ~5e-3 sd off (scratch probe, 2026-11-06). Its unit below,
# the target sd, is then the sd of what the LinReg misses.
#
# `<run>`: a refitted CNN under output/TO_LSTM, one seed (`cnn_mean_fit.jl` with RIKFLOW_CNN_REFIT=1, e.g.
# step3/grid_cap_rmse_wd0.3_s1); its `fits.jld2` gives the network trained again on 1-10 TU and its
# `pred.jld2` that network's predictions on the record (Lux). `<noise>`, all fitted to that network's
# residual on 1-10 TU (record steps 400-4000), in its standardised target (dQ):
#   mvg      the MVG refitted about zero: Σ = Z'Z / N (Rik, 2026-10-23: "about zero")       (a1, a2)
#   head     the head learned with the network (the CRPS's: its sds, no correlation)        (a3)
#   headcorr the learned sds with the correlation of the MVG refit                          (a3')
#   ar2n     LinReg's AR(2) of step 1, `fit_ar(Z, 2; nonneg = true)` of lrs_ar_variant.jl (ACF lags 1-20,
#            φ1 ≥ 0, φ2 ≤ 0), Σ_ξ = D R D with the AR's stationary variance matched to var(Z)    (b1)
#   pl       step 2's power-law scale, `fit_scale` of lrs_scale_variant.jl (ML per QoI, clipped to the
#            1-10 TU range of q*), Σ_ε = Z_f'Z_f / N with Z_f = Z ./ f(q*)                   (c1)
#   mvgnt    a NOISE-TRAINED run (cnn_mean_fit.jl, RIKFLOW_CNN_INNOISE = fixed, 2026-10-26): the MVG it
#            deploys, `S_deploy` in its pred.jld2 -- its residual on 1-10 TU under its own injected window
#            noise, about zero -- not the clean residual's. INNOISE = head (2026-11-03): `S_deploy` is the
#            refitted head's Σ, the noise it trained with -- for the CRPS diagonal: the head as trained (k1b, k2b).
#   headcorrnt  a noise-trained CRPS run in head mode (2026-11-06, step 4): the learned sds (the refitted
#            head) with the correlation of `S_noisy`, its 1-10 TU residual under its own (diagonal) window
#            noise, about zero (k1a, k2a; the noise-trained analogue of headcorr).
#   gen      step 5 (2026-11-06; paper Eq. stochastic net, card analysis/cards/step5.md): a STOCHASTIC SKIP NETWORK
#            (cnn_mean_fit.jl, RIKFLOW_CNN_LOSS = crps_gen; skip=<LinReg> required, the one it was trained on):
#            emission = :generator, the noise skip L (the refitted network's) in `LR`, bd = 0, nothing added after.
#            Its translation check feeds the deployed closure the fit's own deviates (pred.jld2's `chk`: 300 steps
#            of 10-100 TU, one fixed draw each) and compares the SAMPLES.
# mvgnt and headcorrnt are for noise-trained runs only; the other modes refuse them.
# The noise model is stored in the scaling (`ar_phi`, `scale_*`) and the head (`bd`, `LR`), which
# `StochLSTM` applies with the code `LinReg` uses (`ts_noise.jl`, `ts_scale.jl`).
#
# Translation check: the deployed `StochLSTM` (stochastic = false: its mean), run as D6 runs it -- a
# 100-step warm-up replaying the record, then one prediction -- at 300 steps spread over 1-100 TU, against
# the Lux network's prediction in `pred.jld2`. 🔒 Fits read 1-10 TU only.

include(joinpath(@__DIR__, "lrs_scale_variant.jl"))   # fit_ar, fit_scale (and RF, REF, NQ, DT); main guarded
using Random

length(ARGS) in (3, 4, 5) || error("usage: cnn_closure_export.jl <run> <noise> <name> [skip=<LinReg>] [meannet=<closure>]")
const XRUN, XNOISE, XNAME = ARGS[1], ARGS[2], ARGS[3]
const XOPT = Dict(let i = findfirst(==('='), a); i === nothing && error("optional arguments are key=value; got $a"); a[1:(i - 1)] => a[(i + 1):end] end
                  for a in ARGS[4:end])
issubset(keys(XOPT), ("skip", "meannet")) || error("optional arguments: skip=<LinReg>, meannet=<closure>; got $(collect(keys(XOPT)))")
const XSKIP = get(XOPT, "skip", "")
const XMEAN = get(XOPT, "meannet", "")
const XNT_MODES = ("mvgnt", "headcorrnt")
XNOISE in ("mvg", "head", "headcorr", "ar2n", "pl", "gen", XNT_MODES...) ||
    error("noise must be mvg, head, headcorr, ar2n, pl, gen, mvgnt or headcorrnt; got $XNOISE")
const XTO = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
const XNWARM = 100                                   # D6's warm-up (d6_lib.jl)

F = load(joinpath(XTO, XRUN, "fits.jld2"))
P = load(joinpath(XTO, XRUN, "pred.jld2"))
key, r = only(F["res"])
C, lr, wd, seed = key
haskey(r, :refit) || error("$XRUN has no refit (run cnn_mean_fit.jl with RIKFLOW_CNN_REFIT=1)")
const XGEN = P["loss"] === :crps_gen                  # step 5's generator
XGEN == (XNOISE == "gen") || error(XGEN ? "$XRUN is a step-5 generator (crps_gen): export it with noise gen" :
                                          "noise gen is for a step-5 generator (crps_gen); $XRUN is not")
XGEN && XSKIP != P["skip"] && error("$XRUN was trained on the skip $(P["skip"]); got skip=$XSKIP")
const XNETSKIP = XGEN ? get(P, "netskip", "") : ""
isempty(XMEAN) == isempty(XNETSKIP) || error(isempty(XMEAN) ? "$XRUN was trained on the frozen CNN mean of $XNETSKIP: give meannet=<its closure>" :
                                                "meannet= is for a generator trained with GEN_NETSKIP; $XRUN was not")
const XINN = get(P, "innoise", :none)
const XNT = XINN !== :none                           # a noise-trained run
XNT == (XNOISE in XNT_MODES) || error(XNT ? "$XRUN is noise-trained: export it with noise $(join(XNT_MODES, " or "))" :
                                            "noise $XNOISE is for a noise-trained run; $XRUN is not")
XNOISE == "headcorrnt" && !(XINN === :head && P["loss"] === :crps) &&
    error("headcorrnt is for a CRPS run in head mode (INNOISE = head); $XRUN has $(XINN), loss $(P["loss"])")
# the head's START (HEAD = identity | inj) does not enter the deployed closure
(P["input"], P["seq"], P["act"]) == (:full, :lags, :tanh) && (XGEN || P["head"] in (:identity, :inj)) ||
    error("the deployed CNN is SEQ = lags, INPUT = full, ACT = tanh; $XRUN has $((P["input"], P["seq"], P["act"]))")
h, K, D = P["h"], P["K"], P["D"]
scaling = P["scaling"]
nin, nreg = size(scaling.input_map)
cols = P["cols"]
# raw dQ of the refitted network, every step of `cols` (the generator: its sample mean, pred.jld2's fixed draws)
pred3 = XGEN ? P["gen"][seed].mean3 : only(values(P["pred3"]))

# the spec, and the sequence the network reads from the mapped input (cnn_mean_fit.jl's `lagtensor`, from
# step3_data.jl's `lagrows`: rows of dQ^{n-k} and q*^{n-k}; the increments after the regressor)
hist = RF.HistorySpec(; h, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
RF.nfeatures(hist) == nreg || error("regressor width $(RF.nfeatures(hist)), the input map reads $nreg")
spec = RF.LSTMSpec(; hist, n_hidden = D, n_latent = 0, n_encoder = 0, arch = :cnn,
                   emission = XGEN ? :generator : :constant,
                   window = 1, n_in = nin, n_conv = C, kernel = K, skip = !isempty(XSKIP))
lagq(k) = (NQ + 2(k - 1) * NQ + 1):(NQ + (2k - 1) * NQ)
seq_index = zeros(Int, h, 3NQ)
for (t, k) in enumerate(h:-1:1)                      # oldest first
    seq_index[t, :] = vcat(collect(lagq(k)) .+ NQ, collect(lagq(k)), nreg .+ (((k - 1) * NQ + 1):(k * NQ)))
end

# the refitted network's residual on 1-10 TU, standardised target (rows = steps, as fit_ar / fit_scale take it)
so, mo = vec(scaling.out_scaling.sigma), vec(scaling.out_scaling.mu)
dq = REF["q"][:, cols .+ 1] .- REF["q_star"][:, cols]
tr = findall(n -> 400 <= n <= 4000, cols)
all(diff(cols[tr]) .== 1) || error("the 1-10 TU rows are not consecutive steps")
Z = permutedims((dq[:, tr] .- pred3[:, tr]) ./ so)  # (dq - mo)/so - (pred - mo)/so
qs = permutedims(REF["q_star"][:, cols[tr]])
@printf("%s, %s noise: residual on 1-10 TU (%d steps), mean / rms %s\n", XRUN, XNOISE, size(Z, 1),
        fmt(vec(mean(Z; dims = 1)) ./ sqrt.(vec(mean(abs2, Z; dims = 1)))))

# SKIP: the frozen LinReg as `Ws` on the mapped input, checked on every record step of `cols` against the LinReg's
# own predicted correction (its regressor built in its own scaling, as step3_data.jl's `skipq`), in Float64
Ws = nothing
if !isempty(XSKIP)
    lc, lsc, lhl = load(joinpath(XTO, "..", "TO_LRS", XSKIP, "LinReg.jld2"), "c", "scaling", "hist_len")
    lhl == h || error("skip $XSKIP: history $lhl, the network's $h")
    Ws = RF.linreg_skip_map(lc, lsc, scaling, hist)
    a0, b0 = 400, size(REF["q_star"], 2)
    cols[1] - a0 + 1 >= 1 || error("the record columns start before step $a0")
    function xr(s)                                                    # nreg x steps, and the record columns
        X, _, st = RF.build_history(hist, RF.scale_input(REF["q_star"][:, a0:b0], s), RF.scale_input(REF["q"][:, a0:(b0 + 1)], s))
        cs = (a0 - 1) .+ st
        cs == cs[1]:cs[end] || error("the regressor's steps are not contiguous")
        return permutedims(Float64.(X)), cs
    end
    (XL, csl), (XN, csn) = xr(lsc.in_scaling), xr(scaling.in_scaling)
    csl == csn || error("the two regressors cover different steps")
    jc = cols .- csl[1] .+ 1                                          # column of `cols` in XL / XN
    all(1 .<= jc .<= size(XL, 2)) || error("the record columns fall outside the regressor")
    mul, sgl = vec(lsc.in_scaling.mu), vec(lsc.in_scaling.sigma)
    lev = (Float64.(lc) * XL[:, jc]) .* vec(lsc.out_scaling.sigma) .+ vec(lsc.out_scaling.mu)
    qsn = XL[1:NQ, jc] .* sgl .+ mul
    maximum(abs, qsn .- REF["q_star"][:, cols]) < 1e-9 * maximum(abs, REF["q_star"]) ||
        error("skip $XSKIP: the regressor's first rows are not q^{n*} at the record columns")
    skq = (lev .- qsn) ./ so                                          # the LinReg's correction, target units
    e = maximum(abs, Ws * (Float64.(scaling.input_map) * XN[:, jc]) .- skq) / maximum(abs, skq)
    @printf("skip %s: |Ws| max %.3g target sd per input; Ws z against the LinReg's own correction on %d record steps, max rel. diff %.1e\n",
            XSKIP, maximum(abs, Ws), length(cols), e)
    e < 1e-9 || error("skip $XSKIP: Ws differs from the LinReg's predictions by $e (relative)")
    # what is left for the network: the residual of the skip alone on 1-10 TU, against the closure's (Z)
    @printf("  rms on 1-10 TU, target sd: LinReg alone %s | LinReg + network %s\n",
            fmt(sqrt.(vec(mean(abs2, (dq[:, tr] ./ so) .- skq[:, tr]; dims = 2)))), fmt(sqrt.(vec(mean(abs2, Z; dims = 1)))))
end
const XT = isempty(XSKIP) ? Float32 : Float64       # the closure's precision (see the header)

"bd (log sd) and LR (lower Cholesky factor of the correlation) of a covariance S."
function headof(S)
    s = sqrt.(diag(S))
    R = Symmetric(S ./ (s * s'))
    return log.(s), Matrix(cholesky(R).L)
end
S0 = Z' * Z ./ size(Z, 1)                            # the MVG about zero
extra = (;)
prov = ""
if XNOISE == "mvg"
    bd, LR = headof(S0)
    prov = "MVG refitted about zero on the network's 1-10 TU residual"
elseif XNOISE == "mvgnt"
    Sd = P["S_deploy"][seed]
    bd, LR = headof(Sd)
    prov = XINN === :head ?
           "S_deploy of the noise-trained run in head mode: the refitted head's Sigma, the noise it trained with" *
           (P["loss"] === :crps ? " (CRPS: diagonal)" : "") :
           "S_deploy of the noise-trained run: its 1-10 TU residual under its own window noise (Sigma_inj = " *
           "the run's input-noise Sigma), about zero"
    @printf("  S_deploy sd / clean residual rms %s | / injected sd %s\n", fmt(sqrt.(diag(Sd) ./ diag(S0)); d = 3),
            fmt(sqrt.(diag(Sd) ./ diag(P["Sigma_inj"][seed])); d = 3))
    if XINN === :head && P["loss"] === :crps                # the head as trained: its own sds, no correlation
        e = maximum(abs, exp.(Float64.(vec(r.refit.ps.head.bd))) ./ sqrt.(diag(Sd)) .- 1)
        e < 1e-5 && maximum(abs, Sd .- Diagonal(diag(Sd))) == 0 ||
            error("S_deploy is not the refitted CRPS head's diagonal Sigma (sd rel. diff $e)")
    end
    haskey(P, "S_noisy") && @printf("  residual under its own noise, sd / deployed sd %s\n",
                                    fmt(sqrt.(diag(P["S_noisy"][seed]) ./ diag(Sd)); d = 3))
elseif XNOISE == "headcorrnt"
    bd = Float64.(vec(r.refit.ps.head.bd))
    Sn = P["S_noisy"][seed]
    LR = headof(Sn)[2]
    prov = "the learned sds (the refitted CRPS head, the noise it trained with) with the correlation of S_noisy, " *
           "its 1-10 TU residual under that noise, about zero"
    e = maximum(abs, exp.(bd) ./ sqrt.(diag(P["S_deploy"][seed])) .- 1)
    e < 1e-5 || error("the refitted head's sds differ from S_deploy's by $e")
    @printf("  learned sd / sd of the residual under its own noise %s; that residual's correlation, max |off-diagonal| %.3f\n",
            fmt(exp.(bd) ./ sqrt.(diag(Sn)); d = 3), maximum(abs, LR * LR' - I))
elseif XNOISE == "gen"
    bd = zeros(NQ)
    LR = Float64.(r.refit.ps.L) .* [i >= j for i in 1:NQ, j in 1:NQ]          # the noise skip, lower triangular
    maximum(abs, LR .- P["gen"][seed].L3) == 0 || error("the refitted L differs from pred.jld2's")
    S = F["Sigma"]
    prov = "step 5 generator (crps_gen): network on [features; q*; eps] + Ws z + L eps, L learned from chol of " *
           "the skip's Sigma; nothing added after"
    @printf("  L L' sd / skip's Sigma sd %s; corr max |L L' - Sigma| %.3f
", fmt(sqrt.(diag(LR * LR') ./ diag(S)); d = 3),
            maximum(abs, (LR * LR') ./ sqrt.(diag(LR * LR') * diag(LR * LR')') .- S ./ sqrt.(diag(S) * diag(S)')))
elseif XNOISE == "head"
    bd = Float64.(vec(r.refit.ps.head.bd))
    LR = Matrix(1.0I, NQ, NQ)
    prov = "the head learned with the network (sds exp(bd), no correlation)"
    @printf("  learned sd / rms of the residual: %s\n", fmt(exp.(bd) ./ sqrt.(diag(S0))))
elseif XNOISE == "headcorr"
    bd = Float64.(vec(r.refit.ps.head.bd))
    LR = headof(S0)[2]
    prov = "the learned sds with the correlation of the MVG refitted about zero"
    @printf("  learned sd / rms of the residual: %s\n", fmt(exp.(bd) ./ sqrt.(diag(S0))))
elseif XNOISE == "ar2n"
    f = fit_ar(Z, 2; nonneg = true)
    bd, LR = headof(f.S)
    extra = (; ar_phi = f.phi)
    prov = "fit_ar(Z, 2; nonneg = true) (lrs_ar_variant.jl): ACF lags 1-20, phi1 >= 0, phi2 <= 0; Sigma_xi = D R D"
    @printf("  AR(2): phi1 %s\n         phi2 %s\n", fmt(f.p1; d = 3), fmt(f.p2; d = 3))
    for i in 1:NQ
        ra = acf(Z[:, i], 0:20)
        rf = ar_acf(f.p1[i], f.p2[i], 20)
        @printf("  %-8s data rho1 %.3f, AR %.3f; max |AR - data| lags 1-20 %.3f\n", LABELS[i], ra[2], rf[2],
                maximum(abs, rf[2:end] .- ra[2:end]))
    end
elseif XNOISE == "pl"
    f = fit_scale(Z, qs)
    bd, LR = headof(f.S)
    extra = (; scale_beta = f.beta, scale_qref = f.qref, scale_qclip = f.qclip)
    prov = "fit_scale (lrs_scale_variant.jl): RF.fit_powerlaw_scale per QoI, clipped to the 1-10 TU range; Sigma_eps = Z_f'Z_f/N"
    @printf("  power law: beta %s\n", fmt(f.beta))
end
w = RF.cnn_weights(r.refit.ps.net, spec; seq_index, q_index = 1:NQ, bd, LR, Ws, T = XT)
if !isempty(XMEAN)                                  # step 5 B: the step-4 closure's network, frozen
    kd = joinpath(XTO, "step3", "closures", XMEAN)
    kf = only(filter(f -> startswith(f, "StochLSTM_seed") && endswith(f, ".jld2"), readdir(kd)))
    kc = RF.load_stochlstm(joinpath(kd, kf))
    kc.extras.run == XNETSKIP || error("meannet $XMEAN was exported from $(kc.extras.run), the run was trained on $XNETSKIP")
    kc.spec.skip && kc.weights.Ws !== nothing || error("meannet $XMEAN has no linear skip")
    eW = maximum(abs, Float64.(kc.weights.Ws) .- Float64.(Ws)) / maximum(abs, Ws)
    eW < 1e-12 || error("meannet $XMEAN's Ws differs from skip=$(XSKIP)'s by $eW (relative)")
    maximum(abs, Float64.(kc.scaling.input_map) .- Float64.(scaling.input_map)) < 1e-12 &&
        maximum(abs, vec(kc.scaling.out_scaling.mu) .- mo) < 1e-12 * maximum(so) &&
        maximum(abs, vec(kc.scaling.out_scaling.sigma) .- so) < 1e-12 * maximum(so) ||
        error("meannet $XMEAN's scaling is not the run's")
    global w = RF.with_mean_net(w, RF.mean_net_from(kc.weights, kc.spec; T = XT))
    RF.check_shapes(w, spec)
    # frozen_mean + Ws z against the step-4 closure's own deployed mean (no stochastic part) on 300 record steps
    let picks = sort(unique(rand(Xoshiro(3), findall(n -> 400 + h <= n <= 40000 - 1, cols), 300))), Pm = Float64.(scaling.input_map)
        Xreg, csr = xr(scaling.in_scaling)
        stk = RF.LSTMState(kc.spec, XT)
        em = 0.0
        for i in picks
            x = Pm * Xreg[:, cols[i] - csr[1] + 1]
            RF.cnn_window!(stk, kc.weights, kc.spec, reshape(x, :, 1))
            em = max(em, maximum(abs, RF.frozen_mean(RF.cnn_mean_net(w), x, XT) .+ Ws * x .- stk.y))
        end
        @printf("meannet %s (%s): frozen mean + Ws z against its deployed mean on %d record steps, max |diff| %.1e target sd
",
                XMEAN, kc.extras.run, length(picks), em)
        em < 1e-10 || error("the frozen mean differs from $(XMEAN)'s deployed mean by $em")
    end
end
sc = merge(scaling, extra)
dir = joinpath(XTO, "step3", "closures", XNAME)
isdir(dir) && error("$dir exists; refusing to overwrite")
file = joinpath(dir, "StochLSTM_seed$(seed).jld2")
RF.save_stochlstm(file, spec, w, sc; run = XRUN, noise = XNOISE, provenance = prov, seed, C, lr, wd,
                  loss = P["loss"], train_steps = (400, 4000), made = string(now()),
                  skip = isempty(XSKIP) ? "none" : XSKIP, precision = string(XT), innoise = string(XINN),
                  meannet = isempty(XMEAN) ? "none" : XMEAN, netskip = isempty(XNETSKIP) ? "none" : XNETSKIP,
                  hist_noise = string(get(P, "hist_noise", :recorded)))
@printf("wrote %s\n", file)

# the translation check
fit = RF.load_stochlstm(file)
"A stream that returns given standard normals (the generator's check: the fit's own deviates)."
mutable struct FixedNormals <: Random.AbstractRNG
    v::Vector{Float64}
    i::Int
end
Random.randn(r::FixedNormals) = (r.i += 1; r.v[r.i])
Random.randn(r::FixedNormals, ::Type{Float64}) = randn(r)
if XGEN
    chk = P["gen"][seed].chk
    errg = zeros(NQ, length(chk.cols))
    for (j, n) in enumerate(chk.cols)
        warm = REF["q"][:, (n - XNWARM + 1):n] .- REF["q_star"][:, (n - XNWARM):(n - 1)]
        fr = FixedNormals(chk.E[:, j], 0)
        m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = warm, rng = fr, gate = 0.0)
        for t in 1:XNWARM
            RF.get_next_item_timeseries(m, REF["q_star"][:, n - XNWARM + t - 1])
        end
        dQ = RF.get_next_item_timeseries(m, REF["q_star"][:, n])
        fr.i == NQ || error("the deployed generator drew $(fr.i) normals, not N_Q = $NQ")
        errg[:, j] = (dQ .- chk.y[:, j]) ./ so
    end
    @printf("translation check (%s, generator), %d steps of 10-100 TU, the fit's own deviates: max |deployed - Lux| / target sd %.1e, rms %.1e
",
            XT, length(chk.cols), maximum(abs, errg), sqrt(mean(abs2, errg)))
    maximum(abs, errg) < 1e-3 || error("the deployed generator differs from the trained one by $(maximum(abs, errg)) sd")
    exit()
end
rng = Xoshiro(1)
picks = sort(unique(vcat(rand(rng, findall(n -> 400 + XNWARM <= n <= 4000, cols), 50),
                         rand(rng, findall(n -> 4000 < n <= 40000 - 1, cols), 250))))
err = zeros(NQ, length(picks))
for (j, i) in enumerate(picks)
    n = cols[i]
    warm = REF["q"][:, (n - XNWARM + 1):n] .- REF["q_star"][:, (n - XNWARM):(n - 1)]   # dQ of steps n-100..n-1
    m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = warm, rng = Xoshiro(2), gate = 0.0,
                     stochastic = false)
    for t in 1:XNWARM
        RF.get_next_item_timeseries(m, REF["q_star"][:, n - XNWARM + t - 1])
    end
    dQ = RF.get_next_item_timeseries(m, REF["q_star"][:, n])
    err[:, j] = (dQ .- pred3[:, i]) ./ so
end
@printf("translation check (%s), %d steps (%d in 1-10 TU): max |deployed - Lux%s| / target sd %.1e, rms %.1e\n",
        XT, length(picks), count(i -> cols[i] <= 4000, picks), isempty(XSKIP) ? "" : " - skip", maximum(abs, err),
        sqrt(mean(abs2, err)))
# The deployed closure keeps its lag window in Float32 (`StochLSTM`'s `_push_scaled!`, by design), and the
# mapped input forms dQ^{n-k} = q^{n-k} - q^{n-k*} from it: a difference of nearly equal values, so the
# inputs carry ~1e-4 relative error that the training (Float64 map, then Float32) does not. First export
# (a1, 2026-10-23): max 8.9e-5, rms 7.5e-6 sd(dQ), against a residual of 0.05-0.6 sd. Tolerance 1e-3 sd.
# The target sd is sd(dQ) without a skip, and the sd of what the LinReg misses with one (a stricter unit);
# a skip closure runs in Float64 (header), so its check tests the translation, not the rounding.
maximum(abs, err) < 1e-3 || error("the deployed CNN differs from the trained one by $(maximum(abs, err)) sd")
