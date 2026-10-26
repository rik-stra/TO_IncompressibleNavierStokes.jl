# Step 3a closures: a CNN of paper Sec. 6 deployed on the network path (2026-10-23, Rik: "build it on the
# network path ... the AR(p) and power-law code shared with LinReg ... the export from cnn_mean_fit.jl, the
# translation check"). Writes `output/TO_LSTM/step3/closures/<name>/StochLSTM_seed<s>.jld2`, the file
# `d6_lib.jl` resolves for D6_MODEL = step3/closures/<name>, and checks it.
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/cnn_closure_export.jl \
#           <run> <noise> <name>
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
#            noise, about zero -- not the clean residual's. The other modes refuse a noise-trained run.
# The noise model is stored in the scaling (`ar_phi`, `scale_*`) and the head (`bd`, `LR`), which
# `StochLSTM` applies with the code `LinReg` uses (`ts_noise.jl`, `ts_scale.jl`).
#
# Translation check: the deployed `StochLSTM` (stochastic = false: its mean), run as D6 runs it -- a
# 100-step warm-up replaying the record, then one prediction -- at 300 steps spread over 1-100 TU, against
# the Lux network's prediction in `pred.jld2`. 🔒 Fits read 1-10 TU only.

include(joinpath(@__DIR__, "lrs_scale_variant.jl"))   # fit_ar, fit_scale (and RF, REF, NQ, DT); main guarded
using Random

length(ARGS) == 3 || error("usage: cnn_closure_export.jl <run> <noise> <name>")
const XRUN, XNOISE, XNAME = ARGS[1], ARGS[2], ARGS[3]
XNOISE in ("mvg", "head", "headcorr", "ar2n", "pl", "mvgnt") || error("noise must be mvg, head, headcorr, ar2n, pl or mvgnt; got $XNOISE")
const XTO = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
const XNWARM = 100                                   # D6's warm-up (d6_lib.jl)

F = load(joinpath(XTO, XRUN, "fits.jld2"))
P = load(joinpath(XTO, XRUN, "pred.jld2"))
key, r = only(F["res"])
C, lr, wd, seed = key
haskey(r, :refit) || error("$XRUN has no refit (run cnn_mean_fit.jl with RIKFLOW_CNN_REFIT=1)")
const XNT = get(P, "innoise", :none) !== :none        # a noise-trained run
XNT == (XNOISE == "mvgnt") || error(XNT ? "$XRUN is noise-trained: export it with noise mvgnt (its S_deploy)" :
                                          "noise mvgnt is for a noise-trained run; $XRUN is not")
(P["input"], P["seq"], P["act"], P["head"]) == (:full, :lags, :tanh, :identity) ||
    error("the deployed CNN is SEQ = lags, INPUT = full, ACT = tanh; $XRUN has $((P["input"], P["seq"], P["act"]))")
h, K, D = P["h"], P["K"], P["D"]
scaling = P["scaling"]
nin, nreg = size(scaling.input_map)
cols = P["cols"]
pred3 = only(values(P["pred3"]))                    # raw dQ of the refitted network, every step of `cols`

# the spec, and the sequence the network reads from the mapped input (cnn_mean_fit.jl's `lagtensor`, from
# step3_data.jl's `lagrows`: rows of dQ^{n-k} and q*^{n-k}; the increments after the regressor)
hist = RF.HistorySpec(; h, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
RF.nfeatures(hist) == nreg || error("regressor width $(RF.nfeatures(hist)), the input map reads $nreg")
spec = RF.LSTMSpec(; hist, n_hidden = D, n_latent = 0, n_encoder = 0, arch = :cnn, emission = :constant,
                   window = 1, n_in = nin, n_conv = C, kernel = K)
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
    prov = "S_deploy of the noise-trained run: its 1-10 TU residual under its own window noise (Sigma_inj = " *
           "the run's input-noise Sigma), about zero"
    @printf("  S_deploy sd / clean residual rms %s | / injected sd %s\n", fmt(sqrt.(diag(Sd) ./ diag(S0)); d = 3),
            fmt(sqrt.(diag(Sd) ./ diag(P["Sigma_inj"][seed])); d = 3))
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
w = RF.cnn_weights(r.refit.ps.net, spec; seq_index, q_index = 1:NQ, bd, LR)
sc = merge(scaling, extra)
dir = joinpath(XTO, "step3", "closures", XNAME)
isdir(dir) && error("$dir exists; refusing to overwrite")
file = joinpath(dir, "StochLSTM_seed$(seed).jld2")
RF.save_stochlstm(file, spec, w, sc; run = XRUN, noise = XNOISE, provenance = prov, seed, C, lr, wd,
                  loss = P["loss"], train_steps = (400, 4000), made = string(now()))
@printf("wrote %s\n", file)

# the translation check
fit = RF.load_stochlstm(file)
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
@printf("translation check, %d steps (%d in 1-10 TU): max |deployed - Lux| / sd(dQ) %.1e, rms %.1e\n",
        length(picks), count(i -> cols[i] <= 4000, picks), maximum(abs, err), sqrt(mean(abs2, err)))
# The deployed closure keeps its lag window in Float32 (`StochLSTM`'s `_push_scaled!`, by design), and the
# mapped input forms dQ^{n-k} = q^{n-k} - q^{n-k*} from it: a difference of nearly equal values, so the
# inputs carry ~1e-4 relative error that the training (Float64 map, then Float32) does not. First export
# (a1, 2026-10-23): max 8.9e-5, rms 7.5e-6 sd(dQ), against a residual of 0.05-0.6 sd. Tolerance 1e-3 sd.
maximum(abs, err) < 1e-3 || error("the deployed CNN differs from the trained one by $(maximum(abs, err)) sd")
