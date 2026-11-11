# Step 5 (card analysis/cards/step5.md, "reported beside the check": the response to its own draws and the error on
# noisy inputs, as Table tab: own-draws; Rik, 2026-11-11: "Yes, run it"). No fit, no Lux: the EXPORTED closure (plain
# arrays, `cnn_window!`, Float64) and its skip + MVG (LinReg1 by default), each along a path of its OWN draws on 10-100 TU.
#
#     RIKFLOW_MLP_SKIP=LinReg1 RIKFLOW_MLP_SKIP_CENTER=mvg \
#         julia --startup-file=no --project=. exp_square_HIT/tools/gen_owndraws.jl [closure name] [paths]
#
# (from lib/RikFlow; default closure LinReg1_CNN_gen under output/TO_LSTM/step3/closures, 3 paths). Per record step m one
# sample on the RECORDED history; its noise η^m = the sample minus the closure's mean there (16 draws; LinReg1 + MVG:
# η^m = L0 ε^m with the same ε, L0 the closure's starting noise skip = LinReg1's Cholesky factor). The path enters every
# later window through the anchored window noise (`window_noise`, step3_innoise.jl): the history then carries the
# closure's own draws, as online. Per QoI, in units of LinReg1's clean rms on 10-100 TU (the training target, LinReg1's
# residual about its MVG mean): the error of the mean on the recorded history, on the noisy history, and the drawn
# noise's sd; the noisy error over the noise sd (1 = the residual met online is the noise drawn); and the response
# rms(noisy mean - clean mean) / noise sd (the skip's response included for both).

using RikFlow
using JLD2, Printf, Statistics, LinearAlgebra, Random
const RF = RikFlow
include(joinpath(@__DIR__, "step3_data.jl"))
include(joinpath(@__DIR__, "step3_innoise.jl"))
# any skip since 2026-11-11 (card step5_skips.md: LinReg1_nt_CNN_gen with RIKFLOW_MLP_SKIP=LinReg1_nt, LinReg7_CNN_gen
# with LinReg7); the closure's own skip map is checked against SKIP below
isempty(SKIP) && error("set RIKFLOW_MLP_SKIP to the closure's skip (and RIKFLOW_MLP_SKIP_CENTER=mvg)")
const NAME = length(ARGS) >= 1 ? ARGS[1] : "LinReg1_CNN_gen"
const NREP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const MM = 16
fit = RF.load_stochlstm(joinpath(TO_folder, "step3", "closures", NAME, "StochLSTM_seed1.jld2"))
sp, w = fit.spec, fit.weights
RF.is_generator(sp) || error("$NAME is not a step-5 generator")
# B (card step5_skips.md): a generator carrying a step-4 closure's frozen CNN mean m (`Cv.mean_net`). Its comparator is
# that step-4 closure (m + Ws z with its own Gaussian, the headcorrnt Σ = D R D of its file), in its frame
# (RIKFLOW_MLP_SKIP_CENTER=data); the unit is that closure's clean rms. Otherwise the comparator is the skip + MVG.
const MN = RF.cnn_mean_net(w)
const KC = MN === nothing ? nothing : let kd = joinpath(TO_folder, "step3", "closures", String(fit.extras.meannet))
    RF.load_stochlstm(joinpath(kd, only(filter(f -> startswith(f, "StochLSTM_seed"), readdir(kd)))))
end
strip(get(ENV, "RIKFLOW_MLP_SKIP_CENTER", "data")) == (MN === nothing ? "mvg" : "data") ||
    error("set RIKFLOW_MLP_SKIP_CENTER=" * (MN === nothing ? "mvg" : "data (a B generator: the step-4 closure's frame)"))
Pm = Float64.(fit.scaling.input_map)
maximum(abs, Pm .- P) < 1e-10 * maximum(abs, P) || error("the closure's input map is not step3_data.jl's")
maximum(abs, vec(fit.scaling.out_scaling.sigma) .- so) < 1e-10 * maximum(so) &&
    maximum(abs, vec(fit.scaling.out_scaling.mu) .- mo) < 1e-12 * maximum(abs, so) ||
    error("the closure's target scaling is not step3_data.jl's (SKIP_CENTER = mvg?)")
# the skip + MVG's draw in the training frame: diag(sigma_out / so) chol(Sigma) eps (as cnn_mean_fit.jl's GEN_L0)
const L0LR = KC !== nothing ? let sd = exp.(Float64.(KC.weights.bd)), R = Float64.(KC.weights.LR) * Float64.(KC.weights.LR)'
    Matrix(cholesky(Symmetric(Diagonal(sd) * R * Diagonal(sd))).L)        # the step-4 closure's noise (target units)
end : let ds = load(joinpath(TO_folder, "..", "TO_LRS", SKIP, "LinReg.jld2"), "stoch_distr", "scaling")
    d, sc_ = ds
    sgo = vec(sc_.out_scaling.sigma)
    S = Matrix{Float64}(hasproperty(d.Σ, :mat) ? d.Σ.mat : d.Σ)
    Matrix(cholesky(Symmetric(Diagonal(sgo ./ so) * S * Diagonal(sgo ./ so))).L)
end
all(diff(cols) .== 1) || error("the record columns are not contiguous")
# the closure's frozen skip is SKIP: Ws z = the skip's predicted correction / so on the record
let c = cols[(cfg.h + 1):50:end], b = (cfg.h + 1):50:length(cols)
    e = maximum(abs, w.Ws * Xf[:, b] .- skipq(c) ./ so) / maximum(abs, skipq(c) ./ so)
    e < 1e-8 || error("$NAME's skip map is not $(SKIP)'s (max rel. diff $e)")
end
N = length(cols)
ho = HO[HO .> cfg.h]
"B: the frozen mean m on mapped inputs Z (nin x N), target units; zero otherwise."
fmean(Z) = MN === nothing ? zeros(nq, size(Z, 2)) : reduce(hcat, [RF.frozen_mean(MN, view(Z, :, b), Float64) for b in axes(Z, 2)])
const MC = fmean(Xf)                                                # the comparator's clean mean (relative to the skip)
const CLAB = MN === nothing ? SKIP * " + MVG" : String(fit.extras.meannet)
unit = rmsq(Yf[:, ho] .- MC[:, ho])
fmt3(v) = join((@sprintf("%.3f", x) for x in v), " ")

st = RF.LSTMState(sp, Float64)
"The generator's sample minus the skip (training frame, target units) for mapped input z and deviates e."
function gsample(z, e)
    RF.cnn_window!(st, w, sp, reshape(z, :, 1); eps = e)
    return st.y .- w.Ws * z
end
"Mean over `M` draws of rng (training frame) at every column of the mapped inputs Z (nin x N)."
function gmean(Z, M, rng)
    mu = zeros(nq, size(Z, 2))
    for b in axes(Z, 2), _ in 1:M
        mu[:, b] .+= gsample(view(Z, :, b), randn(rng, nq)) ./ M
    end
    return mu
end
"Window noise (raw, nq x h x N) of a path of noises `et` (target units; column b = record column cols[b])."
function path_eta(et)
    eta = zeros(nq, cfg.h, N)
    for b in (cfg.h + 1):N, j in 1:cfg.h
        eta[:, j, b] = so .* et[:, b - j]
    end
    return eta
end

t0 = time()
mu_c = gmean(Xf, MM, Xoshiro(31))                                  # the generator's mean on the recorded history
@printf("clean mean: %.1f min\n", (time() - t0) / 60); flush(stdout)
acc = Dict(k => zeros(nq, 4) for k in (CLAB, "generator"))
for rep in 1:NREP
    E = randn(Xoshiro(4000 + rep), nq, N)
    et_g = reduce(hcat, [gsample(view(Xf, :, b), E[:, b]) for b in 1:N]) .- mu_c
    et_l = L0LR * E
    for (lab, et) in (("generator", et_g), (CLAB, et_l))
        dx = window_noise(path_eta(et))
        sk = skip_dy(dx)                                            # the skip's response (target units)
        if lab == "generator"
            mu_n = gmean(Xf .+ P * dx, MM, Xoshiro(5000 + rep)) .+ sk
            mc = mu_c
        else
            mu_n, mc = sk .+ fmean(Xf .+ P * dx), MC                # B: the step-4 closure on the noisy history
        end
        acc[lab] .+= hcat(rmsq(Yf[:, ho] .- mc[:, ho]) .^ 2, rmsq(Yf[:, ho] .- mu_n[:, ho]) .^ 2,
                          rmsq(et[:, ho]) .^ 2, rmsq(mu_n[:, ho] .- mc[:, ho]) .^ 2) ./ NREP
    end
    @printf("path %d of %d done (%.1f min)\n", rep, NREP, (time() - t0) / 60); flush(stdout)
end
@printf("\nOWNDRAWS %s: 10-100 TU (%d steps), %d paths, means of %d draws; per QoI Z[0,6] E[0,6] Z[7,15] E[7,15] Z[16,32] E[16,32]\n",
        NAME, length(ho), NREP, MM)
@printf("  unit: %s's clean rms on 10-100 TU (target units) %s\n", SKIP, fmt3(unit))
for lab in (CLAB, "generator")
    a = sqrt.(acc[lab])
    @printf("  %-18s error, recorded history %s | error, own draws in the history %s | drawn noise sd %s\n", lab,
            fmt3(a[:, 1] ./ unit), fmt3(a[:, 2] ./ unit), fmt3(a[:, 3] ./ unit))
    @printf("  %-18s noisy error / drawn noise (target 1) %s | response rms(noisy - clean mean) / noise sd %s\n", "",
            fmt3(a[:, 2] ./ a[:, 3]), fmt3(a[:, 4] ./ a[:, 3]))
end
