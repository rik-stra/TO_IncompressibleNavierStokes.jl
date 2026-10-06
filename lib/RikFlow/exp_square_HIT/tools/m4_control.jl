# The CONTROL run's model: a deployed LinReg (default LinReg1) written as an M4 fit, so it runs
# through the second implementation (`StochLSTM`, `D6_CLOSURE=lstm`, `12_online_StochLSTM.jl`)
# unchanged. Paper: `closures.tex`, "control run"; `todo.md` R2; WORKFLOW.md closure #0.
#
#     julia --startup-file=no --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/tools/m4_control.jl [LinReg1]
#
# Writes `output/TO_LSTM/diag/control_<LinReg>/StochLSTM_seed1.jld2`. Deploy it as any diag fit:
# `RIKFLOW_M4_MODEL_DIR=<that dir> ... 12_online_StochLSTM.jl 2 [replica]` (cell 2, `StochLSTM2`,
# whose row supplies n_replicas and the seeds) or `D6_CLOSURE=lstm D6_MODEL=<file>`.
#
# The model, and nothing fitted here:
#   - mean: the LinReg's map, translated EXACTLY into this path's regressor and scaled-dQ target.
#     Both are affine in the same raw regressor, so the translation is closed-form (no refit; a
#     refit on 80 % of the rows is what `m4_linear_eta.jl` did, and it is a different model --
#     coefficients 83 % apart at h = 5, results_LSTMS.md R2). Asserted on every row of the record.
#   - LSTM off: `arch = :lstm`, `V1 = 0`, so the recurrence never reaches the output.
#   - noise: a `:constant` head equal to the LinReg's own MVG covariance (fitted on 1-10 TU),
#     translated into the scaled-dQ units. Its mean is zero; the MVG's is reported (it is round-off).
#   - weights stored in Float32, the precision every M4 fit deploys at.
#
# 🔑 GATE 2 is run here, after the file is written, on the file as the driver loads it: the deployed
# closure (`StochLSTM`, `stochastic = false`) is replayed for 100 recorded steps from many starts and
# its first prediction compared with the LinReg's mean on the same history. Pass: rms < 1e-3 sd(dQ)
# in every QoI (Float32 round-off is ~1e-4; the 80 %-window refit differs by 5e-3 to 5e-2).

using RikFlow
using Lux, Optimisers, Zygote          # the Lux extension: `init_lstm_params`, `LSTMWeights(ps, spec)`
using JLD2, Printf, Statistics, LinearAlgebra, Random
const RF = RikFlow
Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")
include(joinpath(@__DIR__, "m4_data.jl"))

src = isempty(ARGS) ? "LinReg1" : ARGS[1]
TO = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
lrs_file = normpath(joinpath(TO, "..", "TO_LRS", src, "LinReg.jld2"))
lrs = load(lrs_file)
base = load(joinpath(TO, "inputs_lstm.jld2"), "inputs")[2]
cfg = merge(base, (; arch = :lstm, emission = :constant, h = lrs["hist_len"], hist_var = lrs["hist_var"],
                   include_predictor = lrs["include_predictor"], beta = 0.0))
cfg.train_range == (400, 4000) || error("inputs_lstm.jld2 cell 2 has train_range $(cfg.train_range), expected (400, 4000)")
rec = load_m4_qois(; track_file = joinpath(TO, "..", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
hist = RF.HistorySpec(; h = cfg.h, n_qoi = size(rec.q, 1), cfg.hist_var, cfg.include_predictor)
hist.include_predictor || error("the translation needs q*^n in the regressor (include_predictor)")
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, arch = :lstm, cfg.uclip,
                   emission = :constant, skip = true)
dat = m4_training_data(rec, cfg, hist; target = :dQ)
nq, nin = RF.n_output(spec), RF.n_input(spec)

# --- the mean: y = Ws x, x this path's scaled regressor (bias last), y the scaled dQ ----------------
mi, si = vec(dat.scaling.in_scaling.mu), vec(dat.scaling.in_scaling.sigma)
ml, sl = vec(lrs["scaling"].in_scaling.mu), vec(lrs["scaling"].in_scaling.sigma)
mo, so = vec(lrs["scaling"].out_scaling.mu), vec(lrs["scaling"].out_scaling.sigma)
md, sd_ = vec(dat.scaling.out_scaling.mu), vec(dat.scaling.out_scaling.sigma)
c = Matrix{Float64}(lrs["c"])                       # nq x nin, the LinReg's scaled level map
size(c) == (nq, nin) || error("$src: c is $(size(c)), expected $((nq, nin))")
lrs["fitted_qois"] == 1:nq || collect(lrs["fitted_qois"]) == collect(1:nq) || error("$src fits a subset of QoIs")
# x_lrs = M x: feature j carries QoI mod1(j, nq) (blocks of nq in QoI order, as `build_history`)
M = zeros(nin, nin); M[nin, nin] = 1
for j in 1:(nin - 1)
    i = mod1(j, nq)
    M[j, j] = si[i] / sl[i]
    M[j, nin] = (mi[i] - ml[i]) / sl[i]
end
S = zeros(nq, nin)                                   # raw q*^n = S x (the first nq features)
for i in 1:nq
    S[i, i] = si[i]; S[i, nin] = mi[i]
end
E = zeros(1, nin); E[nin] = 1
# raw dQhat = diag(so) c M x + mo - q*;  y = (dQhat - md) ./ sd_
Ws = Diagonal(1 ./ sd_) * (Diagonal(so) * c * M .- S .+ (mo .- md) * E)

# asserted on every row of the record (not only the training rows): Ws against the LinReg evaluated
# in its own scaling, through the raw units
b = size(rec.q_star, 2)
Xm, _, stp = RF.build_history(hist, RF.scale_input(rec.q_star, dat.scaling.in_scaling),
                              RF.scale_input(rec.q, dat.scaling.in_scaling))
Xl, _, _ = RF.build_history(hist, RF.scale_input(rec.q_star, lrs["scaling"].in_scaling),
                            RF.scale_input(rec.q, lrs["scaling"].in_scaling))
dq_lrs = (c * Xl') .* so .+ mo .- rec.q_star[:, stp]
dq_ws = (Ws * Xm') .* sd_ .+ md
tres = maximum(abs.(dq_ws .- dq_lrs) ./ vec(std(rec.dQ; dims = 2)))
tres < 1e-9 || error("translation residual $tres sd(dQ): the two regressors are not affine images")

# --- the noise: the LinReg's MVG covariance in scaled-dQ units ---------------------------------------
sdist = lrs["stoch_distr"]
Sig_l = Matrix{Float64}(hasproperty(sdist.Σ, :mat) ? sdist.Σ.mat : sdist.Σ)
mu_l = Vector{Float64}(sdist.μ)
# cross-check against the MLE covariance of the LinReg's residuals on its training rows (what
# `5_train_LinReg.jl` fits), so the object read above is the one that is deployed
a0, b0 = cfg.train_range
Xt, Yt, _ = RF.build_history(hist, RF.scale_input(rec.q_star[:, a0:(b0 - 1)], lrs["scaling"].in_scaling),
                             RF.scale_input(rec.q[:, a0:b0], lrs["scaling"].in_scaling))
Rt = Yt' .- c * Xt'
Sig_chk = (Rt .- mean(Rt; dims = 2)) * (Rt .- mean(Rt; dims = 2))' ./ size(Rt, 2)
schk = maximum(abs.(Sig_chk .- Sig_l)) / maximum(abs.(Sig_l))
schk < 1e-8 || error("$src's stoch_distr is not the MLE covariance of its training residuals (rel $schk)")
G = Diagonal(so ./ sd_)
Sig_d = Symmetric(G * Sig_l * G)
sdv = sqrt.(diag(Sig_d))
P = inv(Symmetric(Matrix(Sig_d) ./ (sdv * sdv')))
J = reverse(Matrix{Float64}(I, nq, nq); dims = 1)
A = J * cholesky(Symmetric(J * P * J)).U * J          # lower triangular, A'A = R^{-1}
Araw = tril(A, -1) + Diagonal(log.(diag(A)))

p0 = RF.init_lstm_params(Xoshiro(1), spec)
ps = merge(p0, (; Ws = Float32.(Ws), V1 = zero(p0.V1), cdec = zero(p0.cdec), Wd = zero(p0.Wd),
                bd = Float32.(log.(sdv)), Araw = Float32.(Araw)))
tag = "control_$src"
out = joinpath(TO, "diag", tag, "StochLSTM_seed1.jld2")
RF.save_stochlstm(out, spec, RF.LSTMWeights(ps, spec), dat.scaling;
                  cfg, seed = 1, train_range = cfg.train_range, qoi_source = rec.source, target = :dQ,
                  lambda = 0.0, val = (; tf = NaN, ro = NaN, Kx = 0), ps,
                  control = (; from = src, lrs_file, translation_residual = tres,
                             mvg_mean_scaled = mu_l, built = string(Base.Libc.strftime("%Y-%m-%d", time()))))
@printf("wrote %s\n  translation residual (all %d rows) %.2e sd(dQ) | MVG cross-check %.1e | MVG mean max %.1e (scaled)\n",
        out, size(Xm, 1), tres, schk, maximum(abs, mu_l))

# --- GATE 2 on the file as deployed -------------------------------------------------------------------
fit = RF.load_stochlstm(out)
w = fit.weights
Sig_dep = Diagonal(exp.(Float64.(w.bd))) * (Float64.(w.LR) * Float64.(w.LR)') * Diagonal(exp.(Float64.(w.bd)))
sig_err = maximum(abs.(Sig_dep .- Sig_d) ./ (sdv * sdv'))
nwarm = 100
sddq = vec(std(rec.dQ[:, a0:b0]; dims = 2))
names = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
fmt(x) = join((@sprintf("%9.1e", v) for v in x), "")
ok = sig_err < 1e-5
@printf("GATE 2  Sigma: max |Sigma_dep - Sigma_LinReg| / (sd sd') = %.1e  (pass < 1e-5)\n", sig_err)
println("        one-step mean, deployed StochLSTM (Float32) minus the LinReg's mean (Float64), in sd(dQ):")
println("        ", rpad("", 18), join((lpad(n, 9) for n in names), ""))
for (lab, lo, hi) in (("1-10 TU", 1.0, 10.0), ("10-100 TU", 10.0, 99.0))
    starts = round.(Int, range(lo / 2.5e-3, hi / 2.5e-3 - nwarm - 1; length = 400))
    D = zeros(nq, length(starts))
    for (r, s) in enumerate(starts)
        m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = rec.dQ[:, s:(s + nwarm - 1)],
                         rng = Xoshiro(1), stochastic = false)
        for n in s:(s + nwarm - 1)
            RF.get_next_item_timeseries(m, rec.q_star[:, n])
        end
        n = s + nwarm
        dqm = RF.get_next_item_timeseries(m, rec.q_star[:, n])
        # the LinReg's mean on the SAME history the closure holds: q^{n-k} = q*^{n-k} + dQ^{n-k}
        x = zeros(nin); x[1:nq] .= rec.q_star[:, n]; x[nin] = 1
        for k in 1:hist.h
            x[(2k - 1) * nq .+ (1:nq)] .= rec.q_star[:, n - k] .+ rec.dQ[:, n - k]
            x[2k * nq .+ (1:nq)] .= rec.q_star[:, n - k]
        end
        xl = vcat(((x[1:(nin - 1)] .- ml[mod1.(1:(nin - 1), nq)]) ./ sl[mod1.(1:(nin - 1), nq)]), 1.0)
        dql = (c * xl) .* so .+ mo .- rec.q_star[:, n]
        D[:, r] .= (dqm .- dql) ./ sddq
    end
    rms = vec(sqrt.(mean(abs2, D; dims = 2)))
    global ok &= all(rms .< 1e-3)
    @printf("        %-9s mean %s\n        %-9s rms  %s\n        %-9s max  %s\n", lab, fmt(vec(mean(D; dims = 2))),
            "", fmt(rms), "", fmt(vec(maximum(abs, D; dims = 2))))
end
println(ok ? "GATE 2 PASS" : "GATE 2 FAIL")
ok || exit(1)
