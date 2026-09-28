# Linear-mean + eta models on the `dQ` target, ridge-fitted in closed form and written as deployable
# M4 fits -- M0's structure in the M4 code path (`results_LSTMS.md` §7d).
#
#     RH=<h> LAMS=0,1e-5,... julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/tools/m4_linear_eta.jl
#
# Each lambda gives `output/TO_LSTM/diag/rdg_h<h>_l<lambda>/StochLSTM_seed1.jld2`: arch `:lstm`, linear
# skip `Ws` = the ridge map, `V1 = 0` (the recurrence contributes nothing), `:constant` emission seeded
# at the fit's own training-residual covariance. Deploy with `RIKFLOW_M4_MODEL_DIR` like any M4 fit.
# 🔑 The penalty is `lambda * N` on every coefficient but the bias, in the standardised design; the
# online bias crosses zero near lambda = 1e-5 at h = 1 (§7d point 5).
#
# Options (2026-09-28, plan step 1; all default to the old behaviour):
#   TRAIN_TU=50        end of the training range in TU (default: inputs_lstm.jld2's, 1-10 TU)
#   SCORE_TU=52,74     held-out window in TU (default 50 TU to the end). 🔴 Plan step 1: never past 74
#   OUTSUB=p4          subdirectory of output/TO_LSTM (default `diag`)
#   TAGPFX=m0          the fit's name prefix (default `rdg`), giving <TAGPFX>_h<h>_l<lambda>

using RikFlow, Lux, Optimisers, Zygote, JLD2, Printf, Statistics, LinearAlgebra, Random
const RF = RikFlow
include(joinpath(@__DIR__, "m4_data.jl"))
TO = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
base = load(joinpath(TO, "inputs_lstm.jld2"), "inputs")[2]
rec = load_m4_qois(; track_file = "none")
h = parse(Int, get(ENV, "RH", "1"))
cfg = merge(base, (; arch = :lstm, emission = :constant, h, beta = 0.0))
haskey(ENV, "TRAIN_TU") &&
    (cfg = merge(cfg, (; train_range = (base.train_range[1], round(Int, parse(Float64, ENV["TRAIN_TU"]) / 2.5e-3)))))
outsub, tagpfx = get(ENV, "OUTSUB", "diag"), get(ENV, "TAGPFX", "rdg")
sw = get(ENV, "SCORE_TU", "")
score_lo, score_hi = isempty(sw) ? (50.0, Inf) : Tuple(parse.(Float64, split(sw, ",")))
score_hi > 74 && @warn "held-out window reaches past 74 TU (confirmation block under the 2026-09-28 partition)"
hist = RF.HistorySpec(; h, n_qoi = 6, cfg.hist_var, cfg.include_predictor)
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, arch = :lstm, cfg.uclip, emission = :constant, skip = true)
dat = m4_training_data(rec, cfg, hist; target = :dQ)
ntr = floor(Int, 0.8 * size(dat.Xc, 2))
X, Y = Float64.(dat.Xc[:, 1:ntr]), Float64.(dat.Yc[:, 1:ntr])
# held-out 50-100 TU, as m4_diag_fit
b = size(rec.q_star, 2); a = 400
qs = RF.scale_input(rec.q[:, a:(b + 1)], dat.scaling.in_scaling); qss = RF.scale_input(rec.q_star[:, a:b], dat.scaling.in_scaling)
Xf, _, st = RF.build_history(hist, qss, qs); cols = (a - 1) .+ st
Yf = RF.scale_input(rec.dQ[:, cols], dat.scaling.out_scaling); Xf = permutedims(Xf)
H = findall(c -> score_lo / 2.5e-3 <= c <= score_hi / 2.5e-3, cols)
first(cols[H]) > cfg.train_range[2] || error("held-out window overlaps the training range")
@info "windows (TU)" train = (cfg.train_range[1] * 2.5e-3, cfg.train_range[2] * 2.5e-3) fit_rows = ntr held = (cols[H[1]] * 2.5e-3, cols[H[end]] * 2.5e-3)
nin = size(X, 1)
for lam in parse.(Float64, split(get(ENV, "LAMS", "0,1e-3,1e-2,1e-1,1,10"), ","))
    # ridge in the standardised design, per row normalised: lambda * N * I, the bias column unpenalised
    P = Diagonal([fill(lam * ntr, nin - 1); 0.0])
    C = (X * X' + P) \ (X * Y')                         # n_in x N_Q
    R = Y .- C' * X
    rh = Yf[:, H] .- C' * Xf[:, H]
    sd = vec(std(R; dims = 2)); Pr = inv(Symmetric(cor(permutedims(R))))
    J = reverse(Matrix{Float64}(I, 6, 6); dims = 1); A = J * cholesky(Symmetric(J * Pr * J)).U * J
    p0 = RF.init_lstm_params(Xoshiro(1), spec)
    ps = merge(p0, (; Ws = Float32.(permutedims(C)), V1 = zero(p0.V1), bd = Float32.(log.(sd)),
                    Araw = Float32.(tril(A, -1) + Diagonal(log.(diag(A))))))
    tag = @sprintf("%s_h%d_l%g", tagpfx, h, lam)
    RF.save_stochlstm(joinpath(TO, outsub, tag, "StochLSTM_seed1.jld2"), spec, RF.LSTMWeights(ps, spec), dat.scaling;
                      cfg, seed = 1, train_range = cfg.train_range, qoi_source = rec.source, target = :dQ, lambda = lam,
                      val = (; tf = NaN, ro = NaN, Kx = 0), ps)
    @printf("%-14s held mean %.3f | train %.3f | max|C| %6.1f | resid sd %s\n", tag, 0.5 * sum(abs2, rh) / size(rh, 2),
            0.5 * sum(abs2, R) / size(R, 2), maximum(abs, C[1:(end - 1), :]), join((@sprintf("%.3f", x) for x in sd), " "))
end
