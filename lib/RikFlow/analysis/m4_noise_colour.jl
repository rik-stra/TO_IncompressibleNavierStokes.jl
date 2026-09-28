# Does a closure's noise have the COLOUR the data's residual has? (Rik, 2026-09-27)
#
#     julia --project=lib/RikFlow/training lib/RikFlow/analysis/m4_noise_colour.jl fitdir...
#
# The deployed closure is run TEACHER-FORCED over held-out 50-100 TU (recorded q* supplied, and the
# level it pushes overwritten with the recorded one, so every run sees the same inputs) with K noise
# seeds. Per step: the ensemble mean m_n = mean_k dQhat_n^(k); the model's NOISE e_n^(k) = dQhat_n^(k)
# - m_n; the data's RESIDUAL r_n = dQ_n - m_n. The residual's autocorrelation is the colour the data
# asks for, the noise's the colour the model produces. Printed per QoI at lags 1, 2, 5, 10, 20, plus
# the noise-to-residual sd ratio (1 = calibrated one-step spread).
#   - linear + eta (white MVG): noise ACF ~ 0 at every lag by construction.
#   - window CVAE, tied latent: consecutive predictions share W-1 draws -> a learned moving average.
#   - `tie_noise = false` ablation (RIKFLOW_COLOUR_UNTIED=1): white by construction.

using RikFlow, JLD2, Statistics, Printf, Random
const RF = RikFlow
const K = parse(Int, get(ENV, "RIKFLOW_COLOUR_K", "20"))
const UNTIED = get(ENV, "RIKFLOW_COLOUR_UNTIED", "0") == "1"
TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
ref = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
qr, qsr, dQr = ref["q"], ref["q_star"], ref["dQ"]
nq = size(qr, 1)
# window: held-out 50-100 TU by default; `RIKFLOW_COLOUR_TRAIN=1` uses the training rows 400-3999
# instead -- the only window a coloured-noise PARAMETER may be fitted on
const TRAIN = get(ENV, "RIKFLOW_COLOUR_TRAIN", "0") == "1"
const S0, NW, N = TRAIN ? (400, 100, 3399) : (20_000, 100, 19_800)
cols = (S0 + NW):(S0 + NW + N - 1)

function run(fit, seed)
    m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = dQr[:, S0:(S0 + NW - 1)],
                     rng = Xoshiro(seed), gate = 0.0, tie_noise = !UNTIED)
    for n in S0:(S0 + NW - 1)
        RF.get_next_item_timeseries(m, qsr[:, n])
    end
    D = zeros(nq, N)
    for (j, n) in enumerate(cols)
        D[:, j] = RF.get_next_item_timeseries(m, qsr[:, n])
        fit.spec.hist.h > 0 && (m.buf.q[:, 1] .= Float32.(vec(RF.scale_input(qr[:, n + 1], fit.scaling.in_scaling))))
    end
    return D
end
acf(x, L) = (y = x .- mean(x); v = sum(abs2, y); [sum(y[1:(end - l)] .* y[(1 + l):end]) / v for l in L])
const LAGS = (1, 2, 5, 10, 20)
names = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
for d in ARGS
    fit = RF.load_stochlstm(joinpath(TO, d, "StochLSTM_seed1.jld2"))
    Ds = [run(fit, 1000 + k) for k in 1:K]
    M = sum(Ds) ./ K
    R = dQr[:, cols] .- M
    @printf("\n%s%s%s  (arch %s, posterior %s, window %d, K = %d)\n", d, UNTIED ? " [UNTIED]" : "", TRAIN ? " [TRAINING 1-10 TU]" : "", fit.spec.arch,
            fit.spec.posterior, fit.spec.window, K)
    @printf("  %-9s %-34s %-34s %s\n", "QoI", "residual ACF at lags 1 2 5 10 20", "noise ACF at lags 1 2 5 10 20", "sd(noise)/sd(resid)")
    for q in 1:nq
        ra = acf(R[q, :], LAGS)
        na = mean(acf(Ds[k][q, :] .- M[q, :], LAGS) for k in 1:K)
        sr = mean(std(Ds[k][q, :] .- M[q, :]) for k in 1:K) / std(R[q, :])
        @printf("  %-9s %s   %s   %.2f\n", names[q], join((@sprintf("%6.2f", x) for x in ra), ""),
                join((@sprintf("%6.2f", x) for x in na), ""), sr)
    end
    flush(stdout)
end
