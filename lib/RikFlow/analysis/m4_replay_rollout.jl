# Does a rollout with REPLAYED q* expose the closed-loop bias? (Rik, 2026-09-25)
#
#     julia --project=lib/RikFlow/training lib/RikFlow/analysis/m4_replay_rollout.jl fitdir...
#
# The deployed closure pushes its own `q = q* + dQhat` into its history, so running it over the
# record with the RECORDED q* supplied each step is exactly the replayed-q* rollout: the level lags
# are the model's own, the predictor is the record's and does not respond. Started at 50 TU after the
# usual 100-step recorded warm-up, run for 50 TU. Compared with teacher forcing on the same rows:
#   - the one-step loss 0.5 x SSE per step on the standardised dQ (teacher-forced = the fit's own
#     `diag.held`, recomputed here on the same rows with the recorded lags);
#   - the mean error of dQhat - dQ per QoI, in dQ sd (does it carry the ONLINE bias's sign?);
#   - the mean offset of the rollout's q = q* + dQhat from the record, in q sd.
# `mean` runs switch the model's noise off (a zero RNG: latent and emission draws are 0); three seeds
# run it stochastic.

using RikFlow, JLD2, Statistics, Printf, Random
const RF = RikFlow

"An RNG whose normal draws are all zero: the closure then deploys its mean."
struct ZeroRNG <: Random.AbstractRNG end
Random.randn(::ZeroRNG) = 0.0
Random.randn(::ZeroRNG, ::Type{T}) where {T} = zero(T)

TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
ref = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
qr, qsr, dQr = ref["q"], ref["q_star"], ref["dQ"]
nq = size(qr, 1)
const S0 = 20_000                   # 50 TU
const NW = 100
const N = 19_800                     # ~49.5 TU of rollout (the record ends at 40 000)
cols = (S0 + NW):(S0 + NW + N - 1)  # predicted steps (record columns of q_star / dQ)

function run(fit, rng; teacher = false)
    m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = dQr[:, S0:(S0 + NW - 1)],
                     rng, gate = 0.0)
    for n in S0:(S0 + NW - 1)
        RF.get_next_item_timeseries(m, qsr[:, n])
    end
    D = zeros(nq, N)
    for (j, n) in enumerate(cols)
        d = RF.get_next_item_timeseries(m, qsr[:, n])
        D[:, j] = d
        if teacher
            # overwrite what the closure pushed with the RECORDED level: teacher forcing
            # (the newest history column is what the closure just pushed, q* + dQhat)
            if fit.spec.hist.h > 0
                m.buf.q[:, 1] .= Float32.(vec(RF.scale_input(qr[:, n + 1], fit.scaling.in_scaling)))
            end
        end
    end
    return D
end

osc(fit) = fit.scaling.out_scaling
sdD = vec(std(dQr[:, cols]; dims = 2)); sdq = vec(std(qr; dims = 2))
@printf("%-20s %-10s %8s  %-44s %s\n", "fit", "run", "0.5SSE", "mean(dQhat - dQ) / sd(dQ) per QoI", "mean(q - q_rec) / sd(q)")
for d in ARGS
    fit = RF.load_stochlstm(joinpath(TO, d, "StochLSTM_seed1.jld2"))
    for (lab, rng, tf) in (("TF mean", ZeroRNG(), true), ("RO mean", ZeroRNG(), false),
                           ("RO s1", Xoshiro(1), false), ("RO s2", Xoshiro(2), false))
        D = run(fit, rng; teacher = tf)
        Z = RF.scale_input(D, osc(fit)) .- RF.scale_input(dQr[:, cols], osc(fit))
        sse = 0.5 * sum(abs2, Z) / N
        be = vec(mean(D .- dQr[:, cols]; dims = 2)) ./ sdD
        qro = qsr[:, cols] .+ D
        qo = vec(mean(qro .- qr[:, cols .+ 1]; dims = 2)) ./ sdq
        @printf("%-20s %-10s %8.3f  %s   %s\n", basename(d), lab, sse, join((@sprintf("%6.2f", x) for x in be), ""),
                join((@sprintf("%6.2f", x) for x in qo), ""))
    end
end
