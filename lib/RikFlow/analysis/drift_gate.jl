# Drift gate (mechanism test 3, card analysis/cards/step5_reach_k1a.md; Rik, 2026-11-13: "do all of them"; the critical
# reviews meta_files/step5_critical_review_2026-11-13.md D2 and step4_critical_review_2026-11-13.md): the signed mean push
# E_ε g of a generator that carries a frozen step-4 mean (B and the reach-k1a variants), against that mean (= k1a's), per
# window, from FRESH antithetic deviate pairs per window (the linear ε-response cancels exactly; one fixed set reused across
# windows is biased, step-4 review), on two state sets:
#   - the recorded track, 10–100 TU, every 10th step (the inputs of step3_data.jl's Xf);
#   - k1a's five long runs (out/TO_LSTM/step3/closures/LinReg1_nt_CNN_crps_nt_headcorr), every RIKFLOW_DRIFT_STRIDE-th step
#     (default 20), inputs rebuilt from q and dQ (q* = q^{n+1} − dQ^n), as the reviewers' b_bins.jl.
# Per QoI, binned by the window's predictor level q*^n at the reference's 0–1–5–25–50–75–95–99–100 % quantiles, in 1e-3 (‰)
# of the reference's mean level per step (so ⊙ g / level). Summaries for E[0,6]: the BULK offset (windows with level at or
# below the 75 % quantile) and the LOW TAIL (at or below the 5 % quantile), with an SE from the spread over the five runs
# (long runs) or over 10 contiguous blocks (record). g = the deployed sample − Ws z − m(z) − L ε (`cnn_window!`), so a
# centred network's Jensen term is included. W0 (g ≡ 0) must give exactly 0.
#
#   RIKFLOW_MLP_SKIP=LinReg1_nt RIKFLOW_MLP_SKIP_CENTER=data julia --project=. analysis/drift_gate.jl <closure> [<closure> ...]
#   (from lib/RikFlow; closures under out/TO_LSTM/step3/closures). RIKFLOW_DRIFT_PAIRS (default 32) antithetic pairs.
using RikFlow
using JLD2, Printf, Statistics, LinearAlgebra, Random
const RF = RikFlow
include(joinpath(pwd(), "exp_square_HIT", "tools", "step3_data.jl"))
const OUTD = joinpath(pwd(), "exp_square_HIT", "output")
const QREF = load(joinpath(pwd(), "analysis", "data", "hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2"), "q_ref")
const LEV = vec(mean(QREF; dims = 2))
const NP = parse(Int, get(ENV, "RIKFLOW_DRIFT_PAIRS", "32"))
const STRIDE = parse(Int, get(ENV, "RIKFLOW_DRIFT_STRIDE", "20"))
const PQ = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
const EDGES = [begin e = quantile(QREF[i, :], PQ); e[1] = -Inf; e[end] = Inf; e end for i in 1:nq]
const Q75 = [quantile(QREF[i, :], 0.75) for i in 1:nq]
const Q05 = [quantile(QREF[i, :], 0.05) for i in 1:nq]
const K1A = "TO_LSTM/step3/closures/LinReg1_nt_CNN_crps_nt_headcorr"
f3(v) = join((isnan(x) ? "      --" : @sprintf("%8.4f", x) for x in v), " ")

"Load a generator closure with a frozen mean (B / reach-k1a variants)."
function load_gen(name)
    cdir = joinpath(OUTD, "TO_LSTM", "step3", "closures", name)
    fit = RF.load_stochlstm(joinpath(cdir, only(filter(f -> startswith(f, "StochLSTM_seed") && endswith(f, ".jld2"), readdir(cdir)))))
    RF.is_generator(fit.spec) || error("$name is not a generator")
    mn = RF.cnn_mean_net(fit.weights)
    mn === nothing && error("$name carries no frozen mean (the drift gate is against the step-4 mean)")
    maximum(abs, Float64.(fit.scaling.input_map) .- P) < 1e-10 * maximum(abs, P) || error("$name: input map is not step3_data.jl's")
    maximum(abs, vec(fit.scaling.out_scaling.sigma) .- so) < 1e-10 * maximum(so) || error("$name: target scaling (SKIP_CENTER = data?)")
    return (; name, sp = fit.spec, w = fit.weights, mn, L = LowerTriangular(Float64.(fit.weights.LR)))
end

"E_ε g at mapped input z from NP fresh antithetic pairs of `rng` (target units)."
function egen(G, st, z, rng)
    base = G.w.Ws * z .+ RF.frozen_mean(G.mn, z, Float64)
    m = zeros(nq)
    for _ in 1:NP
        e = randn(rng, nq)
        for s in (1, -1)
            RF.cnn_window!(st, G.w, G.sp, reshape(z, :, 1); eps = s .* e)
            m .+= (st.y .- base .- G.L * (s .* e)) ./ (2NP)
        end
    end
    return m
end

"Accumulate ‰ per step per QoI and bin; and per group (run / block) the bulk and low-tail E[0,6] sums."
function accumulate!(acc, i_group, g, qlev)
    pm = 1e3 .* so .* g ./ LEV
    for i in 1:nq
        b = clamp(searchsortedlast(EDGES[i], qlev[i]), 1, 8)
        acc.s[b, i] += pm[i]; acc.n[b, i] += 1
    end
    qlev[2] <= Q75[2] && (acc.bulk[i_group] += pm[2]; acc.nbulk[i_group] += 1)
    qlev[2] <= Q05[2] && (acc.tail[i_group] += pm[2]; acc.ntail[i_group] += 1)
    return acc
end
newacc(ng) = (; s = zeros(8, nq), n = zeros(Int, 8, nq), bulk = zeros(ng), nbulk = zeros(Int, ng), tail = zeros(ng), ntail = zeros(Int, ng))

function summary(acc)
    gm(s, n) = (k = n .> 0; v = s[k] ./ n[k]; (sum(s) / max(sum(n), 1), length(v) > 1 ? std(v) / sqrt(length(v)) : NaN))
    return (; bulk = gm(acc.bulk, acc.nbulk), tail = gm(acc.tail, acc.ntail), prof = acc.s ./ max.(acc.n, 1),
            share = 100 .* acc.n ./ sum(acc.n[:, 1]))
end

function run_closure(name)
    G = load_gen(name)
    st = RF.LSTMState(G.sp, Float64)
    rng = Xoshiro(20261113)
    # (1) the recorded track, 10–100 TU, every 10th step; 10 contiguous blocks for the SE
    ho = HO[HO .> cfg.h][1:10:end]
    acc_r = newacc(10)
    for (j, b) in enumerate(ho)
        accumulate!(acc_r, 1 + (10 * (j - 1)) ÷ length(ho), egen(G, st, Xf[:, b], rng), rec.q_star[:, cols[b]])
    end
    # (2) k1a's long runs, every STRIDE-th step; the five runs for the SE
    p = joinpath(OUTD, K1A)
    fs = sort(filter(f -> occursin(r"^data_online_tsim100\.0_replica\d\.jld2$", f), readdir(p)))
    acc_k = newacc(length(fs))
    nwk = 0
    for (r, f) in enumerate(fs)
        q, dq = jldopen(io -> (Float64.(io["data_online"].q), Float64.(io["data_online"].dQ)), joinpath(p, f))
        qs = q[:, 2:end] .- dq
        X, _, stp = RF.build_history(hist, RF.scale_input(qs, dat.scaling.in_scaling), RF.scale_input(q, dat.scaling.in_scaling))
        Z = P * permutedims(Float64.(X))
        for j in 1:STRIDE:size(Z, 2)
            accumulate!(acc_k, r, egen(G, st, Z[:, j], rng), qs[:, stp[j]])
            nwk += 1
        end
    end
    sr, sk = summary(acc_r), summary(acc_k)
    @printf("\n== %s: E_eps g against the frozen step-4 mean (k1a), %d antithetic pairs per window, 1e-3 of the reference mean level per step\n", name, NP)
    for (lab, s, nw) in (("record 10-100 TU", sr, length(ho)), ("k1a's long runs", sk, nwk))
        @printf("  %-18s (%5d windows) E[0,6] BULK (<= 75 %% q.) %+.4f ± %.4f | LOW TAIL (<= 5 %%) %+.4f ± %.4f\n", lab, nw,
                s.bulk[1], s.bulk[2], s.tail[1], s.tail[2])
    end
    for (lab, s) in (("record", sr), ("k1a runs", sk))
        println("  profile, $lab; bins 0-1-5-25-50-75-95-99-100 % of the reference level (per QoI: ‰/step | share of windows %)")
        for (i, l) in enumerate(("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"))
            @printf("    %-8s %s | %s\n", l, f3(s.prof[:, i]), join((@sprintf("%5.1f", x) for x in s.share[:, i]), " "))
        end
    end
    flush(stdout)
    return (; name, record = sr, k1a = sk)
end

res = [run_closure(n) for n in ARGS]
println("\nDRIFT GATE SUMMARY (E[0,6], ‰ per step; BULK = level <= 75 % quantile, TAIL = <= 5 %)")
@printf("  %-22s %22s %22s %22s %22s\n", "closure", "record bulk", "record tail", "k1a-runs bulk", "k1a-runs tail")
for r in res
    @printf("  %-22s %+.4f ± %.4f      %+.4f ± %.4f      %+.4f ± %.4f      %+.4f ± %.4f\n", r.name, r.record.bulk..., r.record.tail...,
            r.k1a.bulk..., r.k1a.tail...)
end
