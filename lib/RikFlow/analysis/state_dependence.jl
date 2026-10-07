# Offline check: does the residual's variance depend on the state? (2026-10-07, before step 4 is chosen)
#
#     julia --startup-file=no --project=lib/RikFlow/analysis lib/RikFlow/analysis/state_dependence.jl
#
# Step 4 (paper Sec. 6, 4a/4c) replaces the constant Σ of LinReg + MVG by a state-dependent scale. This
# asks, offline and before any such closure is chosen, whether there is state dependence to learn, and
# where. For LinReg1 (4p and 4a) and LinReg7 (4c's mean), the deployed coefficients applied
# teacher-forced to R1's tracked record, residual standardised by its training-window sd (the constant
# MVG scale); fits on 1-10 TU (steps 400-4000, the training window), evaluated there and on 10-100 TU
# (no fit uses it). Per QoI:
#
# 1. volatility clustering: ACF of e² at lags 1, 10, 50, minus ρ_k² (what e² of a Gaussian process
#    with e's own ACF ρ would show; LinReg7's residual is coloured, so raw e² ACFs would mislead);
# 2. conditional sd by quintile of (a) the QoI's own predicted level q*^n and (b) that of Z[16,32]
#    (the deep downward excursions of Sec. 4.2 live there), quintile edges from 1-10 TU;
# 3. a linear log-variance head, log σ²(x) = w·x, fitted by Gaussian maximum likelihood (Newton, small
#    ridge) on 1-10 TU, on (i) the six current levels and (ii) the closure's whole regressor (h = 5 lags
#    of q and q*, 66 + bias): held-out NLL gain over the constant variance, nats per step.
#
# ⚠️ An offline gain is necessary, not sufficient: the pilot's LSTM head gained +0.6 nats/step offline
# and was worse online (results_LSTMS.md §13j, memory #75). A null here argues against step 4; a gain
# does not argue for it.
include(joinpath(@__DIR__, "..", "src", "ts_history.jl"))   # HistorySpec, build_history
include(joinpath(@__DIR__, "..", "src", "ts_scaling.jl"))   # scale_input
include(joinpath(@__DIR__, "..", "src", "ts_score.jl"))     # autocorr
include(joinpath(@__DIR__, "..", "src", "ts_scale.jl"))     # fit_powerlaw_scale: step 4p's own fit
using JLD2, LinearAlgebra, Statistics, Printf

const DT = 2.5e-3
const NQ = 6
const LAB = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const OUTD = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const REC = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))

"Teacher-forced residual of a deployed LinReg on the whole record; rows split into 1-10 and 10-100 TU."
function residual(name)
    m = load(joinpath(OUTD, "TO_LRS", name, "LinReg.jld2"))
    C = permutedims(Matrix{Float64}(m["c"]))
    s = m["scaling"].in_scaling
    hist = HistorySpec(; h = m["hist_len"], n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    X, Y, st = build_history(hist, scale_input(REC["q_star"], s), scale_input(REC["q"], s))
    tr = findall(n -> 400 <= n <= 4000, st)                     # = colour_tables.jl's training rows
    ho = findall(n -> 10 < n * DT <= 100, st)
    # the state: q*^n, the predicted level the closure (step 4p) and its guard read -- the first block
    # of `build_history`'s row [q*^n; q^{n-1}; q*^{n-1}; ...; 1], scaled. The latest corrected level
    # q^{n-1} (the next block) gives the same numbers (2026-10-07: 0.481 vs 0.483 nats/step held out).
    lev = X[:, 1:NQ]
    raw = lev .* vec(s.sigma)' .+ vec(s.mu)'                     # the same level in physical units
    return (; R = Y .- X * C, X, lev, raw, tr, ho)
end

"Gaussian ML for log σ² = F w with a fixed zero mean (Newton, ridge λ on all but the first column)."
function logvar_fit(e, F; λ = 1e-3, iters = 50)
    w = zeros(size(F, 2)); w[1] = log(mean(abs2, e))
    P = Diagonal([0.0; fill(λ, size(F, 2) - 1)])
    L(w) = 0.5 * sum(F * w .+ e .^ 2 .* exp.(-(F * w))) + 0.5 * w' * P * w
    for _ in 1:iters
        u = e .^ 2 .* exp.(-(F * w))
        g = 0.5 * F' * (1 .- u) .+ P * w
        H = 0.5 * F' * (F .* u) + P
        d = H \ g
        t = 1.0
        while L(w - t * d) > L(w) && t > 1e-6
            t /= 2
        end
        w -= t * d
        norm(t * d) < 1e-10 && break
    end
    return w
end
nll(e, lv) = mean(0.5 .* (log(2pi) .+ lv .+ e .^ 2 .* exp.(-lv)))

fmt(v; d = 2) = join((@sprintf("%6.*f", d, x) for x in v), " ")

for name in ("LinReg1", "LinReg7")
    r = residual(name)
    sd = vec(std(r.R[r.tr, :]; dims = 1))
    E = r.R ./ sd'                                               # standardised by the constant scale
    println("\n==== $name: residual standardised by its 1-10 TU sd (the constant MVG scale)")

    println("1. volatility clustering: ACF(e²)_k - ρ_k², k = 1, 10, 50   [1-10 TU | 10-100 TU]")
    for i in 1:NQ
        ex(rows) = (e = E[rows, i]; a2 = autocorr(e .^ 2, 50); a = autocorr(e, 50);
                    [a2[k + 1] - a[k + 1]^2 for k in (1, 10, 50)])
        @printf("   %-8s %s | %s\n", LAB[i], fmt(ex(r.tr)), fmt(ex(r.ho)))
    end

    println("2. sd ratio (to the 1-10 TU sd) by quintile, lowest .. highest   [1-10 TU | 10-100 TU]")
    for (lab, col) in (("own level", 0), ("Z[16,32] level", 5))
        println("   by $lab:")
        for i in 1:NQ
            z = r.lev[:, col == 0 ? i : col]
            edges = quantile(z[r.tr], 0.2:0.2:0.8)
            bin(x) = searchsortedfirst(edges, x)
            ratio(rows) = [sqrt(mean(abs2, E[[t for t in rows if bin(z[t]) == b], i])) for b in 1:5]
            @printf("     %-8s %s | %s\n", LAB[i], fmt(ratio(r.tr)), fmt(ratio(r.ho)))
        end
    end

    println("3. linear log-variance head (ML on 1-10 TU), NLL gain over the constant variance, nats/step")
    μF = mean(r.X[r.tr, 1:(end - 1)]; dims = 1); σF = std(r.X[r.tr, 1:(end - 1)]; dims = 1)
    Ffull = hcat(ones(size(r.X, 1)), (r.X[:, 1:(end - 1)] .- μF) ./ σF)
    μL = mean(r.lev[r.tr, :]; dims = 1); σL = std(r.lev[r.tr, :]; dims = 1)
    Flev = hcat(ones(size(r.lev, 1)), (r.lev .- μL) ./ σL)
    @printf("   %-8s %24s %24s\n", "", "6 levels: train | held", "full regressor: train | held")
    tot = zeros(4)
    for i in 1:NQ
        g = Float64[]
        for F in (Flev, Ffull)
            w = logvar_fit(E[r.tr, i], F[r.tr, :])
            for rows in (r.tr, r.ho)
                c = log(mean(abs2, E[r.tr, i]))                  # the constant scale, from 1-10 TU
                push!(g, nll(E[rows, i], fill(c, length(rows))) - nll(E[rows, i], F[rows, :] * w))
            end
        end
        tot .+= g
        @printf("   %-8s %11.3f | %8.3f %14.3f | %8.3f\n", LAB[i], g...)
    end
    @printf("   %-8s %11.3f | %8.3f %14.3f | %8.3f   (sum over QoIs)\n", "total", tot...)

    # 4. the simplest physical form, step 4p: σ_i ∝ (q*_i^n)^β_i, fitted by `fit_powerlaw_scale` and
    #    evaluated as deployed (state clipped to its 1-10 TU range)
    println("4. power law in the own level, σ ∝ q^β (ML on 1-10 TU): β, and NLL gain, nats/step [train | held]")
    tot4 = zeros(2)
    for i in 1:NQ
        p = fit_powerlaw_scale(E[r.tr, i], r.raw[r.tr, i])
        lv(rows) = p.logvar .+ 2p.beta .* log.(clamp.(r.raw[rows, i], p.qclip...) ./ p.qref)
        c = log(mean(abs2, E[r.tr, i]))
        g = [nll(E[rows, i], fill(c, length(rows))) - nll(E[rows, i], lv(rows)) for rows in (r.tr, r.ho)]
        tot4 .+= g
        @printf("   %-8s β = %5.2f   %8.3f | %8.3f\n", LAB[i], p.beta, g...)
    end
    @printf("   %-8s            %8.3f | %8.3f   (sum over QoIs)\n", "total", tot4...)
end
