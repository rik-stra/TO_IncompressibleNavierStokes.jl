# Ridge + colour: three CPU checks on the ridge-stabilised M0 (LinReg7, λ = 1, h = 5), 2026-09-28.
#
#     julia --startup-file=no --project=training analysis/m0c_ridge_colour.jl
#
# Hypothesis (results_LSTMS §10g): ridge moves the mean's dynamics into a COLOURED residual, and
# LinReg7's white Σ then under-disperses in closed loop (D6 spread-skill median 0.559). Checks:
#  (1) kernel-weighted power of AR fits to LinReg7's residual vs its white Σ at the same variance,
#      and vs LinReg1's white Σ (absolute) -- diagonal and full 6x6 measured solver kernel.
#  (2) the D1 joint (mean, AR(p)) fit at LinReg7's λ, p = 1, 2, against LinReg7 (two-stage, white)
#      and LinReg7's mean with an AR fitted to its residual (fixed_C): held-out one-step NLL on
#      52-74 TU, innovation ACF, how far C moves, and the AR coefficients.
#  (3) online dQ ACF of LinReg7's and LinReg1's existing D6 members vs the tracked record on the
#      same steps. Only ICs whose forecast ends by 74 TU are read.
# 🔒 Nothing past 74 TU is read (guard() and the IC filter).

include(joinpath(@__DIR__, "m0c_ridge.jl"))    # deployed_resid, LRS, and via it m0c_checks.jl

# ---- the deployed model's scaled design, exactly as LinReg sees it ---------------------------
function scaled_design(name, r)
    m = load(joinpath(LRS, name, "LinReg.jld2"))
    X, Y = design(m["hist_len"], r)
    si, so = m["scaling"].in_scaling, m["scaling"].out_scaling
    mu, sg = vec(si.mu), vec(si.sigma)
    Xs = copy(X)
    for j in 1:(size(X, 2) - 1)
        i = mod1(j, NQ)
        Xs[:, j] = (X[:, j] .- mu[i]) ./ sg[i]
    end
    Ys = (Y .- vec(so.mu)') ./ vec(so.sigma)'
    return Xs, Ys, Matrix(m["c"])', m
end

function gauss_nll(E, S)
    L = cholesky(Symmetric(S)).L
    Z = L \ permutedims(E)
    return 0.5 * (size(E, 2) * log(2pi) + 2 * sum(log, diag(L))) + 0.5 * mean(sum(abs2, Z; dims = 1))
end

function check1(; io)
    G = load(joinpath(TO, "response", "response_kernel.jld2"), "G")
    K = size(G, 3)
    r1 = deployed_resid("LinReg1", steps_of(1, 10)).R
    r7 = deployed_resid("LinReg7", steps_of(1, 10)).R
    println(io, "\n==== (1) kernel-weighted power of LinReg7's residual (h = 5, λ = 1, 1-10 TU), diagonal kernel")
    @printf(io, "  %-9s %6s | %8s %8s %8s | %10s %10s\n", "band", "rho1", "AR(1)", "AR2-LS", "data",
            "white7/white1", "data7/white1")
    for i in 1:NQ
        e = r7[:, i]; s2 = var(e); r = acf(e, 0:(K - 1)); g = G[i, i, :]
        Pw = kpower(g, [1.0; zeros(K - 1)], s2)
        P1 = kpower(g, [r[2]^k for k in 0:(K - 1)], s2)
        l1, l2 = ar2_ls(r; L = 20)
        Pls = kpower(g, ar2_acf(l1, l2, K - 1), s2)
        Pd = kfilter_var(g, e)
        Pw1 = kpower(g, [1.0; zeros(K - 1)], var(r1[:, i]))
        @printf(io, "  %-9s %6.2f | %8.2f %8.2f %8.2f | %10.2f %10.2f\n", LABELS[i], r[2], P1 / Pw, Pls / Pw,
                Pd / Pw, Pw / Pw1, Pd / Pw1)
    end
    # full 6x6 kernel
    rng = Xoshiro(20260928)
    s2v = vec(var(r7; dims = 1)); Cor = cor(r7); n = 200_000
    pw = kfilter_var_full(G, sim_ar(zeros(NQ), zeros(NQ), Cor, s2v, n, rng))
    pw1 = kfilter_var_full(G, sim_ar(zeros(NQ), zeros(NQ), cor(r1), vec(var(r1; dims = 1)), n, rng))
    a1 = [acf(r7[:, i], 1:1)[1] for i in 1:NQ]
    pls = [ar2_ls(acf(r7[:, i], 0:20); L = 20) for i in 1:NQ]
    pa1 = kfilter_var_full(G, sim_ar(a1, zeros(NQ), Cor, s2v, n, rng))
    pa2 = kfilter_var_full(G, sim_ar(first.(pls), last.(pls), Cor, s2v, n, rng))
    pd = kfilter_var_full(G, permutedims(r7))
    println(io, "  full 6x6 kernel, per output band:")
    @printf(io, "    AR(1)/white7 %s\n    AR2-LS/white7 %s\n    data/white7 %s\n    white7/white1 %s\n    data7/white1 %s\n",
            fmt(pa1 ./ pw), fmt(pa2 ./ pw), fmt(pd ./ pw), fmt(pw ./ pw1), fmt(pd ./ pw1))
end

function check2(; io)
    fitr, held = steps_of(1, 10), WIN_HELD
    X, Y, c7, m7 = scaled_design("LinReg7", fitr)
    Xh, Yh, _, _ = scaled_design("LinReg7", held)
    spec = RF.HistorySpec(; h = 5, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    println(io, "\n==== (2) joint (mean, AR) fit at LinReg7's λ, h = 5, scaled design, 1-10 TU fit, 52-74 held out")
    # λ convention: find the fit_ridge λ that reproduces the deployed c
    best = nothing
    for lam in (0.5, 1.0, 2.0, 1.0 * size(X, 1), 0.5 * size(X, 1))
        C = RF.fit_ridge(X, Y; lambda = lam)
        d = maximum(abs, C .- c7) / maximum(abs, c7)
        @printf(io, "  fit_ridge(lambda = %g) vs deployed LinReg7 c: max rel diff %.2e\n", lam, d)
        (best === nothing || d < best[2]) && (best = (lam, d))
    end
    lam = best[1]
    @printf(io, "  -> using fit_ridge lambda = %g (reproduces LinReg7 to %.1e)\n", lam, best[2])
    Rw = Y .- X * c7; Rwh = Yh .- Xh * c7
    S = cov(Rw)
    @printf(io, "  LinReg7 white (two-stage):   held-out NLL %.4f   [fit %.4f]\n", gauss_nll(Rwh, S), gauss_nll(Rw, S))
    for p in (1, 2)
        for (lab, fc) in (("fixed LinReg7 mean + AR", c7), ("JOINT mean + AR       ", nothing))
            model, hist = RF.fit_joint(X, Y, spec; ar_order = p, lambda = lam, fixed_C = fc, iters = 40)
            a = RF.ar_from_psi(model.psi)
            C = model.C
            nf = RF.nll(X, Y, C, model.W, model.R, a, ones(Int, size(X, 1)), RF.valid_rows(ones(Int, size(X, 1)), p))
            nh = RF.nll(Xh, Yh, C, model.W, model.R, a, ones(Int, size(Xh, 1)), RF.valid_rows(ones(Int, size(Xh, 1)), p))
            # innovation on held-out
            E = Yh .- Xh * C
            Xi = copy(E)
            for k in 1:p, i in 1:NQ
                Xi[(k + 1):end, i] .-= a[k, i] .* E[1:(end - k), i]
            end
            Xi = Xi[(p + 1):end, :]
            @printf(io, "  p = %d  %s  held-out NLL %.4f [fit %.4f]  ‖C-c7‖/‖c7‖ %.3f\n", p, lab, nh / size(Xh, 1) * 1,
                    nf / size(X, 1), norm(C .- c7) / norm(c7))
            @printf(io, "      AR a_1 %s%s\n", fmt(a[1, :]), p > 1 ? "   a_2 " * fmt(a[2, :]) : "")
            @printf(io, "      held-out mean-residual lag1 %s | innovation lag1 %s | innovation sd / white sd %s\n",
                    fmt([acf(E[:, i], 1:1)[1] for i in 1:NQ]), fmt([acf(Xi[:, i], 1:1)[1] for i in 1:NQ]),
                    fmt(vec(std(Xi; dims = 1)) ./ vec(std(Rwh; dims = 1))))
        end
    end
    println(io, "  (NLLs are per row, scaled units; the white line uses a full-covariance Gaussian, the AR lines",
            " RikFlow's nll with R concentrated; compare lines of the same kind, and the ordering)")
end

function check3(; io)
    println(io, "\n==== (3) online dQ ACF of the existing D6 members vs the tracked record on the same steps")
    println(io, "  ICs whose forecast ends by 74 TU; forecast columns nwarm+1:end (1200 steps); lags 1 2 5 10 20")
    lags = (1, 2, 5, 10, 20)
    for cl in ("D6_LinReg1", "D6_LinReg7")
        dir = joinpath(HERE, "output", cl)
        on = [zeros(length(lags)) for _ in 1:NQ]; tr = [zeros(length(lags)) for _ in 1:NQ]
        sdr = zeros(NQ); n = 0; nic = Set{Int}()
        for f in readdir(dir)
            startswith(f, "d6_online_ic") || continue
            d = load(joinpath(dir, f))
            (d["t_k"] + d["tsim"] <= T_MAX_READ) || continue
            haskey(d, "diverged") && d["diverged"] && continue
            nw, nk = d["nwarm"], d["n_k"]
            cols = (nw + 1):size(d["dQ"], 2)
            tcols = guard(nk .+ cols)
            for i in 1:NQ
                on[i] .+= acf(d["dQ"][i, cols], lags)
                tr[i] .+= acf(DQR[i, tcols], lags)
                sdr[i] += std(d["dQ"][i, cols]) / std(DQR[i, tcols])
            end
            n += 1; push!(nic, d["k"])
        end
        @printf(io, "  %s: %d members, %d ICs\n", cl, n, length(nic))
        @printf(io, "    %-9s %-31s %-31s %s\n", "band", "online dQ ACF", "tracked dQ ACF (same steps)", "sd(dQ) on/trk")
        for i in 1:NQ
            @printf(io, "    %-9s %s   %s   %.2f\n", LABELS[i], fmt(on[i] ./ n), fmt(tr[i] ./ n), sdr[i] / n)
        end
    end
    println(io, "  (per-member ACF over 3 TU, averaged: short-window ACFs are biased low at long lags, equally for",
            " online and tracked)")
end

function main2()
    io = stdout
    check1(; io); check2(; io); check3(; io)
end

main2()
