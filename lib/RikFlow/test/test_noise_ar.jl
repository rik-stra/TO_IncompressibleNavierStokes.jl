# V97 -- q1 / q2: the mean model trained with its own AR(2) (and power-law scaled) draws (card
# `analysis/cards/q1q2.md`, builder `exp_square_HIT/tools/lrs_noise_ar_variant.jl`). Added 2026-11-06.
#
# What is pinned, each against an answer known before the code runs:
#   * `ar_lagcov`: per component the closed-form AR(2) variance and ACF; the VAR Yule-Walker identities; a long
#     simulation with the builder's own simulator (`sim_ar_S`); white noise gives Γ_0 = S, Γ_k = 0;
#   * `window_map` against Eq. `eq: window noise` applied to explicit levels (q^{n-k} and q^{n-k*} shifted, the
#     regressor rebuilt), and `window_lagcov` = `fit_ar_noisy`'s S_k ⊗ Γ_0 for white draws;
#   * the white limit: `fit_noise_ar` with φ ≡ 0 IS `lrs_noise_variant.jl`'s `fit_noise`, and the AR fitted after it IS
#     `fit_ar_noisy` (both evaluated from their own files);
#   * q2's row-dependent pieces with constant rows are q1's; `noise_penalty` and `noise_var_rows` against the explicit
#     per-row sums; `fit_noise_ar` with the scale held at β = 0 is q1;
#   * the fixed point: one more step changes nothing, the deployed AR's marginal variance is diag Σ (q1) or diag Σ_ε
#     (q2), diag Σ_ε = exp(logvar);
#   * `resid_lagcov` against a Monte Carlo of the noisy history (AR draws, row-dependent scales);
#   * `lrv_ratio` against the AR(1) closed form.
#
# As `test_round1.jl`, the tools cannot be `include`d here (they load RikFlow and the R1 record at top level), so the
# named top-level definitions are evaluated from the files' own text. `RF` is a stand-in module holding only
# `src/ts_scale.jl` (stdlib only), which is all the builder calls `RF.` for in these functions.

@testmodule NoiseAR begin
    using LinearAlgebra
    using Statistics
    using Random
    using Printf
    using Distributions

    const RFDIR = normpath(joinpath(@__DIR__, ".."))

    module RF
    include(joinpath(@__DIR__, "..", "src", "ts_scale.jl"))
    end

    function defined_name(e)
        e isa Expr || return nothing
        if e.head === :macrocall && e.args[1] == GlobalRef(Core, Symbol("@doc"))
            return defined_name(e.args[end])
        elseif e.head === :const
            a = e.args[1]
            return a isa Expr && a.head === :(=) && a.args[1] isa Symbol ? a.args[1] : nothing
        elseif e.head === :function || (e.head === :(=) && e.args[1] isa Expr)
            s = e.args[1]
            while s isa Expr && s.head in (:where, :(::))
                s = s.args[1]
            end
            return s isa Expr && s.head === :call && s.args[1] isa Symbol ? s.args[1] : nothing
        end
        return nothing
    end

    "As `test_round1.jl`'s: evaluate the named top-level definitions of `file` into `mod`, nothing else."
    function defs_from!(mod::Module, file::AbstractString, names)
        want = Set(Symbol.(names))
        found = Set{Symbol}()
        lnn = LineNumberNode(1, Symbol(file))
        for e in Meta.parseall(read(file, String); filename = file).args
            e isa LineNumberNode && (lnn = e; continue)
            n = defined_name(e)
            n in want || continue
            Core.eval(mod, Expr(:toplevel, lnn, e))
            push!(found, n)
        end
        isempty(setdiff(want, found)) ||
            error("defs_from!: $(sort(collect(setdiff(want, found)))) not defined at top level of $file")
        return mod
    end

    const TOOLS = joinpath(RFDIR, "exp_square_HIT", "tools")
    defs_from!(@__MODULE__, joinpath(RFDIR, "analysis", "m0c_checks.jl"), (:NQ, :acf, :ar2_acf, :ar2_stationary, :ar2_ls))
    defs_from!(@__MODULE__, joinpath(TOOLS, "lrs_ar_variant.jl"),
               (:ar_acf, :ar_innovation_var, :window_map, :fit_ar_noisy, :sim_ar_S))
    defs_from!(@__MODULE__, joinpath(TOOLS, "lrs_noise_variant.jl"), (:window_cov, :ridge_pen, :fit_noise))
    defs_from!(@__MODULE__, joinpath(TOOLS, "lrs_noise_ar_variant.jl"),
               (:ar_lagcov, :lagcov, :window_lagcov, :ar_from_lagcov, :noise_penalty, :noise_var_rows, :resid_lagcov,
                :scale_rows, :lrv_ratio, :fit_noise_ar))

    "Closed-form stationary variance of the AR(2) with innovation variance `s2xi`."
    ar2_gamma0(s2xi, p1, p2) = s2xi * (1 - p2) / ((1 + p2) * ((1 - p2)^2 - p1^2))

    "A random SPD n × n matrix with unit-order diagonal and correlations up to ~0.6."
    function rand_spd(rng, n)
        A = randn(rng, n, n)
        S = A * A' ./ n + I
        d = Diagonal(1 ./ sqrt.(diag(S)))
        sc = Diagonal(0.5 .+ rand(rng, n))
        return Matrix(Symmetric(sc * (d * S * d) * sc))
    end

    "Six restricted stationary AR(2)s (φ1 ≥ 0, φ2 ≤ 0), one persistent, one AR(1), one white."
    const PHI = [0.95 0.6 1.2 0.0 0.3 1.5; 0.0 -0.1 -0.4 0.0 -0.2 -0.6]

    """
    A synthetic noise-training problem: N rows, history h, NQ = 6 QoIs; the regressor layout of the LinReg path
    (q*^n, then q^{n-k}, q^{n-k*} for k = 1..h, then the bias), scaled; positive levels for the scale.
    """
    function synth(; N = 500, h = 2, seed = 1)
        rng = Xoshiro(seed)
        T = N + h + 1
        qs = exp.(0.3 .* cumsum(0.1 .* randn(rng, NQ, T); dims = 2)) .* (1:NQ)       # positive predictors q*
        q = qs .+ 0.05 .* randn(rng, NQ, T)
        isg, osg = 0.5 .+ rand(rng, NQ), 0.5 .+ rand(rng, NQ)
        ns = (h + 1):(h + N)
        X = reduce(vcat, [permutedims([qs[:, n]; reduce(vcat, [[q[:, n - k]; qs[:, n - k]] for k in 1:h]); 1.0]) for n in ns])
        Xs = copy(X)
        for j in 1:(size(X, 2) - 1)
            Xs[:, j] = (X[:, j] .- mean(X[:, j])) ./ isg[mod1(j, NQ)]
        end
        C = 0.3 .* randn(rng, size(Xs, 2), NQ)
        e = zeros(N, NQ)
        for t in 2:N
            e[t, :] = 0.6 .* e[t - 1, :] .+ 0.3 .* randn(rng, NQ)
        end
        Ys = Xs * C .+ e
        qwin = reduce(hcat, [permutedims(qs[:, ns .- j]) for j in 1:h])
        return (; Xs, Ys, isg, osg, qs = permutedims(qs[:, ns]), qwin, h, N, X, q, qsall = qs, ns)
    end
end

@testitem "V97 ar_lagcov: closed form per component, VAR Yule-Walker, simulation, white" default_imports = false setup = [NoiseAR] begin
    using Test, LinearAlgebra, Statistics, Random
    using .NoiseAR: ar_lagcov, ar2_gamma0, ar2_acf, rand_spd, sim_ar_S, PHI, NQ
    rng = Xoshiro(7)
    S = rand_spd(rng, NQ)
    G = ar_lagcov(PHI, S, 30)
    F1, F2 = Diagonal(PHI[1, :]), Diagonal(PHI[2, :])
    for i in 1:NQ, k in 0:30
        g0 = ar2_gamma0(S[i, i], PHI[1, i], PHI[2, i])
        @test G[k + 1][i, i] ≈ g0 * ar2_acf(PHI[1, i], PHI[2, i], 30)[k + 1] rtol = 1e-9 atol = 1e-12
    end
    @test G[1] ≈ F1 * G[1] * F1 .+ F2 * G[1] * F2 .+ F1 * G[2] * F2 .+ F2 * G[2]' * F1 .+ S rtol = 1e-10
    @test G[2] ≈ F1 * G[1] .+ F2 * G[2]' rtol = 1e-10
    @test G[1] ≈ G[1]'
    # a long simulation with the builder's own simulator: lag-0, 1, 3 cross-covariances (correlation units)
    E = sim_ar_S(PHI[1, :], PHI[2, :], S, 400_000, Xoshiro(11))
    sd = sqrt.(diag(G[1]))
    for k in (0, 1, 3)
        Ck = E[:, (k + 1):end] * E[:, 1:(end - k)]' ./ (size(E, 2) - k)
        @test maximum(abs, (Ck .- G[k + 1]) ./ (sd * sd')) < 0.03
    end
    # white: Γ_0 = S, Γ_k = 0
    W = ar_lagcov(zeros(2, NQ), S, 3)
    @test W[1] ≈ S rtol = 1e-12
    @test all(maximum(abs, W[k]) < 1e-14 for k in 2:4)
end

@testitem "V97 window_map is Eq. window noise on explicit levels; window_lagcov is S_k ⊗ Γ_0 for white draws" default_imports = false setup = [NoiseAR] begin
    using Test, LinearAlgebra, Statistics, Random
    using .NoiseAR: window_map, window_lagcov, ar_lagcov, rand_spd, NQ
    rng = Xoshiro(3)
    for h in (1, 2, 5)
        isg = 0.5 .+ rand(rng, NQ)
        M = window_map(h, isg)
        T = h + 3
        q, qs = randn(rng, NQ, T), randn(rng, NQ, T)
        n = T
        x(q, qs) = [qs[:, n]; reduce(vcat, [[q[:, n - k]; qs[:, n - k]] for k in 1:h]); 1.0] ./ [repeat(isg, 2h + 1); 1.0]
        for _ in 1:5
            eta = randn(rng, NQ, h)                      # eta[:, j] = η^{n-j}, raw
            qt, qst = copy(q), copy(qs)
            for k in 1:h                                 # Eq. eq: window noise, anchored at the prediction step
                qt[:, n - k] .-= k > 1 ? vec(sum(eta[:, 1:(k - 1)]; dims = 2)) : zeros(NQ)
                qst[:, n - k] .-= vec(sum(eta[:, 1:k]; dims = 2))
            end
            @test x(qt, qst) .- x(q, qs) ≈ M * vec(eta) atol = 1e-12
            # every correction dQ^{n-k} = q^{n-k} - q^{n-k*} carries η^{n-k} once
            @test all((qt[:, n - k] .- qst[:, n - k]) .- (q[:, n - k] .- qs[:, n - k]) ≈ eta[:, k] for k in 1:h)
        end
        # white draws: W_k = S_k ⊗ Γ_0, S_k[j, j'] = 1 iff j - j' = k (fit_ar_noisy's)
        S = rand_spd(rng, NQ)
        G = ar_lagcov(zeros(2, NQ), S, 2h + 2)
        for k in 0:(h + 1)
            Sk = zeros(h, h)
            for jp in 1:(h - k)
                Sk[jp + k, jp] = 1.0
            end
            @test window_lagcov(G, h, k) ≈ kron(Sk, S) atol = 1e-14
        end
    end
end

@testitem "V97 the white limit is fit_noise, and the AR fitted after it is fit_ar_noisy" default_imports = false setup = [NoiseAR] begin
    using Test, LinearAlgebra, Statistics, Random, Distributions
    using .NoiseAR: synth, window_map, fit_noise_ar, fit_noise, fit_ar_noisy, NQ
    s = synth(; N = 600, h = 2)
    d = (; s.Xs, s.Ys, s.osg)
    M = window_map(s.h, s.isg)
    S0 = cov(s.Ys .- s.Xs * (s.Xs \ s.Ys); corrected = false)
    w = fit_noise_ar(d, M, s.h; phi0 = zeros(2, NQ), S0, white = true, tol = 1e-12)
    f = fit_noise(d, M, S0, s.h; lam = 0.0)
    @test w.Ct ≈ f.Ct rtol = 1e-10
    @test w.Sig ≈ f.Sig rtol = 1e-10
    @test w.mu ≈ f.mu atol = 1e-12
    @test all(w.phi .== 0)
    @test w.S ≈ w.Sig rtol = 1e-10
    m = Dict("noise_provenance" => (;), "hist_len" => s.h, "c" => Matrix(f.Ct'),
             "scaling" => (; in_scaling = (; sigma = s.isg), out_scaling = (; sigma = s.osg)),
             "stoch_distr" => MvNormal(f.mu, Matrix(Symmetric(f.Sig))))
    a = fit_ar_noisy((; m, Z = s.Ys .- s.Xs * f.Ct .- f.mu'), 2; nonneg = true)
    @test a.phi == w.a.phi
    @test a.S ≈ w.a.S rtol = 1e-9
end

@testitem "V97 q2's row pieces with constant rows are q1's; the penalty and response sums; q2 at β = 0 is q1" default_imports = false setup = [NoiseAR] begin
    using Test, LinearAlgebra, Statistics, Random
    using .NoiseAR: synth, window_map, window_lagcov, ar_lagcov, noise_penalty, noise_var_rows, resid_lagcov, fit_noise_ar,
                    rand_spd, PHI, NQ
    s = synth(; N = 300, h = 2)
    rng = Xoshiro(5)
    M = window_map(s.h, s.isg)
    nh = NQ * s.h
    G = ar_lagcov(PHI, rand_spd(rng, NQ), 20 + s.h)
    Wz = [window_lagcov(G, s.h, k) for k in 0:20]
    AM = 0.2 .* randn(rng, NQ, size(M, 1)) * M
    Z = randn(rng, s.N, NQ)
    g = repeat(s.osg, s.h)
    Gc0, Gc = resid_lagcov(Z, AM, Wz, g, nothing)
    Gr0, Gr = resid_lagcov(Z, AM, Wz, repeat(g', s.N), ones(s.N, NQ))
    @test Gr0 ≈ Gc0 rtol = 1e-12
    @test all(Gr[k] ≈ Gc[k] for k in eachindex(Gc))
    @test noise_penalty(M, Wz[1], repeat(g', s.N), s.N) ≈ noise_penalty(M, Wz[1], g, s.N) rtol = 1e-12
    # row-dependent scales: the closed sums against the explicit per-row ones
    gr = 0.5 .+ rand(rng, s.N, nh)
    @test noise_penalty(M, Wz[1], gr, s.N) ≈ sum(M * Diagonal(gr[t, :]) * Wz[1] * Diagonal(gr[t, :]) * M' for t in 1:s.N) rtol = 1e-11
    v = noise_var_rows(AM, Wz[1], gr)
    @test all(v[t, :] ≈ diag((AM * Diagonal(gr[t, :])) * Wz[1] * (AM * Diagonal(gr[t, :]))') for t in 1:s.N)
    @test vec(noise_var_rows(AM, Wz[1], g)) ≈ diag((AM .* g') * Wz[1] * (AM .* g')') rtol = 1e-12
    # the whole fit: q2's machinery with β held at 0 is q1 (to the stop tolerance)
    d = (; s.Xs, s.Ys, s.osg)
    S0 = Matrix(Symmetric(cov(s.Ys .- s.Xs * (s.Xs \ s.Ys))))
    q1 = fit_noise_ar(d, M, s.h; phi0 = zeros(2, NQ), S0)
    q2 = fit_noise_ar(d, M, s.h; phi0 = zeros(2, NQ), S0, scale = (; s.qs, s.qwin, beta0 = zeros(NQ), fixbeta = true))
    @test q2.phi == q1.phi
    @test maximum(abs, s.Xs * (q2.Ct .- q1.Ct)) / maximum(std(s.Ys; dims = 1)) < 1e-7
    @test q2.S ≈ q1.S rtol = 1e-7
    @test q2.Sig ≈ q1.Sig rtol = 1e-7
    @test all(q2.f .== 1)
end

@testitem "V97 the fixed point: one more step is still, the AR's marginal variance is diag Σ (Σ_ε), Σ_ε is the ML variance" default_imports = false setup = [NoiseAR] begin
    using Test, LinearAlgebra, Statistics, Random
    using .NoiseAR: synth, window_map, ar_lagcov, fit_noise_ar, ar_innovation_var, NQ
    s = synth(; N = 500, h = 2, seed = 2)
    d = (; s.Xs, s.Ys, s.osg)
    M = window_map(s.h, s.isg)
    S0 = Matrix(Symmetric(cov(s.Ys .- s.Xs * (s.Xs \ s.Ys))))
    q1 = fit_noise_ar(d, M, s.h; phi0 = zeros(2, NQ), S0)
    @test q1.fixed[2] && q1.fixed[1] < 1e-7
    @test all(q1.phi[1, :] .>= 0) && all(q1.phi[2, :] .<= 0)
    @test diag(ar_lagcov(q1.phi, q1.S, 0)[1]) ≈ diag(q1.Sig) rtol = 1e-6
    @test q1.Sig ≈ q1.Ge0 rtol = 1e-12
    @test q1.Sig ≈ q1.Gef[1] rtol = 1e-12
    @test any(q1.phi[1, :] .> 0)                       # the synthetic residual is coloured (AR(1) 0.6)
    q2 = fit_noise_ar(d, M, s.h; phi0 = q1.phi, S0 = q1.S, scale = (; s.qs, s.qwin, beta0 = zeros(NQ)))
    @test q2.fixed[2] && q2.fixed[1] < 1e-7
    @test diag(q2.Seps) ≈ exp.(q2.logvar) rtol = 1e-6
    @test diag(ar_lagcov(q2.phi, q2.S, 0)[1]) ≈ diag(q2.Seps) rtol = 1e-6
    @test maximum(abs, q2.beta .- [r.beta for r in q2.sfit]) < 1e-7
end

@testitem "V97 resid_lagcov against a Monte Carlo of the noisy history (AR draws, row scales)" default_imports = false setup = [NoiseAR] begin
    using Test, LinearAlgebra, Statistics, Random
    using .NoiseAR: ar_lagcov, window_lagcov, resid_lagcov, window_map, sim_ar_S, rand_spd, PHI, NQ
    rng = Xoshiro(9)
    h, N, R, L = 2, 200, 3000, 4
    S = rand_spd(rng, NQ)
    G = ar_lagcov(PHI, S, L + h)
    Wz = [window_lagcov(G, h, k) for k in 0:L]
    M = window_map(h, 0.5 .+ rand(rng, NQ))
    AM = 0.3 .* randn(rng, NQ, size(M, 1)) * M
    Z = 0.1 .* randn(rng, N, NQ)
    g = 0.5 .+ rand(rng, N, NQ * h)
    f = 0.5 .+ rand(rng, N, NQ)
    Ge0, Gef = resid_lagcov(Z, AM, Wz, g, f; L)
    acc = [zeros(NQ, NQ) for _ in 0:L]
    acc0 = zeros(NQ, NQ)
    for r in 1:R
        zp = sim_ar_S(PHI[1, :], PHI[2, :], S, N + h, rng)        # z^1 … z^{N+h}; row t's window is z^{t+h-1} … z^t
        e = similar(Z)
        for t in 1:N
            w = reduce(vcat, [zp[:, t + h - j] for j in 1:h])
            e[t, :] = Z[t, :] .- (AM .* g[t, :]') * w
        end
        ef = e ./ f
        for k in 0:L
            acc[k + 1] .+= ef[(k + 1):end, :]' * ef[1:(end - k), :] ./ (N - k)
        end
        acc0 .+= e' * e ./ N
    end
    # the estimator here averages every pair (1/(N-k)); resid_lagcov's data part is (1/N): put it on the same footing
    for k in 0:L
        data = (Z ./ f)[(k + 1):end, :]' * (Z ./ f)[1:(end - k), :]
        target = Gef[k + 1] .- data ./ N .+ data ./ (N - k)
        sc = sqrt.(diag(Gef[1]))
        @test maximum(abs, (acc[k + 1] ./ R .- target) ./ (sc * sc')) < 0.03
    end
    sc0 = sqrt.(diag(Ge0))
    @test maximum(abs, (acc0 ./ R .- Ge0) ./ (sc0 * sc0')) < 0.03
end

@testitem "V97 lrv_ratio: the AR(1) closed form, and 1 for white" default_imports = false setup = [NoiseAR] begin
    using Test
    using .NoiseAR: lrv_ratio
    for p in (0.0, 0.3, 0.9, 0.99)
        @test lrv_ratio(p, 0.0) ≈ 1 + 2 * (p == 0 ? 0.0 : p * (1 - p^200) / (1 - p)) rtol = 1e-12
    end
end

@testitem "V97 ar_from_lagcov with a given φ: ar2_ls's own φ reproduces the free fit; another φ keeps the marginal variance" default_imports = false setup = [NoiseAR] begin
    using Test, LinearAlgebra, Random
    using .NoiseAR: ar_lagcov, ar_from_lagcov, rand_spd, PHI, NQ
    G = ar_lagcov(PHI, rand_spd(Xoshiro(4), NQ), 22)
    a = ar_from_lagcov(G)
    b = ar_from_lagcov(G; phi = a.phi)
    @test b.phi == a.phi && b.S == a.S && b.racf == a.racf
    p = a.phi .+ [0.0002; 0.0] .* (a.phi[1, :] .> 0.1)'      # one lattice step off, as a φ cycle's candidates
    c = ar_from_lagcov(G; phi = p)
    @test c.phi == p
    @test diag(ar_lagcov(p, c.S, 0)[1]) ≈ diag(G[1]) rtol = 1e-10
end
