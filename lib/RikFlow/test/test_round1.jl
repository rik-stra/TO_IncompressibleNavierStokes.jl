# V81 -- round 1 of the LinReg closure: the AR(2) residual, the per-QoI ridge splice, and the
# paper's improvement criterion (Sec. 3.7). Added 2026-10-06.
#
# What is pinned, each against an answer known before the code runs:
#
#   * `ar2_ls` (analysis/m0c_checks.jl) recovers (φ1, φ2) from the exact ACF of known stationary
#     AR(2)s, and returns a pair its own stationarity test accepts for ACFs no stationary AR(2) fits;
#   * Σ_ξ (exp_square_HIT/tools/lrs_ar_variant.jl): the innovation variance gives the AR(2) the
#     residual's marginal variance -- in closed form, and by a long simulation with the builder's own
#     simulator -- and `fit_ar` assembles Σ_ξ = D R D from it;
#   * the splice (exp_square_HIT/tools/lrs_splice.jl): one source reproduces that source; a row taken
#     from a ridge fit at λ IS the single-column ridge solution at λ; Σ is the MLE (/N) of the spliced
#     residual, which is also what the trainer's own per-QoI λ path produces;
#   * `criterion_clauses` and `ks_guard` (analysis/score_d6.jl): the strict inequalities of S, C and G;
#   * `load_members`' two exclusion policies, read from the environment at include time.
#
# 🔑 The three tools, and the trainer `5_train_LinReg.jl` the splice is checked against, cannot be
# `include`d here: each loads `using RikFlow` (which this environment deliberately lacks,
# `test/Project.toml`), `m0c_checks.jl` loads the R1 tracking cache at top level, and the trainer is
# a script that reads ARGS. `Round1` therefore evaluates **only the named top-level definitions** of
# each file, parsed from the source -- the code under test is the file's own text, not a copy, and a
# definition that is renamed or gains a dependency on something not extracted fails loudly here.
#
# ⚠️ Five `@test_broken` record two measured limitations of `ar2_ls` (2026-10-06), not wanted
# behaviour: it stops one ulp inside the stationarity boundary on persistent ACFs, and off the 0.01
# grid it misses (φ1, φ2) by more than 5e-4 for ~15 % of stationary AR(2)s. Fix it and they turn
# into "unexpected pass" errors, which is the point.

@testmodule Round1 begin
    using LinearAlgebra
    using Statistics
    using Random
    using JLD2
    using Distributions

    const RF = normpath(joinpath(@__DIR__, ".."))

    "Name a top-level expression defines -- function (long or short form, documented or not) or `const`."
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

    """
        defs_from!(mod, file, names)

    Evaluate into `mod` the top-level definitions of `file` named in `names` (every method of each,
    with its docstring and its source line numbers) and nothing else. Raises when a name is absent.
    """
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

    defs_from!(@__MODULE__, joinpath(RF, "analysis", "m0c_checks.jl"),
               (:NQ, :acf, :ar2_acf, :ar2_stationary, :ar2_ls))
    defs_from!(@__MODULE__, joinpath(RF, "exp_square_HIT", "tools", "lrs_ar_variant.jl"),
               (:ar_acf, :ar_poles, :ar2_ls_lag1, :ar_innovation_var, :fit_ar, :sim_ar_S))
    defs_from!(@__MODULE__, joinpath(RF, "exp_square_HIT", "tools", "lrs_splice.jl"),
               (:mle, :splice_rows))
    # the trainer's fit, so the splice is checked against the models `5_train_LinReg.jl` writes
    defs_from!(@__MODULE__, joinpath(RF, "exp_square_HIT", "5_train_LinReg.jl"), (:fit_model,))

    "Closed-form stationary variance of the AR(2) with innovation variance `s2xi`."
    ar2_gamma0(s2xi, p1, p2) = s2xi * (1 - p2) / ((1 + p2) * ((1 - p2)^2 - p1^2))

    """
        write_member(dir, k, member; nwarm = 100, nlead = 100, nq = 6, extra...)

    A minimal D6 output file with every key `load_members` reads (`t_k`, and `nlead` for
    `d6_run_identity`), in `D6Score.write_run`'s planted-step layout. `extra` adds keys, e.g.
    `diverged = true`.
    """
    function write_member(dir, k, member; nwarm = 100, nlead = 100, nq = 6, extra...)
        nt = nwarm + nlead
        q = Float32[c - 1 for _ in 1:nq, c in 1:(nt + 1)]
        dQ = Float32[c for _ in 1:nq, c in 1:nt]
        p = joinpath(dir, "d6_online_ic$(k)_m$(member).jld2")
        jldsave(p; q, dQ, tau = dQ, k, n_k = 100 * (k - 1), t_k = 0.25 * (k - 1), ordinal = 1,
                member, seed = UInt64(member), ou_advance = 100 * (k - 1), nwarm, nlead, M = 4,
                extra...)
        return p
    end
end

# 🔴 `EXCLUDE_ICS`, `THIN_MEMBERS` and `THIN_TO` are `const`s read from ENV when score_d6.jl is
# INCLUDED, so each policy needs its own copy of the file. `withenv` sets exactly the four policy
# variables for the include and restores the caller's environment afterwards, so neither module
# leaks into `D6Score` (test_d6_score.jl) or into the other, whatever order they are built in.
@testmodule D6PolicyA begin
    const ENV_BEFORE = Dict(k => get(ENV, k, nothing) for k in
                            ("D6_EXCLUDE_ICS", "D6_THIN_MEMBERS", "D6_THIN_TO", "D6_POLICY_CONTROL"))
    withenv("D6_EXCLUDE_ICS" => "46", "D6_THIN_MEMBERS" => nothing, "D6_THIN_TO" => nothing,
            "D6_POLICY_CONTROL" => nothing) do
        include(normpath(joinpath(@__DIR__, "..", "analysis", "score_d6.jl")))
    end
end

@testmodule D6PolicyB begin
    const ENV_BEFORE = Dict(k => get(ENV, k, nothing) for k in
                            ("D6_EXCLUDE_ICS", "D6_THIN_MEMBERS", "D6_THIN_TO", "D6_POLICY_CONTROL"))
    withenv("D6_EXCLUDE_ICS" => nothing, "D6_THIN_MEMBERS" => "44:2", "D6_THIN_TO" => "2",
            "D6_POLICY_CONTROL" => nothing) do
        include(normpath(joinpath(@__DIR__, "..", "analysis", "score_d6.jl")))
    end
end

# ---------------------------------------------------------------------------------------------
# 1. ar2_ls
# ---------------------------------------------------------------------------------------------

@testitem "V81 ar2_ls recovers known stationary AR(2)s from their exact ACF" default_imports = false setup = [Round1] begin
    using Test
    using .Round1: ar2_acf, ar2_ls, ar2_stationary
    # A damped oscillation with a negative lobe, a pure decay, and the rest of the triangle: both
    # real roots, complex roots, φ1 < 0, φ2 > 0, an AR(1) near the unit root and white noise.
    @test minimum(ar2_acf(1.2, -0.5, 20)) < -0.1           # the lobe is really there ...
    @test all(>(0), ar2_acf(0.6, 0.2, 20))                 # ... and the decay really has none
    for (p1, p2) in ((1.2, -0.5), (0.6, 0.2), (1.5, -0.7), (0.5, -0.3), (-0.4, 0.3), (0.3, 0.6),
                     (-0.6, -0.2), (1.83, -0.85), (0.99, 0.0), (0.0, 0.0))
        @test ar2_stationary(p1, p2)                       # the target itself
        a, b = ar2_ls(ar2_acf(p1, p2, 20); L = 20)
        @test ar2_stationary(a, b)
        @test abs(a - p1) < 5e-4
        @test abs(b - p2) < 5e-4
    end
end

@testitem "V81 ar2_ls returns a stationary pair for ACFs no stationary AR(2) fits" default_imports = false setup = [Round1] begin
    using Test
    using .Round1: ar2_acf, ar2_ls, ar2_stationary, ar_poles
    # ⚠️ White noise and an AR(1) at rho = 0.99 ARE fitted exactly, by (0, 0) and (0.99, 0), both
    # stationary -- so they are recovery cases, kept for that. The genuinely unfittable ones are the
    # unit root (rho = 1 at every lag), an undamped oscillation, and a sequence that is not an ACF
    # at all (rho1 = 0.9 with rho2 = -0.9 violates rho2 >= 2 rho1^2 - 1).
    for (r, want) in (([1.0; zeros(20)], (0.0, 0.0)), (0.99 .^ (0:20), (0.99, 0.0)))
        a, b = ar2_ls(r; L = 20)
        @test ar2_stationary(a, b)
        @test abs(a - want[1]) < 5e-4 && abs(b - want[2]) < 5e-4
    end
    for r in (ones(21), cos.(0.3 .* (0:20)), [1.0; 0.9; fill(-0.9, 19)])
        a, b = ar2_ls(r; L = 20)
        @test ar2_stationary(a, b)                         # by the code's own (strict) test
        # 🔴 ... but only to round-off. The local refinement walks onto the stationarity boundary
        # and stops one ulp inside it: rho = 1 returns (1.15, -0.15), a root at z = 1 to round-off;
        # the cosine and the non-ACF return phi2 = -1 + 1e-16. Such a pair has sigma_xi^2 ~ 0 in
        # `fit_ar`. Harmless for a decaying residual ACF, recorded here so it is not rediscovered.
        @test_broken maximum(ar_poles(a, b)) < 1 - 1e-6
    end
end

@testitem "V81 ar2_ls off the 0.01 grid: what it does and does not recover" default_imports = false setup = [Round1] begin
    using Test
    using Random
    using .Round1: ar2_acf, ar2_ls, ar2_stationary, ar_poles
    # 🔴 Measured 2026-10-06, and the reason the recovery test above uses grid values: the
    # refinement is three bounded pattern-search passes per step size, so it can move at most
    # ~0.037 from the best 0.01-grid point. In the long, flat valleys of the ACF misfit the best grid
    # point can lie further than that from the minimiser, and then the true pair -- misfit 0 -- is
    # never reached. AR(1) at rho = 0.985 comes back as (1.359, -0.369).
    a1, b1 = ar2_ls(0.985 .^ (0:20); L = 20)
    @test ar2_stationary(a1, b1)
    @test_broken abs(a1 - 0.985) < 5e-4 && abs(b1) < 5e-4
    # Random stationary AR(2)s with poles inside 0.97: every result stationary; how many miss.
    rng = Xoshiro(81)
    errs, misfit = Float64[], Float64[]
    allstat = Ref(true)
    while length(errs) < 200
        p1, p2 = 4 * rand(rng) - 2, 2 * rand(rng) - 1
        (ar2_stationary(p1, p2) && maximum(ar_poles(p1, p2)) < 0.97) || continue
        r = ar2_acf(p1, p2, 20)
        a, b = ar2_ls(r; L = 20)
        allstat[] &= ar2_stationary(a, b)
        push!(errs, max(abs(a - p1), abs(b - p2)))
        push!(misfit, maximum(abs, ar2_acf(a, b, 20)[2:end] .- r[2:end]))
    end
    @test allstat[]
    nmiss = count(>(5e-4), errs)
    @info "V81 ar2_ls on 200 off-grid AR(2)s: $nmiss miss (phi) by > 5e-4, worst $(round(maximum(errs); sigdigits = 3)); " *
          "worst ACF misfit $(round(maximum(misfit); sigdigits = 3))"
    @test_broken nmiss == 0
end

# ---------------------------------------------------------------------------------------------
# 2. Σ_ξ
# ---------------------------------------------------------------------------------------------

@testitem "V81 Σ_ξ gives the AR its residual's variance: closed form and simulation" default_imports = false setup = [Round1] begin
    using Test
    using Random
    using Statistics
    using LinearAlgebra
    using .Round1: ar_innovation_var, ar2_gamma0, sim_ar_S, NQ
    # σ_ξ² = var(z) (1 − φ1 ρ1 − φ2 ρ2), checked against γ0 = σ² (1 − φ2) / ((1 + φ2)((1 − φ2)² − φ1²)),
    # which is derived independently of the ACF.
    for (p1, p2) in ((1.2, -0.5), (0.6, 0.2), (1.5, -0.7), (-0.4, 0.3), (0.9, 0.0), (0.0, 0.0)),
        v in (0.37, 1.0, 2.5e-3)
        s = ar_innovation_var(v, p1, p2)
        @test 0 < s <= v
        @test ar2_gamma0(s, p1, p2) ≈ v rtol = 1e-12
    end
    @test ar_innovation_var(2.0, 0.9, 0.0) ≈ 2.0 * (1 - 0.9^2) rtol = 1e-14   # AR(1)
    @test ar_innovation_var(2.0, 0.0, 0.0) == 2.0                              # white

    # Simulated with the builder's own `sim_ar_S`, six bands at once, Σ_ξ = D R D with a full R.
    @test NQ == 6
    p1 = [1.2, 0.6, 1.5, -0.4, 0.9, 0.0]
    p2 = [-0.5, 0.2, -0.7, 0.3, 0.0, 0.0]
    v = [1.0, 0.37, 2.5, 0.05, 1.0, 3.0]
    A = randn(Xoshiro(1), 6, 6)
    C = A * A' + 6I
    R = C ./ sqrt.(diag(C) * diag(C)')
    D = Diagonal(sqrt.(ar_innovation_var.(v, p1, p2)))
    E = sim_ar_S(p1, p2, Matrix(Symmetric(D * R * D)), 400_000, Xoshiro(2))
    for i in 1:NQ
        @test var(E[i, :]) ≈ v[i] rtol = 0.03
    end
end

@testitem "V81 fit_ar assembles Σ_ξ = D R D with D matched to var(z)" default_imports = false setup = [Round1] begin
    using Test
    using Random
    using Statistics
    using LinearAlgebra
    using .Round1: ar_innovation_var, ar2_gamma0, sim_ar_S, fit_ar, acf, NQ
    # a residual drawn from a known diagonal AR(2) with correlated innovations, N x NQ as fit_ar takes it
    p1 = [1.2, 0.6, 1.5, -0.4, 0.9, 0.0]
    p2 = [-0.5, 0.2, -0.7, 0.3, 0.0, 0.0]
    A = randn(Xoshiro(3), 6, 6)
    S0 = Matrix(Symmetric(A * A' / 6 + I))
    Z = permutedims(sim_ar_S(p1, p2, S0, 200_000, Xoshiro(4)))
    for p in (2, 1)
        f = fit_ar(Z, p)
        @test size(f.phi) == (p, NQ)
        @test f.s2 == vec(var(Z; dims = 1))
        @test f.sxi2 == ar_innovation_var.(f.s2, f.p1, f.p2)
        @test f.R ≈ cor(f.Xi)
        @test all(i -> f.R[i, i] ≈ 1, 1:NQ)
        Dx = Diagonal(sqrt.(f.sxi2))
        @test f.S ≈ Dx * f.R * Dx rtol = 1e-12
        @test diag(f.S) ≈ f.sxi2 rtol = 1e-12
        @test isposdef(Symmetric(f.S))
        # 🔑 the paper's claim: the fitted AR, driven by Σ_ξ, has each band's residual variance
        for i in 1:NQ
            @test ar2_gamma0(f.sxi2[i], f.p1[i], f.p2[i]) ≈ f.s2[i] rtol = 1e-10
        end
        if p == 1
            @test f.p1 ≈ [acf(Z[:, i], 1:1)[1] for i in 1:NQ]
            @test all(iszero, f.p2)
        end
        # ... and so does its simulation
        Ef = sim_ar_S(f.p1, f.p2, f.S, 400_000, Xoshiro(5))
        for i in 1:NQ
            @test var(Ef[i, :]) ≈ f.s2[i] rtol = 0.03
        end
    end
end

# ---------------------------------------------------------------------------------------------
# 3. the splice
# ---------------------------------------------------------------------------------------------

@testitem "V81 splice: one source is that source; a ridge row is the single-column ridge; Σ is the MLE" default_imports = false setup = [Round1] begin
    using Test
    using Random
    using Statistics
    using LinearAlgebra
    using Distributions
    using .Round1: fit_model, splice_rows
    # 300 rows, 6 correlated features + the bias = 7 regressors, 3 outputs with correlated noise.
    rng = Xoshiro(81)
    N, nf, nq = 300, 6, 3
    inputs = randn(rng, nf, nf) * randn(rng, nf, N) .+ 0.3
    X = vcat(inputs, ones(1, N))                       # the tool's layout: regressors x N, bias last
    Y = randn(rng, nq, nf + 1) * X .+ [1.0 0 0; 0.6 0.8 0; -0.3 0.2 0.9] * randn(rng, nq, N)
    lams = Dict("A" => 0.0, "B" => 0.5, "C" => 50.0)

    # argmin_c ||y - c X||^2 + λ ||P c||^2 for ONE output column, solved on its own
    function ridge_col(y, λ, pen)
        m = size(X, 1)
        P = sqrt(λ) * Matrix(1.0I, m, m)
        pen || (P[m, m] = 0)
        return vec([X'; P] \ [y; zeros(m)])
    end

    for pen in (false, true)          # the exactness does not hinge on the intercept convention
        # the sources, as `5_train_LinReg.jl` fits and stores them (`c'`, and `fit(MvNormal, ·)`)
        fits = Dict(s => fit_model(inputs, Y, 1:nq; lambda = λ, ridge_solver = :exact,
                                   penalize_intercept = pen) for (s, λ) in lams)
        cs = Dict(s => fits[s][1]' for s in keys(lams))
        sd = Dict(s => fits[s][2] for s in keys(lams))

        # (a) all rows from one source: that source's c, bitwise, and its stored noise
        for s in keys(lams)
            r = splice_rows(cs, fill(s, nq), X, Y)
            @test r.c == Matrix(cs[s])
            @test r.mu ≈ mean(sd[s]) atol = 1e-12
            @test r.S ≈ cov(sd[s]) rtol = 1e-10
        end

        # (b) a mixed splice: row i is copied from its source, and IS the single-column ridge at
        #     that source's λ -- the ridge problem decouples by output column
        srcs = ["A", "C", "B"]
        r = splice_rows(cs, srcs, X, Y)
        Dp = Diagonal([ones(nf); pen ? 1.0 : 0.0])
        for i in 1:nq
            λ = lams[srcs[i]]
            @test r.c[i, :] == Matrix(cs[srcs[i]])[i, :]
            @test r.c[i, :] ≈ ridge_col(Y[i, :], λ, pen) rtol = 1e-10
            g = X * (X' * r.c[i, :] .- Y[i, :]) .+ λ .* (Dp * r.c[i, :])   # the normal equations
            @test norm(g) <= 1e-9 * norm(X * Y[i, :])
        end
        @test !(r.c[2, :] ≈ Matrix(cs["A"])[2, :])     # it is a splice, not one of the sources
        # the trainer's own per-QoI λ path (one exact solve per column) gives the same model
        cq, sq = fit_model(inputs, Y, 1:nq; lambda_per_qoi = [lams[s] for s in srcs],
                           penalize_intercept = pen)
        @test cq' ≈ r.c rtol = 1e-10

        # (c) Σ is the MLE of the spliced residual: mean, and covariance over N, not N - 1
        Rs = Y .- r.c * X
        Rc = Rs .- mean(Rs; dims = 2)
        @test r.mu ≈ vec(mean(Rs; dims = 2)) atol = 1e-12
        @test r.S ≈ Rc * Rc' / N rtol = 1e-12
        @test r.S ≈ cov(Rs; dims = 2) * (N - 1) / N rtol = 1e-12
        @test !(r.S ≈ cov(Rs; dims = 2))
        @test r.S ≈ cov(fit(MvNormal, Rs)) rtol = 1e-10
        @test r.S ≈ cov(sq) rtol = 1e-10                # = the trainer's refit of the per-QoI model
        # the diagonal travels with the rows; the off-diagonal in general does not, which is why Σ
        # is refitted. Between the two ridge rows (C and B) it matches neither source ...
        for i in 1:nq
            @test r.S[i, i] ≈ cov(sd[srcs[i]])[i, i] rtol = 1e-10
        end
        @test !(r.S[2, 3] ≈ cov(sd["B"])[2, 3]) && !(r.S[2, 3] ≈ cov(sd["C"])[2, 3])
        # ... while the unpenalised row A's residual is orthogonal to the design (bias included),
        # so its covariance with any other row is A's own, whichever source that row comes from.
        @test r.S[1, 2] ≈ cov(sd["A"])[1, 2] rtol = 1e-10
        @test r.S[1, 3] ≈ cov(sd["A"])[1, 3] rtol = 1e-10
        @test_throws ErrorException splice_rows(cs, ["A", "B"], X, Y)     # one source per QoI
    end
end

# ---------------------------------------------------------------------------------------------
# 5./6. the criterion: S, C and the KS guard
# ---------------------------------------------------------------------------------------------

@testitem "V81 criterion_clauses: S and C, strict at 0 and at the margin" default_imports = false setup = [D6Score] begin
    using Test
    cc = D6Score.criterion_clauses
    # S iff the CRPS_0.5 interval's upper end is below 0
    @test cc(-1e-9, 1.0, -5.0).S
    @test !cc(0.0, 1.0, -5.0).S                       # AT 0 is not below it
    @test !cc(-0.0, 1.0, -5.0).S
    @test !cc(0.3, 1.0, -5.0).S
    # hi_pct = 100 p05_hi / score_b
    @test cc(0.05, 2.0, 1.0).hi_pct ≈ 2.5
    @test cc(-1, 10, 1).hi_pct == -10.0               # integers are fine
    # C needs BOTH the count interval above 0 and hi_pct below the margin
    @test cc(0.05, 1.0, 1.0).C                        # 5 % < 10 %, count resolved
    @test !cc(0.05, 1.0, 0.0).C                       # count CI touching 0 is not above it
    @test !cc(0.05, 1.0, -2.0).C
    @test !cc(0.5, 1.0, 3.0).C                        # 50 % is past the margin
    @test cc(-0.5, 1.0, 3.0).C && cc(-0.5, 1.0, 3.0).S
    # exactly at the margin: hi_pct == 10 must fail, one ulp below must pass
    at = cc(2.0, 20.0, 1.0)
    @test at.hi_pct == 10.0
    @test !at.C
    @test cc(prevfloat(2.0), 20.0, 1.0).C
    # the margin is a keyword, and S does not depend on it (values exact in binary: 5 % and -1 %)
    @test cc(0.25, 5.0, 1.0).hi_pct == 5.0
    @test !cc(0.25, 5.0, 1.0; margin = 5.0).C
    @test cc(0.25, 5.0, 1.0; margin = 5.0 + 1e-9).C
    @test cc(-0.5, 50.0, 1.0).hi_pct == -1.0
    @test !cc(-0.5, 50.0, 1.0; margin = -1.0).C && cc(-0.5, 50.0, 1.0; margin = -1.0).S
    @test !cc(0.0, 1.0, 1.0; margin = 0.0).C
    @test keys(cc(0.1, 1.0, 1.0)) == (:S, :C, :hi_pct)
end

@testitem "V81 ks_guard fails only when every closure run is above every reference run" default_imports = false setup = [D6Score] begin
    using Test
    g = D6Score.ks_guard
    ref = [1.0, 1.2, 1.1, 0.9, 1.3]
    @test g([0.5, 0.6, 0.7, 0.8, 0.85], ref).pass         # closure better
    @test g([1.0, 1.5, 1.6, 1.7, 1.8], ref).pass          # overlapping ranges
    @test g(ref, ref).pass                                # identical
    @test !g([1.31, 1.4, 2.0, 1.5, 1.6], ref).pass        # separated: every closure value above
    # the boundary: min(closure) == max(ref) is NOT above it, so it passes
    @test g([1.3, 1.4, 2.0, 1.5, 1.6], ref).pass
    @test !g([nextfloat(1.3), 1.4, 2.0, 1.5, 1.6], ref).pass
    r = g([1.31, 1.4, 2.0], [0.2, 1.2])                   # lengths need not match
    @test r == (; pass = false, min_closure = 1.31, max_ref = 1.2)
    @test_throws ArgumentError g(Float64[], ref)
    @test_throws ArgumentError g(ref, Float64[])
    # 🔑 p = 1/252: over all C(10, 5) = 252 ways to split ten distinct values 5 against 5 --
    # each equally likely for exchangeable runs -- exactly one fails the guard.
    vals = collect(1.0:10.0)
    splits = [m for m in 0:1023 if count_ones(m) == 5]
    @test length(splits) == 252
    nfail = count(splits) do m
        c = [vals[j] for j in 1:10 if (m >> (j - 1)) & 1 == 1]
        rr = [vals[j] for j in 1:10 if (m >> (j - 1)) & 1 == 0]
        !g(c, rr).pass
    end
    @test nfail == 1
end

@testitem "V81 paired_compare's S and C are criterion_clauses of its own interval ends" default_imports = false setup = [D6Score] begin
    using Test
    using JLD2
    using Random
    # Synthetic pair: members around a per-cell shifted truth, so A (member sd = shift sd) is
    # calibrated and B (member sd 8 against 30) under-dispersed. Same planted layout as V71.
    truth = D6Score.planted_truth(6, 30000)
    ks = collect(209:2:247)
    nwarm, nlead, nq, M = 100, 400, 6, 3
    nt = nwarm + nlead
    function run!(dir, sd; seed)
        rng, drng = Xoshiro(seed), Xoshiro(99)
        for k in ks
            n_k = 100 * (k - 1)
            del = 30 .* randn(drng, nq, nt + 1)
            for m in 1:M
                q = [Float64(n_k + c - 1) + del[i, c] + sd * randn(rng) for i in 1:nq, c in 1:(nt + 1)]
                jldsave(joinpath(dir, "d6_online_ic$(k)_m$(m).jld2"); q, dQ = zeros(nq, nt),
                        tau = zeros(nq, nt), k, n_k, t_k = 0.25 * (k - 1), ordinal = 1, member = m,
                        seed = UInt64(m), nwarm, nlead, M, closure = "lstm", model_name = "sd$(sd)",
                        block = "selection")
            end
        end
    end
    mktempdir() do root
        a, b = mkpath(joinpath(root, "a")), mkpath(joinpath(root, "b"))
        run!(a, 30.0; seed = 4)
        run!(b, 8.0; seed = 5)
        for (x, y) in ((a, b), (b, a))
            io = IOBuffer()
            r = D6Score.paired_compare(x, y; truth, nboot = 300, ndraw = 10, io)
            want = D6Score.criterion_clauses(r.crps05.hi, r.crps05.score_b, r.calibration.lo;
                                             margin = r.margin)
            @test (r.S, r.C, r.crps05_hi_pct) === (want.S, want.C, want.hi_pct)
            txt = String(take!(io))
            @test occursin("S  CRPS_0.5 CI upper end", txt) && occursin("-> improves on B iff G and (S or C)", txt)
            x == a && @test r.C              # the calibrated side resolves the count clause
        end
    end
end

# ---------------------------------------------------------------------------------------------
# 7. load_members' exclusion policies
# ---------------------------------------------------------------------------------------------

@testitem "V81 policy A: D6_EXCLUDE_ICS drops the IC, after the divergence census" default_imports = false setup = [Round1, D6PolicyA] begin
    using Test
    P = D6PolicyA
    @test P.EXCLUDE_ICS == Set([46])
    @test isempty(P.THIN_MEMBERS) && P.THIN_TO == 0 && !P.THINNING
    @test P.POLICY_TAG == ""
    # the include did not leave the policy in the process environment
    @test all(get(ENV, k, nothing) == v for (k, v) in P.ENV_BEFORE)
    filt = P.ic_filter()
    mktempdir() do dir
        for k in (42, 44, 46, 48), m in 1:4
            Round1.write_member(dir, k, m)
        end
        ens = P.load_members(dir; filt)
        @test ens.ks == [42, 44, 48]                    # 46 is gone ...
        @test !haskey(ens.files, 46)
        @test ens.M == 4                                # ... and nothing else changed
        @test all(ens.member_ids[k] == 1:4 for k in ens.ks)
        @test isempty(ens.thinned) && isempty(ens.divergences)
    end
    # the excluded IC's divergence is still counted: the census describes the run
    mktempdir() do dir
        for k in (42, 44, 46, 48), m in 1:4
            Round1.write_member(dir, k, m; (k == 46 && m == 3 ? (; diverged = true) : (;))...)
        end
        ens = P.load_members(dir; filt)
        @test ens.ks == [42, 44, 48]
        @test ens.divergences == Dict(46 => [3])
        @test isempty(ens.incomplete)
    end
    # excluding an IC the run does not hold is not an error
    mktempdir() do dir
        for k in (42, 44), m in 1:4
            Round1.write_member(dir, k, m)
        end
        @test P.load_members(dir; filt).ks == [42, 44]
    end
end

@testitem "V81 policy B: D6_THIN_MEMBERS drops the named member, D6_THIN_TO sets M" default_imports = false setup = [Round1, D6PolicyB] begin
    using Test
    P = D6PolicyB
    @test P.THIN_MEMBERS == Dict(44 => Set([2])) && P.THIN_TO == 2 && P.THINNING
    @test isempty(P.EXCLUDE_ICS)
    @test P.POLICY_TAG == "_thin"
    @test all(get(ENV, k, nothing) == v for (k, v) in P.ENV_BEFORE)
    filt = P.ic_filter()
    mktempdir() do dir
        for k in (42, 44, 46, 48), m in 1:4
            Round1.write_member(dir, k, m)
        end
        ens = P.load_members(dir; filt)
        @test ens.ks == [42, 44, 46, 48]                # every IC kept
        # 🔑 M = THIN_TO = 2, not the derived minimum (3 after the named drop)
        @test ens.M == 2
        @test ens.member_ids[44] == [1, 3]              # member 2 named, then the highest id
        @test ens.thinned[44] == [2, 4]
        for k in (42, 46, 48)
            @test ens.member_ids[k] == [1, 2]           # thinned from the top
            @test ens.thinned[k] == [3, 4]
        end
        @test [first(e) for e in ens.files[44]] == [1, 3]
        @test isempty(ens.divergences)
    end
    # the named member already diverged: not an error, and not a second drop
    mktempdir() do dir
        for k in (42, 44, 46, 48), m in 1:4
            Round1.write_member(dir, k, m; (k == 44 && m == 2 ? (; diverged = true) : (;))...)
        end
        ens = P.load_members(dir; filt)
        @test ens.divergences == Dict(44 => [2])
        @test ens.member_ids[44] == [1, 3] && ens.thinned[44] == [4]
        @test ens.M == 2
    end
    # a spec that does not match the run is refused
    mktempdir() do dir                                   # no IC 44
        for k in (42, 46), m in 1:4
            Round1.write_member(dir, k, m)
        end
        @test_throws ErrorException P.load_members(dir; filt)
    end
    mktempdir() do dir                                   # IC 44 without a member 2, not diverged
        for k in (42, 44), m in (1, 3, 4)
            Round1.write_member(dir, k, m)
        end
        @test_throws ErrorException P.load_members(dir; filt)
    end
end
