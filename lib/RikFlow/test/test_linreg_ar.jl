# V73 -- the optional AR(p) residual of `LinReg` (M0ᶜ-ridge, analysis/results_LSTMS.md §12).
#
# `time_series_methods.jl` is included bare with a stub `adapt` (this suite has no Adapt/Lux; only
# the CPU path, `ArrayType = Array`, is exercised, where `adapt(Array, x) === x` anyway). Pinned:
#
#   * 🔴 without the AR keys, `LinReg` is BIT-IDENTICAL to the pre-AR code -- outputs and RNG stream --
#     against a verbatim copy of that code (`legacy_linreg.jl`, from the 2026-09-28 HEAD);
#   * the AR path produces the AR's ACF and marginal variance on a synthetic model;
#   * the warm start: the state after the replay is the residual the record realised on the last p
#     warm-up steps; the fallback (short warm-up) draws a stationary state;
#   * the turbulence gate zeroes dQ but still advances the AR state (the stream is not shifted);
#   * a variant file (source keys + ar_*) round-trips, and malformed/non-stationary AR is refused.

@testmodule LinRegAR begin
    using LinearAlgebra
    using Statistics
    using Random
    using JLD2
    using Distributions

    const RF = normpath(joinpath(@__DIR__, ".."))
    const SRC = joinpath(RF, "src")
    const LRS = joinpath(RF, "exp_square_HIT", "output", "TO_LRS")

    # stubs for the non-stdlib name the file uses on the CPU path (Adapt is identity on Array)
    adapt(::Any, x) = x
    adapt(::Any) = identity
    include(joinpath(SRC, "ts_scaling.jl"))
    include(joinpath(SRC, "time_series_methods.jl"))
    include(joinpath(@__DIR__, "legacy_linreg.jl"))

    acf(x, L) = (y = x .- mean(x); v = sum(abs2, y); [sum(y[1:(end - l)] .* y[(1 + l):end]) / v for l in L])

    "AR(2) ACF, lags 0:L."
    function ar2_acf(p1, p2, L)
        r = zeros(L + 1); r[1] = 1; r[2] = p1 / (1 - p2)
        for k in 2:L
            r[k + 1] = p1 * r[k] + p2 * r[k - 1]
        end
        return r
    end

    """
    A synthetic `:q_star_q` LinReg file (h, n QoIs, predictor included) with random `c` and a
    full-covariance white Σ; optional AR keys.
    """
    function synth_file(path; n = 3, h = 2, seed = 11, phi = nothing, S_xi = nothing, cscale = 0.05,
                        sig_eta = 0.1)
        rng = Xoshiro(seed)
        c = cscale .* randn(rng, n, n * (2h + 1) + 1)
        A = randn(rng, n, n)
        Sw = sig_eta^2 .* (A * A' ./ n + I)
        stoch_distr = MvNormal(0.01 .* randn(rng, n), Sw)
        sc = (; mu = reshape(1 .+ rand(rng, n), n, 1), sigma = reshape(0.2 .+ rand(rng, n), n, 1))
        jldopen(path, "w") do f
            f["c"] = c; f["stoch_distr"] = stoch_distr
            f["scaling"] = (; in_scaling = sc, out_scaling = sc)
            f["hist_var"] = :q_star_q; f["hist_len"] = h; f["include_predictor"] = true
            f["fitted_qois"] = collect(1:n)
            phi === nothing || (f["ar_phi"] = phi)
            S_xi === nothing || (f["ar_sigma_xi"] = S_xi)
        end
        return (; c, stoch_distr, sc, n, h)
    end

    "q* trajectory around the scaling mean; `gate_at` steps get one component below the gate."
    function qstar_path(n, N; seed = 3, gate_at = Int[])
        rng = Xoshiro(seed)
        Q = 1.0 .+ 0.3 .* rand(rng, n, N)
        for t in gate_at
            Q[end, t] = 1e-3
        end
        return Q
    end

    "Drive a closure along q* columns, returning dQ (n x N)."
    drive(m, Q) = reduce(hcat, [copy(get_next_item_timeseries(m, Q[:, t])) for t in axes(Q, 2)])

    new_closure(T, file, seed, n, h, spin) = T(file, Xoshiro(seed), Array; q_hist = zeros(2n, h),
                                               spinnup_data = copy(spin))
end

@testitem "V73 LinReg without AR is bit-identical to the pre-AR code (outputs and RNG)" default_imports = false setup = [LinRegAR] begin
    using Test, Random
    using .LinRegAR: synth_file, qstar_path, drive, new_closure, LegacyLinReg, LinReg, LRS, ar_order,
                     TURBULENCE_GATE
    mktempdir() do d
        f = joinpath(d, "LinReg.jld2")
        s = synth_file(f; n = 3, h = 2)
        spin = 0.01 .* randn(Xoshiro(9), 3, 10)
        Q = qstar_path(3, 300; gate_at = [50, 51, 200])
        for seed in (1, 2, 12345)
            a = new_closure(LinReg, f, seed, 3, 2, spin)
            b = new_closure(LegacyLinReg, f, seed, 3, 2, spin)
            @test a.ar === nothing && ar_order(a) == 0
            A, B = drive(a, Q), drive(b, Q)
            @test A == B                                 # bit-identical dQ
            @test A[:, 1:10] == spin                     # the replay is verbatim
            @test all(A[:, 50] .== 0) && all(A[:, 200] .== 0) && any(A[:, 49] .!= 0)
            @test a.q_hist == b.q_hist
            @test rand(a.rng) === rand(b.rng)            # the RNG stream is at the same point
        end
    end
    # the archived deployed files, where present (gitignored data; skipped on a clean clone)
    nfiles = Ref(0)
    for name in ("LinReg1", "LinReg7")
        f = joinpath(LRS, name, "LinReg.jld2")
        isfile(f) || continue
        nfiles[] += 1
        h = LinRegAR.load(f, "hist_len")
        mu = vec(LinRegAR.load(f, "scaling").in_scaling.mu)
        spin = 0.001 .* mu .* randn(Xoshiro(4), 6, 100)
        # an open loop with no physics drifts off (NaN) after a few hundred steps, so keep it short
        # and compare with isequal (NaN-safe, bitwise on finite values)
        Q = mu .* (1 .+ 0.1 .* randn(Xoshiro(5), 6, 160))
        a = new_closure(LinReg, f, 7, 6, h, spin)
        b = new_closure(LegacyLinReg, f, 7, 6, h, spin)
        A, B = drive(a, Q), drive(b, Q)
        @test isequal(A, B)
        @test all(isfinite, A)
        @test rand(a.rng) === rand(b.rng)
    end
    @info "V73 bit-identity checked on $(nfiles[]) archived LinReg files"
end

@testitem "V73 AR(2) residual reproduces its ACF and marginal variance" default_imports = false setup = [LinRegAR] begin
    using Test, Random, Statistics, LinearAlgebra
    using .LinRegAR: synth_file, qstar_path, new_closure, LinReg, get_next_item_timeseries, acf, ar2_acf,
                     ar_order, linreg_data
    mktempdir() do d
        f = joinpath(d, "LinReg.jld2")
        n, h, nw = 3, 2, 10
        phi = [1.2 0.5 0.3; -0.4 0.2 0.0]          # all stationary
        S_xi = [1.0 0.3 0.0; 0.3 1.0 -0.2; 0.0 -0.2 1.0] .* 1e-2
        s = synth_file(f; n, h, phi, S_xi)
        m = new_closure(LinReg, f, 1, n, h, 0.01 .* randn(Xoshiro(9), n, nw))
        @test ar_order(m) == 2
        N = 100_000
        Q = qstar_path(n, N)
        Z = zeros(n, N - nw)
        for t in 1:N
            mu = vec(s.c * linreg_data(m, Q[:, t]))      # the model's own mean at this step
            dQ = get_next_item_timeseries(m, Q[:, t])
            if t > nw
                lev = (Q[:, t] .+ dQ .- vec(s.sc.mu)) ./ vec(s.sc.sigma)
                Z[:, t - nw] = lev .- mu .- mean(s.stoch_distr)
            end
        end
        for i in 1:n
            r = ar2_acf(phi[1, i], phi[2, i], 5)
            @test acf(Z[i, :], [1, 2, 5]) ≈ r[[2, 3, 6]] atol = 0.03
            v = S_xi[i, i] / (1 - phi[1, i] * r[2] - phi[2, i] * r[3])
            @test var(Z[i, :]) ≈ v rtol = 0.06
        end
        # the innovations recovered from the emitted noise have covariance Σ_ξ
        Xi = Z[:, 3:end] .- phi[1, :] .* Z[:, 2:(end - 1)] .- phi[2, :] .* Z[:, 1:(end - 2)]
        @test cov(Xi; dims = 2) ≈ S_xi atol = 5e-4
    end
end

@testitem "V73 AR warm start from the replayed residual; short-warm-up fallback" default_imports = false setup = [LinRegAR] begin
    using Test, Random, Statistics, LinearAlgebra
    using .LinRegAR: synth_file, qstar_path, new_closure, LinReg, get_next_item_timeseries, linreg_data,
                     AR_BURNIN
    mktempdir() do d
        f = joinpath(d, "LinReg.jld2")
        n, h, nw = 3, 2, 10
        phi = [0.9 0.5 0.3; -0.2 0.1 0.0]
        S_xi = Matrix(0.01I, 3, 3)
        s = synth_file(f; n, h, phi, S_xi)
        spin = 0.05 .* randn(Xoshiro(9), n, nw)
        Q = qstar_path(n, 40)
        m = new_closure(LinReg, f, 1, n, h, spin)
        zref = zeros(n, nw)
        for t in 1:nw
            data = linreg_data(m, Q[:, t])             # the history the model holds BEFORE the step
            lev = (Q[:, t] .+ spin[:, t] .- vec(s.sc.mu)) ./ vec(s.sc.sigma)
            zref[:, t] = lev .- vec(s.c * data) .- mean(s.stoch_distr)
            @test get_next_item_timeseries(m, Q[:, t]) == spin[:, t]     # replay verbatim
            t <= nw - 2 && @test m.ar.nz[] == 0
        end
        @test m.ar.nz[] == 2
        @test m.ar.z[:, 1] ≈ zref[:, nw] rtol = 1e-12
        @test m.ar.z[:, 2] ≈ zref[:, nw - 1] rtol = 1e-12
        @test rand(m.rng) === rand(Xoshiro(1))         # the replay drew nothing

        # the first forecast step continues from the data state: z_1 = φ1 z_0 + φ2 z_-1 + ξ
        m2 = new_closure(LinReg, f, 1, n, h, spin)
        for t in 1:nw
            get_next_item_timeseries(m2, Q[:, t])
        end
        z0, zm1 = copy(m2.ar.z[:, 1]), copy(m2.ar.z[:, 2])
        get_next_item_timeseries(m2, Q[:, nw + 1])
        xi = rand(Xoshiro(1), m2.ar.xi_distr)
        @test m2.ar.z[:, 1] ≈ phi[1, :] .* z0 .+ phi[2, :] .* zm1 .+ xi rtol = 1e-12
        @test m2.ar.z[:, 2] == z0

        # fallback: a warm-up of h + 1 steps fills only ONE lag (the first h have no full history)
        m3 = new_closure(LinReg, f, 1, n, h, spin[:, 1:(h + 1)])
        for t in 1:(h + 1)
            get_next_item_timeseries(m3, Q[:, t])
        end
        @test m3.ar.nz[] == 1
        get_next_item_timeseries(m3, Q[:, h + 2])
        @test m3.ar.nz[] == 2
        r = Xoshiro(1)                                  # AR_BURNIN + 1 innovation draws consumed
        for _ in 1:(AR_BURNIN + 1)
            rand(r, m3.ar.xi_distr)
        end
        @test rand(m3.rng) === rand(r)
    end
end

@testitem "V73 turbulence gate zeroes dQ but advances the AR state" default_imports = false setup = [LinRegAR] begin
    using Test, Random, LinearAlgebra
    using .LinRegAR: synth_file, qstar_path, new_closure, LinReg, get_next_item_timeseries
    mktempdir() do d
        f = joinpath(d, "LinReg.jld2")
        n, h, nw = 3, 2, 10
        s = synth_file(f; n, h, phi = [0.8 0.5 0.3; 0.1 0.0 0.0], S_xi = Matrix(0.01I, 3, 3))
        spin = 0.05 .* randn(Xoshiro(9), n, nw)
        gated = [15, 16, 30]
        Qg = qstar_path(n, 60; gate_at = gated)
        Qu = qstar_path(n, 60)
        a = new_closure(LinReg, f, 5, n, h, spin)
        b = new_closure(LinReg, f, 5, n, h, spin)
        za, zb = [], []
        for t in 1:60
            dg = get_next_item_timeseries(a, Qg[:, t])
            get_next_item_timeseries(b, Qu[:, t])
            t in gated && @test all(dg .== 0)
            t > nw && (push!(za, copy(a.ar.z[:, 1])); push!(zb, copy(b.ar.z[:, 1])))
        end
        # after the warm-up the noise process does not depend on q*: the gated run's AR state is
        # the ungated run's, step for step (the warm start agrees because Q agrees there)
        @test Qg[:, 1:nw] == Qu[:, 1:nw]
        @test za == zb
        @test za[15 - nw] != za[14 - nw]                 # it moved on the gated step
        @test rand(a.rng) === rand(b.rng)
    end
end

@testitem "V73 AR keys: variant file round trip, and malformed AR refused" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2
    using .LinRegAR: synth_file, LinReg, LRS, ar_order, load_ar_residual
    mktempdir() do d
        f = joinpath(d, "LinReg.jld2")
        s = synth_file(f; n = 3, h = 2)
        g = joinpath(d, "LinReg_ar.jld2")
        cp(f, g)
        phi = [0.9 0.5 0.3; -0.2 0.1 0.0]
        S = [1.0 0.2 0.0; 0.2 1.0 0.0; 0.0 0.0 1.0] .* 1e-3
        jldopen(g, "a+") do fh
            fh["ar_phi"] = phi; fh["ar_sigma_xi"] = S
        end
        a, b = load(f), load(g)
        @test all(isequal(a[k], b[k]) || k == "stoch_distr" for k in keys(a))
        @test Matrix(a["stoch_distr"].Σ) == Matrix(b["stoch_distr"].Σ)
        @test b["ar_phi"] == phi && b["ar_sigma_xi"] == S
        m = LinReg(g, Xoshiro(1), Array; q_hist = zeros(6, 2), spinnup_data = zeros(3, 5))
        @test ar_order(m) == 2 && m.ar.phi == phi && Matrix(m.ar.xi_distr.Σ) == S
        # malformed: sigma without phi, phi without sigma, non-stationary, order 3
        for (keys_, vals) in ((("ar_sigma_xi",), (S,)), (("ar_phi",), (phi,)),
                              (("ar_phi", "ar_sigma_xi"), ([0.6 0.5 0.3; 0.5 0.1 0.0], S)),
                              (("ar_phi", "ar_sigma_xi"), (zeros(3, 3), S)))
            h = joinpath(d, "bad.jld2"); cp(f, h; force = true)
            jldopen(h, "a+") do fh
                for (k, v) in zip(keys_, vals)
                    fh[k] = v
                end
            end
            @test_throws ErrorException load_ar_residual(h, s.stoch_distr)
        end
    end
    # the built variant(s), where present: source keys untouched, AR keys stationary and loadable
    for v in ("LinReg7_ar2", "LinReg7_ar1")
        f = joinpath(LRS, v, "LinReg.jld2")
        isfile(f) || continue
        a, b = load(joinpath(LRS, "LinReg7", "LinReg.jld2")), load(f)
        @test all(isequal(a[k], b[k]) || k == "stoch_distr" for k in keys(a))
        @test a["stoch_distr"].μ == b["stoch_distr"].μ && Matrix(a["stoch_distr"].Σ) == Matrix(b["stoch_distr"].Σ)
        @test sort(collect(setdiff(keys(b), keys(a)))) == ["ar_phi", "ar_provenance", "ar_sigma_xi"]
        m = LinReg(f, Xoshiro(1), Array; q_hist = zeros(12, 5), spinnup_data = zeros(6, 100))
        @test ar_order(m) == parse(Int, v[end:end])
        @test b["ar_provenance"].source == "LinReg7"
    end
end
