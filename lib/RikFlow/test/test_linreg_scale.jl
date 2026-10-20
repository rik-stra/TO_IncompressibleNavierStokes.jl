# V83 -- step 2's power-law scale of the LinReg residual (paper Sec. 6.3, Eq. power law; 2026-10-07;
# step 4p before the renumbering of 2026-10-13).
#
# Pinned, each against an answer known before the code runs:
#
#   * `fit_powerlaw_scale` (src/ts_scale.jl) recovers β and the variance at q_ref from synthetic
#     heteroscedastic data; `powerlaw_factor` clamps to the clip range and is exactly 1 at β = 0;
#   * 🔴 a file with the scale keys at β = 0 and Σ_ε = Σ draws BIT-IDENTICALLY to the white file,
#     outputs and RNG stream, through `get_next_item_timeseries` (so step 2 shares LinReg1's random
#     numbers, and β = 0 is LinReg1);
#   * at β ≠ 0 the draw is μ_η + f(q*) ⊙ (the white draw's deviation), with the clamp applied;
#   * at β ≠ 0, THROUGH `get_next_item_timeseries`: the residual each step realises is μ_η + f(q*) ⊙
#     the white closure's deviation with q* the predictor in physical units -- not the scaled q*,
#     and not the previous level in the history (the β = 0 test cannot see the state, and the test
#     above calls `draw_eta` directly; review of 2026-10-15);
#   * the loader refuses a partial key set, AR + scale, wrong sizes and a non-positive reference;
#   * the built `TO_LRS/LinReg1_pl` (where present): LinReg1's keys untouched, exactly the six
#     `scale_*` keys added, the fit the paper quotes, and LinReg1's standard normals at every draw.
#
# Uses V73's `LinRegAR` module (time_series_methods.jl included bare, with ts_scale.jl).

@testitem "V83 fit_powerlaw_scale recovers β and the variance at q_ref; powerlaw_factor clamps" default_imports = false setup = [LinRegAR] begin
    using Test, Random, Statistics
    using .LinRegAR: fit_powerlaw_scale, powerlaw_factor
    rng = Xoshiro(21)
    N = 400_000
    for (beta, a) in ((1.5, log(0.04)), (0.5, log(2.0)), (0.0, log(1.0)), (-0.7, log(0.3)))
        q = 2.0 .* exp.(0.3 .* randn(rng, N))
        qref = exp(mean(log.(q)))
        e = exp(a / 2) .* (q ./ qref) .^ beta .* randn(rng, N)
        f = fit_powerlaw_scale(e, q)
        @test f.qref ≈ qref
        @test f.qclip == (minimum(q), maximum(q))
        @test abs(f.beta - beta) < 0.01
        @test abs(f.logvar - a) < 0.01
    end
    @test_throws ErrorException fit_powerlaw_scale(randn(rng, 10), [1.0; -ones(9)])
    beta, qref, qclip = [0.0, 2.0, 1.0], [1.0, 1.0, 2.0], [0.5 1.5; 0.5 1.5; 1.0 3.0]
    @test powerlaw_factor([7.0, 1.2, 2.5], beta, qref, qclip) == [1.0, 1.2^2, 1.25]
    @test powerlaw_factor([1.0, 9.0, 0.1], beta, qref, qclip) == [1.0, 1.5^2, 0.5]   # clamped above / below
end

@testitem "V83 β = 0 with Σ_ε = Σ is the white LinReg, bit for bit (outputs and RNG)" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2, LinearAlgebra
    using .LinRegAR: synth_file, qstar_path, drive, new_closure, LinReg
    mktempdir() do d
        fw, fs = joinpath(d, "white.jld2"), joinpath(d, "scaled.jld2")
        s = synth_file(fw; n = 3, h = 2)
        cp(fw, fs)
        jldopen(fs, "a+") do f
            f["scale_beta"] = zeros(3); f["scale_qref"] = [1.0, 1.1, 0.9]
            f["scale_qclip"] = [0.5 2.0; 0.5 2.0; 0.5 2.0]
            f["scale_sigma_eps"] = Matrix(s.stoch_distr.Σ)
        end
        spin = 0.01 .* randn(Xoshiro(9), 3, 10)
        Q = qstar_path(3, 300; gate_at = [50, 200])
        for seed in (1, 2, 12345)
            a = new_closure(LinReg, fw, seed, 3, 2, spin)
            b = new_closure(LinReg, fs, seed, 3, 2, spin)
            @test a.scale === nothing && b.scale !== nothing
            A, B = drive(a, Q), drive(b, Q)
            @test A == B
            @test rand(a.rng) === rand(b.rng)
        end
    end
end

@testitem "V83 β ≠ 0: the draw is μ_η + f(q*) ⊙ the white deviation, clamp included" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2, Statistics
    using .LinRegAR: synth_file, new_closure, LinReg, draw_eta, powerlaw_factor
    mktempdir() do d
        fw, fs = joinpath(d, "white.jld2"), joinpath(d, "scaled.jld2")
        s = synth_file(fw; n = 3, h = 2)
        cp(fw, fs)
        beta, qref, qclip = [1.0, 2.0, 0.5], [1.0, 1.2, 1.1], [0.6 1.4; 0.6 1.4; 0.6 1.4]
        jldopen(fs, "a+") do f
            f["scale_beta"] = beta; f["scale_qref"] = qref; f["scale_qclip"] = qclip
            f["scale_sigma_eps"] = Matrix(s.stoch_distr.Σ)
        end
        spin = zeros(3, 10)
        a = new_closure(LinReg, fw, 4, 3, 2, spin)
        b = new_closure(LinReg, fs, 4, 3, 2, spin)
        mu = mean(s.stoch_distr)
        rng = Xoshiro(8)
        for _ in 1:200
            q = 0.3 .+ 1.5 .* rand(rng, 3)                   # reaches both sides of the clip range
            ew, es = draw_eta(a, q), draw_eta(b, q)
            f = powerlaw_factor(q, beta, qref, qclip)
            @test es .- mu ≈ f .* (ew .- mu) rtol = 1e-12
        end
        @test_throws ErrorException draw_eta(b)               # the scaled closure needs q*
    end
end

@testitem "V83 β ≠ 0 through get_next_item_timeseries: the scale reads the predictor q* in physical units" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2, Statistics, LinearAlgebra
    using .LinRegAR: synth_file, qstar_path, new_closure, LinReg, linreg_data, scale_input,
                     get_next_item_timeseries, powerlaw_factor
    mktempdir() do d
        fw, fs = joinpath(d, "white.jld2"), joinpath(d, "scaled.jld2")
        s = synth_file(fw; n = 3, h = 2)
        cp(fw, fs)
        # the clip range holds qstar_path's 1.0-1.3, so for the right state the clamp never binds
        beta, qref, qclip = [1.0, 2.0, 0.5], [1.1, 1.2, 1.05], [0.9 1.4; 0.9 1.4; 0.9 1.4]
        jldopen(fs, "a+") do f
            f["scale_beta"] = beta; f["scale_qref"] = qref; f["scale_qclip"] = qclip
            f["scale_sigma_eps"] = Matrix(s.stoch_distr.Σ)
        end
        spin = 0.01 .* randn(Xoshiro(9), 3, 10)
        Q = qstar_path(3, 300)
        mu = mean(s.stoch_distr)
        sd = sqrt.(diag(Matrix(s.stoch_distr.Σ)))
        # the residual a step realises, scale(q* + dQ) − c [x; 1], with the history held before the step
        function step_eta(m, q)
            cx = vec(m.c * linreg_data(m, q))
            dQ = get_next_item_timeseries(m, q)
            return vec(scale_input(q .+ dQ, m.scaling.out_scaling)) .- cx
        end
        # misfit, in residual sd, of the scaled residual against f(state) ⊙ the white deviation
        miss(es, ew, state) = maximum(abs.((es .- mu) .- powerlaw_factor(state, beta, qref, qclip) .* (ew .- mu)) ./ sd)
        a, b = new_closure(LinReg, fw, 5, 3, 2, spin), new_closure(LinReg, fs, 5, 3, 2, spin)
        right, scaled, previous = Float64[], Float64[], Float64[]
        for t in axes(Q, 2)
            q = Q[:, t]
            if t <= size(spin, 2)                              # the replay: no draw, verbatim dQ
                @test get_next_item_timeseries(a, q) == get_next_item_timeseries(b, q) == spin[:, t]
                continue
            end
            qsc = vec(scale_input(q, b.scaling.in_scaling))   # two wrong states the code could pass
            qprev = b.q_hist[1:3, 1]
            ew, es = step_eta(a, q), step_eta(b, q)
            push!(right, miss(es, ew, q)); push!(scaled, miss(es, ew, qsc)); push!(previous, miss(es, ew, qprev))
        end
        @test length(right) == 290
        @test maximum(right) < 1e-5                            # Float32 round-off of `pred` only
        @test median(scaled) > 1e-2 && median(previous) > 1e-2  # the test would see a wrong state
        @test rand(a.rng) === rand(b.rng)                      # one draw per step in both
    end
end

@testitem "V83 the built LinReg1_pl is LinReg1 + the six scale keys, holds the step-2 fit, shares LinReg1's normals" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2, LinearAlgebra, Statistics
    using .LinRegAR: LRS, LinReg, draw_eta, powerlaw_factor
    fa, fb = joinpath(LRS, "LinReg1", "LinReg.jld2"), joinpath(LRS, "LinReg1_pl", "LinReg.jld2")
    if !(isfile(fa) && isfile(fb))                             # gitignored data; skipped on a clean clone
        @info "V83: TO_LRS/LinReg1_pl or LinReg1 not present, built-artefact check skipped"
    else
        a, b = load(fa), load(fb)
        @test all(isequal(a[k], b[k]) for k in keys(a))        # the mean, Σ, scaling ... untouched
        @test sort(collect(setdiff(keys(b), keys(a)))) ==
              ["scale_beta", "scale_logvar", "scale_provenance", "scale_qclip", "scale_qref", "scale_sigma_eps"]
        @test read(joinpath(LRS, "LinReg1", "parameters.jld2")) == read(joinpath(LRS, "LinReg1_pl", "parameters.jld2"))
        beta, qref, qclip, S = b["scale_beta"], b["scale_qref"], b["scale_qclip"], b["scale_sigma_eps"]
        p = b["scale_provenance"]
        @test p.source == "LinReg1"
        # the fit rows are steps 400-4000; LinReg1's own training rows are 405-3999 (disclosed in
        # analysis/cards/step2.md: Δβ ≤ 0.009, Σ_ε at β = 0 within 0.95 % of Σ in variance)
        @test p.fit_steps == (400, 4000)
        @test round.(beta; digits = 3) == [0.514, 0.554, 1.849, 1.823, 1.276, 1.365]   # paper Sec. 5.5
        @test diag(S) ≈ exp.(b["scale_logvar"]) rtol = 1e-10  # diag Σ_ε = the ML variance at q_ref
        @test issymmetric(S) && isposdef(S)
        @test all(qclip[:, 1] .< qref .< qclip[:, 2])
        # deployed: the scale loads, and each draw maps LinReg1's standard normals (same seed) through
        # the two Cholesky factors -- what "shares LinReg1's random numbers" means at Σ_ε ≠ Σ
        m1 = LinReg(fa, Xoshiro(3), Array; q_hist = zeros(12, 5), spinnup_data = zeros(6, 100))
        mp = LinReg(fb, Xoshiro(3), Array; q_hist = zeros(12, 5), spinnup_data = zeros(6, 100))
        @test m1.scale === nothing && mp.scale.beta == beta && mp.scale.qclip == qclip
        mu = mp.scale.mu_eta
        @test mu == mean(a["stoch_distr"])
        L1, Lp = cholesky(Symmetric(Matrix(a["stoch_distr"].Σ))).L, cholesky(Symmetric(S)).L
        rng = Xoshiro(17)
        dev = map(1:200) do _
            q = qclip[:, 1] .+ (qclip[:, 2] .- qclip[:, 1]) .* (1.2 .* rand(rng, 6) .- 0.1)   # incl. outside the clip
            z1 = L1 \ (draw_eta(m1, q) .- mu)
            zp = Lp \ ((draw_eta(mp, q) .- mu) ./ powerlaw_factor(q, beta, qref, qclip))
            maximum(abs, zp .- z1)
        end
        @test maximum(dev) < 1e-10
        @test rand(m1.rng) === rand(mp.rng)
    end
end

@testitem "V83 the loader refuses partial keys, AR + scale, wrong sizes, a non-positive reference" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2
    using .LinRegAR: synth_file, new_closure, LinReg
    good = (; scale_beta = zeros(3), scale_qref = ones(3), scale_qclip = [0.5 2.0; 0.5 2.0; 0.5 2.0],
            scale_sigma_eps = [1.0 0 0; 0 1.0 0; 0 0 1.0] .* 1e-2)
    function try_file(d, kv; ar = false)
        f = joinpath(d, "m$(rand(UInt32)).jld2")
        synth_file(f; n = 3, h = 2, phi = ar ? [0.5 0.5 0.5] : nothing, S_xi = ar ? good.scale_sigma_eps : nothing)
        jldopen(f, "a+") do fh
            for (k, v) in pairs(kv)
                fh[String(k)] = v
            end
        end
        return () -> new_closure(LinReg, f, 1, 3, 2, zeros(3, 10))
    end
    mktempdir() do d
        @test try_file(d, good)() isa LinReg
        @test_throws ErrorException try_file(d, Base.structdiff(good, (; scale_qclip = 0)))()
        m = try_file(d, good; ar = true)()                   # AR + scale: allowed since 2026-10-16
        @test m isa LinReg && m.ar !== nothing && m.scale !== nothing
        @test_throws ErrorException try_file(d, merge(good, (; scale_beta = zeros(2))))()
        @test_throws ErrorException try_file(d, merge(good, (; scale_qref = [1.0, 0.0, 1.0])))()
        @test_throws ErrorException try_file(d, merge(good, (; scale_qclip = [2.0 0.5; 0.5 2.0; 0.5 2.0])))()
    end
end

@testitem "V83 AR + scale at β = 0 is the AR LinReg bit for bit (outputs, RNG, warm start)" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2
    using .LinRegAR: synth_file, qstar_path, drive, new_closure, LinReg
    mktempdir() do d
        fa, fs = joinpath(d, "ar.jld2"), joinpath(d, "ar_scaled.jld2")
        phi = [0.9 0.5 0.3; -0.2 0.0 -0.1]
        S_xi = [1.0 0.3 0.0; 0.3 1.0 -0.2; 0.0 -0.2 1.0] .* 1e-2
        s = synth_file(fa; n = 3, h = 2, phi, S_xi)
        cp(fa, fs)
        jldopen(fs, "a+") do f
            f["scale_beta"] = zeros(3); f["scale_qref"] = [1.0, 1.1, 0.9]
            f["scale_qclip"] = [0.5 2.0; 0.5 2.0; 0.5 2.0]
            f["scale_sigma_eps"] = Matrix(s.stoch_distr.Σ)
        end
        spin = 0.01 .* randn(Xoshiro(9), 3, 10)                # the last 2 replayed steps fill the AR state
        Q = qstar_path(3, 300; gate_at = [50, 200])
        for seed in (1, 2, 12345)
            a = new_closure(LinReg, fa, seed, 3, 2, spin)
            b = new_closure(LinReg, fs, seed, 3, 2, spin)
            @test a.scale === nothing && b.scale !== nothing && b.ar !== nothing
            A, B = drive(a, Q), drive(b, Q)
            @test A == B
            @test a.ar.z == b.ar.z
            @test rand(a.rng) === rand(b.rng)
        end
    end
end

@testitem "V83 AR + scale at β ≠ 0: η − μ_η = f(q*) ⊙ z, z the AR on the same innovations, warm state ÷ f" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2, Statistics, LinearAlgebra
    using .LinRegAR: synth_file, qstar_path, new_closure, LinReg, linreg_data, scale_input,
                     get_next_item_timeseries, powerlaw_factor
    mktempdir() do d
        fa, fs = joinpath(d, "ar.jld2"), joinpath(d, "ar_scaled.jld2")
        phi = [0.9 0.5 0.3; -0.2 0.0 -0.1]
        S_xi = [1.0 0.3 0.0; 0.3 1.0 -0.2; 0.0 -0.2 1.0] .* 1e-2
        s = synth_file(fa; n = 3, h = 2, phi, S_xi)
        cp(fa, fs)
        beta, qref, qclip = [1.0, 2.0, 0.5], [1.1, 1.2, 1.05], [0.9 1.4; 0.9 1.4; 0.9 1.4]
        jldopen(fs, "a+") do f
            f["scale_beta"] = beta; f["scale_qref"] = qref; f["scale_qclip"] = qclip
            f["scale_sigma_eps"] = Matrix(s.stoch_distr.Σ)
        end
        spin = 0.01 .* randn(Xoshiro(9), 3, 10)
        nw = size(spin, 2)
        Q = qstar_path(3, 300)                                # 1.0-1.3: inside the clip range
        mu = mean(s.stoch_distr)
        sd = sqrt.(diag(S_xi))
        fac(q) = powerlaw_factor(q, beta, qref, qclip)
        function step_eta(m, q)                               # the residual a step realises
            cx = vec(m.c * linreg_data(m, q))
            dQ = get_next_item_timeseries(m, q)
            return vec(scale_input(q .+ dQ, m.scaling.out_scaling)) .- cx
        end
        a, b = new_closure(LinReg, fa, 5, 3, 2, spin), new_closure(LinReg, fs, 5, 3, 2, spin)
        for t in 1:nw                                         # the replay: verbatim, no draw
            @test get_next_item_timeseries(a, Q[:, t]) == get_next_item_timeseries(b, Q[:, t]) == spin[:, t]
        end
        # warm start: the scaled closure's state is the realised residual (the AR closure's) ÷ f(q*) of its step
        @test b.ar.z[:, 1] ≈ a.ar.z[:, 1] ./ fac(Q[:, nw]) rtol = 1e-12
        @test b.ar.z[:, 2] ≈ a.ar.z[:, 2] ./ fac(Q[:, nw - 1]) rtol = 1e-12
        za, zb, zu = [a.ar.z[:, 2], a.ar.z[:, 1]], [b.ar.z[:, 2], b.ar.z[:, 1]], [b.ar.z[:, 2], b.ar.z[:, 1]]
        miss, unscaled = Float64[], Float64[]
        innov(z) = z[end] .- phi[1, :] .* z[end - 1] .- phi[2, :] .* z[end - 2]
        for t in (nw + 1):size(Q, 2)
            q = Q[:, t]
            ea, eb = step_eta(a, q), step_eta(b, q)
            push!(za, ea .- mu); push!(zb, (eb .- mu) ./ fac(q)); push!(zu, eb .- mu)
            push!(miss, maximum(abs.(innov(zb) .- innov(za)) ./ sd))
            push!(unscaled, maximum(abs.(innov(zu) .- innov(za)) ./ sd))
        end
        @test length(miss) == 290
        @test maximum(miss) < 1e-4                            # the same ξ every step, up to Float32 round-off
        @test median(unscaled) > 1e-2                         # the test would see a missing or misplaced f
        @test rand(a.rng) === rand(b.rng)
    end
end

@testitem "V83 the built LinReg7_pl and LinReg7_ar2n_pl: LinReg7's keys untouched, the step-2 fits, one draw per step" default_imports = false setup = [LinRegAR] begin
    using Test, Random, JLD2, LinearAlgebra
    using .LinRegAR: LRS, LinReg, draw_eta, ar_stationary
    f7, fp, fa, fs1 = (joinpath(LRS, n, "LinReg.jld2") for n in ("LinReg7", "LinReg7_pl", "LinReg7_ar2n_pl", "LinReg7_ar2n"))
    if !all(isfile, (f7, fp, fa, fs1))                         # gitignored data; skipped on a clean clone
        @info "V83: LinReg7, LinReg7_pl, LinReg7_ar2n_pl or LinReg7_ar2n not present, built-artefact check skipped"
    else
        a, p, c = load(f7), load(fp), load(fa)
        sk = ["scale_beta", "scale_logvar", "scale_provenance", "scale_qclip", "scale_qref", "scale_sigma_eps"]
        @test all(isequal(a[k], p[k]) for k in keys(a)) && all(isequal(a[k], c[k]) for k in keys(a))
        @test sort(collect(setdiff(keys(p), keys(a)))) == sk
        @test sort(collect(setdiff(keys(c), keys(a)))) == sort([sk; "ar_phi"; "ar_provenance"; "ar_sigma_xi"])
        for d in ("LinReg7_pl", "LinReg7_ar2n_pl")
            @test read(joinpath(LRS, "LinReg7", "parameters.jld2")) == read(joinpath(LRS, d, "parameters.jld2"))
        end
        # the scale is LinReg7_pl's in both (one fit), as quoted (results_LSTMS.md §17 and §29)
        @test all(isequal(p[k], c[k]) for k in sk if k != "scale_provenance")
        @test round.(p["scale_beta"]; digits = 3) == [1.029, 0.509, 1.358, 0.956, 0.717, 1.058]
        @test diag(p["scale_sigma_eps"]) ≈ exp.(p["scale_logvar"]) rtol = 1e-10
        @test p["scale_provenance"].source == "LinReg7" && p["scale_provenance"].fit_steps == (400, 4000)
        # the AR(2) on the residual / f: constrained (phi1 >= 0, phi2 <= 0), stationary, as quoted
        phi = c["ar_phi"]
        @test size(phi) == (2, 6) && all(>=(0), phi[1, :]) && all(<=(0), phi[2, :]) && ar_stationary(phi)
        @test round.(phi; digits = 3) == [0.826 0.458 1.215 1.573 0.873 0.796; 0.0 -0.052 -0.325 -0.634 0.0 0.0]
        @test issymmetric(c["ar_sigma_xi"]) && isposdef(c["ar_sigma_xi"])
        @test c["ar_provenance"].source == "LinReg7" && occursin("divided by f", c["ar_provenance"].method)
        # deployed: each closure draws exactly as many normals per step as its partner (LinReg7, LinReg7 + AR(2))
        mk(f) = LinReg(f, Xoshiro(3), Array; q_hist = zeros(12, 5), spinnup_data = zeros(6, 100))
        m7, mp, ms1, ma = mk(f7), mk(fp), mk(fs1), mk(fa)
        @test mp.scale !== nothing && mp.ar === nothing && ma.scale !== nothing && ma.ar !== nothing
        q = p["scale_qref"]
        for _ in 1:50
            draw_eta(m7, q); draw_eta(mp, q); draw_eta(ms1, q); draw_eta(ma, q)
        end
        @test rand(m7.rng) === rand(mp.rng)
        @test rand(ms1.rng) === rand(ma.rng)
    end
end
