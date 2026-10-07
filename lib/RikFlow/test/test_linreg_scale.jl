# V83 -- step 4p's power-law scale of the LinReg residual (paper Sec. 6.4, Eq. power law; 2026-10-07).
#
# Pinned, each against an answer known before the code runs:
#
#   * `fit_powerlaw_scale` (src/ts_scale.jl) recovers β and the variance at q_ref from synthetic
#     heteroscedastic data; `powerlaw_factor` clamps to the clip range and is exactly 1 at β = 0;
#   * 🔴 a file with the scale keys at β = 0 and Σ_ε = Σ draws BIT-IDENTICALLY to the white file,
#     outputs and RNG stream, through `get_next_item_timeseries` (so 4p shares LinReg1's random
#     numbers, and β = 0 is LinReg1);
#   * at β ≠ 0 the draw is μ_η + f(q*) ⊙ (the white draw's deviation), with the clamp applied;
#   * the loader refuses a partial key set, AR + scale, wrong sizes and a non-positive reference.
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
        @test_throws ErrorException try_file(d, good; ar = true)()
        @test_throws ErrorException try_file(d, merge(good, (; scale_beta = zeros(2))))()
        @test_throws ErrorException try_file(d, merge(good, (; scale_qref = [1.0, 0.0, 1.0])))()
        @test_throws ErrorException try_file(d, merge(good, (; scale_qclip = [2.0 0.5; 0.5 2.0; 0.5 2.0])))()
    end
end
