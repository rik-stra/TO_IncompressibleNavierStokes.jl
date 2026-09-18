# M4 -- the stochastic LSTM. V39, V40, V42, V43.
#
# V41 (training forward == lstm_step!) and V44 (the warm-up returns replayed columns unconverted)
# need the Lux extension and the `StochLSTM` closure respectively, and live with those.
#
# ⚠️ The verification registry in `plan.md` §14 covers V0-V28 while the suite already uses V29-V38
# with no rows. These numbers continue that sequence and need rows adding when the gap is closed.

# ---------------------------------------------------------------------------------------------
# V39 -- segmentation for BPTT
# ---------------------------------------------------------------------------------------------

@testitem "V39 segment_indices covers, never crosses a block, and burns in" default_imports = false setup = [TSLayer] begin
    using Test

    seg = TSLayer.segment_indices(collect(1:100); L = 20, burn = 5)

    for s in seg
        @test length(s.rows) <= 20
        @test first(s.score) == first(s.rows) + 5
        @test last(s.score) == last(s.rows)
        @test length(s.score) >= 1
    end

    # the scored ranges tile the scorable rows without a gap or an overlap
    scored = reduce(vcat, [collect(s.score) for s in seg])
    @test scored == sort(scored)
    @test allunique(scored)
    @test minimum(scored) == 6        # the first 5 rows are burn-in and are never scored
    @test maximum(scored) == 100      # the tail is reached
    @test scored == collect(6:100)    # and nothing in between is missed

    # 🔴 a segment must never straddle a discontinuity in the step index
    steps = vcat(1:10, 51:60)         # two blocks, a 40-step gap between them
    for s in TSLayer.segment_indices(steps; L = 8, burn = 2)
        @test all(diff(steps[collect(s.rows)]) .== 1)   # contiguous in PHYSICAL step
    end

    # a block shorter than the burn-in produces nothing rather than an unscorable segment
    @test isempty(TSLayer.segment_indices(collect(1:4); L = 10, burn = 5))
    @test isempty(TSLayer.segment_indices(Int[]; L = 10, burn = 5))

    # and the arguments are policed
    @test_throws ErrorException TSLayer.segment_indices(collect(1:20); L = 5, burn = 5)
    @test_throws ErrorException TSLayer.segment_indices(collect(1:20); L = 10, burn = 2, stride = 0)
end

# ---------------------------------------------------------------------------------------------
# shared fixture
# ---------------------------------------------------------------------------------------------

@testmodule LSTMFix begin
    using LinearAlgebra
    using Random

    const SRC = normpath(joinpath(@__DIR__, "..", "src"))
    include(joinpath(SRC, "ts_scaling.jl"))
    include(joinpath(SRC, "ts_history.jl"))
    include(joinpath(SRC, "ts_lstm.jl"))

    """
        build(arch; n_encoder, T, seed, n_qoi, h, n_hidden, n_latent, uclip)

    A spec and a random weight set that match it. Small dimensions by default -- the point is the
    arithmetic, not the capacity.
    """
    function build(arch; n_encoder = 6, T = Float64, seed = 42, n_qoi = 3, h = 2,
                   n_hidden = 7, n_latent = 4, uclip = nothing,
                   emission = :state_dependent)
        # `emission` defaults to the Gaussian head HERE and to `:none` in `LSTMSpec`: the head is
        # off in production (Rik, 2026-09-18) but is still in the code, so the arithmetic tests
        # that exercise it have to name it rather than inherit it.
        spec = LSTMSpec(; hist = HistorySpec(; h, n_qoi), n_hidden, n_latent, n_encoder, arch,
                        emission, uclip)
        H = spec.n_hidden
        nin, nout, nz = n_input(spec), n_output(spec), spec.n_latent
        ncin, nenc = n_cell_input(spec), n_encoder_out(spec)
        rng = Xoshiro(seed)
        w = LSTMWeights{T}(
            randn(rng, T, 4H, ncin), randn(rng, T, 4H, H), randn(rng, T, 4H),
            (n_encoder > 0 && latent_sampled(spec)) ? randn(rng, T, nenc, nin) : nothing,
            (n_encoder > 0 && latent_sampled(spec)) ? randn(rng, T, nenc) : nothing,
            randn(rng, T, nz, nenc), randn(rng, T, nz, nenc),
            randn(rng, T, nout, H),
            latent_to_decoder(spec) ? randn(rng, T, nout, nz) : nothing,
            randn(rng, T, nout),
            randn(rng, T, nout, H), randn(rng, T, nout),
            Matrix{T}(I, nout, nout))
        return spec, w
    end
end

# ---------------------------------------------------------------------------------------------
# V40 -- the hand-written cell against an independent reference
# ---------------------------------------------------------------------------------------------
#
# `lstm_step!` is written for the solver: in place, pre-allocated scratch, fused loops. That is
# exactly the shape of code that gets an index or an accumulation wrong without failing loudly, so
# it is compared against a naive transcription of the equations written separately below.

@testitem "V40 lstm_step! matches a naive reference forward pass" default_imports = false setup = [LSTMFix] begin
    using Test
    using Random
    using LinearAlgebra

    sig(x) = 1 / (1 + exp(-x))
    sp(x) = log(1 + exp(x))

    for arch in (:lstm, :vaernn, :storn, :vrnn), n_encoder in (0, 6)
        spec, w = LSTMFix.build(arch; n_encoder)
        @test LSTMFix.check_shapes(w, spec)

        H = spec.n_hidden
        nin, nout, nz = LSTMFix.n_input(spec), LSTMFix.n_output(spec), spec.n_latent
        T = Float64
        xs = [randn(Xoshiro(100 + t), T, nin) for t in 1:6]

        # --- reference: the equations, transcribed ---------------------------------------------
        h = zeros(T, H); c = zeros(T, H)
        ref_y, ref_logd, ref_sig = Vector{T}[], Vector{T}[], Vector{T}[]
        for x in xs
            local z, sg
            if LSTMFix.latent_sampled(spec)
                enc = spec.n_encoder > 0 ? tanh.(w.We * x + w.be) : x
                z = w.Bmu * enc              # posterior mean: sample_latent = false
                sg = sp.(w.Bsig * enc)
            else
                z = zeros(T, nz); sg = zeros(T, nz)
            end
            xin = LSTMFix.latent_to_cell(spec) ? vcat(x, z) : x
            g = w.Wx * xin + w.Wh * h + w.b
            i = sig.(g[1:H]); f = sig.(g[(H + 1):(2H)])
            gg = tanh.(g[(2H + 1):(3H)]); o = sig.(g[(3H + 1):(4H)])
            c = f .* c + i .* gg
            h = o .* tanh.(c)
            y = w.V1 * h + w.cdec
            LSTMFix.latent_to_decoder(spec) && (y += w.V2 * z)
            push!(ref_y, copy(y)); push!(ref_logd, w.Wd * h + w.bd); push!(ref_sig, copy(sg))
        end

        # --- the deployed path -----------------------------------------------------------------
        st = LSTMFix.LSTMState(spec, T)
        for (t, x) in enumerate(xs)
            y, logd, _ = LSTMFix.lstm_step!(st, w, spec, x; sample_latent = false)
            @test y ≈ ref_y[t]
            @test logd ≈ ref_logd[t]
            LSTMFix.latent_sampled(spec) && @test st.sigz ≈ ref_sig[t]
        end
        @test st.nstep == length(xs)

        # reset! really does return it to the start
        LSTMFix.reset!(st)
        @test all(iszero, st.h) && all(iszero, st.c) && st.nstep == 0
        y1, _, _ = LSTMFix.lstm_step!(st, w, spec, xs[1]; sample_latent = false)
        @test y1 ≈ ref_y[1]
    end
end

@testitem "V40 the uclip bound is applied to the emission log-scale" default_imports = false setup = [LSTMFix] begin
    using Test
    using Random

    spec, w = LSTMFix.build(:vrnn; uclip = (-0.25, 0.25))
    st = LSTMFix.LSTMState(spec, Float64)
    x = randn(Xoshiro(5), Float64, LSTMFix.n_input(spec))
    _, logd, _ = LSTMFix.lstm_step!(st, w, spec, x; sample_latent = false)
    @test all(-0.25 .<= logd .<= 0.25)

    # and without the clip the same weights leave the range, so the test above is not vacuous
    spec2, w2 = LSTMFix.build(:vrnn; uclip = nothing)
    st2 = LSTMFix.LSTMState(spec2, Float64)
    _, logd2, _ = LSTMFix.lstm_step!(st2, w2, spec2, x; sample_latent = false)
    @test any(abs.(logd2) .> 0.25)
end

@testitem "V40 lstm_step! allocates nothing per step" default_imports = false setup = [LSTMFix] begin
    using Test
    using Random

    # 🔑 S4 is launch-bound, not FLOP-bound. A per-step allocation is what would actually cost the
    # budget, so it is asserted rather than hoped for.
    #
    # ⚠️ Measured from inside a function on purpose. At top level the arguments are globals and
    # `@allocated` reports the boxing of the globals rather than anything `lstm_step!` does -- 96
    # bytes of it, which is a measurement of the test and not of the code. The solver calls this
    # from inside `get_next_item_timeseries`, so a local scope is also the representative one.
    function measure(st, w, spec, x, rng, sample_latent)
        LSTMFix.lstm_step!(st, w, spec, x; rng, sample_latent)        # compile
        return @allocated LSTMFix.lstm_step!(st, w, spec, x; rng, sample_latent)
    end

    for arch in (:lstm, :vaernn, :storn, :vrnn), n_encoder in (0, 60)
        spec, w = LSTMFix.build(arch; n_encoder, T = Float32, n_qoi = 6, h = 1,
                                n_hidden = 60, n_latent = 60)
        st = LSTMFix.LSTMState(spec, Float32)
        rng = Xoshiro(7)
        x = randn(rng, Float32, LSTMFix.n_input(spec))
        @test measure(st, w, spec, x, rng, true) == 0
        @test measure(st, w, spec, x, rng, false) == 0
    end
end

# ---------------------------------------------------------------------------------------------
# V42 -- the warm-up does not touch the RNG
# ---------------------------------------------------------------------------------------------

@testitem "V42 sample_latent = false consumes no randomness" default_imports = false setup = [LSTMFix] begin
    using Test
    using Random

    # 🔴 V38's invariant, extended to M4. A member seed has to mean the same thing across closures
    # and across warm-up lengths, which it only does if the warm-up replay returns before the first
    # draw. `LinReg` and `MVG_sampler` get this by returning early; M4 cannot, because its warm-up
    # has to charge the recurrence -- so it uses the posterior mean instead and draws nothing.
    spec, w = LSTMFix.build(:vrnn)
    nin = LSTMFix.n_input(spec)
    xs = [randn(Xoshiro(t), Float64, nin) for t in 1:4]

    st1 = LSTMFix.LSTMState(spec, Float64)
    r1 = Xoshiro(999)
    for t in 1:3
        LSTMFix.lstm_step!(st1, w, spec, xs[t]; rng = r1, sample_latent = false)
    end
    after_warmup = randn(r1)

    r2 = Xoshiro(999)
    no_warmup = randn(r2)
    @test after_warmup == no_warmup

    # and sampling genuinely does consume it, so the assertion above is not vacuous
    st3 = LSTMFix.LSTMState(spec, Float64)
    r3 = Xoshiro(999)
    LSTMFix.lstm_step!(st3, w, spec, xs[1]; rng = r3, sample_latent = true)
    @test randn(r3) != no_warmup

    # sample_latent = true without an rng is refused rather than silently made deterministic
    st4 = LSTMFix.LSTMState(spec, Float64)
    @test_throws ErrorException LSTMFix.lstm_step!(st4, w, spec, xs[1]; sample_latent = true)

    # :lstm has no latent path, so it never draws whatever it is asked
    sl, wl = LSTMFix.build(:lstm)
    stl = LSTMFix.LSTMState(sl, Float64)
    r5 = Xoshiro(999)
    LSTMFix.lstm_step!(stl, wl, sl, randn(Xoshiro(1), Float64, LSTMFix.n_input(sl));
                       rng = r5, sample_latent = true)
    @test randn(r5) == no_warmup
end

# ---------------------------------------------------------------------------------------------
# V43 -- the likelihood arithmetic
# ---------------------------------------------------------------------------------------------

@testitem "V43 gauss_logpdf against the closed form" default_imports = false setup = [TSLayer] begin
    using Test
    using Random
    using LinearAlgebra

    rng = Xoshiro(11)
    n = 5
    A = randn(rng, n, n)
    R0 = A * A'
    d0 = sqrt.(diag(R0))
    R = R0 ./ (d0 * d0')                       # a genuine correlation matrix
    LR = Matrix(cholesky(Symmetric(R)).L)
    logd = randn(rng, n) .* 0.3
    r = randn(rng, n)

    D = Diagonal(exp.(logd))
    Sig = D * R * D
    @test TSLayer.gauss_logpdf(r, logd, LR) ≈
          -0.5 * (n * log(2pi) + logdet(Sig) + dot(r, Sig \ r))

    # the R = I case reduces to independent normals, the form the tex quotes as convex in u
    LI = Matrix{Float64}(I, n, n)
    @test TSLayer.gauss_logpdf(r, logd, LI) ≈
          sum(-logd[k] - 0.5log(2pi) - 0.5r[k]^2 * exp(-2logd[k]) for k in 1:n)
end

@testitem "V43 the KL terms against their closed forms" default_imports = false setup = [TSLayer] begin
    using Test
    using Random

    rng = Xoshiro(13)
    n = 4
    muq, mup = randn(rng, n), randn(rng, n)
    sigq, sigp = exp.(randn(rng, n) .* 0.2), exp.(randn(rng, n) .* 0.2)

    @test TSLayer.kl_diag_gaussian(muq, sigq, mup, sigp) ≈
          sum(log(sigp[k] / sigq[k]) + (sigq[k]^2 + (muq[k] - mup[k])^2) / (2sigp[k]^2) - 0.5
              for k in 1:n)

    @test TSLayer.kl_diag_gaussian(muq, sigq, muq, sigq) ≈ 0 atol = 1e-12
    @test TSLayer.kl_diag_gaussian(muq, sigq, mup, sigp) >= 0

    # 🔑 the objective's KL is to a FIXED standard normal -- the source's prior, settled against
    # ML_Code/networks_qg.py -- so the specialised form must agree with the general one
    @test TSLayer.kl_to_standard_normal(muq, sigq) ≈
          TSLayer.kl_diag_gaussian(muq, sigq, zeros(n), ones(n))
    @test TSLayer.kl_to_standard_normal(zeros(n), ones(n)) ≈ 0 atol = 1e-12
    @test TSLayer.kl_to_standard_normal(muq, sigq) >= 0
end

@testitem "V43 iwae_bound is >= the ELBO and is numerically stable" default_imports = false setup = [TSLayer] begin
    using Test
    using Random
    using Statistics

    logw = randn(Xoshiro(17), 64) .* 2

    # Jensen: log(mean(w)) >= mean(log(w)). The ELBO is the right-hand side, so IWAE bounds it.
    @test TSLayer.iwae_bound(logw) >= mean(logw)

    # and it tightens with K on average -- the reason K = 64 rather than 1
    small = [TSLayer.iwae_bound(randn(Xoshiro(s), 2) .* 2) for s in 1:200]
    large = [TSLayer.iwae_bound(randn(Xoshiro(s), 64) .* 2) for s in 1:200]
    @test mean(large) > mean(small)

    # stability: a naive exp/sum/log overflows here, the shifted form does not
    @test TSLayer.iwae_bound(fill(1.0e4, 8)) ≈ 1.0e4
    @test isfinite(TSLayer.iwae_bound([-1.0e4, -1.0e4, -1.0e4]))
    @test TSLayer.iwae_bound([3.25]) ≈ 3.25
end

# ---------------------------------------------------------------------------------------------
# Architecture nesting
# ---------------------------------------------------------------------------------------------

@testitem "V43 the four architectures differ only where they are supposed to" default_imports = false setup = [TSLayer] begin
    using Test

    mk(arch) = TSLayer.LSTMSpec(; hist = TSLayer.HistorySpec(; h = 2, n_qoi = 6),
                                n_hidden = 60, n_latent = 6, n_encoder = 6, arch,
                                emission = :state_dependent)

    @test !TSLayer.latent_sampled(mk(:lstm))
    @test all(TSLayer.latent_sampled, (mk(:vaernn), mk(:storn), mk(:vrnn)))

    # "upstream stochasticity" -- the property Sørensen et al. report as the one that matters -- is
    # exactly z-into-the-cell, and it is STORN and VRNN that have it
    @test !TSLayer.latent_to_cell(mk(:lstm))
    @test !TSLayer.latent_to_cell(mk(:vaernn))
    @test TSLayer.latent_to_cell(mk(:storn))
    @test TSLayer.latent_to_cell(mk(:vrnn))

    # the V2 decoder skip
    @test !TSLayer.latent_to_decoder(mk(:lstm))
    @test TSLayer.latent_to_decoder(mk(:vaernn))
    @test !TSLayer.latent_to_decoder(mk(:storn))
    @test TSLayer.latent_to_decoder(mk(:vrnn))

    # only the architectures that feed z upstream widen the cell input
    @test TSLayer.n_cell_input(mk(:lstm)) == TSLayer.n_input(mk(:lstm))
    @test TSLayer.n_cell_input(mk(:vaernn)) == TSLayer.n_input(mk(:vaernn))
    @test TSLayer.n_cell_input(mk(:storn)) == TSLayer.n_input(mk(:storn)) + 6

    # n_encoder = 0 is the tex's linear encoder; anything else is the repository's dense layer
    lin = TSLayer.LSTMSpec(; hist = TSLayer.HistorySpec(; h = 2, n_qoi = 6), n_encoder = 0)
    @test TSLayer.n_encoder_out(lin) == TSLayer.n_input(lin)
    @test TSLayer.n_encoder_out(mk(:vrnn)) == 6

    # a typo in arch is refused at construction, not at the first forward pass on the cluster
    @test_throws ErrorException TSLayer.LSTMSpec(; hist = TSLayer.HistorySpec(; h = 1, n_qoi = 6),
                                                 arch = :storm)
end
