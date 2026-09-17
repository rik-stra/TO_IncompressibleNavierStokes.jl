# V41 and the extension's own acceptance tests.
#
# Run as
#
#     julia --startup-file=no --project=lib/RikFlow/training lib/RikFlow/training/runtests_lux.jl
#
# 🔑 Separate from `test/runtests.jl` on purpose, and the separation is the same one the whole
# package is built around: `test/` is stdlib-only so it stays cheap and runs without a GPU, and
# everything that needs Lux lives here. A failure here means M4 cannot be *trained*; a failure
# there means M4 cannot be *run*, which is the more serious of the two.

using Test
using RikFlow
using Lux, Optimisers, Zygote
using Random, LinearAlgebra, Statistics

const RF = RikFlow

@testset "M4 Lux extension" begin

    @testset "the extension actually loaded" begin
        # 🔴 `hasmethod` is NOT the check. The fallbacks in ts_lstm.jl are `(args...; kwargs...)`,
        # so `hasmethod(RF.init_lstm_params, Tuple{Xoshiro, LSTMSpec})` is true whether the
        # extension loaded or not -- this assertion was written that way first and passed while
        # every test below it errored on the stub. Ask the module system instead.
        @test Base.get_extension(RikFlow, :RikFlowLuxExt) !== nothing
        # and belt-and-braces: the stub errors, so a successful call proves the real method ran
        @test RF.init_lstm_params(Xoshiro(1),
                                  RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = 2))) isa
              NamedTuple
    end

    mkspec(arch; n_qoi = 3, h = 2, n_hidden = 5, n_latent = 4, n_encoder = 5, uclip = nothing) =
        RF.LSTMSpec(; hist = RF.HistorySpec(; h, n_qoi), n_hidden, n_latent, n_encoder, arch,
                    uclip)

    # -----------------------------------------------------------------------------------------
    # V41 -- the training forward pass and the deployed one are the same model
    # -----------------------------------------------------------------------------------------

    @testset "V41 lstm_forward == lstm_step! ($arch, n_encoder=$ne)" for
            arch in (:lstm, :vaernn, :storn, :vrnn), ne in (0, 5)

        spec = mkspec(arch; n_encoder = ne)
        T = Float32
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)

        # perturb away from the initialisation, or Wd = 0 and Araw = 0 make the test trivial
        ps = merge(ps, (; Wd = randn(Xoshiro(11), T, RF.n_output(spec), spec.n_hidden) ./ 5,
                        bd = randn(Xoshiro(12), T, RF.n_output(spec)) ./ 5,
                        Araw = randn(Xoshiro(13), T, RF.n_output(spec), RF.n_output(spec)) ./ 5))

        L = 7
        X = randn(Xoshiro(21), T, RF.n_input(spec), L)
        epsz = zeros(T, spec.n_latent, L)          # z = mu, i.e. the `sample_latent = false` path

        out = RF.lstm_forward(spec, ps, X, epsz)
        w = RF.LSTMWeights(ps, spec)
        @test RF.check_shapes(w, spec)

        st = RF.LSTMState(spec, T)
        for t in 1:L
            y, logd, _ = RF.lstm_step!(st, w, spec, view(X, :, t); sample_latent = false)

            # the MEAN must agree exactly -- nothing is reparametrised on this path
            @test y ≈ out.Y[:, t] rtol = 1e-5

            # the log-scales do NOT agree: training carries a free precision factor and the
            # deployed form carries a correlation matrix, so `bd` absorbs a per-coordinate
            # constant. What has to agree is the DENSITY.
            r = randn(Xoshiro(100 + t), T, RF.n_output(spec))
            dep = RF.gauss_logpdf(r, logd, w.LR)

            A = ps.Araw .* [i > j ? one(T) : zero(T) for i in 1:RF.n_output(spec),
                                                         j in 1:RF.n_output(spec)] .+
                Diagonal(exp.(diag(ps.Araw)))
            u = out.LOGD[:, t]
            train = -0.5 * (RF.n_output(spec) * log(2pi) + 2 * sum(u) - 2 * sum(log.(diag(A))) +
                            sum(abs2, A * (r ./ exp.(u))))
            @test dep ≈ train rtol = 1e-4
        end
    end

    @testset "V41 the covariance conversion produces a genuine correlation matrix" begin
        spec = mkspec(:vrnn)
        T = Float32
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        ps = merge(ps, (; Araw = randn(Xoshiro(31), T, RF.n_output(spec), RF.n_output(spec)) ./ 3))
        w = RF.LSTMWeights(ps, spec)
        R = w.LR * w.LR'
        @test all(isapprox.(diag(R), 1; atol = 1e-5))      # unit diagonal
        @test issymmetric(round.(R; digits = 6))
        @test isposdef(Symmetric(Float64.(R)))
    end

    @testset "V41 uclip survives the round trip" begin
        spec = mkspec(:vrnn; uclip = (-0.3, 0.3))
        T = Float32
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        ps = merge(ps, (; Wd = randn(Xoshiro(41), T, RF.n_output(spec), spec.n_hidden)))
        X = randn(Xoshiro(42), T, RF.n_input(spec), 4)
        out = RF.lstm_forward(spec, ps, X, zeros(T, spec.n_latent, 4))
        # ⚠️ compare in Float32. `spec.uclip` is Float64 and the clamp happens at T = Float32, and
        # Float32(0.3) is one ulp ABOVE Float64(0.3) -- so a correctly clamped value fails a
        # Float64 bound. The clip is doing its job; the naive assertion was wrong.
        @test all(T(-0.3) .<= out.LOGD .<= T(0.3))
        @test any(abs.(out.LOGD) .> T(0.29))       # and it is actually binding, not vacuous
    end

    # -----------------------------------------------------------------------------------------
    # the objective and the loop
    # -----------------------------------------------------------------------------------------

    @testset "elbo is finite, and beta scales only the KL" begin
        spec = mkspec(:vrnn)
        T = Float32
        ps = RF.init_lstm_params(Xoshiro(5), spec; T)
        L = 12
        X = randn(Xoshiro(51), T, RF.n_input(spec), L)
        Y = randn(Xoshiro(52), T, RF.n_output(spec), L)
        epsz = randn(Xoshiro(53), T, spec.n_latent, L)

        l0 = RF.elbo(spec, ps, X, Y, 4:L, epsz; beta = 0.0)
        l1 = RF.elbo(spec, ps, X, Y, 4:L, epsz; beta = 1.0)
        l2 = RF.elbo(spec, ps, X, Y, 4:L, epsz; beta = 2.0)
        @test isfinite(l0) && isfinite(l1)
        @test l1 > l0                                # the KL is positive
        @test (l2 - l1) ≈ (l1 - l0) rtol = 1e-4      # and enters linearly in beta

        # :lstm has no latent path, so beta does nothing at all
        sl = mkspec(:lstm)
        pl = RF.init_lstm_params(Xoshiro(5), sl; T)
        @test RF.elbo(sl, pl, X, Y, 4:L, epsz; beta = 0.0) ==
              RF.elbo(sl, pl, X, Y, 4:L, epsz; beta = 5.0)
    end

    @testset "gradients flow to every trained block" begin
        spec = mkspec(:vrnn)
        T = Float32
        ps = RF.init_lstm_params(Xoshiro(7), spec; T)
        L = 10
        X = randn(Xoshiro(61), T, RF.n_input(spec), L)
        Y = randn(Xoshiro(62), T, RF.n_output(spec), L)
        epsz = randn(Xoshiro(63), T, spec.n_latent, L)

        g = Zygote.gradient(p -> RF.elbo(spec, p, X, Y, 3:L, epsz; beta = 1e-4), ps)[1]
        for k in (:Wx, :Wh, :b, :We, :be, :Bmu, :Bsig, :V1, :V2, :cdec, :Wd, :bd, :Araw)
            @test getproperty(g, k) !== nothing
            @test any(!iszero, getproperty(g, k))
        end
    end

    @testset "training reduces the loss on a learnable signal" begin
        # A stream with real temporal structure: an AR(1) the recurrence can actually latch on to.
        # The assertion is only that the optimiser moves downhill -- this is a smoke test for the
        # loop, not a claim about M4's skill.
        T = Float32
        nq, N = 3, 900
        rng = Xoshiro(71)
        q = zeros(Float64, nq, N)
        for t in 2:N
            q[:, t] = 0.85 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        spec = mkspec(:storn; n_qoi = nq, h = 1, n_hidden = 8, n_latent = 4, n_encoder = 8)
        hs = spec.hist
        X, Yb, steps = RF.build_history(hs, q[:, 1:(N - 1)], q)
        ps, hist = RF.train_stochlstm(spec, permutedims(X), permutedims(Yb), steps;
                                      L = 60, burn = 15, epochs = 12, batch = 4, lr = 5e-3,
                                      seed = 2, verbose = false, T)
        @test length(hist.train) == 12
        @test all(isfinite, hist.train)
        @test mean(hist.train[end-2:end]) < mean(hist.train[1:3])
    end

    # -----------------------------------------------------------------------------------------
    # V49 -- the fit that is RETURNED is the best iterate, on a reproducible curve
    # -----------------------------------------------------------------------------------------
    #
    # 🔴 These three properties are not conveniences. Without them, on R1's record, all three
    # latent architectures reached val ~= -12 near epoch 91 and were at ~= -4 by epoch 100, and
    # the last-iterate fits that got saved INVERTED the architecture ranking. A conclusion was
    # drawn from that and had to be retracted; this testset is what stops it recurring.
    @testset "V49 train_stochlstm returns the best iterate, not the last" begin
        T = Float32
        nq, N = 3, 700
        rng = Xoshiro(91)
        q = zeros(Float64, nq, N)
        for t in 2:N
            q[:, t] = 0.8 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        spec = mkspec(:vrnn; n_qoi = nq, h = 1, n_hidden = 8, n_latent = 4, n_encoder = 8)
        X, Yb, steps = RF.build_history(spec.hist, q[:, 1:(N - 1)], q)
        Xc, Yc = permutedims(X), permutedims(Yb)

        kw = (; L = 60, burn = 15, epochs = 20, batch = 4, lr = 5e-3, verbose = false, T)
        ps, hist = RF.train_stochlstm(spec, Xc, Yc, steps; seed = 3, kw...)

        # (1) the recorded best really is the minimum of the curve, and the returned fit is it
        @test hist.best_val == minimum(hist.val)
        @test hist.val[hist.best_epoch] == hist.best_val
        @test 1 <= hist.best_epoch <= 20
        @test hist.best_val <= hist.val[end]        # never worse than the last iterate

        # (2) the validation curve is a function of the parameters alone. Two runs at the same
        # seed must agree exactly -- if the val epsilons were redrawn each epoch they would not,
        # and "best validation" would be selecting partly on a lucky noise draw.
        _, hist2 = RF.train_stochlstm(spec, Xc, Yc, steps; seed = 3, kw...)
        @test hist.val == hist2.val
        @test hist.best_epoch == hist2.best_epoch

        # (3) the learning rate decays on plateau rather than staying put
        _, hplateau = RF.train_stochlstm(spec, Xc, Yc, steps; seed = 3, patience = 1,
                                         lr_decay = 0.5, kw...)
        @test length(hplateau.lr) == 20
        @test hplateau.lr[end] < hplateau.lr[1]
        @test all(hplateau.lr .>= 1e-5)
    end

    # -----------------------------------------------------------------------------------------
    # V50 -- batching segments is a SPEED change and nothing else
    # -----------------------------------------------------------------------------------------
    #
    # The recurrence runs over `H x B` matrices so a chunk of segments costs `L` traced steps
    # instead of `B * L`. That is only legitimate if the numbers are unchanged, so: the batched
    # forward must equal the per-segment forward on every segment, and the batched objective must
    # equal the per-segment objective averaged with the right weights.
    @testset "V50 batched == per-segment ($arch)" for arch in (:lstm, :vaernn, :storn, :vrnn)
        spec = mkspec(arch)
        T = Float32
        ps = RF.init_lstm_params(Xoshiro(15), spec; T)
        ps = merge(ps, (; Wd = randn(Xoshiro(16), T, RF.n_output(spec), spec.n_hidden) ./ 5,
                        Araw = randn(Xoshiro(17), T, RF.n_output(spec), RF.n_output(spec)) ./ 5))

        L, B, nin, nz, nout = 9, 4, RF.n_input(spec), spec.n_latent, RF.n_output(spec)
        X = randn(Xoshiro(18), T, nin, L, B)
        Y = randn(Xoshiro(19), T, nout, L, B)
        E = randn(Xoshiro(20), T, nz, L, B)
        sc = 4:L

        ob = RF.lstm_forward(spec, ps, X, E)
        for b in 1:B
            os = RF.lstm_forward(spec, ps, X[:, :, b], E[:, :, b])
            @test os.Y ≈ ob.Y[:, :, b] rtol = 1e-5
            @test os.LOGD ≈ ob.LOGD[:, :, b] rtol = 1e-5
            @test os.Hm ≈ ob.Hm[:, :, b] rtol = 1e-5
        end

        # the objective: per-scored-step normalisation makes the batch the plain mean here,
        # because every segment contributes the same number of scored steps
        lb = RF.elbo(spec, ps, X, Y, sc, E; beta = 1e-3)
        ls = mean(RF.elbo(spec, ps, X[:, :, b], Y[:, :, b], sc, E[:, :, b]; beta = 1e-3)
                  for b in 1:B)
        @test lb ≈ ls rtol = 1e-4

        # and the gradients agree, which is what actually trains the model
        gb = Zygote.gradient(p -> RF.elbo(spec, p, X, Y, sc, E; beta = 1e-3), ps)[1]
        gs = Zygote.gradient(ps) do p
            mean(RF.elbo(spec, p, X[:, :, b], Y[:, :, b], sc, E[:, :, b]; beta = 1e-3)
                 for b in 1:B)
        end[1]
        for k in (:Wx, :Wh, :b, :V1, :cdec, :Wd, :bd, :Araw)
            @test getproperty(gb, k) ≈ getproperty(gs, k) rtol = 1e-3
        end
    end

    # -----------------------------------------------------------------------------------------
    # V51 -- emission = :none is the source's design: the latent path is the ONLY noise
    # -----------------------------------------------------------------------------------------
    @testset "V51 emission = :none removes the second noise channel" begin
        T = Float32
        spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = 6), n_hidden = 16,
                           n_latent = 4, n_encoder = 0, arch = :storn, emission = :none)
        @test !RF.emission_noise(spec)
        ps = RF.init_lstm_params(Xoshiro(31), spec; T)

        L, B, nin, nz, nout = 12, 3, RF.n_input(spec), spec.n_latent, RF.n_output(spec)
        X = randn(Xoshiro(32), T, nin, L, B)
        Y = randn(Xoshiro(33), T, nout, L, B)
        E = randn(Xoshiro(34), T, nz, L, B)
        sc = 5:L

        # the reconstruction term is a plain sum of squares, and beta still scales only the KL
        out = RF.lstm_forward(spec, ps, X, E)
        ns = length(sc) * B
        expect = 0.5 * sum(abs2, Y[:, sc, :] .- out.Y[:, sc, :]) / ns
        @test RF.elbo(spec, ps, X, Y, sc, E; beta = 0.0) ≈ expect rtol = 1e-4
        @test RF.elbo(spec, ps, X, Y, sc, E; beta = 1.0) > expect      # the KL is positive

        # 🔴 no predictive density means no likelihood, and it refuses rather than inventing one
        @test_throws ErrorException RF.iwae_nll(spec, ps, X[:, :, 1], Y[:, :, 1], sc)

        # the deployed step: log-scale is identically zero and the emission draw is the mean,
        # so the ONLY source of ensemble spread is the latent draw
        w = RF.LSTMWeights(ps, spec)
        st = RF.LSTMState(spec, T)
        rng = Xoshiro(35)
        y, logd, _ = RF.lstm_step!(st, w, spec, view(X, :, 1, 1); rng, sample_latent = true)
        @test all(iszero, logd)
        outv = zeros(T, nout)
        before = copy(rng)
        RF.sample_emission!(outv, st, w, spec, rng)
        @test outv == y                       # the prediction IS the mean
        @test rand(rng) == rand(before)       # and nothing was drawn

        # a deterministic backbone with no emission noise has no stochasticity at all, and is
        # refused at construction rather than silently producing a zero-spread "ensemble"
        @test_throws ErrorException RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = 6),
                                                arch = :lstm, emission = :none)
    end

    @testset "iwae_nll runs and tightens with K" begin
        spec = mkspec(:vrnn)
        T = Float32
        ps = RF.init_lstm_params(Xoshiro(9), spec; T)
        L = 8
        X = randn(Xoshiro(81), T, RF.n_input(spec), L)
        Y = randn(Xoshiro(82), T, RF.n_output(spec), L) ./ 4

        n1 = RF.iwae_nll(spec, ps, X, Y, 3:L; K = 1, rng = Xoshiro(1))
        n64 = RF.iwae_nll(spec, ps, X, Y, 3:L; K = 64, rng = Xoshiro(1))
        @test isfinite(n1) && isfinite(n64)
        # more samples => a tighter lower bound on log p => a smaller (better) NLL bound
        @test n64 <= n1
    end

    @testset "train_stochlstm refuses a record it cannot segment" begin
        spec = mkspec(:vrnn)
        X = randn(Xoshiro(91), Float64, RF.n_input(spec), 20)
        Y = randn(Xoshiro(92), Float64, RF.n_output(spec), 20)
        @test_throws ErrorException RF.train_stochlstm(spec, X, Y, collect(1:20);
                                                       L = 200, burn = 150, verbose = false)
        @test_throws ErrorException RF.train_stochlstm(spec, X, Y, collect(1:19); verbose = false)
    end
end
