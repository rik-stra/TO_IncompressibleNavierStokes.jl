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
# V54 only. `JLArrays` is GPUArrays' host-backed reference array: it refuses scalar
# indexing exactly as `CuArray` does, which is what makes it a real proxy for a device
# on a machine that has none.
# ⚠️ CUDA is deliberately NOT imported here: it is a dependency of `RikFlow`, not of this test
# environment, and `using` it would be gotcha #53's class of defect in the test that exists to
# catch that class. `m4_device`'s CUDA branch is tested through its behaviour instead.
using JLArrays

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

    # `emission` defaults to the Gaussian head here and to `:none` in `LSTMSpec`: the head is off
    # in production (Rik, 2026-09-18) but still in the code and still has to be tested.
    mkspec(arch; n_qoi = 3, h = 2, n_hidden = 5, n_latent = 4, n_encoder = 5, uclip = nothing,
           emission = :state_dependent) =
        RF.LSTMSpec(; hist = RF.HistorySpec(; h, n_qoi), n_hidden, n_latent, n_encoder, arch,
                    emission, uclip)

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
    # V60 -- the linear skip `y += Ws x` is the same model in training, deployment and on disk
    # -----------------------------------------------------------------------------------------

    @testset "V60 linear skip: lstm_forward == lstm_step!, save/load ($arch)" for
            arch in (:lstm, :vaernn, :storn, :vrnn)

        T = Float32
        spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 2, n_qoi = 3), n_hidden = 5, n_latent = 4,
                           n_encoder = 0, arch, emission = :constant, skip = true)
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        @test size(ps.Ws) == (RF.n_output(spec), RF.n_input(spec))
        ps = merge(ps, (; Ws = randn(Xoshiro(61), T, size(ps.Ws)...)))
        L = 7
        X = randn(Xoshiro(62), T, RF.n_input(spec), L)
        epsz = zeros(T, spec.n_latent, L)
        out = RF.lstm_forward(spec, ps, X, epsz)
        # positive control: the skip actually contributes, so agreement below is not vacuous
        out0 = RF.lstm_forward(spec, merge(ps, (; Ws = zero(ps.Ws))), X, epsz)
        @test maximum(abs, out.Y .- out0.Y) > 0.1

        w = RF.LSTMWeights(ps, spec)
        @test RF.check_shapes(w, spec)
        st = RF.LSTMState(spec, T)
        for t in 1:L
            y, _, _ = RF.lstm_step!(st, w, spec, view(X, :, t); sample_latent = false)
            @test y ≈ out.Y[:, t] rtol = 1e-5
        end

        path = joinpath(mktempdir(), "skip.jld2")
        RF.save_stochlstm(path, spec, w, (; in_scaling = nothing, out_scaling = nothing))
        back = RF.load_stochlstm(path)
        @test back.spec.skip
        @test back.weights.Ws == w.Ws

        # a no-skip spec keeps `Ws = nothing` all the way through
        spec0 = RF.LSTMSpec(; hist = spec.hist, n_hidden = 5, n_latent = 4, n_encoder = 0, arch,
                            emission = :constant)
        w0 = RF.LSTMWeights(RF.init_lstm_params(Xoshiro(3), spec0; T), spec0)
        @test w0.Ws === nothing && RF.check_shapes(w0, spec0)
    end

    @testset "V60 linear skip: the rollout forward at K = 1 is the teacher-forced forward" begin
        T = Float32
        spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = 3), n_hidden = 5, n_latent = 4,
                           n_encoder = 0, arch = :vrnn, skip = true)
        ps = merge(RF.init_lstm_params(Xoshiro(3), spec; T),
                   (; Ws = randn(Xoshiro(63), T, RF.n_output(spec), RF.n_input(spec))))
        L = 6
        X = randn(Xoshiro(64), T, RF.n_input(spec), L, 2)
        epsz = randn(Xoshiro(65), T, spec.n_latent, L, 2)
        tf = RF.lstm_forward(spec, ps, X, epsz)
        ext = Base.get_extension(RikFlow, :RikFlowLuxExt)
        ro = ext._rollout_forward(spec, ps, X, epsz, falses(L))
        @test ro.Y ≈ tf.Y rtol = 1e-5
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
        # ⚠️ The history has one row per VALIDATION, not per epoch, so its length is not the
        # epoch count -- it is `updates / val_every`. Asserting the invariants rather than a
        # number keeps this honest if `val_every` ever moves again.
        @test length(hist.train) == length(hist.val) == length(hist.update)
        @test hist.updates == 12 * hist.upd_per_epoch
        @test hist.update[end] == hist.updates
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
        @test hist.val[hist.best_index] == hist.best_val
        @test hist.update[hist.best_index] == hist.best_update
        @test 1 <= hist.best_index <= length(hist.val)
        @test hist.best_val <= hist.val[end]        # never worse than the last iterate

        # (2) the validation curve is a function of the parameters alone. Two runs at the same
        # seed must agree exactly -- if the val epsilons were redrawn each epoch they would not,
        # and "best validation" would be selecting partly on a lucky noise draw.
        _, hist2 = RF.train_stochlstm(spec, Xc, Yc, steps; seed = 3, kw...)
        @test hist.val == hist2.val
        @test hist.best_update == hist2.best_update
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

    # -----------------------------------------------------------------------------------------
    # V52 -- the hand-written adjoint against finite differences
    # -----------------------------------------------------------------------------------------
    #
    # 🔴 **V50 cannot catch a wrong `_timeslices` adjoint, and this is the gap it leaves.** V50
    # compares the batched objective against the per-segment one, but since the time-slicing rule
    # was introduced *both* sides go through it, so a bug in the custom pullback moves them
    # together and V50 still passes. `_timeslices` is the only hand-written Zygote rule in the
    # package; everything else composes from rules that are already tested upstream. So it gets an
    # independent check, against the one reference that shares no code with it: the function
    # itself, differenced numerically.
    #
    # Float64 throughout -- a central difference on Float32 parameters is dominated by round-off
    # and would pass against almost anything.
    @testset "V52 gradients match central differences ($em)" for em in (:state_dependent, :none)
        T = Float64
        spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = 4), n_hidden = 6,
                           n_latent = 3, n_encoder = 5, arch = :vrnn, emission = em)
        ps = RF.init_lstm_params(Xoshiro(61), spec; T)
        # move off the zero initialisation, or Wd/Araw sit at a point where the check is vacuous
        ps = merge(ps, (; Wd = randn(Xoshiro(62), T, RF.n_output(spec), spec.n_hidden) ./ 10,
                        bd = randn(Xoshiro(63), T, RF.n_output(spec)) ./ 10,
                        Araw = randn(Xoshiro(64), T, RF.n_output(spec), RF.n_output(spec)) ./ 10))

        # L well past a couple of steps, so the recurrence -- and the slice rule -- is exercised
        L, B, nin, nz, nout = 40, 3, RF.n_input(spec), spec.n_latent, RF.n_output(spec)
        X = randn(Xoshiro(65), T, nin, L, B)
        Y = randn(Xoshiro(66), T, nout, L, B)
        E = randn(Xoshiro(67), T, nz, L, B)
        sc = 11:L

        f(p) = RF.elbo(spec, p, X, Y, sc, E; beta = 1e-3)
        g = Zygote.gradient(f, ps)[1]

        bump(p, k, i, d) = merge(p, NamedTuple{(k,)}((begin
            a = copy(getproperty(p, k)); a[i] += d; a
        end,)))

        δ = 1e-6
        keys_to_check = em === :none ? (:Wx, :Wh, :b, :V1, :V2, :cdec, :Bmu, :Bsig, :We, :be) :
                        (:Wx, :Wh, :b, :V1, :V2, :cdec, :Bmu, :Bsig, :We, :be, :Wd, :bd, :Araw)
        for k in keys_to_check
            arr = getproperty(ps, k)
            arr === nothing && continue
            gk = getproperty(g, k)
            @test gk !== nothing
            # a handful of entries per array, deterministically chosen
            for i in unique(round.(Int, range(1, length(arr); length = min(4, length(arr)))))
                fd = (f(bump(ps, k, i, δ)) - f(bump(ps, k, i, -δ))) / (2δ)
                @test isapprox(gk[i], fd; rtol = 1e-4, atol = 1e-7)
            end
        end

        # and with emission = :none the three unused blocks get no gradient signal at all, which
        # is what lets them stay at their zero initialisation without being frozen by hand
        if em === :none
            for k in (:Wd, :bd, :Araw)
                gk = getproperty(g, k)
                @test gk === nothing || all(iszero, gk)
            end
        end
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

    # -----------------------------------------------------------------------------------------
    # V54 -- the device path, without a device
    # -----------------------------------------------------------------------------------------
    #
    # 🔴 **This workstation has no GPU (`claude_memory.md` #56), so the CUDA path cannot be run
    # here -- and "it compiles" was never the question anyway.** `JLArray` is GPUArrays' reference
    # array: host-backed, but it **refuses scalar indexing** exactly as `CuArray` does and its
    # `similar` returns its own type. That makes it a real proxy for the two failure modes a GPU
    # port actually has -- a host array silently mixed into a device graph, and a scalar `getindex`
    # inside the recurrence -- both of which a plain CPU run cannot see.
    #
    # 🔑 **The positive control comes first.** A proxy that does not enforce the property proves
    # nothing, which is V31's lesson in a new place: assert that `JLArray` really does refuse
    # scalar indexing before believing anything the rest of this testset says.
    #
    # ⚠️ What this does NOT establish: that CUDA.jl compiles these kernels, that the performance is
    # anything but worse, or that `m4_device("cuda")` finds a device. Those need a GPU node.
    @testset "V54 the device path runs and agrees with the host ($arch)" for
            arch in (:storn, :vrnn, :vaernn, :lstm)

        # positive control: the proxy has the property the test relies on
        probe = JLArray(randn(Float32, 2, 2))
        @test_throws Exception probe[1, 1]
        @test similar(probe, Float32, 3, 4) isa JLArray

        em = arch === :lstm ? :constant : :none
        nq, N = 3, 400
        rng = Xoshiro(91)
        q = zeros(Float64, nq, N)
        for t in 2:N
            q[:, t] = 0.8 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = nq), n_hidden = 8,
                           n_latent = 4, n_encoder = 0, arch, emission = em)
        X, Y, steps = RF.build_history(spec.hist, q[:, 1:(N - 1)], q)
        Xc, Yc = permutedims(X), permutedims(Y)
        kw = (; L = 60, burn = 15, epochs = 5, batch = 4, lr = 5e-3, seed = 3, verbose = false)

        pc, hc = RF.train_stochlstm(spec, Xc, Yc, steps; kw...)
        pd, hd = RF.train_stochlstm(spec, Xc, Yc, steps; kw..., device = JLArray)

        # (1) the device fit is the SAME fit. The host draws every random number and only then
        # moves it, so the two differ by reduction order alone -- Float32 round-off, not a
        # different trajectory. A device RNG would show up here as a completely different curve.
        @test length(hd.val) == length(hc.val)
        @test isapprox(hd.val, hc.val; rtol = 1e-4)
        @test isapprox(hd.train, hc.train; rtol = 1e-4)
        @test hd.best_update == hc.best_update

        # (2) parameters come back on the HOST whatever the device was: `LSTMWeights`,
        # `save_stochlstm` and JLD2 all want plain arrays, and a fit only readable on a GPU node
        # is not a fit.
        @test all(v -> v === nothing || v isa Array, values(pd))
        @test all(isapprox(pd[k], pc[k]; rtol = 1e-3, atol = 1e-5)
                  for k in keys(pc) if pc[k] isa Array)
    end

    @testset "V54 m4_device resolves names and refuses a missing device" begin
        @test RF.m4_device("cpu") === identity
        @test RF.m4_device("CPU ") === identity
        @test_throws ErrorException RF.m4_device("tpu")
        # 🔴 `cuda` must THROW when no device is functional, never fall back to the host: a
        # silent fallback reports a GPU run that took CPU time and puts that in a table. Written
        # without importing CUDA (see the header): accept either outcome, but pin WHICH -- on a
        # machine with no device it must be the refusal, with a message naming the reason, and
        # never `identity`.
        res = try
            RF.m4_device("cuda")
        catch e
            e
        end
        @test !(res === identity)
        if res isa ErrorException
            @test occursin("CUDA.functional() is false", res.msg)
        else
            @test res isa Type || res isa Function      # a device array constructor
        end
    end

    # -----------------------------------------------------------------------------------------
    # V55 -- host->device transfers per optimiser update
    # -----------------------------------------------------------------------------------------
    #
    # 🔑 **Counted, not inspected.** `device` is any `Array -> AbstractArray` function, so a
    # counting one measures exactly how much crosses the boundary — on a machine with no GPU, and
    # without profiling tools. The property under test is a MARGINAL: how many calls a further ten
    # epochs add, which is what a 3000-update point pays over and over. One-time staging of the
    # parameters and the record is not a per-update cost and is deliberately not counted.
    #
    # 🔴 The record is staged on the device once and batches are gathered THERE, so `X` and `Y`
    # cost nothing per update. What remains by default is the reparametrisation noise, one draw
    # per update, kept on the host because a device stream would break seed comparability (V54).
    # `device_rng = true` removes that too, at that cost.
    @testset "V55 the record is staged once, not copied per update" begin
        calls = Ref(0)
        counting(x) = (calls[] += 1; copy(x))

        nq, N = 3, 900
        rng = Xoshiro(91)
        q = zeros(Float64, nq, N)
        for t in 2:N
            q[:, t] = 0.8 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 1, n_qoi = nq), n_hidden = 8,
                           n_latent = 4, n_encoder = 0, arch = :storn, emission = :none)
        X, Y, steps = RF.build_history(spec.hist, q[:, 1:(N - 1)], q)
        Xc, Yc = permutedims(X), permutedims(Y)
        kw = (; L = 120, burn = 30, batch = 4, lr = 5e-3, beta = 1e-4, seed = 3, verbose = false)

        count_for(epochs; device_rng) = (calls[] = 0;
            RF.train_stochlstm(spec, Xc, Yc, steps; kw..., epochs, device = counting, device_rng);
            calls[])

        # the marginal cost of ten further epochs, which is what a long fit actually pays
        base = count_for(1; device_rng = false)
        more = count_for(11; device_rng = false)
        per_epoch = (more - base) / 10

        # ⚠️ Updates per epoch is `fld(nfull, min(batch, nfull))`, and every part of that has
        # been got wrong here before. The trailing SHORT segment is dropped -- it cannot share a
        # recurrence with the full-length ones, so batching it alone would spend a whole update on
        # one segment -- and so is the final partial batch, leaving every update exactly
        # `batch_eff` segments wide. `m4_geometry` is the same arithmetic, shared.
        segs = RF.segment_indices(view(steps, 1:floor(Int, 0.8 * length(steps)));
                                  L = 120, burn = 30, stride = 90, anchor = :end)
        nfull = count(sg -> length(sg.rows) == 120, segs)
        n_upd = fld(nfull, min(4, nfull))

        # exactly ONE transfer per update -- the noise draw -- and nothing else
        @test per_epoch == n_upd
        @test base > 0                       # staging did happen

        # 🔴 `device_rng = true` makes the marginal ZERO: nothing crosses per update at all.
        b2 = count_for(1; device_rng = true)
        m2 = count_for(11; device_rng = true)
        @test m2 == b2
        @test b2 < base                      # and it stages no more than the default does
    end

    @testset "train_stochlstm refuses a record it cannot segment" begin
        spec = mkspec(:vrnn)
        X = randn(Xoshiro(91), Float64, RF.n_input(spec), 20)
        Y = randn(Xoshiro(92), Float64, RF.n_output(spec), 20)
        @test_throws ErrorException RF.train_stochlstm(spec, X, Y, collect(1:20);
                                                       L = 200, burn = 150, verbose = false)
        @test_throws ErrorException RF.train_stochlstm(spec, X, Y, collect(1:19); verbose = false)
    end

    # -----------------------------------------------------------------------------------------
    # V53 -- overlapping training segments, and the split that makes them safe
    # -----------------------------------------------------------------------------------------
    #
    # 🔴 The train/val split is by ROW, not by position in the segment list. Splitting the list is
    # safe ONLY at `stride == L - burn`, where the scored windows exactly tile the record; at any
    # shorter stride they overlap, and scored validation rows land inside scored training rows --
    # a validation loss that is quietly a training loss. This testset reproduces the geometry of
    # the split directly, because it is the property that has to hold, not the code that produces
    # it.
    @testset "V53 scored train and val rows stay disjoint at every stride" begin
        ncol, L, burn, val_frac = 1000, 200, 50, 0.2
        ntrain = floor(Int, (1 - val_frac) * ncol)
        steps = collect(1:ncol)

        for stride in (L - burn, 75, 50, 25, 1)
            # training's geometry exactly: tiled from the end, short stub dropped
            tr = filter(s -> length(s.rows) == L,
                        RF.segment_indices(view(steps, 1:ntrain); L, burn, stride, anchor = :end))
            va = [(; rows = s.rows .+ ntrain, score = s.score .+ ntrain)
                  for s in RF.segment_indices(view(steps, (ntrain + 1):ncol); L, burn)]
            @test !isempty(tr) && !isempty(va)

            scored_tr = reduce(union, (collect(s.score) for s in tr))
            scored_va = reduce(union, (collect(s.score) for s in va))
            @test isempty(intersect(scored_tr, scored_va))
            # and no segment of EITHER side reaches a row belonging to the other
            @test maximum(maximum(s.rows) for s in tr) <= ntrain
            @test minimum(minimum(s.rows) for s in va) > ntrain
            # the embargo: the first scored val row is `burn` steps past the last training row
            @test minimum(scored_va) == ntrain + burn + 1
            # 🔴 and the last training row IS scored, at every stride: the dropped short segment
            # is the oldest stretch, not the rows adjacent to validation
            @test maximum(scored_tr) == ntrain
        end

        # a shorter stride really does make more segments, and they cover the same rows
        n_tile = length(RF.segment_indices(view(steps, 1:ntrain); L, burn, stride = L - burn))
        n_over = length(RF.segment_indices(view(steps, 1:ntrain); L, burn, stride = 25))
        @test n_over > 4 * n_tile

        # a stride past `L - burn` would leave rows nothing ever scores, and is refused
        spec = mkspec(:vrnn; h = 1, n_qoi = 2)
        X = randn(Xoshiro(93), Float64, RF.n_input(spec), ncol)
        Y = randn(Xoshiro(94), Float64, RF.n_output(spec), ncol)
        @test_throws ErrorException RF.train_stochlstm(spec, X, Y, steps;
                                                       L, burn, stride = L - burn + 1,
                                                       epochs = 1, verbose = false)

        # and a strided fit runs and improves, which the geometry above does not by itself prove
        _, h = RF.train_stochlstm(spec, X, Y, steps; L, burn, stride = 50, epochs = 8, batch = 4,
                                  lr = 5e-3, seed = 2, verbose = false)
        @test length(h.train) == length(h.update)
        @test h.updates == 8 * h.upd_per_epoch
        @test h.update[end] == h.updates
        @test all(isfinite, h.train)
    end

    # -----------------------------------------------------------------------------------------
    # V56 -- a loss spike costs ONE decay, not a cascade to min_lr
    # -----------------------------------------------------------------------------------------
    #
    # 🔴 The 2026-09-23 stride scan: the `(400, b2)` control spiked, and a patience rule counted
    # against the ALL-TIME best then fired every `patience` validations -- five decays in 160
    # updates, lr at the floor, fit stopped at 15x its neighbours' loss. `_plateau_step` counts
    # against the best since the last decay. Tested on a synthetic validation sequence, because
    # the property is about the rule, and a fit that happens to spike is not a reproducible input.
    @testset "V56 patience counts from the last decay, so a spike does not cascade" begin
        ext = Base.get_extension(RikFlow, :RikFlowLuxExt)
        patience = 3
        descent = collect(range(1.0, 0.5; length = 10))
        recovery = [5.0 * 0.9^k for k in 0:40]        # a spike, then a slow monotone recovery
        vals = vcat(descent, recovery)

        # the new rule, exactly as `train_stochlstm` drives it
        function decays_new(vals)
            pl, n = ext._PLATEAU_FRESH, 0
            for v in vals
                pl, d = ext._plateau_step(pl, v, patience)
                d && (n += 1; pl = ext._PLATEAU_FRESH)
            end
            return n
        end
        # the OLD rule, reproduced as the positive control: all-time best, counter reset at decay
        function decays_old(vals)
            best, since, n = Inf, 0, 0
            for v in vals
                v < best ? (best = v; since = 0) : (since += 1)
                since >= patience && (n += 1; since = 0)
            end
            return n
        end

        # 🔑 positive control: the old rule really does cascade on this sequence -- a test whose
        # failure mode it cannot reproduce proves nothing
        @test decays_old(vals) >= 5
        # the spike itself reads as a plateau and costs one decay; the recovery costs none
        @test decays_new(vals) == 1

        # the rule still does its job on a genuine plateau, after a decay as well as before
        flat = vcat(descent, fill(0.5, 4 * (patience + 1)))
        @test decays_new(flat) >= 3
        # and before the first decay it is the old rule: same first decay on any sequence
        first_decay(f, vals) = findfirst(k -> f(vals[1:k]) > 0, eachindex(vals))
        @test first_decay(decays_new, vals) == first_decay(decays_old, vals)
        @test first_decay(decays_new, flat) == first_decay(decays_old, flat)
    end

    # -----------------------------------------------------------------------------------------
    # V57 -- the windowed stop: < stop_rel gain over the last stop_window updates
    # -----------------------------------------------------------------------------------------
    @testset "V57 the windowed stop reads a RATE, and survives a negative loss" begin
        ext = Base.get_extension(RikFlow, :RikFlowLuxExt)
        ws = ext._window_stalled
        u = collect(2:2:1000)                        # validations every 2 updates
        # geometric best-so-far: a fixed fractional gain per validation
        bsf_rate(g) = [1.0 * (1 - g)^k for k in 0:(length(u) - 1)]
        # 250 validations span 500 updates: 1e-4/validation is ~2.5% per window -> not stalled;
        # 1e-5/validation is ~0.25% -> stalled
        @test !ws(u, bsf_rate(1e-4), 500, 0.005)
        @test ws(u, bsf_rate(1e-5), 500, 0.005)
        # not before the history spans the window, and never when disabled
        @test !ws(u[1:100], bsf_rate(1e-5)[1:100], 500, 0.005)
        @test !ws(u, bsf_rate(1e-5), 0, 0.005)
        @test !ws(u, bsf_rate(1e-5), 500, 0.0)
        # 🔴 a NEGATIVE loss (the `:lstm` control's log-density, ~ -18): a steady improvement must
        # not be read as a loss. Without `abs` in the denominator the first line would be true.
        neg_fast = [-18.0 - 0.01k for k in 0:(length(u) - 1)]     # 2.5 nats per window: 14%
        neg_slow = [-18.0 - 1e-5k for k in 0:(length(u) - 1)]     # 0.0025 nats: 0.014%
        @test !ws(u, neg_fast, 500, 0.005)
        @test ws(u, neg_slow, 500, 0.005)

        # and inside a real fit: a window the fit can span, with a threshold no gain can meet,
        # stops it before the cap and says why
        T = Float32
        nq, N = 3, 700
        rng = Xoshiro(97)
        q = zeros(Float64, nq, N)
        for t in 2:N
            q[:, t] = 0.8 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        spec = mkspec(:storn; n_qoi = nq, h = 1, n_hidden = 8, n_latent = 4, n_encoder = 8)
        X, Yb, steps = RF.build_history(spec.hist, q[:, 1:(N - 1)], q)
        kw = (; L = 60, burn = 15, epochs = 40, batch = 4, lr = 5e-3, seed = 3, verbose = false, T)
        _, h = RF.train_stochlstm(spec, permutedims(X), permutedims(Yb), steps; kw...,
                                  stop_window = 4, stop_rel = 10.0)
        @test h.stopped_early && h.stop_reason === :window
        @test h.updates < 40 * h.upd_per_epoch
        # the default window cannot be spanned by a fit this short, so it runs to the cap
        _, h0 = RF.train_stochlstm(spec, permutedims(X), permutedims(Yb), steps; kw...)
        @test !h0.stopped_early && h0.stop_reason === :cap
        @test h0.updates == 40 * h0.upd_per_epoch
    end

    # -----------------------------------------------------------------------------------------
    # V58 -- rollout training: the model's own output fed back into its level lags
    # -----------------------------------------------------------------------------------------
    #
    # 🔴 Three properties, each of which a plausible-looking implementation can get wrong silently:
    # K = 1 must BE teacher forcing (so `rollout = 1` changes nothing); a free run must be the
    # DEPLOYED closure's loop, `lstm_step!` fed its own `y` in the lag columns; and the gradient
    # must flow through the feedback, checked against central differences.
    @testset "V58 rollout training: K = 1 is teacher forcing, K > 1 is the deployed feedback loop ($arch, n_encoder=$ne)" for
            arch in (:vaernn, :storn, :vrnn), ne in (0, 5)
        ext = Base.get_extension(RikFlow, :RikFlowLuxExt)
        T = Float64
        spec = mkspec(arch; n_qoi = 3, h = 2, n_encoder = ne, emission = :none)
        ps = RF.init_lstm_params(Xoshiro(41), spec; T)
        nin, nout, nz = RF.n_input(spec), RF.n_output(spec), spec.n_latent
        L, burn, B = 30, 8, 3
        X = randn(Xoshiro(42), T, nin, L, B); X[end, :, :] .= 1       # bias column
        Y = randn(Xoshiro(43), T, nout, L, B)
        E = randn(Xoshiro(44), T, nz, L, B)
        sc = (burn + 1):L

        # (1) K = 1: no step is free, so the rollout pass IS the teacher-forced pass
        f1 = ext._rollout_mask(L, burn, 1)
        @test !any(f1)
        @test RF.elbo(spec, ps, X, Y, sc, E; beta = 1e-3, free = f1) ≈
              RF.elbo(spec, ps, X, Y, sc, E; beta = 1e-3) rtol = 1e-12

        # (2) a full free run equals the DEPLOYED step fed its own output, at z = mu (eps = 0)
        ff = ext._rollout_mask(L, burn, L)
        @test !any(ff[1:burn]) && all(ff[(burn + 1):(L - 1)])
        o = ext._rollout_forward(spec, ps, X, zero(E), ff)
        w = RF.LSTMWeights(ps, spec)
        for b in 1:B
            st = RF.LSTMState(spec, T)
            ys = zeros(T, nout, L)
            for t in 1:L
                x = X[:, t, b]
                for k in 1:spec.hist.h
                    s = t - k
                    (s >= 1 && ff[s]) && (x[RF.qlag_columns(spec.hist, k)] .= ys[:, s])
                end
                RF.lstm_step!(st, w, spec, x; sample_latent = false)
                ys[:, t] .= st.y
            end
            @test o.Y[:, :, b] ≈ ys rtol = 1e-10
        end
        # and the feedback really changes the answer -- a rollout that fed the record would pass (2)
        @test !(o.Y ≈ RF.lstm_forward(spec, ps, X, zero(E)).Y)

        # (2b) the :dQ target: the fed-back level is q*_lag + a .* y .+ b, not y itself
        lm = (; a = T[0.5, 2.0, 1.5], b = T[0.1, -0.2, 0.3])
        od = ext._rollout_forward(spec, ps, X, zero(E), ff; lagmap = lm)
        for b in 1:B
            st = RF.LSTMState(spec, T)
            ys = zeros(T, nout, L)
            for t in 1:L
                x = X[:, t, b]
                for k in 1:spec.hist.h
                    s = t - k
                    if s >= 1 && ff[s]
                        x[RF.qlag_columns(spec.hist, k)] .=
                            X[RF.qstarlag_columns(spec.hist, k), t, b] .+ lm.a .* ys[:, s] .+ lm.b
                    end
                end
                RF.lstm_step!(st, w, spec, x; sample_latent = false)
                ys[:, t] .= st.y
            end
            @test od.Y[:, :, b] ≈ ys rtol = 1e-10
        end

        # (3) the gradient THROUGH the feedback, against central differences in a random direction
        fK = ext._rollout_mask(L, burn, 7)
        loss(p) = RF.elbo(spec, p, X, Y, sc, E; beta = 1e-3, free = fK)
        g = Zygote.gradient(loss, ps)[1]
        v = map(a -> a === nothing ? nothing : randn(Xoshiro(45), T, size(a)), ps)
        dir(sgn, h) = map((a, d) -> a === nothing ? nothing : a .+ sgn * h .* d, ps, v)
        fd = (loss(dir(1, 1e-6)) - loss(dir(-1, 1e-6))) / 2e-6
        ad = sum(sum(gi .* vi) for (gi, vi) in zip(values(g), values(v)) if gi !== nothing && vi !== nothing)
        @test ad ≈ fd rtol = 1e-5
    end

    @testset "V58 rollout training runs, warm-starts, and refuses an emission head" begin
        T = Float32
        nq, N = 3, 700
        rng = Xoshiro(98)
        q = zeros(Float64, nq, N)
        for t in 2:N
            q[:, t] = 0.8 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        spec = mkspec(:storn; n_qoi = nq, h = 1, n_hidden = 8, n_latent = 4, n_encoder = 0,
                      emission = :none)
        X, Yb, steps = RF.build_history(spec.hist, q[:, 1:(N - 1)], q)
        Xc, Yc = permutedims(X), permutedims(Yb)
        kw = (; L = 60, burn = 15, batch = 4, seed = 3, verbose = false, T)
        _, h = RF.train_stochlstm(spec, Xc, Yc, steps; kw..., epochs = 6, lr = 5e-3, rollout = 45,
                                  clip = 1.0)
        @test h.rollout == 45 && h.clip == 1.0 && all(isfinite, h.val) && all(isfinite, h.gmax)
        @test length(h.gmax) == length(h.val)
        # warm start: at lr = 0 Adam does not move, so the returned fit IS init_ps
        p0, _ = RF.train_stochlstm(spec, Xc, Yc, steps; kw..., epochs = 3, lr = 5e-3)
        p1, h1 = RF.train_stochlstm(spec, Xc, Yc, steps; kw..., epochs = 2, lr = 0.0,
                                    rollout = 45, init_ps = p0)
        @test h1.warm_start && p1.Wx ≈ p0.Wx && p1.V1 ≈ p0.V1
        # the warm start is validated at update 0 and can be returned: a fine-tune never comes back
        # worse than it started (lr = 0.5 wrecks it within an update)
        @test h1.update[1] == 0 && isnan(h1.train[1])
        p2, h2 = RF.train_stochlstm(spec, Xc, Yc, steps; kw..., epochs = 3, lr = 0.5,
                                    rollout = 45, init_ps = p0)
        @test h2.best_val <= h2.val[1]
        h2.best_update == 0 && @test p2.Wx ≈ p0.Wx
        # a wrong-shaped warm start is refused, and so is an emission head
        bad = merge(p0, (; Wh = zeros(T, 3, 3)))
        @test_throws ErrorException RF.train_stochlstm(spec, Xc, Yc, steps; kw..., epochs = 1,
                                                       init_ps = bad)
        spec_e = mkspec(:storn; n_qoi = nq, h = 1, n_hidden = 8, n_latent = 4, n_encoder = 0,
                        emission = :constant)
        @test_throws ErrorException RF.train_stochlstm(spec_e, Xc, Yc, steps; kw..., epochs = 1,
                                                       rollout = 10)
    end

    # -----------------------------------------------------------------------------------------
    # V59 -- emission = :constant means constant: Wd is never trained
    # -----------------------------------------------------------------------------------------
    #
    # 🔴 Until 2026-09-23 `lstm_forward` computed `Wd * h + bd` for every emission mode, so a
    # `:constant` head learned `Wd` like a state-dependent one and a `:constant` and a
    # `:state_dependent` fit from one seed came out bit-identical. The deployed step computes
    # `Wd * h + bd` too, so the only way `:constant` can be constant is `Wd` staying at zero.
    @testset "V59 a :constant emission head never trains Wd" begin
        T = Float32
        nq, N = 3, 500
        rng = Xoshiro(99)
        q = zeros(Float64, nq, N)
        for t in 2:N
            q[:, t] = 0.8 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        kw = (; L = 60, burn = 15, epochs = 4, batch = 4, lr = 5e-3, seed = 3, verbose = false, T)
        function fit(em)
            sp = mkspec(:storn; n_qoi = nq, h = 1, n_hidden = 6, n_latent = 3, n_encoder = 0,
                        emission = em)
            X, Yb, st = RF.build_history(sp.hist, q[:, 1:(N - 1)], q)
            return RF.train_stochlstm(sp, permutedims(X), permutedims(Yb), st; kw...)[1]
        end
        pc, psd = fit(:constant), fit(:state_dependent)
        @test all(iszero, pc.Wd)                 # constant: the scale head never moved
        @test !all(iszero, psd.Wd)               # positive control: state-dependent does train it
        @test pc.bd != psd.bd || pc.Wx != psd.Wx # and the two are no longer the same fit
    end

    # -----------------------------------------------------------------------------------------
    # V61 -- window mode: the training segment IS the deployed window
    # -----------------------------------------------------------------------------------------

    @testset "V61 window: lstm_forward over the stored window == the deployed replay ($arch)" for
            arch in (:lstm, :storn, :vrnn)

        T = Float32
        nq, W = 3, 5
        hist = RF.HistorySpec(; h = 0, n_qoi = nq, hist_var = :q_star)
        spec = RF.LSTMSpec(; hist, n_hidden = 6, n_latent = 4, n_encoder = 0, arch, window = W,
                           emission = arch === :lstm ? :constant : :none, skip = true)
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        ps = merge(ps, (; Ws = randn(Xoshiro(4), T, size(ps.Ws)...) ./ 3))
        w = RF.LSTMWeights(ps, spec)
        mu, sd = reshape([0.1, 0.2, 0.3], nq, 1), reshape([2.0, 2.5, 3.0], nq, 1)
        sc = (; in_scaling = (; mu, sigma = sd), out_scaling = (; mu, sigma = sd), target = :dQ)
        rng = Xoshiro(11)
        q_star = randn(rng, nq, 20) .+ 1.0
        dQ = randn(rng, nq, 7) ./ 10
        m = RF.StochLSTM(spec, w, sc; spinnup_data = dQ, rng = Xoshiro(5), gate = 0.0,
                         stochastic = false)
        for k in 1:20
            RF.get_next_item_timeseries(m, q_star[:, k])
            k > size(dQ, 2) || continue
            # the training forward pass, from h = 0, on the closure's own window and draws
            out = RF.lstm_forward(spec, ps, copy(m.xwin), copy(m.ewin))
            @test out.Y[:, end] ≈ m.state.y rtol = 1e-5
        end
        RF.latent_sampled(spec) && @test any(!iszero, m.ewin)   # the draws are real

        path = joinpath(mktempdir(), "win.jld2")
        RF.save_stochlstm(path, spec, w, sc)
        @test RF.load_stochlstm(path).spec.window == W
    end

    @testset "V61 window: train_stochlstm fits windows and refuses L != window" begin
        T = Float32
        nq, W, N = 3, 6, 400
        rng = Xoshiro(21)
        q = zeros(nq, N)
        for t in 2:N
            q[:, t] = 0.8 .* q[:, t - 1] .+ 0.3 .* randn(rng, nq)
        end
        hist = RF.HistorySpec(; h = 0, n_qoi = nq, hist_var = :q_star)
        spec = RF.LSTMSpec(; hist, n_hidden = 5, n_latent = 3, n_encoder = 0, arch = :vrnn,
                           window = W)
        X, Yb, st = RF.build_history(hist, q[:, 1:(N - 1)], q)
        Xc, Yc = permutedims(X), permutedims(Yb)
        @test_throws ErrorException RF.train_stochlstm(spec, Xc, Yc, st; L = W + 1, burn = W,
                                                       stride = 1, epochs = 1, verbose = false)
        ps, h = RF.train_stochlstm(spec, Xc, Yc, st; L = W, burn = W - 1, stride = 1, epochs = 2,
                                   batch = 16, verbose = false, T)
        # every training row that can end a full window does: ~0.8 N windows, not N / L
        @test h.nseg == floor(Int, 0.8 * length(st)) - (W - 1)
        @test all(isfinite, h.val)
        @test RF.check_shapes(RF.LSTMWeights(ps, spec), spec)
    end

    # -----------------------------------------------------------------------------------------
    # V62 -- kl_mode = :reference is the source's `vae_loss_2D`
    # -----------------------------------------------------------------------------------------

    @testset "V62 kl_mode = :reference reproduces vae_loss_2D ($arch)" for arch in (:storn, :vrnn)
        T = Float32
        nq, L, B, lam = 3, 8, 4, 1e-2
        spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 0, n_qoi = nq, hist_var = :q_star),
                           n_hidden = 5, n_latent = 4, n_encoder = 0, arch, window = L)
        ps = RF.init_lstm_params(Xoshiro(7), spec; T)
        ps = merge(ps, (; Bsig = randn(Xoshiro(8), T, size(ps.Bsig)...)))   # sig not all log 2
        X = randn(Xoshiro(9), T, RF.n_input(spec), L, B)
        Y = randn(Xoshiro(10), T, nq, L, B)
        E = randn(Xoshiro(11), T, spec.n_latent, L, B)
        o = RF.lstm_forward(spec, ps, X, E)

        # the source, written out: yloss per (b, t) summed over features, KL one scalar over
        # everything, loss = mean over (b, t) of (yloss + lam * kl)
        for sc in (1:L, L:L)
            yl = [sum(abs2, Y[:, t, b] .- o.Y[:, t, b]) for t in sc, b in 1:B]
            kl = 0.5 * sum(o.SIG .^ 2 .+ o.MU .^ 2 .- 1 .- log.(1e-8 .+ o.SIG .^ 2))
            want = sum(yl .+ lam * kl) / length(yl)
            got = RF.elbo(spec, ps, X, Y, sc, E; beta = lam, kl_mode = :reference, kl_batch = B)
            @test got ≈ want rtol = 1e-5
        end

        # splitting a chunk into smaller batches (validation) gives the same mean loss
        whole = RF.elbo(spec, ps, X, Y, L:L, E; beta = lam, kl_mode = :reference, kl_batch = B)
        halves = [RF.elbo(spec, ps, X[:, :, r], Y[:, :, r], L:L, E[:, :, r]; beta = lam,
                          kl_mode = :reference, kl_batch = B) for r in (1:2, 3:4)]
        @test sum(halves) / 2 ≈ whole rtol = 1e-5
        # and the KL really carries weight: the per-step objective is a different number
        @test RF.elbo(spec, ps, X, Y, L:L, E; beta = lam) != whole

        spec_e = RF.LSTMSpec(; hist = spec.hist, n_hidden = 5, n_latent = 4, n_encoder = 0, arch,
                             emission = :constant)
        pe = RF.init_lstm_params(Xoshiro(7), spec_e; T)
        @test_throws ErrorException RF.elbo(spec_e, pe, X, Y, L:L, E; beta = lam,
                                            kl_mode = :reference, kl_batch = B)
        # a short fit runs end to end under it
        Xc = randn(Xoshiro(12), T, RF.n_input(spec), 200); Yc = randn(Xoshiro(13), T, nq, 200)
        _, h = RF.train_stochlstm(spec, Xc, Yc, collect(1:200); L, burn = L - 1, stride = 1,
                                  epochs = 1, batch = 8, verbose = false, kl_mode = :reference, T)
        @test h.kl_mode === :reference && all(isfinite, h.val)
    end

    @testset "V64 an explicit split: segments never cross a gap, validation is the given block" begin
        T = Float32
        nq, W, N = 3, 6, 300
        hist = RF.HistorySpec(; h = 0, n_qoi = nq, hist_var = :q_star)
        spec = RF.LSTMSpec(; hist, n_hidden = 4, n_latent = 2, n_encoder = 0, arch = :vrnn,
                           window = W)
        Xc = randn(Xoshiro(1), T, RF.n_input(spec), N); Yc = randn(Xoshiro(2), T, nq, N)
        steps = collect(1:N)
        va = collect(121:180)                              # a middle block validates
        tr = [c for c in 1:N if !(111 <= c <= 190)]         # 10-row embargo on each side
        _, h = RF.train_stochlstm(spec, Xc, Yc, steps; L = W, burn = W - 1, stride = 1,
                                  epochs = 1, batch = 8, verbose = false, T,
                                  split = (; train = tr, val = va))
        # full windows inside [1, 110] and [191, 300] only: (110 - W + 1) + (110 - W + 1)
        @test h.nseg == 2 * (110 - W + 1)
        @test all(isfinite, h.val)
        @test_throws ErrorException RF.train_stochlstm(spec, Xc, Yc, steps; L = W, burn = W - 1,
                                                       stride = 1, epochs = 1, verbose = false,
                                                       split = (; train = tr, val = [100, 101]))
    end

    # -----------------------------------------------------------------------------------------
    # V65 -- posterior = :xy, the conditional-VAE encoder: q(z | x, y) in training, prior online
    # -----------------------------------------------------------------------------------------

    @testset "V65 posterior = :xy ($arch)" for arch in (:storn, :vrnn)
        T = Float32
        nq, L, B = 3, 6, 4
        hist = RF.HistorySpec(; h = 0, n_qoi = nq, hist_var = :q_star)
        spec = RF.LSTMSpec(; hist, n_hidden = 5, n_latent = 4, n_encoder = 0, arch, window = L,
                           emission = :constant, posterior = :xy)
        @test RF.n_encoder_in(spec) == RF.n_input(spec) + nq
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        @test size(ps.Bmu) == (4, RF.n_input(spec) + nq)
        X = randn(Xoshiro(4), T, RF.n_input(spec), L, B)
        Y = randn(Xoshiro(5), T, nq, L, B)
        E = randn(Xoshiro(6), T, 4, L, B)

        # without a target the forward pass IS the prior: z = eps, the encoder weights unused
        o1 = RF.lstm_forward(spec, ps, X, E)
        o2 = RF.lstm_forward(spec, merge(ps, (; Bmu = 10 .* ps.Bmu, Bsig = ps.Bsig .+ 1)), X, E)
        @test o1.Y == o2.Y && all(o1.MU .== 0) && all(o1.SIG .== 1)
        # with one, the encoder reads it: a different target moves the latent
        a = RF.lstm_forward(spec, ps, X, E; Y)
        b = RF.lstm_forward(spec, ps, X, E; Y = Y .+ 1)
        @test maximum(abs, a.MU .- b.MU) > 1e-3
        # and elbo trains the encoder on it: nonzero gradient on Bmu
        g = Zygote.gradient(p -> RF.elbo(spec, p, X, Y, 1:L, E; beta = 1.0), ps)[1]
        @test sum(abs, g.Bmu) > 0

        # the deployed step draws from the prior and ignores the encoder
        w = RF.LSTMWeights(ps, spec)
        st = RF.LSTMState(spec, T)
        for t in 1:L
            y, _, z = RF.lstm_step!(st, w, spec, view(X, :, t, 1); eps = view(E, :, t, 1))
            @test z == E[:, t, 1]
            @test y ≈ o1.Y[:, t, 1] rtol = 1e-5
        end
        path = joinpath(mktempdir(), "xy.jld2")
        RF.save_stochlstm(path, spec, w, (; in_scaling = nothing, out_scaling = nothing))
        @test RF.load_stochlstm(path).spec.posterior === :xy
        @test_throws ErrorException RF.LSTMSpec(; hist, arch = :lstm, emission = :constant,
                                                posterior = :xy)
    end

    # -----------------------------------------------------------------------------------------
    # V66 -- arch = :dense, the no-recurrence window VAE (Phase D)
    # -----------------------------------------------------------------------------------------

    @testset "V66 arch = :dense: training forward == deployed window map (posterior $post)" for
            post in (:x, :xy)
        T = Float32
        nq, W = 3, 5
        hist = RF.HistorySpec(; h = 1, n_qoi = nq, hist_var = :q_star_q)
        spec = RF.LSTMSpec(; hist, n_hidden = 7, n_latent = 2, n_encoder = 0, arch = :dense,
                           window = W, emission = :constant, skip = true, posterior = post)
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        @test size(ps.Wx) == (7, W * (RF.n_input(spec) + 2)) && length(ps.b) == 14
        ps = merge(ps, (; Ws = randn(Xoshiro(4), T, size(ps.Ws)...) ./ 3, V1 = randn(Xoshiro(5), T, nq, 7)))
        w = RF.LSTMWeights(ps, spec)
        @test RF.check_shapes(w, spec)
        mu, sd = reshape([0.1, 0.2, 0.3], nq, 1), reshape([2.0, 2.5, 3.0], nq, 1)
        sc = (; in_scaling = (; mu, sigma = sd), out_scaling = (; mu, sigma = sd), target = :dQ)
        rng = Xoshiro(11)
        q_star = randn(rng, nq, 16) .+ 1.0
        m = RF.StochLSTM(spec, w, sc; spinnup_data = randn(rng, nq, 6) ./ 10, rng = Xoshiro(5),
                         gate = 0.0, stochastic = false)
        for k in 1:16
            RF.get_next_item_timeseries(m, q_star[:, k])
            k > 6 || continue
            o = RF.lstm_forward(spec, ps, reshape(copy(m.xwin), :, W, 1), reshape(copy(m.ewin), :, W, 1))
            @test o.Y[:, end, 1] ≈ m.state.y rtol = 1e-5
            @test all(iszero, o.Y[:, 1:(end - 1), 1])
        end
        path = joinpath(mktempdir(), "dense.jld2")
        RF.save_stochlstm(path, spec, w, sc)
        back = RF.load_stochlstm(path)
        @test back.spec.arch === :dense && back.spec.posterior === post
        # a short fit (last-step score) runs end to end
        N = 200
        Xc = randn(Xoshiro(12), T, RF.n_input(spec), N); Yc = randn(Xoshiro(13), T, nq, N)
        _, h = RF.train_stochlstm(spec, Xc, Yc, collect(1:N); L = W, burn = W - 1, stride = 1,
                                  epochs = 1, batch = 8, verbose = false, T)
        @test all(isfinite, h.val)
        @test_throws ErrorException RF.LSTMSpec(; hist, arch = :dense, window = 0)
    end

    # V71 -- arch = :dense with n_latent = 0: M3ᶠ's deterministic residual MLP + constant head
    @testset "V71 arch = :dense, n_latent = 0: no latent, constant noise head, zero-init net = skip (W = $W)" for
            W in (1, 3)
        T = Float32
        nq = 3
        hist = RF.HistorySpec(; h = 2, n_qoi = nq, hist_var = :q_star_q)
        spec = RF.LSTMSpec(; hist, n_hidden = 5, n_latent = 0, n_encoder = 0, arch = :dense,
                           window = W, emission = :constant, skip = true)
        @test RF.n_cell_input(spec) == W * RF.n_input(spec)
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        @test size(ps.Bmu, 1) == 0
        Ws = randn(Xoshiro(4), T, nq, RF.n_input(spec)) ./ 3
        # zero-initialised output (the fit driver's V1 = 0): the model IS the skip
        p0 = merge(ps, (; Ws, V1 = zero(ps.V1)))
        X = randn(Xoshiro(6), T, RF.n_input(spec), W, 4)
        o0 = RF.lstm_forward(spec, p0, X, zeros(T, 0, W, 4))
        @test o0.Y[:, end, :] ≈ Ws * X[:, end, :]
        # training forward == deployed window map, with a live network
        p1 = merge(p0, (; V1 = randn(Xoshiro(5), T, nq, 5), bd = T[-1, -0.5, 0]))
        w = RF.LSTMWeights(p1, spec)
        @test RF.check_shapes(w, spec)
        mu, sd = reshape([0.1, 0.2, 0.3], nq, 1), reshape([2.0, 2.5, 3.0], nq, 1)
        sc = (; in_scaling = (; mu, sigma = sd), out_scaling = (; mu, sigma = sd), target = :dQ)
        rng = Xoshiro(11)
        q_star = randn(rng, nq, 12) .+ 1.0
        m = RF.StochLSTM(spec, w, sc; spinnup_data = randn(rng, nq, 6) ./ 10, rng = Xoshiro(5),
                         gate = 0.0, stochastic = true)
        for k in 1:12
            RF.get_next_item_timeseries(m, q_star[:, k])
            k > 6 || continue
            o = RF.lstm_forward(spec, p1, reshape(copy(m.xwin), :, W, 1), zeros(T, 0, W, 1))
            @test o.Y[:, end, 1] ≈ m.state.y rtol = 1e-5
            @test o.LOGD[:, end, 1] ≈ T[-1, -0.5, 0]      # constant head: the scale is bd alone
        end
        path = joinpath(mktempdir(), "dense0.jld2")
        RF.save_stochlstm(path, spec, w, sc)
        back = RF.load_stochlstm(path)
        @test back.spec.arch === :dense && back.spec.n_latent == 0
        # a short fit runs, with weight decay and the skip frozen, and leaves Ws untouched
        N = 200
        Xc = randn(Xoshiro(12), T, RF.n_input(spec), N); Yc = randn(Xoshiro(13), T, nq, N)
        pf, h = RF.train_stochlstm(spec, Xc, Yc, collect(1:N); L = W, burn = W - 1, stride = 1,
                                   epochs = 2, batch = 8, verbose = false, T, init_ps = p0,
                                   freeze = (:Ws,), weight_decay = 1e-3)
        @test all(isfinite, h.val)
        @test pf.Ws == Ws
        # `decay_exclude`: with a zero learning signal on nothing but decay, an exempt leaf is
        # untouched and a decayed one shrinks. Y = the model's own output makes every gradient ~0
        # at init except through the head, so compare a huge decay with and without the exemption.
        pz = merge(p1, (; Wx = ps.Wx))
        Xz = randn(Xoshiro(14), T, RF.n_input(spec), N)
        kw = (; L = W, burn = W - 1, stride = 1, epochs = 1, batch = 8, verbose = false, T,
              init_ps = pz, freeze = (:Ws, :V1, :Wx, :Wh, :b, :cdec, :Araw), weight_decay = 10.0,
              val_every = 1000)
        Yz = Float32.(p1.Ws * Xz)          # a V1 h term remains, so bd sees a small gradient
        pa, _ = RF.train_stochlstm(spec, Xz, Yz, collect(1:N); kw...)
        pb, _ = RF.train_stochlstm(spec, Xz, Yz, collect(1:N); kw..., decay_exclude = (:bd,))
        @test norm(pa.bd .- p1.bd) > 3 * norm(pb.bd .- p1.bd)
        # no stochasticity at all, or a conditional encoder with nothing to encode: refused
        @test_throws ErrorException RF.LSTMSpec(; hist, arch = :dense, window = W, n_latent = 0)
        @test_throws ErrorException RF.LSTMSpec(; hist, arch = :dense, window = W, n_latent = 0,
                                                emission = :constant, posterior = :xy)
        @test_throws ErrorException RF.LSTMSpec(; hist, arch = :vrnn, n_latent = 0)
    end

    # -----------------------------------------------------------------------------------------
    # V70 -- prior = :learned, the VRNN prior p(z_t | h_{t-1}): learned colour
    # -----------------------------------------------------------------------------------------

    @testset "V70 prior = :learned ($arch)" for arch in (:storn, :vrnn)
        T = Float32
        nq, L, B = 3, 6, 2
        hist = RF.HistorySpec(; h = 0, n_qoi = nq, hist_var = :q_star)
        spec = RF.LSTMSpec(; hist, n_hidden = 5, n_latent = 3, n_encoder = 0, arch, window = L,
                           emission = :constant, posterior = :xy, prior = :learned)
        ps = RF.init_lstm_params(Xoshiro(3), spec; T)
        @test all(iszero, ps.Pm) && all(RF._softplus.(ps.Pbs) .≈ 1)    # starts at N(0, I)
        ps = merge(ps, (; Pm = randn(Xoshiro(4), T, 3, 5) ./ 2, Ps = randn(Xoshiro(5), T, 3, 5) ./ 2,
                        Pbm = randn(Xoshiro(6), T, 3) ./ 3))
        w = RF.LSTMWeights(ps, spec)
        @test RF.check_shapes(w, spec)
        X = randn(Xoshiro(7), T, RF.n_input(spec), L, B)
        E = randn(Xoshiro(8), T, 3, L, B)
        o = RF.lstm_forward(spec, ps, X, E)                     # prior mode: z from p(z | h)
        @test o.MU == o.MUP
        # replay with the DEPLOYED step: z_t = learned_prior_z!(h_{t-1}, eps_t), then lstm_step!
        st = RF.LSTMState(spec, T)
        z = zeros(T, 3)
        for t in 1:L
            RF.learned_prior_z!(z, st, w, E[:, t, 1])
            @test z ≈ o.MU[:, t, 1] .+ o.SIG[:, t, 1] .* E[:, t, 1] rtol = 1e-5
            y, _, _ = RF.lstm_step!(st, w, spec, view(X, :, t, 1); eps = z)
            @test y ≈ o.Y[:, t, 1] rtol = 1e-5
        end
        # the KL trains the prior
        Y = randn(Xoshiro(9), T, nq, L, B)
        g = Zygote.gradient(p -> RF.elbo(spec, p, X, Y, 1:L, E; beta = 1.0), ps)[1]
        @test sum(abs, g.Pm) > 0 && sum(abs, g.Bmu) > 0
        # the deployed closure: its output equals a replay of its own stored window of latents
        mu, sd = reshape([0.1, 0.2, 0.3], nq, 1), reshape([2.0, 2.5, 3.0], nq, 1)
        sc = (; in_scaling = (; mu, sigma = sd), out_scaling = (; mu, sigma = sd), target = :dQ)
        q_star = randn(Xoshiro(11), nq, 14) .+ 1.0
        m = RF.StochLSTM(spec, w, sc; spinnup_data = randn(Xoshiro(12), nq, 7) ./ 10, rng = Xoshiro(5),
                         gate = 0.0, stochastic = false)
        for k in 1:14
            RF.get_next_item_timeseries(m, q_star[:, k])
        end
        st2 = RF.LSTMState(spec, T)
        for j in 1:L
            RF.lstm_step!(st2, w, spec, view(m.xwin, :, j); eps = view(m.ewin, :, j))
        end
        @test st2.y ≈ m.state.y rtol = 1e-6
        path = joinpath(mktempdir(), "lp.jld2")
        RF.save_stochlstm(path, spec, w, sc)
        back = RF.load_stochlstm(path)
        @test back.spec.prior === :learned && back.weights.P.Wm == w.P.Wm
        @test_throws ErrorException RF.LSTMSpec(; hist, arch = :vrnn, prior = :learned)   # needs :xy
    end
end
