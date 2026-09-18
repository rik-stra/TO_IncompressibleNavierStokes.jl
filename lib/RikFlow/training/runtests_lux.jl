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
        @test hd.best_epoch == hc.best_epoch

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
            tr = RF.segment_indices(view(steps, 1:ntrain); L, burn, stride)
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
        @test length(h.train) == 8
        @test all(isfinite, h.train)
    end
end
