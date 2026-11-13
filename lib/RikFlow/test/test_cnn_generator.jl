# Step 5 (2026-11-06; paper Eq. stochastic net, closures.tex sec:lever-stoch; card analysis/cards/step5.md): the
# stochastic skip network, `y = g(x, ε) + Ws x + L ε`, on the network path (`emission = :generator`, V98). Pinned:
#
#   * the spec: `:generator` for `arch = :cnn` only; the dense layers read N_Q more inputs (the deviates);
#     the weights' L = LR lower triangular, bd = Wd = 0;
#   * the deployed forward pass: the deviates join [conv features; q*] and L ε is added, nothing after;
#   * N_Q standard normals per predicted step from the member's stream, before the forward pass (one scalar
#     randn each, LinReg + MVG's count and order), none in the warm-up, also on gated steps;
#   * 🔴 with the output layer at zero and L the Cholesky factor of a LinReg's Σ (in the network's target
#     units), the Float64 closure IS that LinReg + MVG -- the real `LinReg` path, same member seed, step by
#     step (synthetic file, and the archived LinReg1 where present);
#   * zero noise weights and L = 0: a deterministic closure (the seed does not matter);
#   * no noise option acts after the generator;
#   * `crps_fair_samples` (the training loss) is `crps_ensemble(fair = true)`, summed over the QoIs and
#     averaged over the steps.

@testmodule GenFix begin
    using LinearAlgebra
    using Statistics
    using Random
    using JLD2
    using Distributions

    const RF = normpath(joinpath(@__DIR__, ".."))
    const SRC = joinpath(RF, "src")
    const LRS = joinpath(RF, "exp_square_HIT", "output", "TO_LRS")

    adapt(::Any, x) = x                                  # as V73: Adapt is identity on Array
    adapt(::Any) = identity
    include(joinpath(SRC, "ts_scaling.jl"))
    include(joinpath(SRC, "ts_history.jl"))
    include(joinpath(SRC, "ts_scale.jl"))
    include(joinpath(SRC, "ts_noise.jl"))
    include(joinpath(SRC, "time_series_methods.jl"))
    include(joinpath(SRC, "ts_lstm.jl"))
    include(joinpath(SRC, "ts_lstm_online.jl"))
    include(joinpath(SRC, "ts_score.jl"))
    include(joinpath(SRC, "ts_lstm_io.jl"))           # V101: the save/load round trip

    const C, K, D = 5, 2, 7

    hist(nq, h) = HistorySpec(; h, n_qoi = nq, hist_var = :q_star_q, include_predictor = true)
    nin(nq, h) = nfeatures(hist(nq, h)) + h * nq

    spec(nq = 3, h = 4; kw...) = LSTMSpec(; hist = hist(nq, h), n_hidden = D, n_latent = 0, n_encoder = 0,
                                          arch = :cnn, emission = :generator, window = 1, n_in = nin(nq, h),
                                          n_conv = C, kernel = K, kw...)

    "cnn_mean_fit.jl's sequence layout: per lag (oldest first) q*, dQ and the predictor increment."
    function seq_index(nq, h)
        nreg = nfeatures(hist(nq, h))
        lagq(k) = (nq + 2(k - 1) * nq + 1):(nq + (2k - 1) * nq)
        s = zeros(Int, h, 3nq)
        for (t, k) in enumerate(h:-1:1)
            s[t, :] = vcat(collect(lagq(k)) .+ nq, collect(lagq(k)), nreg .+ (((k - 1) * nq + 1):(k * nq)))
        end
        return s
    end

    """
    Weights in the Lux layout `cnn_weights` reads, emission = :generator: `L` the noise skip; `zero_out` the
    output layer at zero (as the step-5 fit starts); `zero_noise` the deviates' columns of the first dense layer
    at zero; `Ws` the skip (`spec(...; skip = true)`).
    """
    function weights(sp; L, seed = 1, Ws = nothing, T = Float64, zero_out = false, zero_noise = false)
        nq, h = n_output(sp), sp.hist.h
        rng = Xoshiro(seed)
        o = (; weight = randn(rng, nq, D) ./ 3, bias = randn(rng, nq) ./ 3)
        zero_out && (o = (; weight = zero(o.weight), bias = zero(o.bias)))
        W1 = randn(rng, D, n_cell_input(sp)) ./ 3
        zero_noise && (W1[:, (end - nq + 1):end] .= 0)
        net = (; layer_1 = (; layer_1 = (; layer_1 = (; weight = randn(rng, K, 3nq, C) ./ 3, bias = randn(rng, C) ./ 3),
                                           layer_2 = (; weight = randn(rng, K, C, C) ./ 3, bias = randn(rng, C) ./ 3),
                                           layer_3 = (;)), layer_2 = (;), layer_3 = (;)),
               layer_2 = (; weight = W1, bias = randn(rng, D) ./ 3),
               layer_3 = (; weight = randn(rng, D, D) ./ 3, bias = randn(rng, D) ./ 3),
               layer_4 = o)
        return cnn_weights(net, sp; seq_index = seq_index(nq, h), q_index = 1:nq, bd = zeros(nq), LR = L, Ws, T)
    end

    "The network's scalings: an input map whose regressor block is not the identity, a target mean `mo`."
    function scaling(nq, h; mo = zeros(nq), so = fill(0.1, nq), seed = 4, extra...)
        nreg = nfeatures(hist(nq, h))
        rng = Xoshiro(seed)
        P0 = Matrix{Float64}(I, nreg, nreg) .+ 0.1 .* randn(rng, nreg, nreg)
        P = vcat(P0, randn(rng, h * nq, nreg) ./ 4)
        sc = (; in_scaling = (; mu = reshape(collect(range(0.1, 0.3; length = nq)), nq, 1), sigma = fill(2.0, nq, 1)),
              out_scaling = (; mu = reshape(mo, nq, 1), sigma = reshape(so, nq, 1)), target = :dQ, input_map = P)
        return merge(sc, (; extra...))
    end

    "A synthetic `:q_star_q` LinReg file (as V73's), with a nonzero MVG mean and full covariance."
    function synth_linreg(path; n = 3, h = 4, seed = 11)
        rng = Xoshiro(seed)
        c = hcat(Matrix{Float64}(I, n, n), zeros(n, n * 2h + 1)) .+ 0.05 .* randn(rng, n, n * (2h + 1) + 1)
        A = randn(rng, n, n)
        stoch_distr = MvNormal(0.01 .* randn(rng, n), 0.01 .* (A * A' ./ n + I))
        sc = (; mu = reshape(1 .+ rand(rng, n), n, 1), sigma = reshape(0.2 .+ rand(rng, n), n, 1))
        jldopen(path, "w") do f
            f["c"] = c; f["stoch_distr"] = stoch_distr
            f["scaling"] = (; in_scaling = sc, out_scaling = sc)
            f["hist_var"] = :q_star_q; f["hist_len"] = h; f["include_predictor"] = true
            f["fitted_qois"] = collect(1:n)
        end
        return path
    end

    """
    The generator at its start for a LinReg file: Ws its map on the network's mapped input, the network's target
    centred at the LinReg's MVG mean (`mo = σ_out ⊙ μ`), `L = diag(σ_out / so) chol(Σ)`, output layer at zero.
    """
    function generator_of(file; seed = 1, so = nothing, zero_noise = false)
        c, lsc, h, d = JLD2.load(file, "c", "scaling", "hist_len", "stoch_distr")
        nq = size(c, 1)
        sgo = vec(lsc.out_scaling.sigma)
        so = so === nothing ? 0.5 .* sgo : so
        sc = scaling(nq, h; mo = sgo .* mean(d), so)
        sp = spec(nq, h; skip = true)
        Ws = linreg_skip_map(Matrix{Float64}(c), lsc, sc, hist(nq, h))
        L = Diagonal(sgo ./ so) * Matrix(cholesky(Symmetric(Matrix(cov(d)))).L)
        return sp, weights(sp; L, Ws, seed, zero_out = true, zero_noise), sc
    end

    "q* columns about the scaling mean, and a warm-up of recorded corrections."
    function path(mu, N; nwarm = 10, seed = 3)
        Q = mu .* (1 .+ 0.05 .* randn(Xoshiro(seed), length(mu), N))
        return Q, 0.001 .* mu .* randn(Xoshiro(seed + 1), length(mu), nwarm)
    end

    drive(m, Q) = reduce(hcat, [copy(get_next_item_timeseries(m, Q[:, t])) for t in axes(Q, 2)])

    "A step-4 closure (emission :constant, the linear skip Ws) and the generator that carries its network frozen."
    function bpair(nq = 3, h = 4; L = Matrix(cholesky([1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1.0]).L) ./ 4, seed_k = 21)
        Ws = randn(Xoshiro(3), nq, nin(nq, h)) ./ 5
        spk = spec(nq, h; emission = :constant, skip = true)
        wk = weights(spk; L = Matrix(1.0I, nq, nq), Ws, seed = seed_k)
        spg = spec(nq, h; skip = true)
        wg0 = weights(spg; L, Ws, zero_out = true)
        return (; spk, wk, spg, wg = with_mean_net(wg0, mean_net_from(wk, spk)), wg0, L, Ws)
    end
end

@testitem "V98 emission = :generator: the CNN only, N_Q more inputs to the dense layers, L lower, bd = Wd = 0" default_imports = false setup = [GenFix] begin
    using Test, LinearAlgebra
    using .GenFix: spec, weights, n_cell_input, check_shapes, LSTMSpec, hist, is_generator
    sp = spec()
    @test is_generator(sp)
    @test n_cell_input(sp) == (4 - 2) * 5 + 3 + 3                     # positions x channels + q* + ε
    @test n_cell_input(spec(; emission = :constant)) == (4 - 2) * 5 + 3
    @test_throws ErrorException LSTMSpec(; hist = hist(3, 4), n_hidden = 4, n_latent = 0, n_encoder = 0,
                                         arch = :dense, window = 1, emission = :generator)
    L = Matrix(cholesky([1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1.0]).L)
    @test check_shapes(weights(sp; L), sp)
    @test_throws ErrorException weights(sp; L = Matrix(L'))           # upper: refused
    w = weights(sp; L)
    w.bd .= 0.1
    @test_throws ErrorException check_shapes(w, sp)                    # an emission scale on top: refused
end

@testitem "V98 the deployed generator: [features; q*; ε] into the dense layers, + Ws x + L ε, nothing after" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: spec, weights, scaling, path, cnn_features, cnn_window!, LSTMState, StochLSTM,
                   get_next_item_timeseries, nin
    nq, h = 3, 4
    L = Matrix(cholesky([1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1.0]).L) ./ 4
    Ws = randn(Xoshiro(3), nq, nin(nq, h)) ./ 5
    sp = spec(nq, h; skip = true)
    for T in (Float32, Float64)
        w = weights(sp; L, Ws, T)
        x = randn(Xoshiro(8), nin(nq, h))
        e = randn(Xoshiro(9), nq)
        st = LSTMState(sp, T)
        cnn_window!(st, w, sp, reshape(x, :, 1); eps = e)
        F = vcat(cnn_features(w, sp, x), T.(e))
        h1 = tanh.(w.Wx * F .+ w.b[1:7]); h2 = tanh.(w.Wh * h1 .+ w.b[8:14])
        ref = w.V1 * h2 .+ w.cdec .+ T.(Ws) * T.(x) .+ T.(L) * T.(e)
        @test st.y ≈ ref rtol = (T === Float32 ? 1e-5 : 1e-12)
        @test all(iszero, st.logd)
        @test_throws ErrorException cnn_window!(st, w, sp, reshape(x, :, 1))     # the generator needs ε
    end
    # the closure's step: the same, with the ε it drew from its own stream
    w = weights(sp; L, Ws)
    Q, warm = path(fill(3.0, nq), 16; nwarm = 6)
    m = StochLSTM(sp, w, scaling(nq, h); spinnup_data = warm, rng = Xoshiro(5), gate = 0.0)
    ref_rng = Xoshiro(5)
    for k in 1:16
        dQ = get_next_item_timeseries(m, Q[:, k])
        k <= 6 && (@test dQ == warm[:, k]; continue)
        e = [randn(ref_rng) for _ in 1:nq]
        st = LSTMState(sp, Float64)
        cnn_window!(st, w, sp, m.xwin[:, 1:1]; eps = e)
        @test m.state.y ≈ st.y rtol = 1e-12
        @test dQ ≈ st.y .* 0.1 rtol = 1e-12                                       # out_scaling: so = 0.1, mo = 0
    end
end

@testitem "V98 N_Q normals per predicted step, before the forward pass; none in the warm-up; gated steps too" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: spec, weights, scaling, path, StochLSTM, drive
    nq, h = 3, 4
    L = Matrix(1.0I, nq, nq) ./ 3
    sp = spec(nq, h)
    for gate in (0.0, 100.0)                                       # 100: every step is gated
        Q, warm = path(fill(3.0, nq), 16; nwarm = 6)
        m = StochLSTM(sp, weights(sp; L), scaling(nq, h); spinnup_data = warm, rng = Xoshiro(9), gate)
        out = drive(m, Q)
        @test out[:, 1:6] == warm
        gate > 0 && @test all(iszero, out[:, 7:end])
        ref = Xoshiro(9)
        for _ in 1:((16 - 6) * nq)
            randn(ref)
        end
        @test rand(m.rng) == rand(ref)
    end
    # stochastic = false: ε = 0, nothing drawn
    Q, warm = path(fill(3.0, nq), 16; nwarm = 6)
    m = StochLSTM(sp, weights(sp; L), scaling(nq, h); spinnup_data = warm, rng = Xoshiro(9), gate = 0.0, stochastic = false)
    drive(m, Q)
    @test rand(m.rng) == rand(Xoshiro(9))
end

@testitem "V98 at its start the Float64 generator IS LinReg + MVG: the real LinReg path, same seed, step by step" default_imports = false setup = [GenFix] begin
    using Test, Random, Statistics
    using .GenFix: generator_of, synth_linreg, path, drive, StochLSTM, LinReg, LRS
    function compare(file, mu; N = 120, nwarm = 10, seeds = (1, 7))
        sp, w, sc = generator_of(file)
        h = sp.hist.h
        errs = Float64[]
        for seed in seeds
            Q, warm = path(mu, N; nwarm)
            g = StochLSTM(sp, w, sc; spinnup_data = warm, rng = Xoshiro(seed), gate = 0.0)
            l = LinReg(file, Xoshiro(seed), Array; q_hist = zeros(2length(mu), h), spinnup_data = copy(warm), gate = 0.0)
            A, B = drive(g, Q), drive(l, Q)
            @test A[:, 1:nwarm] == B[:, 1:nwarm]
            # in units of the MVG's sd (raw): LinReg rounds its scaled level to Float32 (#77)
            sd = vec(sqrt.(var(B[:, (nwarm + 1):end]; dims = 2)))
            push!(errs, maximum(abs.(A .- B) ./ sd))
            @test rand(g.rng) === rand(l.rng)                    # the member streams at the same point
        end
        return maximum(errs)
    end
    mktempdir() do d
        f = synth_linreg(joinpath(d, "LinReg.jld2"))
        mu = vec(GenFix.JLD2.load(f, "scaling").in_scaling.mu)
        e = compare(f, mu)
        @info "V98 generator vs LinReg + MVG (synthetic), max |ΔdQ| / sd" e
        @test e < 1e-5
    end
    f = joinpath(LRS, "LinReg1", "LinReg.jld2")
    if isfile(f)
        mu = vec(GenFix.JLD2.load(f, "scaling").in_scaling.mu)
        e = compare(f, mu; N = 60)                                # an open loop drifts: keep it short
        @info "V98 generator vs LinReg1 + MVG (archived file), max |ΔdQ| / sd" e
        @test e < 1e-4
    end
end

@testitem "V98 zero noise weights and L = 0: a deterministic closure" default_imports = false setup = [GenFix] begin
    using Test, Random
    using .GenFix: spec, weights, scaling, path, StochLSTM, drive
    nq, h = 3, 4
    sp = spec(nq, h)
    w = weights(sp; L = zeros(nq, nq), zero_noise = true)
    Q, warm = path(fill(3.0, nq), 16; nwarm = 6)
    A = drive(StochLSTM(sp, w, scaling(nq, h); spinnup_data = warm, rng = Xoshiro(1), gate = 0.0), Q)
    B = drive(StochLSTM(sp, w, scaling(nq, h); spinnup_data = warm, rng = Xoshiro(2), gate = 0.0), Q)
    @test A == B
    # and with noise weights, the seed matters
    w2 = weights(sp; L = zeros(nq, nq))
    A2 = drive(StochLSTM(sp, w2, scaling(nq, h); spinnup_data = warm, rng = Xoshiro(1), gate = 0.0), Q)
    B2 = drive(StochLSTM(sp, w2, scaling(nq, h); spinnup_data = warm, rng = Xoshiro(2), gate = 0.0), Q)
    @test A2[:, 7:end] != B2[:, 7:end]
end

@testitem "V98 no noise option acts after the generator" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: spec, weights, scaling, path, StochLSTM
    nq, h = 3, 4
    sp = spec(nq, h)
    w = weights(sp; L = Matrix(1.0I, nq, nq))
    _, warm = path(fill(3.0, nq), 16; nwarm = 6)
    for extra in ((; ar_phi = [0.5 0.5 0.5]), (; eta_ar = [0.5, 0.5, 0.5]), (; noise_scale = 1.2),
                  (; emission_scale = ones(3)), (; dq_offset = zeros(3)),
                  (; scale_beta = ones(3), scale_qref = ones(3), scale_qclip = [0.1 9.0; 0.1 9.0; 0.1 9.0]))
        @test_throws ErrorException StochLSTM(sp, w, scaling(nq, h; extra...); spinnup_data = warm, rng = Xoshiro(1))
    end
    @test StochLSTM(sp, w, scaling(nq, h); spinnup_data = warm, rng = Xoshiro(1)) isa StochLSTM
end

@testitem "V98 crps_fair_samples is crps_ensemble(fair = true), summed over the QoIs, averaged over the steps" default_imports = false setup = [GenFix] begin
    using Test, Random
    using .GenFix: crps_fair_samples, crps_ensemble
    rng = Xoshiro(2)
    for M in (2, 4, 7)
        X, Y = randn(rng, 3, 11, M), randn(rng, 3, 11)
        ref = sum(crps_ensemble(X[i, b, :], Y[i, b]; fair = true) for i in 1:3, b in 1:11) / 11
        @test crps_fair_samples(X, Y) ≈ ref rtol = 1e-12
        @test crps_fair_samples(Float32.(X), Float32.(Y)) ≈ ref rtol = 1e-5
    end
    @test_throws ErrorException crps_fair_samples(randn(3, 5, 1), randn(3, 5))
    @test_throws ErrorException crps_fair_samples(randn(3, 5, 4), randn(3, 4))
end

@testitem "V98 energy_fair_samples: the fair energy score by explicit sums; N_Q = 1 is the fair CRPS; it sees the dependence" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra, Statistics
    using .GenFix: energy_fair_samples, crps_fair_samples
    rng = Xoshiro(3)
    for M in (2, 4, 7)
        X, Y = randn(rng, 3, 9, M), randn(rng, 3, 9)
        ref = sum(sum(norm(X[:, b, m] - Y[:, b]) for m in 1:M) / M -
                  sum(norm(X[:, b, m] - X[:, b, k]) for m in 1:M, k in 1:M) / (2M * (M - 1)) for b in 1:9) / 9
        @test energy_fair_samples(X, Y) ≈ ref rtol = 1e-12
        X1, Y1 = randn(rng, 1, 9, M), randn(rng, 1, 9)
        @test energy_fair_samples(X1, Y1) ≈ crps_fair_samples(X1, Y1) rtol = 1e-12
    end
    # truth with correlated components: samples with the right correlation score better than samples with the
    # same marginals and the opposite correlation (the summed marginal CRPS cannot tell them apart)
    C = [1 0.9; 0.9 1.0]; Cw = [1 -0.9; -0.9 1.0]
    L, Lw = cholesky(C).L, cholesky(Cw).L
    Y = L * randn(Xoshiro(4), 2, 4000)
    E = randn(Xoshiro(5), 2, 4000 * 8)
    Xr, Xw = reshape(L * E, 2, 4000, 8), reshape(Lw * E, 2, 4000, 8)
    @test energy_fair_samples(Xr, Y) < 0.95 * energy_fair_samples(Xw, Y)
    @test isapprox(crps_fair_samples(Xr, Y), crps_fair_samples(Xw, Y); rtol = 0.02)
    @test_throws ErrorException energy_fair_samples(randn(3, 5, 1), randn(3, 5))
end

# V101 -- step 5 variant B (2026-11-11, card analysis/cards/step5_skips.md; Rik: "A2, both skips; and B"): a generator
# that carries a step-4 closure's FROZEN CNN mean, `y = g(x, ε) + m(x) + Ws x + L ε` (`Cv.mean_net`). Pinned: at its
# start (output layer at zero) it IS the step-4 closure's deployed mean plus L ε, in the forward pass and step by step
# online with the member's own ε; a closure without `mean_net` is untouched; the refusals; a save/load round trip.

@testitem "V101 a generator carrying a frozen CNN mean: at its start = the step-4 closure's mean + L ε" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: cnn_window!, LSTMState, cnn_mean_net, check_shapes
    b = GenFix.bpair()
    @test cnn_mean_net(b.wg0) === nothing
    @test cnn_mean_net(b.wg) !== nothing
    @test check_shapes(b.wg, b.spg)
    for seed in 1:5
        x = randn(Xoshiro(seed), size(b.Ws, 2))
        e = randn(Xoshiro(100 + seed), 3)
        sk, sg = LSTMState(b.spk, Float64), LSTMState(b.spg, Float64)
        cnn_window!(sk, b.wk, b.spk, reshape(x, :, 1))
        cnn_window!(sg, b.wg, b.spg, reshape(x, :, 1); eps = e)
        @test sg.y ≈ sk.y .+ b.L * e rtol = 1e-12 atol = 1e-14
        # without the frozen mean: the network (zero at its start) + Ws x + L ε, as V98
        s0 = LSTMState(b.spg, Float64)
        cnn_window!(s0, b.wg0, b.spg, reshape(x, :, 1); eps = e)
        @test s0.y ≈ b.Ws * x .+ b.L * e rtol = 1e-12
    end
end

@testitem "V101 online: the B generator at its start = the step-4 closure's mean + L ε with the member's own ε" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: scaling, path, StochLSTM, get_next_item_timeseries, cnn_window!, LSTMState
    b = GenFix.bpair()
    nq, h = 3, 4
    mo = [0.01, -0.02, 0.03]
    sc = scaling(nq, h; mo)
    Q, warm = path(fill(3.0, nq), 20; nwarm = 6)
    g = StochLSTM(b.spg, b.wg, sc; spinnup_data = warm, rng = Xoshiro(5), gate = 0.0)
    ref = Xoshiro(5)
    for t in 1:20
        dg = get_next_item_timeseries(g, Q[:, t])
        t <= 6 && (@test dg == warm[:, t]; continue)
        e = [randn(ref) for _ in 1:nq]
        sk = LSTMState(b.spk, Float64)
        cnn_window!(sk, b.wk, b.spk, g.xwin[:, 1:1])           # the step-4 closure's mean on the same window
        @test dg ≈ (sk.y .+ b.L * e) .* 0.1 .+ mo rtol = 1e-10  # so = 0.1
    end
    @test rand(g.rng) == rand(ref)                             # the same N_Q normals per predicted step
end

@testitem "V101 a frozen CNN mean: refused off a generator with a linear skip, or with wrong shapes; round trip" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: check_shapes, with_mean_net, mean_net_from, spec, weights, cnn_mean_net, save_stochlstm,
                   load_stochlstm, scaling, cnn_window!, LSTMState
    b = GenFix.bpair()
    mn = mean_net_from(b.wk, b.spk)
    @test_throws ErrorException check_shapes(with_mean_net(b.wk, mn), b.spk)            # not a generator
    spn = spec(3, 4)                                                                     # a generator without skip
    @test_throws ErrorException check_shapes(with_mean_net(weights(spn; L = b.L), mn), spn)
    @test_throws ErrorException mean_net_from(b.wg0, b.spg)                              # a generator is no mean model
    bad = merge(mn, (; V1 = zeros(3, 2)))
    @test_throws ErrorException check_shapes(with_mean_net(b.wg0, bad), b.spg)
    mktempdir() do d
        f = save_stochlstm(joinpath(d, "g.jld2"), b.spg, b.wg, scaling(3, 4))
        r = load_stochlstm(f)
        @test cnn_mean_net(r.weights) !== nothing
        x, e = randn(Xoshiro(2), size(b.Ws, 2)), randn(Xoshiro(3), 3)
        s1, s2 = LSTMState(b.spg, Float64), LSTMState(r.spec, Float64)
        cnn_window!(s1, b.wg, b.spg, reshape(x, :, 1); eps = e)
        cnn_window!(s2, r.weights, r.spec, reshape(x, :, 1); eps = e)
        @test s1.y == s2.y
    end
end

# V102 -- step 5 reaching k1a (2026-11-13, card analysis/cards/step5_reach_k1a.md): a generator CENTRED on ε (`Cv.centred`),
# y = g(x, ε) - g(x, 0) [+ m(x)] + Ws x + L ε. Pinned: the forward pass by hand; at ε = 0 the output is the skip exactly (with
# a frozen mean: the step-4 closure's mean); online with the member's ε; refused off a generator; a save/load round trip.

@testitem "V102 a generator centred on ε: g(x, ε) - g(x, 0) + Ws x + L ε; at ε = 0 the skip exactly" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: spec, weights, nin, cnn_window!, cnn_features, LSTMState, with_centred, is_centred, check_shapes,
                   with_mean_net, mean_net_from
    nq, h = 3, 4
    L = Matrix(cholesky([1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1.0]).L) ./ 4
    Ws = randn(Xoshiro(3), nq, nin(nq, h)) ./ 5
    sp = spec(nq, h; skip = true)
    w0 = weights(sp; L, Ws)                                    # a trained-looking generator (output layer not zero)
    w = with_centred(w0)
    @test is_centred(w) && !is_centred(w0)
    @test check_shapes(w, sp)
    net(x, e) = (F = vcat(cnn_features(w0, sp, x), e); h1 = tanh.(w0.Wx * F .+ w0.b[1:7]);
                 w0.V1 * tanh.(w0.Wh * h1 .+ w0.b[8:14]) .+ w0.cdec)
    for s in 1:5
        x, e = randn(Xoshiro(s), nin(nq, h)), randn(Xoshiro(50 + s), nq)
        st = LSTMState(sp, Float64)
        cnn_window!(st, w, sp, reshape(x, :, 1); eps = e)
        @test st.y ≈ net(x, e) .- net(x, zeros(nq)) .+ Ws * x .+ L * e rtol = 1e-12
        cnn_window!(st, w, sp, reshape(x, :, 1); eps = zeros(nq))
        @test maximum(abs, st.y .- Ws * x) < 1e-12               # at ε = 0: the skip
    end
    # with a frozen CNN mean (B): at ε = 0 the step-4 closure's deployed mean
    b = GenFix.bpair()
    wc = with_centred(with_mean_net(weights(b.spg; L, Ws = b.Ws), mean_net_from(b.wk, b.spk)))
    @test check_shapes(wc, b.spg)
    x = randn(Xoshiro(9), size(b.Ws, 2))
    sk, sg = LSTMState(b.spk, Float64), LSTMState(b.spg, Float64)
    cnn_window!(sk, b.wk, b.spk, reshape(x, :, 1))
    cnn_window!(sg, wc, b.spg, reshape(x, :, 1); eps = zeros(3))
    @test maximum(abs, sg.y .- sk.y) < 1e-12
end

@testitem "V102 centred: online with the member's ε; refused off a generator; round trip" default_imports = false setup = [GenFix] begin
    using Test, Random, LinearAlgebra
    using .GenFix: spec, weights, nin, scaling, path, StochLSTM, get_next_item_timeseries, cnn_window!, LSTMState,
                   with_centred, is_centred, check_shapes, save_stochlstm, load_stochlstm
    nq, h = 3, 4
    L = Matrix(1.0I, nq, nq) ./ 3
    Ws = randn(Xoshiro(3), nq, nin(nq, h)) ./ 5
    sp = spec(nq, h; skip = true)
    w = with_centred(weights(sp; L, Ws))
    Q, warm = path(fill(3.0, nq), 16; nwarm = 6)
    m = StochLSTM(sp, w, scaling(nq, h); spinnup_data = warm, rng = Xoshiro(5), gate = 0.0)
    ref = Xoshiro(5)
    for k in 1:16
        dQ = get_next_item_timeseries(m, Q[:, k])
        k <= 6 && continue
        e = [randn(ref) for _ in 1:nq]
        st = LSTMState(sp, Float64)
        cnn_window!(st, w, sp, m.xwin[:, 1:1]; eps = e)
        @test dQ ≈ st.y .* 0.1 rtol = 1e-12
    end
    spk = spec(nq, h; emission = :constant, skip = true)
    @test_throws ErrorException check_shapes(with_centred(weights(spk; L = Matrix(1.0I, nq, nq), Ws)), spk)
    mktempdir() do d
        r = load_stochlstm(save_stochlstm(joinpath(d, "c.jld2"), sp, w, scaling(nq, h)))
        @test is_centred(r.weights)
    end
end
