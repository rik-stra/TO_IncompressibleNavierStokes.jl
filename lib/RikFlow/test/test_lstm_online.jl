# M4 deployed -- `StochLSTM` as `to_sgs_term` sees it. V44, plus the warm-up and ordering
# obligations M4 inherits from V2 and V38.

@testmodule OnlineFix begin
    using LinearAlgebra
    using Random

    const SRC = normpath(joinpath(@__DIR__, "..", "src"))
    include(joinpath(SRC, "ts_scaling.jl"))
    include(joinpath(SRC, "ts_history.jl"))
    include(joinpath(SRC, "ts_lstm.jl"))
    include(joinpath(SRC, "ts_lstm_online.jl"))

    const NQ = 3
    const H = 2

    "A scaling pair shaped the way `fit_scaling` produces one: `mu`/`sigma` are `N_Q x 1`."
    function scaling(; nq = NQ)
        mu = reshape(collect(range(0.1, 0.3; length = nq)), nq, 1)
        sigma = reshape(collect(range(2.0, 3.0; length = nq)), nq, 1)
        return (in_scaling = (; mu, sigma), out_scaling = (; mu, sigma))
    end

    function weights(spec; T = Float32, seed = 5)
        Hh = spec.n_hidden
        nin, nout, nz = n_input(spec), n_output(spec), spec.n_latent
        ncin, nenc = n_cell_input(spec), n_encoder_out(spec)
        rng = Xoshiro(seed)
        return LSTMWeights{T}(
            randn(rng, T, 4Hh, ncin) ./ 5, randn(rng, T, 4Hh, Hh) ./ 5, randn(rng, T, 4Hh) ./ 5,
            (spec.n_encoder > 0 && latent_sampled(spec)) ? randn(rng, T, nenc, nin) ./ 5 : nothing,
            (spec.n_encoder > 0 && latent_sampled(spec)) ? randn(rng, T, nenc) ./ 5 : nothing,
            randn(rng, T, nz, nenc) ./ 5, randn(rng, T, nz, nenc) ./ 5,
            randn(rng, T, nout, Hh) ./ 5,
            latent_to_decoder(spec) ? randn(rng, T, nout, nz) ./ 5 : nothing,
            randn(rng, T, nout) ./ 5,
            randn(rng, T, nout, Hh) ./ 5, randn(rng, T, nout) ./ 5,
            Matrix{T}(I, nout, nout))
    end

    # `emission` defaults to the Gaussian head here and to `:none` in `LSTMSpec`: the head is off
    # in production (Rik, 2026-09-18) but still in the code, so the deployed-path tests that
    # exercise an emission draw have to name it rather than inherit it.
    spec(arch = :vrnn; n_qoi = NQ, h = H, n_hidden = 4, n_latent = 3, n_encoder = 4,
         emission = :state_dependent) =
        LSTMSpec(; hist = HistorySpec(; h, n_qoi), n_hidden, n_latent, n_encoder, arch, emission)

    "A synthetic predictor stream and a warm-up record, both Float64 as a real record is."
    function stream(; n = 12, nwarm = 5, nq = NQ, seed = 21)
        rng = Xoshiro(seed)
        q_star = randn(rng, Float64, nq, n) .+ 1.0
        dQ = randn(rng, Float64, nq, nwarm) ./ 10
        return q_star, dQ
    end

    # --- V61, window mode -----------------------------------------------------------------
    module WindowFix
    using Random
    import ..LSTMSpec, ..HistorySpec, ..NQ, ..weights, ..stream, ..StochLSTM,
           ..get_next_item_timeseries, ..scale_input, ..scale_output, ..scaling, ..LSTMState,
           ..lstm_step!

    const W = 4
    const NWARM = 5
    const N = 14
    const SEED = 77

    # q*-only input, [q*^n; 1], the simple window model
    spec(arch = :vrnn; emission = :none, window = W) =
        LSTMSpec(; hist = HistorySpec(; h = 0, n_qoi = NQ, hist_var = :q_star),
                 n_hidden = 4, n_latent = 3, n_encoder = 0, arch, emission, window)

    "Run the closure over a synthetic stream; returns (closure, outputs, q_star, dQ warm-up)."
    function run(spec; seed = SEED, nwarm = NWARM, n = N)
        w = weights(spec)
        q_star, dQ = stream(; n, nwarm)
        m = StochLSTM(spec, w, scaling(); spinnup_data = dQ, rng = Xoshiro(seed), gate = 0.0)
        got = [get_next_item_timeseries(m, q_star[:, k]) for k in 1:n]
        return m, w, got, q_star
    end

    "The scaled input row [q*; 1] the closure builds, in the model's precision."
    xrow(q_star, k) = Float32.(vcat(vec(scale_input(q_star[:, k], scaling().in_scaling)), 1))

    "Reference: fresh state per prediction, replaying the given per-step draws."
    function reference(spec, w, q_star, eps; nwarm = NWARM, n = N)
        out = Vector{Vector{Float64}}()
        for k in (nwarm + 1):n
            st = LSTMState(spec, Float32)
            for t in (k - W + 1):k
                lstm_step!(st, w, spec, xrow(q_star, t); eps = eps[t])
            end
            qhat = Float64.(vec(scale_output(st.y, scaling().out_scaling)))
            push!(out, qhat .- q_star[:, k])
        end
        return out
    end
    end # WindowFix
end

# ---------------------------------------------------------------------------------------------
# V44 -- the replayed window is returned unconverted and bit-identical
# ---------------------------------------------------------------------------------------------

@testitem "V44 the warm-up replays dQ bit-identically and unconverted" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random

    # 🔴 D6's validation gate is `dQ` bit-identity over the replayed window (claude_memory.md #48).
    # The model is Float32; the record is Float64. A conversion on the way out would pass every
    # `≈` test in this file and fail that gate.
    spec = OnlineFix.spec(:vrnn)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream()
    nwarm = size(dQ, 2)

    m = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling();
                            spinnup_data = dQ, rng = Xoshiro(1))

    for n in 1:nwarm
        out = OnlineFix.get_next_item_timeseries(m, q_star[:, n])
        @test out == dQ[:, n]                      # bit-identical, not approximately
        @test eltype(out) === Float64              # unconverted
    end
    @test !OnlineFix.in_warmup(m)
    @test OnlineFix.nwarm(m) == nwarm

    # and the recurrence really was charged while that happened
    @test m.state.nstep == nwarm
    @test any(!iszero, m.state.h)
end

@testitem "V44 a warm-up shorter than the lag window is refused at construction" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random

    # `LinReg` asserts the same thing. Without it the first prediction is built from a part-zero
    # history and nothing says so.
    spec = OnlineFix.spec(:vrnn; h = 6)
    w = OnlineFix.weights(spec)
    _, dQ = OnlineFix.stream(; nwarm = 3)
    @test_throws ErrorException OnlineFix.StochLSTM(spec, w, OnlineFix.scaling();
                                                    spinnup_data = dQ, rng = Xoshiro(1))
end

# ---------------------------------------------------------------------------------------------
# V42 (deployed) -- the warm-up draws nothing
# ---------------------------------------------------------------------------------------------

@testitem "V42 a member's first sampled value does not depend on the warm-up length" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random

    # 🔑 This is the property that makes a replica seed mean the same thing across closures. It is
    # asserted on the deployed object, not just on `lstm_step!`, because that is where it can be
    # lost -- one stray `rand` in the replay branch would do it.
    spec = OnlineFix.spec(:vrnn)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream(; nwarm = 5)

    m = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling();
                            spinnup_data = dQ, rng = Xoshiro(4242))
    for n in 1:size(dQ, 2)
        OnlineFix.get_next_item_timeseries(m, q_star[:, n])
    end
    after_replay = randn(m.rng)

    @test after_replay == randn(Xoshiro(4242))
end

# ---------------------------------------------------------------------------------------------
# The M4 analogue of V2 -- online regressor == batch regressor
# ---------------------------------------------------------------------------------------------

@testitem "V40 the online closure reproduces a batch forward pass on the same stream" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random

    # Run with `arch = :lstm` and `stochastic = false` so the whole path is deterministic, then
    # rebuild it from `build_history` + a bare `lstm_step!` loop. Any disagreement is an off-by-one
    # in the lag window or a scaling applied on the wrong side -- exactly what V2 catches on the
    # linear cells, and it has to hold here too before any D6 number means anything.
    spec = OnlineFix.spec(:lstm)
    w = OnlineFix.weights(spec)
    sc = OnlineFix.scaling()
    q_star, dQ = OnlineFix.stream(; n = 10, nwarm = 6)
    nwarm = size(dQ, 2)
    T = Float32

    m = OnlineFix.StochLSTM(spec, w, sc; spinnup_data = dQ, rng = Xoshiro(1),
                            stochastic = false)
    got = [OnlineFix.get_next_item_timeseries(m, q_star[:, n]) for n in 1:(nwarm + 1)]

    # --- rebuild it ---------------------------------------------------------------------------
    # the warm-up determines q exactly: q^n = q*^n + dQ^n over the replayed window
    q = hcat([q_star[:, n] .+ dQ[:, n] for n in 1:nwarm]...)

    # `build_history` wants `q` with a t = 0 column and `q_star` one column shorter. The closure
    # starts from an empty buffer, i.e. from an implicit zero t = 0 column in scaled units.
    qs = OnlineFix.scale_input(q, sc.in_scaling)
    qss = OnlineFix.scale_input(q_star, sc.in_scaling)
    Xb, _, steps = OnlineFix.build_history(spec.hist,
                                           T.(qss[:, 1:nwarm]),
                                           T.(hcat(zeros(size(qs, 1)), qs)))

    st = OnlineFix.LSTMState(spec, T)
    # replay the same number of leading steps the closure took before `steps[1]`
    buf = OnlineFix.HistoryBuffer(spec.hist, T)
    for n in 1:(nwarm + 1)
        x = OnlineFix.inputvec(buf, T.(vec(qss[:, n])))
        OnlineFix.lstm_step!(st, w, spec, x; sample_latent = false)
        if n <= nwarm
            OnlineFix.push!(buf, T.(vec(qs[:, n])), T.(vec(qss[:, n])))
        end
    end
    qhat = vec(OnlineFix.scale_output(st.y, sc.out_scaling))
    expected_last = qhat .- q_star[:, nwarm + 1]

    @test got[nwarm + 1] ≈ expected_last
    @test length(steps) > 0     # the batch builder saw the same stream shape
end

# ---------------------------------------------------------------------------------------------
# The gate, and the deterministic mode
# ---------------------------------------------------------------------------------------------

@testitem "V44 the turbulence gate zeroes the whole correction" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random

    # M4 sits in the q*-consuming path and inherits `LinReg`'s gate rather than introducing a third
    # convention. `any(...)` and a whole-vector zero are correct BY INTENT -- the question the gate
    # asks is "is the flow turbulent yet", over all bands at once.
    spec = OnlineFix.spec(:vrnn)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream(; n = 8, nwarm = 4)

    m = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling();
                            spinnup_data = dQ, rng = Xoshiro(9), gate = 1e-2)
    for n in 1:size(dQ, 2)
        OnlineFix.get_next_item_timeseries(m, q_star[:, n])
    end

    laminar = copy(q_star[:, 5])
    laminar[2] = 1e-6                       # one band below the gate is enough
    @test all(iszero, OnlineFix.get_next_item_timeseries(m, laminar))

    # with the gate off, the same input gives a nonzero correction
    m2 = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling();
                             spinnup_data = dQ, rng = Xoshiro(9), gate = 0.0)
    for n in 1:size(dQ, 2)
        OnlineFix.get_next_item_timeseries(m2, q_star[:, n])
    end
    @test any(!iszero, OnlineFix.get_next_item_timeseries(m2, laminar))
end

@testitem "V44 stochastic = false returns the emission mean" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random

    spec = OnlineFix.spec(:vrnn)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream(; n = 8, nwarm = 4)

    function run(stochastic, seed)
        m = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling();
                                spinnup_data = dQ, rng = Xoshiro(seed), stochastic)
        for n in 1:(size(dQ, 2) + 1)
            OnlineFix.get_next_item_timeseries(m, q_star[:, n])
        end
        return m
    end

    # the mean path is seed-independent up to the latent draw only; two different seeds give
    # different draws when sampling and the same emission offset when not
    a = run(false, 1)
    b = run(false, 1)
    @test a.scratch == b.scratch

    c = run(true, 1)
    @test c.scratch != a.scratch      # an emission draw really was added
end

# ---------------------------------------------------------------------------------------------
# V61 -- window mode: reset + replay of the last W inputs, latent draws tied to the physical step
# ---------------------------------------------------------------------------------------------

@testitem "V61 window mode: each prediction is a reset + replay with the step's own draw" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random
    WF = OnlineFix.WindowFix

    spec = WF.spec(:vrnn)
    m, w, got, q_star = WF.run(spec)
    nz = spec.n_latent

    # the draws the closure must have made: none during the warm-up (eps = 0 there), then
    # n_latent scalar draws per predicted step, in step order
    rng = Xoshiro(WF.SEED)
    eps = [zeros(Float32, nz) for _ in 1:WF.N]
    for k in (WF.NWARM + 1):WF.N
        eps[k] = [Float32(randn(rng)) for _ in 1:nz]
    end
    ref = WF.reference(spec, w, q_star, eps)
    for (j, k) in enumerate((WF.NWARM + 1):WF.N)
        @test got[k] ≈ ref[j] rtol = 1e-6
    end

    # 🔑 the tying itself, read off the closure: the stored draws ARE the last W steps' draws
    for (j, t) in enumerate((WF.N - WF.W + 1):WF.N)
        @test m.ewin[:, j] == eps[t]
    end

    # 🔴 positive control: re-drawing every step's latent in every window (the bug this mode must
    # not have) gives different predictions, so the agreement above is not vacuous
    rng2 = Xoshiro(WF.SEED)
    redrawn = [[Float32(randn(rng2)) for _ in 1:nz] for _ in 1:WF.N]
    alt = WF.reference(spec, w, q_star, redrawn)
    @test maximum(maximum(abs, got[k] .- alt[j]) for (j, k) in enumerate((WF.NWARM + 1):WF.N)) > 1e-3
end

@testitem "V61 window mode: the output depends on exactly the last W inputs" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random
    WF = OnlineFix.WindowFix

    # deterministic path (`:lstm` + emission mean), so only the inputs matter
    spec = WF.spec(:lstm; emission = :constant)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream(; n = WF.N, nwarm = WF.NWARM)
    function last_out(qs)
        m = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling(); spinnup_data = dQ,
                                rng = Xoshiro(1), gate = 0.0, stochastic = false)
        return [OnlineFix.get_next_item_timeseries(m, qs[:, k]) for k in 1:WF.N][end]
    end
    base = last_out(q_star)

    # changing an input OUTSIDE the last window changes nothing (the state was reset) ...
    q1 = copy(q_star); q1[:, WF.N - WF.W] .+= 0.5
    @test last_out(q1) == base
    # ... and changing the OLDEST input inside it does (the replay really covers W steps)
    q2 = copy(q_star); q2[:, WF.N - WF.W + 1] .+= 0.5
    @test maximum(abs, last_out(q2) .- base) > 1e-4
end

@testitem "V61 window mode: the warm-up draws nothing and must fill the window" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random
    WF = OnlineFix.WindowFix

    spec = WF.spec(:vrnn)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream(; n = WF.N, nwarm = WF.NWARM)
    m = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling(); spinnup_data = dQ,
                            rng = Xoshiro(4242))
    replayed = [OnlineFix.get_next_item_timeseries(m, q_star[:, k]) for k in 1:WF.NWARM]
    @test randn(m.rng) == randn(Xoshiro(4242))           # V38: the replay is RNG-free
    @test all(replayed[k] == dQ[:, k] for k in 1:WF.NWARM) # and returns the record unconverted

    # a warm-up shorter than W - 1 would make the first prediction see a partial window
    short = dQ[:, 1:(WF.W - 2)]
    @test_throws ErrorException OnlineFix.StochLSTM(spec, w, OnlineFix.scaling();
                                                    spinnup_data = short, rng = Xoshiro(1))
    @test OnlineFix.StochLSTM(spec, w, OnlineFix.scaling(); spinnup_data = dQ[:, 1:(WF.W - 1)],
                              rng = Xoshiro(1)) isa OnlineFix.StochLSTM
end

@testitem "V63 the input_map is applied to every regressor the closure builds" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random
    using LinearAlgebra
    WF = OnlineFix.WindowFix

    # `scaling.input_map = P` must turn [q*_n; q*_{n-1}; 1] into P * that, in the window too
    spec = OnlineFix.LSTMSpec(; hist = OnlineFix.HistorySpec(; h = 1, n_qoi = OnlineFix.NQ,
                                                           hist_var = :q_star),
                              n_hidden = 4, n_latent = 3, n_encoder = 0, arch = :lstm,
                              emission = :constant, window = WF.W)
    w = OnlineFix.weights(spec)
    nin = OnlineFix.n_input(spec)
    P = Matrix{Float64}(I, nin, nin) .+ 0.3 .* randn(Xoshiro(3), nin, nin)
    sc = merge(OnlineFix.scaling(), (; input_map = P))
    q_star, dQ = OnlineFix.stream(; n = WF.N, nwarm = WF.NWARM)
    m = OnlineFix.StochLSTM(spec, w, sc; spinnup_data = dQ, rng = Xoshiro(1), gate = 0.0,
                            stochastic = false)
    for k in 1:WF.N
        OnlineFix.get_next_item_timeseries(m, q_star[:, k])
    end
    s(k) = vec(OnlineFix.scale_input(q_star[:, k], sc.in_scaling))
    for (j, k) in enumerate((WF.N - WF.W + 1):WF.N)
        @test m.xwin[:, j] ≈ Float32.(P * vcat(s(k), s(k - 1), 1.0)) rtol = 1e-5
    end
    # positive control: without the map the window holds the raw rows, which differ
    m0 = OnlineFix.StochLSTM(spec, w, OnlineFix.scaling(); spinnup_data = dQ, rng = Xoshiro(1),
                             gate = 0.0, stochastic = false)
    for k in 1:WF.N
        OnlineFix.get_next_item_timeseries(m0, q_star[:, k])
    end
    @test maximum(abs, m0.xwin .- m.xwin) > 1e-2
end

@testitem "V67 noise_scale scales every draw (window mode)" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random
    WF = OnlineFix.WindowFix

    spec = WF.spec(:vrnn; emission = :constant)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream(; n = WF.N, nwarm = WF.NWARM)
    function run(sc, seed)
        m = OnlineFix.StochLSTM(spec, w, sc; spinnup_data = dQ, rng = Xoshiro(seed), gate = 0.0)
        return [OnlineFix.get_next_item_timeseries(m, q_star[:, k]) for k in 1:WF.N][end]
    end
    base = OnlineFix.scaling()
    # scale 0: no noise at all, so the seed no longer matters
    s0 = merge(base, (; noise_scale = 0.0))
    @test run(s0, 1) == run(s0, 2)
    # scale 1 is the unscaled closure, bit for bit
    @test run(merge(base, (; noise_scale = 1.0)), 7) == run(base, 7)
    # positive control: the unscaled closure does depend on the seed
    @test run(base, 1) != run(base, 2)
end

@testitem "V68 dq_offset: constant, and state-proportional with offset_ref" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random
    WF = OnlineFix.WindowFix

    spec = WF.spec(:vrnn; emission = :constant)
    w = OnlineFix.weights(spec)
    q_star, dQ = OnlineFix.stream(; n = WF.N, nwarm = WF.NWARM)
    function run(sc)
        m = OnlineFix.StochLSTM(spec, w, sc; spinnup_data = dQ, rng = Xoshiro(3), gate = 0.0)
        return [OnlineFix.get_next_item_timeseries(m, q_star[:, k]) for k in 1:WF.N]
    end
    base = OnlineFix.scaling()
    c = [0.1, -0.2, 0.3]
    a = run(base)
    b = run(merge(base, (; dq_offset = c)))
    ref = [2.0, 1.0, 4.0]
    r = run(merge(base, (; dq_offset = c, offset_ref = ref)))
    for k in (WF.NWARM + 1):WF.N
        @test b[k] ≈ a[k] .+ c
        @test r[k] ≈ a[k] .+ c .* q_star[:, k] ./ ref
    end
    # the replayed warm-up is never offset
    @test all(b[k] == dQ[:, k] for k in 1:WF.NWARM)
end

@testitem "V69 eta_ar colours the emission noise: AR(1), same marginal variance" default_imports = false setup = [OnlineFix] begin
    using Test
    using Random
    using Statistics
    WF = OnlineFix.WindowFix

    # `:lstm` + constant head: all the noise is emission noise, so its ACF is the AR(1)'s
    spec = WF.spec(:lstm; emission = :constant)
    w = OnlineFix.weights(spec)
    n = 6000
    q_star = 1.0 .+ 0.1 .* randn(Xoshiro(2), OnlineFix.NQ, n)
    dQ = zeros(OnlineFix.NQ, WF.NWARM)
    a = [0.0, 0.5, 0.9]
    function noise(sc, seed)
        m = OnlineFix.StochLSTM(spec, w, sc; spinnup_data = dQ, rng = Xoshiro(seed), gate = 0.0)
        mm = OnlineFix.StochLSTM(spec, w, sc; spinnup_data = dQ, rng = Xoshiro(seed), gate = 0.0,
                                 stochastic = false)
        d = [OnlineFix.get_next_item_timeseries(m, q_star[:, k]) for k in 1:n]
        d0 = [OnlineFix.get_next_item_timeseries(mm, q_star[:, k]) for k in 1:n]
        return reduce(hcat, d[(WF.NWARM + 1):end] .- d0[(WF.NWARM + 1):end])
    end
    base = OnlineFix.scaling()
    E = noise(merge(base, (; eta_ar = a)), 5)
    E0 = noise(base, 5)
    lag1(x) = (y = x .- mean(x); sum(y[1:(end - 1)] .* y[2:end]) / sum(abs2, y))
    for k in 1:3
        @test abs(lag1(E[k, :]) - a[k]) < 0.05
        @test std(E[k, :]) / std(E0[k, :]) ≈ 1 atol = 0.08
    end
    @test abs(lag1(E0[3, :])) < 0.05            # without it: white
end
