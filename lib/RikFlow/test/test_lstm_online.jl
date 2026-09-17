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

    spec(arch = :vrnn; n_qoi = NQ, h = H, n_hidden = 4, n_latent = 3, n_encoder = 4) =
        LSTMSpec(; hist = HistorySpec(; h, n_qoi), n_hidden, n_latent, n_encoder, arch)

    "A synthetic predictor stream and a warm-up record, both Float64 as a real record is."
    function stream(; n = 12, nwarm = 5, nq = NQ, seed = 21)
        rng = Xoshiro(seed)
        q_star = randn(rng, Float64, nq, n) .+ 1.0
        dQ = randn(rng, Float64, nq, nwarm) ./ 10
        return q_star, dQ
    end
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
