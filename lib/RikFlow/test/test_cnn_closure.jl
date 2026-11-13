# Step 3a on the network path (2026-10-23): `arch = :cnn` in `LSTMSpec` / `StochLSTM` (V93) and the
# emission noise options shared with `LinReg` -- the AR(p) of step 1 and the power-law scale of step 2
# (V94). The CNN's forward pass against Lux is V92 (training/runtests_lux.jl); `LinReg` drawing the same
# as before the AR recursion moved to `ts_noise.jl` is V73 (test_linreg_ar.jl).

@testmodule CnnFix begin
    using LinearAlgebra
    using Random

    const SRC = normpath(joinpath(@__DIR__, "..", "src"))
    include(joinpath(SRC, "ts_scaling.jl"))
    include(joinpath(SRC, "ts_history.jl"))
    include(joinpath(SRC, "ts_lstm.jl"))
    include(joinpath(SRC, "ts_noise.jl"))
    include(joinpath(SRC, "ts_scale.jl"))
    include(joinpath(SRC, "ts_lstm_online.jl"))

    const NQ, H, C, K, D = 3, 4, 5, 2, 7
    const HIST = HistorySpec(; h = H, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    const NREG = nfeatures(HIST)
    const NIN = NREG + H * NQ

    spec(; kw...) = LSTMSpec(; hist = HIST, n_hidden = D, n_latent = 0, n_encoder = 0, arch = :cnn,
                             emission = :constant, window = 1, n_in = NIN, n_conv = C, kernel = K, kw...)

    "The sequence layout of cnn_mean_fit.jl: per lag (oldest first) q*, dQ and the predictor increment."
    function seq_index()
        lagq(k) = (NQ + 2(k - 1) * NQ + 1):(NQ + (2k - 1) * NQ)
        s = zeros(Int, H, 3NQ)
        for (t, k) in enumerate(H:-1:1)
            s[t, :] = vcat(collect(lagq(k)) .+ NQ, collect(lagq(k)), NREG .+ (((k - 1) * NQ + 1):(k * NQ)))
        end
        return s
    end

    "Random weights in the Lux parameter layout `cnn_weights` reads (`zero_out`: the output layer at zero, as
    `cnn_mean_fit.jl` starts a skip fit; `Ws`: the skip, step 4)."
    function weights(sp = spec(); seed = 1, Ws = nothing, T = Float32, zero_out = false)
        rng = Xoshiro(seed)
        o = (; weight = randn(rng, NQ, D) ./ 3, bias = randn(rng, NQ) ./ 3)
        zero_out && (o = (; weight = zero(o.weight), bias = zero(o.bias)))
        net = (; layer_1 = (; layer_1 = (; layer_1 = (; weight = randn(rng, K, 3NQ, C) ./ 3, bias = randn(rng, C) ./ 3),
                                           layer_2 = (; weight = randn(rng, K, C, C) ./ 3, bias = randn(rng, C) ./ 3),
                                           layer_3 = (;)), layer_2 = (;)),
               layer_2 = (; weight = randn(rng, D, n_cell_input(sp)) ./ 3, bias = randn(rng, D) ./ 3),
               layer_3 = (; weight = randn(rng, D, D) ./ 3, bias = randn(rng, D) ./ 3),
               layer_4 = o)
        return cnn_weights(net, sp; seq_index = seq_index(), q_index = 1:NQ, bd = fill(-1.0, NQ),
                           LR = Matrix(cholesky([1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1.0]).L), Ws, T)
    end

    function scaling(; extra...)
        mu = reshape(collect(range(0.1, 0.3; length = NQ)), NQ, 1)
        sd = reshape(fill(2.0, NQ), NQ, 1)
        P = vcat(Matrix{Float64}(I, NREG, NREG), randn(Xoshiro(4), H * NQ, NREG) ./ 4)
        return merge((; in_scaling = (; mu, sigma = sd), out_scaling = (; mu = 0 .* mu, sigma = sd ./ 10),
                      target = :dQ, input_map = P), (; extra...))
    end

    stream(; n = 16, nwarm = 6, seed = 21) =
        (randn(Xoshiro(seed), NQ, n) .+ 3.0, randn(Xoshiro(seed + 1), NQ, nwarm) ./ 10)

    "Run a closure over the synthetic stream; returns (closure, outputs)."
    function run(sc; seed = 5, gate = 0.0, stochastic = true, nwarm = 6, n = 16)
        q_star, warm = stream(; n, nwarm)
        m = StochLSTM(spec(), weights(), sc; spinnup_data = warm, rng = Xoshiro(seed), gate, stochastic)
        return m, [get_next_item_timeseries(m, q_star[:, k]) for k in 1:n]
    end
end

@testitem "V93 arch = :cnn: what the spec needs, and n_conv / kernel only there" default_imports = false setup = [CnnFix] begin
    using Test
    using .CnnFix
    using .CnnFix: LSTMSpec, HIST, NIN
    @test CnnFix.spec().arch === :cnn
    @test CnnFix.n_cell_input(CnnFix.spec()) == (4 - 2) * 5 + 3          # positions x channels + N_Q
    @test_throws ErrorException CnnFix.spec(window = 2)
    @test_throws ErrorException CnnFix.spec(emission = :none)
    @test_throws ErrorException CnnFix.spec(n_in = 0)
    @test_throws ErrorException CnnFix.spec(kernel = 3)                 # 4 - 2 x 2 < 1: no position left
    @test_throws ErrorException CnnFix.spec(n_latent = 2)
    @test_throws ErrorException LSTMSpec(; hist = HIST, n_hidden = 4, n_latent = 0, arch = :dense, window = 1,
                                         emission = :constant, n_in = NIN, n_conv = 3, kernel = 2)
end

@testitem "V93 conv1d_tanh is Lux's (flipped-kernel) convolution, written out" default_imports = false setup = [CnnFix] begin
    using Test
    using .CnnFix: conv1d_tanh
    A = Float32[1 10; 2 20; 3 30]                                       # 3 steps x 2 channels
    W = zeros(Float32, 2, 2, 1); W[1, 1, 1] = 0.5; W[2, 1, 1] = -0.25; W[1, 2, 1] = 0.01
    b = Float32[0.1]
    Y = conv1d_tanh(A, W, b)
    # NNlib's conv: y[t] = b + Σ_c Σ_j W[j, c] x[t + K - j, c]; for K = 2, W[1] meets the LATER step
    @test Y[:, 1] ≈ tanh.(0.1f0 .+ 0.5f0 .* A[2:3, 1] .- 0.25f0 .* A[1:2, 1] .+ 0.01f0 .* A[2:3, 2])
    @test size(Y) == (2, 1)
end

@testitem "V93 StochLSTM with a CNN: N_Q normals per predicted step, none in the warm-up" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using .CnnFix
    m, out = CnnFix.run(CnnFix.scaling(); seed = 9)
    @test all(o -> all(isfinite, o), out)
    # the stream after the run is a fresh stream advanced by N_Q normals per predicted step
    ref = Xoshiro(9)
    for _ in 1:((16 - 6) * 3)
        randn(ref)
    end
    @test rand(m.rng) == rand(ref)
    # the warm-up returns the recorded corrections unconverted
    _, warm = CnnFix.stream()
    @test out[1:6] == [warm[:, k] for k in 1:6]
end

@testitem "V94 ar_phi is LinReg's AR recursion on the emission draw" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using .CnnFix
    using .CnnFix: ar_advance!, ar_state, _shared_noise!, get_next_item_timeseries
    phi = [0.6 0.3 0.0; -0.2 0.0 0.0]
    m, _ = CnnFix.run(CnnFix.scaling(; ar_phi = phi))
    @test m.ar.nz[] == 2                                                # warm-started from the record
    # one step of _shared_noise!: the drawn noise is the innovation of the shared recursion
    st = ar_state(phi); st.z .= m.ar.z
    e0 = Float32[0.3, -0.1, 0.2]
    m.scratch .= m.state.y .+ e0
    _shared_noise!(m, fill(3.0, 3))
    z = ar_advance!(st, Float64.(m.state.y .+ e0) .- Float64.(m.state.y))
    @test m.ar.z == st.z
    @test m.scratch ≈ m.state.y .+ Float32.(z) rtol = 1e-6
    # φ = 0 is the white closure, up to Float32 rounding of y + (scratch - y)
    _, white = CnnFix.run(CnnFix.scaling())
    _, ar0 = CnnFix.run(CnnFix.scaling(; ar_phi = zeros(1, 3)))
    @test all(k -> isapprox(ar0[k], white[k]; rtol = 1e-4), eachindex(white))
    # refused: both AR options, a non-stationary AR
    @test_throws ErrorException CnnFix.run(CnnFix.scaling(; ar_phi = phi, eta_ar = [0.5, 0.5, 0.5]))
    @test_throws ErrorException CnnFix.run(CnnFix.scaling(; ar_phi = [1.2 0.0 0.0]))
end

@testitem "V94 the AR warm start pushes the realised residual of the last warm-up steps" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using .CnnFix
    using .CnnFix: StochLSTM, get_next_item_timeseries, scale_input
    q_star, warm = CnnFix.stream()
    sc = CnnFix.scaling(; ar_phi = [0.5 0.5 0.5])
    m = StochLSTM(CnnFix.spec(), CnnFix.weights(), sc; spinnup_data = warm, rng = Xoshiro(5), gate = 0.0)
    for k in 1:6
        get_next_item_timeseries(m, q_star[:, k])
    end
    # after the last warm-up step, m.state.y is the network's mean for it (the replay, no draw)
    r = Float64.(vec(scale_input(warm[:, 6], sc.out_scaling))) .- Float64.(m.state.y)
    @test m.ar.z[:, 1] ≈ r
    @test m.ar.nz[] == 1
end

@testitem "V94 the power-law option multiplies the noise by f(q*)" default_imports = false setup = [CnnFix] begin
    using Test
    using .CnnFix
    using .CnnFix: _shared_noise!, powerlaw_factor
    pl = (; scale_beta = [1.0, 0.5, 2.0], scale_qref = [3.0, 3.0, 3.0], scale_qclip = [1.0 9.0; 1.0 9.0; 2.5 3.5])
    m, _ = CnnFix.run(CnnFix.scaling(; pl...))
    e0 = Float32[0.3, -0.1, 0.2]
    m.scratch .= m.state.y .+ e0
    q = [4.0, 2.0, 5.0]                                                 # the last is clipped at 3.5
    _shared_noise!(m, q)
    f = powerlaw_factor(q, pl.scale_beta, pl.scale_qref, pl.scale_qclip)
    @test f[3] ≈ (3.5 / 3.0)^2
    @test m.scratch ≈ m.state.y .+ Float32.(f .* (Float64.(m.state.y .+ e0) .- Float64.(m.state.y))) rtol = 1e-6
    # β = 0 is the white closure, up to Float32 rounding
    _, white = CnnFix.run(CnnFix.scaling())
    _, pl0 = CnnFix.run(CnnFix.scaling(; scale_beta = zeros(3), scale_qref = ones(3), scale_qclip = [0.1 9.0; 0.1 9.0; 0.1 9.0]))
    @test all(k -> isapprox(pl0[k], white[k]; rtol = 1e-4), eachindex(white))
    @test_throws ErrorException CnnFix.run(CnnFix.scaling(; scale_beta = ones(3)))   # a partial key set
end

@testitem "V94 a gated step still advances the AR state" default_imports = false setup = [CnnFix] begin
    using Test
    using .CnnFix
    m, out = CnnFix.run(CnnFix.scaling(; ar_phi = [0.5 0.5 0.5]); gate = 100.0)   # every q* is below the gate
    @test all(o -> all(iszero, o), out[7:end])
    z1 = copy(m.ar.z)
    CnnFix.get_next_item_timeseries(m, fill(3.0, 3))
    @test m.ar.z != z1
end

# Step 4 (2026-11-06): the CNN on a frozen LinReg, `y = g(z) + Ws z` (paper Eq. skip net).

@testitem "V95 arch = :cnn takes a skip; the weights must carry Ws exactly when the spec asks" default_imports = false setup = [CnnFix] begin
    using Test
    using .CnnFix
    using .CnnFix: NQ, NIN, check_shapes
    sp = CnnFix.spec(skip = true)
    @test sp.skip
    @test_throws ErrorException CnnFix.weights(sp)                              # skip without Ws
    @test_throws ErrorException CnnFix.weights(; Ws = zeros(NQ, NIN))          # Ws without skip
    @test_throws ErrorException CnnFix.weights(sp; Ws = zeros(NQ, NIN - 1))    # N_Q x n_in, n_in the mapped width
    @test check_shapes(CnnFix.weights(sp; Ws = randn(NQ, NIN)), sp)
    @test_throws ErrorException CnnFix.spec(skip = true, n_latent = 1)
end

@testitem "V95 the deployed CNN with a skip is the network plus Ws times the mapped input, step by step" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using .CnnFix
    using .CnnFix: NQ, NIN, StochLSTM, LSTMState, cnn_window!, get_next_item_timeseries
    Ws = randn(Xoshiro(3), NQ, NIN) ./ 5
    sp, sp0 = CnnFix.spec(skip = true), CnnFix.spec()
    for T in (Float32, Float64)
        w, w0 = CnnFix.weights(sp; Ws, T), CnnFix.weights(sp0; T)
        q_star, warm = CnnFix.stream()
        m = StochLSTM(sp, w, CnnFix.scaling(); spinnup_data = warm, rng = Xoshiro(5), gate = 0.0, stochastic = false)
        @test eltype(m.state.y) == T
        for k in 1:16
            get_next_item_timeseries(m, q_star[:, k])
            k <= 6 && continue
            st = LSTMState(sp0, T)
            cnn_window!(st, w0, sp0, m.xwin[:, 1:1])                          # the network alone, same input
            @test m.state.y ≈ st.y .+ T.(Ws) * m.xwin[:, 1] rtol = (T === Float32 ? 1e-5 : 1e-12)
        end
    end
end

@testitem "V95 linreg_skip_map: with the network at zero, the Float64 closure IS the frozen LinReg" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using LinearAlgebra
    using .CnnFix
    using .CnnFix: NQ, H, NREG, NIN, HIST, StochLSTM, get_next_item_timeseries, linreg_skip_map
    rng = Xoshiro(11)
    # the network's scalings: an input map whose regressor block is NOT the identity (as FEAT = dqinc's),
    # a nonzero target mean; the LinReg's own in- and out-scalings differ from them and from each other
    P0 = Matrix{Float64}(I, NREG, NREG) .+ 0.1 .* randn(rng, NREG, NREG)
    P = vcat(P0, randn(rng, H * NQ, NREG) ./ 4)
    mo = reshape([0.01, -0.02, 0.005], NQ, 1)
    sc = merge(CnnFix.scaling(), (; input_map = P, out_scaling = (; mu = mo, sigma = reshape([0.3, 0.2, 0.1], NQ, 1))))
    lin = (; in_scaling = (; mu = reshape([0.2, 0.25, 0.35], NQ, 1), sigma = reshape([1.5, 1.8, 2.2], NQ, 1)),
           out_scaling = (; mu = reshape([0.15, 0.3, 0.2], NQ, 1), sigma = reshape([1.4, 1.9, 2.0], NQ, 1)))
    c = hcat(Matrix{Float64}(I, NQ, NQ), zeros(NQ, NREG - NQ)) .+ 0.05 .* randn(rng, NQ, NREG)
    Ws = linreg_skip_map(c, lin, sc, HIST)
    @test size(Ws) == (NQ, NIN)
    @test all(iszero, Ws[:, (NREG + 1):end])                                   # the extra rows are not read
    sp = CnnFix.spec(skip = true)
    m = StochLSTM(sp, CnnFix.weights(sp; Ws, T = Float64, zero_out = true), sc; spinnup_data = CnnFix.stream()[2],
                  rng = Xoshiro(5), gate = 0.0, stochastic = false)
    q_star, warm = CnnFix.stream()
    # the LinReg by hand: x_L = [q*^n; q^{n-1}; q*^{n-1}; ...; q^{n-h}; q*^{n-h}; 1] in its own in-scaling, the level
    # (c x_L) in its out-scaling; the closure adds the network target's mean mo to the correction
    s(v) = (v .- vec(lin.in_scaling.mu)) ./ vec(lin.in_scaling.sigma)
    q = similar(q_star)
    errs = Float64[]
    for k in 1:16
        dQ = get_next_item_timeseries(m, q_star[:, k])
        if k <= 6
            q[:, k] = q_star[:, k] .+ warm[:, k]
            continue
        end
        x = vcat(s(q_star[:, k]), reduce(vcat, [vcat(s(q[:, k - j]), s(q_star[:, k - j])) for j in 1:H]), 1.0)
        lev = (c * x) .* vec(lin.out_scaling.sigma) .+ vec(lin.out_scaling.mu)
        ref = lev .- q_star[:, k] .+ vec(mo)
        push!(errs, maximum(abs, dQ .- ref) / maximum(abs, ref))
        q[:, k] = q_star[:, k] .+ ref
    end
    @test length(errs) == 10 && maximum(errs) < 1e-12
    @test_throws ErrorException linreg_skip_map(c[:, 1:(end - 1)], lin, sc, HIST)  # a LinReg of another width
end

# ---------------------------------------------------------------------------------------------------------------
# V103 (2026-11-13, card step4.md "Mechanism test 2"): a step-4 closure with its NETWORK output gated by each QoI's
# own level (`cnn_level_gate`, `level_gate_weights`, `cnn_window!(...; gate)`); the skip is never gated.
# ---------------------------------------------------------------------------------------------------------------
@testitem "V103 level_gate_weights: 1 / 0 at the thresholds, half cosines between, the two modes sum to one" default_imports = false setup = [CnnFix] begin
    using Test
    using .CnnFix: level_gate_weights
    lg = (; mode = :extremes, lo_full = [1.0, 0.0], lo_edge = [2.0, 1.0], hi_edge = [4.0, 3.0], hi_full = [5.0, 5.0])
    lb = merge(lg, (; mode = :bulk))
    @test level_gate_weights(lg, [0.5, -1.0]) == [1.0, 1.0]                  # below lo_full
    @test level_gate_weights(lg, [1.0, 0.0]) == [1.0, 1.0]                   # at lo_full
    @test level_gate_weights(lg, [2.0, 1.0]) == [0.0, 0.0]                   # at lo_edge
    @test level_gate_weights(lg, [3.0, 2.0]) == [0.0, 0.0]                   # the bulk
    @test level_gate_weights(lg, [4.0, 3.0]) == [0.0, 0.0]                   # at hi_edge
    @test level_gate_weights(lg, [5.0, 5.0]) == [1.0, 1.0]                   # at hi_full
    @test level_gate_weights(lg, [1.5, 4.0]) ≈ [0.5, 0.5]                    # the tapers' midpoints
    for q in ([0.7, 0.3], [1.3, 0.8], [3.0, 2.0], [4.6, 4.1], [9.0, 9.0])
        @test level_gate_weights(lg, q) .+ level_gate_weights(lb, q) == [1.0, 1.0]
    end
    xs = range(0.0, 6.0; length = 601)                                       # continuous and monotone on each side
    w = [first(level_gate_weights(lg, [x, 2.0])) for x in xs]
    @test maximum(abs, diff(w)) < 0.02
    @test issorted(w[xs .<= 3.0]; rev = true) && issorted(w[xs .>= 3.0])
end

@testitem "V103 cnn_window!: gate ones is the ungated closure bit for bit, gate zeros is the skip alone" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using .CnnFix
    using .CnnFix: NQ, NIN, LSTMState, cnn_window!, check_shapes, with_level_gate
    Ws = randn(Xoshiro(3), NQ, NIN) ./ 5
    sp = CnnFix.spec(skip = true)
    for T in (Float32, Float64)
        w, w0 = CnnFix.weights(sp; Ws, T), CnnFix.weights(sp; Ws, T, zero_out = true)
        for s in 1:5
            x = randn(Xoshiro(s), T, NIN, 1)
            a, b, c, d = LSTMState(sp, T), LSTMState(sp, T), LSTMState(sp, T), LSTMState(sp, T)
            cnn_window!(a, w, sp, x); cnn_window!(b, w, sp, x; gate = ones(NQ))
            @test a.y == b.y && a.logd == b.logd && a.h == b.h
            cnn_window!(c, w, sp, x; gate = zeros(NQ)); cnn_window!(d, w0, sp, x)
            @test c.y == d.y && c.logd == d.logd                              # the network at zero: the skip
            g = [0.0, 0.3, 1.0]
            e = LSTMState(sp, T); cnn_window!(e, w, sp, x; gate = g)
            @test e.y ≈ c.y .+ T.(g) .* (a.y .- c.y) rtol = (T === Float32 ? 1e-5 : 1e-12)
        end
    end
    # only on a step-4 closure (skip, not a generator), with ordered thresholds of N_Q entries
    lg = (; mode = :bulk, lo_full = fill(1.0, NQ), lo_edge = fill(2.0, NQ), hi_edge = fill(4.0, NQ), hi_full = fill(5.0, NQ))
    @test check_shapes(with_level_gate(CnnFix.weights(sp; Ws, T = Float64), lg), sp)
    @test_throws ErrorException check_shapes(with_level_gate(CnnFix.weights(), lg), CnnFix.spec())   # no skip
    @test_throws ErrorException check_shapes(with_level_gate(CnnFix.weights(sp; Ws), merge(lg, (; mode = :all))), sp)
    @test_throws ErrorException check_shapes(with_level_gate(CnnFix.weights(sp; Ws), merge(lg, (; lo_edge = fill(0.5, NQ)))), sp)
    @test_throws ErrorException check_shapes(with_level_gate(CnnFix.weights(sp; Ws), merge(lg, (; hi_full = fill(5.0, NQ - 1)))), sp)
end

@testitem "V103 online: the gated closure draws as the ungated one; bulk-only in the bulk IS it, extremes-only IS the skip" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using .CnnFix
    using .CnnFix: NQ, NIN, StochLSTM, LSTMState, cnn_window!, get_next_item_timeseries, with_level_gate, level_gate_weights
    Ws = randn(Xoshiro(3), NQ, NIN) ./ 5
    sp = CnnFix.spec(skip = true)
    q_star, warm = CnnFix.stream()
    wide = (; lo_full = fill(-100.0, NQ), lo_edge = fill(-99.0, NQ), hi_edge = fill(99.0, NQ), hi_full = fill(100.0, NQ))
    function runm(w; stochastic = true)
        m = StochLSTM(sp, w, CnnFix.scaling(); spinnup_data = warm, rng = Xoshiro(5), gate = 0.0, stochastic)
        return m, [get_next_item_timeseries(m, q_star[:, k]) for k in 1:16]
    end
    for T in (Float32, Float64)
        w, w0 = CnnFix.weights(sp; Ws, T), CnnFix.weights(sp; Ws, T, zero_out = true)
        _, full = runm(w)
        _, skip = runm(w0)
        _, bulk = runm(with_level_gate(w, merge(wide, (; mode = :bulk))))
        _, ext = runm(with_level_gate(w, merge(wide, (; mode = :extremes))))
        @test bulk == full                       # every level in the bulk: the closure itself, draws included
        @test ext == skip                        # ... and extremes-only is its skip (plus the same noise)
        @test full != skip
    end
    # mixed: thresholds inside the stream, so each QoI switches between the zones; y = skip + w .* network per step
    lg = (; mode = :extremes, lo_full = fill(2.0, NQ), lo_edge = fill(2.6, NQ), hi_edge = fill(3.4, NQ), hi_full = fill(4.0, NQ))
    w = CnnFix.weights(sp; Ws, T = Float64)
    m = StochLSTM(sp, with_level_gate(w, lg), CnnFix.scaling(); spinnup_data = warm, rng = Xoshiro(5), gate = 0.0,
                  stochastic = false)
    ws = Float64[]
    for k in 1:16
        get_next_item_timeseries(m, q_star[:, k])
        k <= 6 && continue
        g = level_gate_weights(lg, q_star[:, k])
        append!(ws, g)
        a, c = LSTMState(sp, Float64), LSTMState(sp, Float64)
        cnn_window!(a, w, sp, m.xwin[:, 1:1])
        cnn_window!(c, w, sp, m.xwin[:, 1:1]; gate = zeros(NQ))
        @test m.state.y ≈ c.y .+ g .* (a.y .- c.y) rtol = 1e-12
    end
    @test any(iszero, ws) && any(isone, ws) && any(x -> 0 < x < 1, ws)       # all three zones were met
end

@testitem "V103 gate zero everywhere: the step-4 closure on a frozen LinReg IS that LinReg, whatever its network" default_imports = false setup = [CnnFix] begin
    using Test
    using Random
    using LinearAlgebra
    using .CnnFix
    using .CnnFix: NQ, H, NREG, NIN, HIST, StochLSTM, get_next_item_timeseries, linreg_skip_map, with_level_gate
    rng = Xoshiro(11)
    P0 = Matrix{Float64}(I, NREG, NREG) .+ 0.1 .* randn(rng, NREG, NREG)
    P = vcat(P0, randn(rng, H * NQ, NREG) ./ 4)
    # target mean zero here: the closure adds the network target's mean through its scaling, which no gate removes
    sc = merge(CnnFix.scaling(), (; input_map = P, out_scaling = (; mu = zeros(NQ, 1), sigma = reshape([0.3, 0.2, 0.1], NQ, 1))))
    lin = (; in_scaling = (; mu = reshape([0.2, 0.25, 0.35], NQ, 1), sigma = reshape([1.5, 1.8, 2.2], NQ, 1)),
           out_scaling = (; mu = reshape([0.15, 0.3, 0.2], NQ, 1), sigma = reshape([1.4, 1.9, 2.0], NQ, 1)))
    c = hcat(Matrix{Float64}(I, NQ, NQ), zeros(NQ, NREG - NQ)) .+ 0.05 .* randn(rng, NQ, NREG)
    Ws = linreg_skip_map(c, lin, sc, HIST)
    sp = CnnFix.spec(skip = true)
    lg = (; mode = :extremes, lo_full = fill(-100.0, NQ), lo_edge = fill(-99.0, NQ), hi_edge = fill(99.0, NQ),
          hi_full = fill(100.0, NQ))
    w = with_level_gate(CnnFix.weights(sp; Ws, T = Float64), lg)            # a RANDOM network, gated off
    q_star, warm = CnnFix.stream()
    m = StochLSTM(sp, w, sc; spinnup_data = warm, rng = Xoshiro(5), gate = 0.0, stochastic = false)
    s(v) = (v .- vec(lin.in_scaling.mu)) ./ vec(lin.in_scaling.sigma)
    q = similar(q_star)
    errs = Float64[]
    for k in 1:16
        dQ = get_next_item_timeseries(m, q_star[:, k])
        if k <= 6
            q[:, k] = q_star[:, k] .+ warm[:, k]
            continue
        end
        x = vcat(s(q_star[:, k]), reduce(vcat, [vcat(s(q[:, k - j]), s(q_star[:, k - j])) for j in 1:H]), 1.0)
        ref = (c * x) .* vec(lin.out_scaling.sigma) .+ vec(lin.out_scaling.mu) .- q_star[:, k]
        push!(errs, maximum(abs, dQ .- ref) / maximum(abs, ref))
        q[:, k] = q_star[:, k] .+ ref
    end
    @test length(errs) == 10 && maximum(errs) < 1e-12
end
