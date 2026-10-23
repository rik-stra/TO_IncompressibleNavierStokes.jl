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

    "Random weights in the Lux parameter layout `cnn_weights` reads."
    function weights(sp = spec(); seed = 1)
        rng = Xoshiro(seed)
        net = (; layer_1 = (; layer_1 = (; layer_1 = (; weight = randn(rng, K, 3NQ, C) ./ 3, bias = randn(rng, C) ./ 3),
                                           layer_2 = (; weight = randn(rng, K, C, C) ./ 3, bias = randn(rng, C) ./ 3),
                                           layer_3 = (;)), layer_2 = (;)),
               layer_2 = (; weight = randn(rng, D, n_cell_input(sp)) ./ 3, bias = randn(rng, D) ./ 3),
               layer_3 = (; weight = randn(rng, D, D) ./ 3, bias = randn(rng, D) ./ 3),
               layer_4 = (; weight = randn(rng, NQ, D) ./ 3, bias = randn(rng, NQ) ./ 3))
        return cnn_weights(net, sp; seq_index = seq_index(), q_index = 1:NQ, bd = fill(-1.0, NQ),
                           LR = Matrix(cholesky([1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1.0]).L))
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
