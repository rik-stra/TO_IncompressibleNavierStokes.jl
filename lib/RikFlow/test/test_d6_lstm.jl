# V72 -- M4 (`StochLSTM`) through D6 (plan Start here 0(a), 2026-09-28).
#
# `tools/run_d6.jl` loads CUDA and IncompressibleNavierStokes and cannot be loaded here (gotcha
# #47), so the M4 path it deploys lives in `tools/d6_lib.jl`, which this file includes bare with
# the stdlib `ts_*` layer -- the same functions the driver calls, on a tiny synthetic fit saved and
# re-loaded through `save_stochlstm`/`load_stochlstm`. What is pinned is D6's contract for a new
# closure, the one `LinReg` already satisfies:
#
#   * the replayed warm-up is the package's `dQ_warm`, bit-identical and unconverted (#48);
#   * the replay draws nothing, so a member seed means the same thing for every closure (V38/V42);
#   * 🔴 the turbulence gate is `LinReg`'s: the same constant, applied to the final correction, never
#     during the replay, and counted by the same census (#59/#65 -- the DDN lacked it and that
#     confounded `results.md` §4c).

@testmodule D6Lstm begin
    using LinearAlgebra
    using Random
    using JLD2

    const RF = normpath(joinpath(@__DIR__, ".."))
    const SRC = joinpath(RF, "src")
    include(joinpath(SRC, "ts_scaling.jl"))
    include(joinpath(SRC, "ts_history.jl"))
    include(joinpath(SRC, "ts_lstm.jl"))
    include(joinpath(SRC, "ts_lstm_online.jl"))
    include(joinpath(SRC, "ts_lstm_io.jl"))
    include(joinpath(RF, "analysis", "build_d6_ics.jl"))
    include(joinpath(RF, "exp_square_HIT", "tools", "d6_lib.jl"))

    const NQ = 6

    """
    `RikFlow.TURBULENCE_GATE`, read off `time_series_methods.jl` (which needs Distributions/Adapt
    and is not loaded here). The driver passes the constant itself; this is how the test gets the
    same number without a second copy of the literal.
    """
    const TURBULENCE_GATE = let src = read(joinpath(SRC, "time_series_methods.jl"), String)
        parse(Float64, match(r"const TURBULENCE_GATE\s*=\s*([0-9.eE+-]+)", src)[1])
    end

    scaling(; nq = NQ) = (in_scaling = (; mu = fill(0.5, nq, 1), sigma = fill(2.0, nq, 1)),
                          out_scaling = (; mu = fill(0.5, nq, 1), sigma = fill(2.0, nq, 1)))

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

    "The two deployed shapes: persistent state with a lag window, and window mode."
    spec(mode::Symbol) = mode === :persistent ?
        LSTMSpec(; hist = HistorySpec(; h = 2, n_qoi = NQ), n_hidden = 4, n_latent = 3,
                 n_encoder = 4, arch = :vrnn, emission = :state_dependent) :
        LSTMSpec(; hist = HistorySpec(; h = 0, n_qoi = NQ, hist_var = :q_star), n_hidden = 4,
                 n_latent = 3, n_encoder = 0, arch = :vrnn, emission = :state_dependent,
                 window = 4)

    "A fit written and read back the way a real one is, under `root/<sub>/StochLSTM_seed1.jld2`."
    function saved_fit(root, sub, mode)
        sp = spec(mode)
        p = joinpath(root, sub, "StochLSTM_seed1.jld2")
        save_stochlstm(p, sp, weights(sp), scaling())
        return p
    end

    """
        emulate(sampler, q0, nt; drive)

    `to_sgs_term`'s bookkeeping in miniature: at step `n` the closure sees `q* = drive(q, n)`, its
    `dQ` goes through `Array(dQ)` into column `n` of an `N_Q x nt` Float64 output, and
    `q = q* + dQ`. Returns the stored `dQ`, as `run_d6.jl` writes it.
    """
    function emulate(sampler, q0, nt; drive = (q, n) -> q)
        out = Array{Float64}(undef, length(q0), nt)
        q = copy(q0)
        for n in 1:nt
            qs = drive(q, n)
            dQ = Array(get_next_item_timeseries(sampler, qs))
            out[:, n] = dQ
            q = qs .+ dQ
        end
        return out
    end
end

@testitem "V72 D6_CLOSURE and D6_MODEL resolve an M4 fit the way 12_online_StochLSTM.jl does" default_imports = false setup = [D6Lstm] begin
    using Test
    using JLD2
    X = D6Lstm
    @test X.parse_closure("lrs") === :lrs
    @test X.parse_closure(" DDN ") === :ddn
    @test X.parse_closure("lstm") === X.parse_closure("StochLSTM") === X.parse_closure("m4") === :lstm
    @test_throws ErrorException X.parse_closure("lux")

    mktempdir() do root
        p1 = X.saved_fit(root, joinpath("diag", "fitA"), :persistent)
        r = X.resolve_lstm_model("diag/fitA"; root)                  # relative to TO_LSTM
        @test r.file == p1 && r.deploy_seed == 1 && r.name == "diag/fitA"
        @test X.resolve_lstm_model(joinpath(root, "diag", "fitA"); root).name == "diag/fitA"
        @test X.resolve_lstm_model(p1; root).deploy_seed == 1        # a file, deployed as is
        # 🔑 S6: the median-seed fit goes online when the directory names one
        d = dirname(p1)
        cp(p1, joinpath(d, "StochLSTM_seed3.jld2"))
        jldsave(joinpath(d, "seed_summary.jld2"); median_seed = 3)
        r3 = X.resolve_lstm_model("diag/fitA"; root)
        @test r3.deploy_seed == 3 && endswith(r3.file, "StochLSTM_seed3.jld2")
        @test_throws ErrorException X.resolve_lstm_model(""; root)
        @test_throws ErrorException X.resolve_lstm_model("diag/nope"; root)
        mkpath(joinpath(root, "empty"))
        @test_throws ErrorException X.resolve_lstm_model("empty"; root)
        # what the output file records about the fit
        rec = X.lstm_record(X.load_stochlstm(p1))
        @test rec.lstm_arch == "vrnn" && rec.lstm_window == 0 && rec.lstm_target == "q"
        @test rec.lstm_knobs == ""
    end
end

@testitem "V72 the D6 warm-up replay is bit-identical, unconverted and draws nothing" default_imports = false setup = [D6Lstm] begin
    using Test
    using Random
    X = D6Lstm
    rng = Xoshiro(17)
    nwarm, nt = 100, 130
    dQ_warm = randn(rng, Float64, X.NQ, nwarm) ./ 50          # Float64, as an IC package's is
    q0 = 1.0 .+ rand(rng, X.NQ)
    mktempdir() do root
        for mode in (:persistent, :window)
            fit = X.load_stochlstm(X.saved_fit(root, string(mode), mode))
            seed = hash((:d6, 209, 3))                           # `member_seed(k, member)`
            s = X.make_lstm_sampler(fit, dQ_warm, seed; gate = X.TURBULENCE_GATE)
            @test s.gate == X.TURBULENCE_GATE
            @test s.spinnup_data !== dQ_warm                     # a copy, not an alias
            dQ = X.emulate(s, q0, nt)
            # 🔴 the #48 gate, as `run_d6.jl` now checks it on the node
            @test view(dQ, :, 1:nwarm) == dQ_warm
            @test eltype(dQ) === Float64
            @test any(!iszero, view(dQ, :, (nwarm + 1):nt))     # and then it forecasts
            # V38/V42: the replay consumed nothing -- after it, the stream is where a fresh one is
            s2 = X.make_lstm_sampler(fit, dQ_warm, seed; gate = X.TURBULENCE_GATE)
            X.emulate(s2, q0, nwarm)
            @test randn(s2.rng) == randn(Xoshiro(seed))
            # member seeds: same seed, same forecast; another seed, another forecast
            @test X.emulate(X.make_lstm_sampler(fit, dQ_warm, seed; gate = X.TURBULENCE_GATE),
                            q0, nt) == dQ
            @test X.emulate(X.make_lstm_sampler(fit, dQ_warm, seed + 1; gate = X.TURBULENCE_GATE),
                            q0, nt)[:, nwarm + 1] != dQ[:, nwarm + 1]
        end
        fit = X.load_stochlstm(X.saved_fit(root, "p", :persistent))
        @test_throws ErrorException X.make_lstm_sampler(fit, dQ_warm[1:5, :], 1; gate = 1e-2)
    end
end

@testitem "V72 the gate is LinReg's: same constant, fires on small q*, not during the replay" default_imports = false setup = [D6Lstm] begin
    using Test
    using Random
    X = D6Lstm
    g = X.TURBULENCE_GATE
    @test g == 1e-2                    # the value every run to date used (#65); changing it splits them
    rng = Xoshiro(23)
    nwarm, nt = 100, 160
    dQ_warm = randn(rng, Float64, X.NQ, nwarm) ./ 50
    q0 = 1.0 .+ rand(rng, X.NQ)
    # E[16,32] (row 6) is pushed below the gate on steps 101-105 (the replay's last steps too) and
    # on 131-140 -- the shape the rebaselined LinReg1 runs show (#59: one band, short bursts)
    low(n) = (96 <= n <= 105) || (131 <= n <= 140)
    # q* is held at q0 (well above the gate) except on the low steps, so a firing anywhere else
    # could only be the closure's own doing
    drive(q, n) = (qs = copy(q0); low(n) && (qs[6] = 0.4g); qs)
    mktempdir() do root
        for mode in (:persistent, :window)
            fit = X.load_stochlstm(X.saved_fit(root, string(mode), mode))
            dQ = X.emulate(X.make_lstm_sampler(fit, dQ_warm, 7; gate = g), q0, nt; drive)
            # not during the replay: LinReg's warm-up branch returns spinnup_data ungated, and so
            # does StochLSTM's -- the replay stays bit-identical even where q* is below the gate
            @test view(dQ, :, 1:nwarm) == dQ_warm
            fired = [n for n in (nwarm + 1):nt if all(iszero, view(dQ, :, n))]
            @test fired == [n for n in (nwarm + 1):nt if low(n)]      # exactly where q* was low
            c = X.gate_census(dQ, nwarm)
            @test c.nfired == 15 && c.nsteps == nt - nwarm && c.first_lead == 1
            # with the gate off nothing is zeroed -- the zeros above are the gate's
            d0 = X.emulate(X.make_lstm_sampler(fit, dQ_warm, 7; gate = 0.0), q0, nt; drive)
            @test X.gate_census(d0, nwarm).nfired == 0
        end
    end
end

@testitem "V72 the gate sits where LinReg's does, and the driver passes the one constant" default_imports = false setup = [D6Lstm] begin
    using Test
    RF = D6Lstm.RF
    # LinReg: every gate site reads the named constant
    tsm = read(joinpath(RF, "src", "time_series_methods.jl"), String)
    @test count("any(abs.(q_star) .< TURBULENCE_GATE) && (dQ .= 0)", tsm) == 4
    # StochLSTM: the gate is the LAST thing done to the correction -- after the output map and the
    # calibration offset -- and before the step is pushed into the lag window, as in LinReg
    src = read(joinpath(RF, "src", "ts_lstm_online.jl"), String)
    body = src[findfirst("function get_next_item_timeseries(m::StochLSTM", src)[1]:end]
    ig = findfirst("any(abs.(qs_host) .< m.gate) && (dQ .= 0)", body)[1]
    @test ig > findlast("dq_offset", body)[1]
    @test ig > findfirst("scale_output", body)[1]
    @test ig < findlast("_push_scaled!(m, qs_host .+ dQ, qs_host)", body)[1]
    # the D6 driver deploys M4 through d6_lib with RikFlow's constant, and checks it
    drv = read(joinpath(RF, "exp_square_HIT", "tools", "run_d6.jl"), String)
    @test occursin("include(joinpath(@__DIR__, \"d6_lib.jl\"))", drv)
    @test occursin("make_lstm_sampler(lstm_fit", drv)
    @test occursin("gate = RikFlow.TURBULENCE_GATE)", drv)
    @test occursin("s.gate == RikFlow.TURBULENCE_GATE", drv)
    @test occursin("warm_identical", drv) && occursin("gate_census", drv)
    # and it parses (the suite cannot load it; gotcha #47)
    ex = Meta.parseall(drv; filename = "run_d6.jl")
    @test !any(a -> a isa Expr && a.head in (:error, :incomplete), ex.args)
end

@testitem "V72 the gate census and the output-directory guard" default_imports = false setup = [D6Lstm] begin
    using Test
    using JLD2
    X = D6Lstm
    dQ = ones(3, 10)
    dQ[:, 2] .= 0                     # inside the warm-up: not counted
    dQ[:, 7] .= 0
    dQ[:, 9] .= 0
    dQ[1, 8] = 0                      # one band zero is not a gate firing
    c = X.gate_census(dQ, 5)
    @test c == (; nfired = 2, nsteps = 5, first_lead = 2)
    @test X.gate_census(ones(3, 4), 4) == (; nfired = 0, nsteps = 0, first_lead = 0)

    id = (; closure = "lstm", model_name = "diag/x", block = "selection", nlead = 400)
    mktempdir() do od
        @test X.check_out_dir(joinpath(od, "absent"), id) === nothing
        @test X.check_out_dir(od, id) === nothing                     # empty
        jldsave(joinpath(od, "d6_online_ic209_m1.jld2"); closure = "lstm", model_name = "diag/x",
                block = "selection", nlead = 400, model = "/a/b")
        @test X.check_out_dir(od, id) === nothing                     # same experiment
        @test_throws ErrorException X.check_out_dir(od, (; id..., model_name = "diag/y"))
        @test_throws ErrorException X.check_out_dir(od, (; id..., closure = "lrs"))
        @test_throws ErrorException X.check_out_dir(od, (; id..., block = ""))
        @test_throws ErrorException X.check_out_dir(od, (; id..., nlead = 1200))
    end
    # a pre-2026-09-28 LinReg file identifies itself from its model path
    @test X.d6_run_identity(Dict("model" => "/x/TO_LRS/LinReg1/LinReg.jld2", "closure" => "lrs",
                                 "nlead" => 1200)) ==
          (; closure = "lrs", model_name = "LinReg1", block = "", nlead = 1200)
end
