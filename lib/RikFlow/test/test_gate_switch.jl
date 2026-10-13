# V86 -- the turbulence gate as a setting (2026-10-13, handoff_gate_off_2026-10-13 Step 2).
#
# `LinReg(...; gate)` defaults to `TURBULENCE_GATE`; `gate = 0.0` switches it off. The drivers read
# `RIKFLOW_GATE` (`gate_from_env`) and refuse a gate-off run into a directory not ending in
# `_nogate` (`check_gate_dir`). Pinned:
#
#   (a) with the default -- and with `gate = TURBULENCE_GATE` passed explicitly -- `LinReg` is
#       bit-identical to the pre-AR oracle (`legacy_linreg.jl`), outputs and RNG stream, on the
#       synthetic file and on the archived LinReg1/LinReg7 fits where present;
#   (b) gate-off equals gate-on up to the step before the first firing and differs at it; it
#       consumes the RNG identically (common random numbers); with no firing on the path the two
#       are identical bit for bit (white and AR(2) paths);
#   (c) the threshold is recorded in the output: both drivers write `gate_threshold` from the
#       deployed value and parse (they cannot be loaded here, gotcha #47); the D6 pairing helpers.
#
# The same `LinRegAR` setup module as V73 (test_linreg_ar.jl) supplies the closure and the oracle.

@testitem "V86 (a) default gate is bit-identical to the pre-switch code" default_imports = false setup = [LinRegAR] begin
    using Test, Random
    using .LinRegAR: synth_file, qstar_path, drive, new_closure, LegacyLinReg, LinReg, LRS,
                     TURBULENCE_GATE
    mktempdir() do d
        f = joinpath(d, "LinReg.jld2")
        synth_file(f; n = 3, h = 2)
        spin = 0.01 .* randn(Xoshiro(9), 3, 10)
        Q = qstar_path(3, 300; gate_at = [50, 51, 200])
        for seed in (1, 77)
            a = new_closure(LinReg, f, seed, 3, 2, spin)
            e = LinReg(f, Xoshiro(seed), Array; q_hist = zeros(6, 2), spinnup_data = copy(spin),
                       gate = TURBULENCE_GATE)
            b = new_closure(LegacyLinReg, f, seed, 3, 2, spin)
            @test a.gate === TURBULENCE_GATE && e.gate === TURBULENCE_GATE
            A, E, B = drive(a, Q), drive(e, Q), drive(b, Q)
            @test A == B && E == B
            @test all(A[:, 50] .== 0) && all(A[:, 200] .== 0)
            ra, re, rb = rand(a.rng), rand(e.rng), rand(b.rng)
            @test ra === rb && re === rb                 # the RNG streams are at the same point
        end
    end
    n = Ref(0)
    for name in ("LinReg1", "LinReg7")
        f = joinpath(LRS, name, "LinReg.jld2")
        isfile(f) || continue
        n[] += 1
        h = LinRegAR.load(f, "hist_len")
        mu = vec(LinRegAR.load(f, "scaling").in_scaling.mu)
        spin = 0.001 .* mu .* randn(Xoshiro(4), 6, 100)
        Q = mu .* (1 .+ 0.1 .* randn(Xoshiro(5), 6, 160))
        a = LinReg(f, Xoshiro(7), Array; q_hist = zeros(12, h), spinnup_data = copy(spin),
                   gate = TURBULENCE_GATE)
        b = new_closure(LegacyLinReg, f, 7, 6, h, spin)
        @test isequal(drive(a, Q), drive(b, Q))
        @test rand(a.rng) === rand(b.rng)
    end
    @info "V86 (a) checked on $(n[]) archived LinReg files"
end

@testitem "V86 (b) gate-off equals gate-on up to the first firing and differs at it" default_imports = false setup = [LinRegAR] begin
    using Test, Random, LinearAlgebra
    using .LinRegAR: synth_file, qstar_path, drive, new_closure, LinReg, get_next_item_timeseries,
                     TURBULENCE_GATE
    mktempdir() do d
        for (label, kw) in (("white", (;)),
                            ("AR(2)", (; phi = [0.8 0.5 0.3; -0.1 0.0 0.0], S_xi = Matrix(0.01I, 3, 3))))
            f = joinpath(d, "LinReg_$(label == "white" ? "w" : "ar").jld2")
            synth_file(f; n = 3, h = 2, kw...)
            nw = 10
            spin = 0.01 .* randn(Xoshiro(9), 3, nw)
            first_fire = 50
            # a tiny q* input (1e-3 < 1e-2) at steps 50, 51 and 200 trips the gate
            Q = qstar_path(3, 300; gate_at = [first_fire, first_fire + 1, 200])
            on = new_closure(LinReg, f, 3, 3, 2, spin)
            off = LinReg(f, Xoshiro(3), Array; q_hist = zeros(6, 2), spinnup_data = copy(spin),
                         gate = 0.0)
            @test off.gate == 0.0
            A, B = drive(on, Q), drive(off, Q)
            @test A[:, 1:(first_fire - 1)] == B[:, 1:(first_fire - 1)]      # identical before
            @test all(A[:, first_fire] .== 0) && all(B[:, first_fire] .!= 0) # differ AT the firing
            @test all(B[:, 200] .!= 0)                                      # off never fires
            @test rand(on.rng) === rand(off.rng)       # the same number of draws: CRN partners

            # no firing on the path -> gate-off is the gated run, bit for bit
            Qn = qstar_path(3, 300)
            on2 = new_closure(LinReg, f, 3, 3, 2, spin)
            off2 = LinReg(f, Xoshiro(3), Array; q_hist = zeros(6, 2), spinnup_data = copy(spin),
                          gate = 0.0)
            @test drive(on2, Qn) == drive(off2, Qn)
            @test on2.q_hist == off2.q_hist
            @test rand(on2.rng) === rand(off2.rng)
        end
        # abs(x) < 0 holds for no x: a NaN or a zero q* does not fire a gate-off closure
        f = joinpath(d, "LinReg_w.jld2")
        m = LinReg(f, Xoshiro(1), Array; q_hist = zeros(6, 2), spinnup_data = zeros(3, 2), gate = 0.0)
        get_next_item_timeseries(m, [1.0, 1.0, 1.0]); get_next_item_timeseries(m, [1.0, 1.0, 1.0])
        @test any(get_next_item_timeseries(m, [1.0, 1.0, 0.0]) .!= 0)
        # a bad threshold is refused
        @test_throws ErrorException LinReg(f, Xoshiro(1), Array; q_hist = zeros(6, 2),
                                           spinnup_data = zeros(3, 2), gate = -1e-2)
        @test_throws ErrorException LinReg(f, Xoshiro(1), Array; q_hist = zeros(6, 2),
                                           spinnup_data = zeros(3, 2), gate = NaN)
    end
end

@testitem "V86 RIKFLOW_GATE and the _nogate directory guard" default_imports = false setup = [LinRegAR] begin
    using Test
    using .LinRegAR: gate_from_env, check_gate_dir, TURBULENCE_GATE
    @test gate_from_env(Dict{String,String}()) === TURBULENCE_GATE
    @test gate_from_env(Dict("RIKFLOW_GATE" => "")) === TURBULENCE_GATE
    @test gate_from_env(Dict("RIKFLOW_GATE" => "on")) === TURBULENCE_GATE
    @test gate_from_env(Dict("RIKFLOW_GATE" => " OFF ")) === 0.0
    for bad in ("0", "false", "of", "1e-3")
        @test_throws ErrorException gate_from_env(Dict("RIKFLOW_GATE" => bad))
    end
    @test check_gate_dir("out/D6_LinReg1_nogate", 0.0) === nothing
    @test check_gate_dir("out/TO_LRS/LinReg1_nogate/", 0.0) === nothing
    @test check_gate_dir("C:\\out\\D6_LinReg1_nogate\\", 0.0) === nothing
    @test check_gate_dir("out/D6_LinReg1", TURBULENCE_GATE) === nothing
    @test_throws ErrorException check_gate_dir("out/D6_LinReg1", 0.0)
    @test_throws ErrorException check_gate_dir("out/D6_LinReg1_nogate_x", 0.0)
    @test_throws ErrorException check_gate_dir("out/TO_LRS/LinReg1/", 0.0)
    @test_throws ErrorException check_gate_dir("out/D6_LinReg1_nogate", TURBULENCE_GATE)
    @test_throws ErrorException check_gate_dir("out/D6_LinReg1_nogate", 1e-3)
end

@testitem "V86 (c) the drivers record the deployed threshold; D6 pairing helpers" default_imports = false setup = [D6Lstm] begin
    using Test
    X = D6Lstm
    RF = X.RF
    parses(src, name) = (ex = Meta.parseall(src; filename = name);
                         !any(a -> a isa Expr && a.head in (:error, :incomplete), ex.args))
    # D6: the threshold written is the one deployed, the dir is checked, LinReg gets the gate
    drv = read(joinpath(RF, "exp_square_HIT", "tools", "run_d6.jl"), String)
    @test occursin("gate = RikFlow.gate_from_env()", drv)
    @test occursin("RikFlow.check_gate_dir(od, gate)", drv)
    @test occursin("spinnup_data = ArrayType{T}(dQ_warm), gate)", drv)
    @test occursin("gate_threshold = cl === :ddn ? NaN : cl === :lrs ? gate : RikFlow.TURBULENCE_GATE", drv)
    @test occursin("pair_meta...", drv)
    @test parses(drv, "run_d6.jl")
    # long runs: the same, plus the model read from the gated model's own directory
    lr = read(joinpath(RF, "exp_square_HIT", "6_online_TO_LRS.jl"), String)
    @test occursin("gate = RikFlow.gate_from_env()", lr)
    @test occursin("RikFlow.check_gate_dir(out_dir, gate)", lr)
    @test occursin("out_dir = gate == 0 ? TO_folder*\"/\$(name)_nogate/\" : model_src_dir", lr)
    @test occursin("LinReg_file_name = model_src_dir*\"LinReg.jld2\"", lr)
    @test occursin("spinnup_data = ArrayType{T}(dQ_data), gate = gate)", lr)
    @test occursin("gate_threshold = gate, model = LinReg_file_name", lr)
    @test parses(lr, "6_online_TO_LRS.jl")

    # D6_ONLY
    @test X.only_members("") === nothing
    @test X.only_members("42:3, 42:10,46:2") == Dict(42 => Set([3, 10]), 46 => Set([2]))
    @test_throws ErrorException X.only_members("42-3")
    @test_throws ErrorException X.only_members("ic42:3")
    # D6_PAIR_WITH's comparison: a gated original and its gate-off rerun first differ at the firing
    nwarm = 5
    orig = ones(2, 12); orig[:, nwarm + 4] .= 0                     # fired at lead 4
    rer = ones(2, 12); rer[:, (nwarm + 4):end] .= 2.0               # off: diverges from there on
    @test X.first_diff_col(rer, orig) == nwarm + X.gate_census(orig, nwarm).first_lead
    @test X.first_diff_col(orig, copy(orig)) == 0
    @test X.first_diff_col(orig[:, 1:7], orig) == 0                 # a truncated (diverged) member
    nan = copy(orig); nan[1, 3] = NaN
    @test X.first_diff_col(nan, copy(nan)) == 0                     # NaN-safe
    @test X.first_diff_col(nan, orig) == 3
    @test X.first_diff_col(ones(3, 2), ones(2, 2)) == 1
end
