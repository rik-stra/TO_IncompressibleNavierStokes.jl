# V29 -- D6's initial-condition selection and packaging.
#
# The two constraints that define the usable pool cost K, and neither is visible in the data:
# an IC inside M0's fit window measures short-lead spread on data the conditional mean has already
# seen, and an IC too late in the record has no truth to be scored against for the whole 1208-step
# forecast. Both are silent if they fail -- the first inflates early skill, the second truncates the
# verification without saying so -- so both are asserted per IC, and the arithmetic that derives
# them is asserted here against planted values rather than re-derived.
#
# The warm-up slice is the off-by-one this file exists for. `dQ[:, m]` is the correction *at* step
# `m`; a run launched from the field at step `n_k` takes its first solver step at `n_k + 1`. The
# reduction test is the sharp one: at `k = 1` the general slice must be exactly `dQ[:, 1:100]`, the
# archived driver's hard-coded value (`6_online_TO_LRS.jl:62`).

@testmodule D6 begin
    using JLD2, Printf, Dates
    const SRC = normpath(joinpath(@__DIR__, "..", "analysis", "build_d6_ics.jl"))
    include(SRC)
end

@testitem "V29 every selected IC clears the fit window and fits inside the reference" default_imports = false setup = [D6] begin
    using Test
    for K in (5, 17, 168, 180, 336)
        sel = D6.select_ics(; K)
        @test length(sel.k) == K
        @test allunique(sel.k)
        @test issorted(sel.k)
        @test all(i -> sel.n[i] == D6.FIELD_STRIDE * (sel.k[i] - 1), 1:K)
        @test all(i -> sel.t[i] == D6.FIELD_DT * (sel.k[i] - 1), 1:K)
        @test all(>(D6.FIT_END_TU), sel.t)                                   # outside M0's fit
        @test all(n -> n + D6.N_WARM + D6.N_LEAD <= D6.N_REF, sel.n)         # inside the truth
        # The tightest instance of each constraint, stated as the numbers rather than as a relation,
        # so a change to nlead or to the record length has to come through this file.
        @test first(sel.t) >= 10.25
        @test last(sel.n) <= 38692
    end
end

@testitem "V29 K = 180 comes out at 0.48 TU, not the nominal 0.5" default_imports = false setup = [D6] begin
    using Test
    # The pool is k in [42, 377], 336 fields -- shorter than the old [42, 387] because N_LEAD went
    # from 1208 to 2172 steps when the forecast length was reset from the LEVEL's decorrelation time
    # (Rik, 2026-09-15). Every second field is 0.5 TU and yields 168 ICs; 180 needs 0.47.
    #
    # 🔴 **The spacing is now BELOW the slowest QoI's decorrelation time, not above it.** 0.4679 TU
    # against T(q)_max = 0.5430 is a ratio of 0.86; on the old constants it was 1.60 against the
    # dQ-based 0.3017. So at K = 180 adjacent ICs are genuinely correlated and the block bootstrap
    # over initialisation time is not merely advisable but load-bearing. K = 90 (`--array=1-179:2`)
    # doubles the spacing to 1.7x and is the intended production setting.
    sel = D6.select_ics(; K = 180)
    @test sel.pool == 336
    @test sel.spacing_tu ≈ 0.46788 atol = 1e-4
    @test sel.spacing_tu < 0.5
    @test sel.spacing_tu / D6.T_INT_MAX ≈ 0.8617 atol = 5e-3
    @test sel.spacing_tu / D6.T_INT_MAX < 1                     # the ICs are NOT independent

    # 168 is what every second field yields, and the spacing lands within 0.3% of the nominal
    # 0.5 TU. It is not exactly 0.5: the selection includes both ends of the pool, so 168 indices
    # span 335 fields rather than 334 and every gap is 2 or 3 fields.
    s168 = D6.select_ics(; K = 168)
    @test s168.spacing_tu ≈ 0.5 atol = 2e-3
    @test first(s168.k) == 42
    @test last(s168.k) == 377
    @test all(d -> d in (2, 3), diff(s168.k))

    # `report_ics` must print the spacing and the warning, since that is the only place a reader
    # meets either number.
    io = IOBuffer()
    D6.report_ics(sel; io)
    out = String(take!(io))
    @test occursin("0.4679", out)
    @test occursin("0.5430", out)
    @test occursin("block bootstrap", out)
end

@testitem "V29 an impossible K fails loudly rather than silently returning fewer" default_imports = false setup = [D6] begin
    using Test
    # The pool is 336. 337 and 400 must both raise; 336 must not.
    @test length(D6.select_ics(; K = 336).k) == 336
    @test_throws ErrorException D6.select_ics(; K = 337)
    @test_throws ErrorException D6.select_ics(; K = 400)
    @test_throws ErrorException D6.select_ics(; K = 0)

    # The bounds themselves are re-derived, so a stale kmin/kmax is caught too.
    @test_throws ErrorException D6.select_ics(; K = 10, kmin = 41)   # t_41 = 10.0, not > 10
    @test_throws ErrorException D6.select_ics(; K = 10, kmax = 378)  # overruns the reference

    # 🔑 The regression this whole derivation exists for: `kmin`/`kmax` used to be literal 42/387,
    # correct only for nwarm = 100, nlead = 1208. Passing the OLD default against the CURRENT
    # constants must fail rather than forecast past the end of the truth.
    @test_throws ErrorException D6.select_ics(; K = 10, kmax = 387)

    # A longer forecast shrinks the pool further, and the default must track it.
    @test_throws ErrorException D6.select_ics(; K = 10, nlead = 3000, kmax = D6.default_kmax())
    @test last(D6.select_ics(; K = 10, nlead = 3000).n) + D6.N_WARM + 3000 <= 40000
end

@testitem "V29 selection is deterministic and disjoint from the archived runs' IC" default_imports = false setup = [D6] begin
    using Test
    @test D6.select_ics(; K = 180).k == D6.select_ics(; K = 180).k
    @test D6.select_ics(; K = 37).k == D6.select_ics(; K = 37).k

    # 🔑 The pilot's ICs are the first five of the same 180 the full run uses, so scaling
    # `--array=1-5` to `--array=1-180` renumbers nothing.
    @test D6.select_ics(; K = 180).k[1:5] == D6.select_ics(; K = 180).k[1:5]

    # V28: the IC set must be disjoint from every archived confirmatory run's IC, which is
    # `fields[1]` (`6_online_TO_LRS.jl:56-59`). It is, because field 1 is at t = 0, deep inside the
    # fit window -- but assert it, since that is the claim the comparison rests on.
    for K in (5, 180, 336)
        @test !(1 in D6.select_ics(; K).k)
    end
end

@testitem "V29 the warm-up slice reduces to the archived driver's at k = 1" default_imports = false setup = [D6] begin
    using Test
    # The reduction test. `6_online_TO_LRS.jl:62` is `dQ_data = data_track.dQ[:,1:100]`, and the
    # field it launches from is `fields[1]`, i.e. n = 0. A generalisation that does not reproduce
    # that exactly is wrong.
    @test D6.warmup_range(0, D6.N_WARM_DRIVER) == 1:100
    @test D6.warmup_range(0) == 1:D6.N_WARM          # the scored set's own, longer, warm-up
    @test collect(D6.warmup_range(0, D6.N_WARM_DRIVER)) == collect(1:100)

    # And the general form: step n_k, first solver step at n_k + 1.
    @test D6.warmup_range(4100, D6.N_WARM_DRIVER) == 4101:4200
    @test D6.warmup_range(4100) == 4101:(4100 + D6.N_WARM)
    @test D6.warmup_range(38600, D6.N_WARM_DRIVER) == 38601:38700
    for n in (0, 100, 4100, 17900, 38600)
        r = D6.warmup_range(n)
        @test length(r) == D6.N_WARM
        @test first(r) == n + 1
        @test last(r) <= D6.N_REF
    end

    # `q` carries the initial state, so step m is column m + 1 (`RikFlow.jl:294`). The scorer's
    # truth alignment rests on the same offset.
    @test D6.ic_q_column(0) == 1
    @test D6.ic_q_column(4100) == 4101
end

@testitem "V29 warm-up slices against the real record" default_imports = false setup = [D6, TrackedData] begin
    using Test
    rec = TrackedData.hitR1()
    if rec === nothing
        @test_skip "no extracted 100 TU tracked record under analysis/data/"
    else
        dQ, q = rec.dQ, rec.q
        @test size(dQ, 2) == D6.N_REF
        @test size(q, 2) == D6.N_REF + 1

        # The reduction, on the actual arrays rather than on index arithmetic.
        @test dQ[:, D6.warmup_range(0, D6.N_WARM_DRIVER)] == dQ[:, 1:100]

        sel = D6.select_ics(; K = 180)
        slices = [dQ[:, D6.warmup_range(n)] for n in sel.n]
        @test all(s -> size(s) == (size(dQ, 1), D6.N_WARM), slices)
        @test !any(s -> any(isnan, s), slices)
        # The forecast's last scored column exists for every IC.
        @test all(n -> n + D6.N_WARM + D6.N_LEAD + 1 <= size(q, 2), sel.n)
        # The package's `q_at_ic` is a real column, not one past the end.
        @test all(n -> all(isfinite, q[:, D6.ic_q_column(n)]), sel.n)
    end
end

@testitem "V29 the step-to-column convention, measured from the record" default_imports = false setup = [D6, TrackedData] begin
    using Test
    using Statistics
    rec = TrackedData.hitR1()
    if rec === nothing
        @test_skip "no extracted 100 TU tracked record under analysis/data/"
    else
        # 🔑 `ic_q_column` and, later, the scorer's truth alignment both rest on "step m is column
        # m + 1 of `q`", because `qoisaver` fires on the initial state (`RikFlow.jl:294`). That is
        # an assertion about the record, so it is measured against the record rather than
        # re-derived from the source: `q_star[:, m] + dQ[:, m]` is the corrected QoI at step m, and
        # the column of `q` holding it must be the winner by a wide margin.
        #
        # It does not match exactly, and should not: `analysis/results.md` phase-0 check 0.4
        # measured an O(||sgs||^2) gap of 1.89e-2 relative between `q_star + dQ` and the recomputed
        # QoIs. The test is therefore comparative, not absolute.
        q, qs, dQ = rec.q, rec.q_star, rec.dQ
        m = 1000:39000
        err(off) = vec(sqrt.(mean(abs2, (qs[:, m] .+ dQ[:, m]) .- q[:, m .+ off], dims = 2)) ./
                       sqrt.(mean(abs2, q[:, m .+ off], dims = 2)))
        e0, e1, e2 = err(0), err(1), err(2)
        # Measured margin, per QoI: 5.4x on the coarsest enstrophy band up to 400x on the
        # smallest-scale ones. The factor asserted is deliberately below the smallest of those.
        @test all(e1 .< e0 ./ 4)
        @test all(e1 .< e2 ./ 4)
        @test maximum(e1) < 1e-3
    end
end

@testitem "V29 built IC packages agree with the record and with select_ics" default_imports = false setup = [D6, TrackedData] begin
    using Test
    using JLD2
    rec = TrackedData.hitR1()
    mpath = D6.manifest_path()
    if rec === nothing || !isfile(mpath)
        @test_skip "IC packages not built yet (run analysis/build_d6_ics.jl)"
    else
        man = load(mpath)
        sel = D6.select_ics(; K = man["K"])
        @test man["k"] == sel.k
        @test man["n"] == sel.n
        @test man["spacing_tu"] ≈ sel.spacing_tu

        # Spot-check the ends and the middle rather than all 180 files.
        for i in unique((1, 2, cld(man["K"], 2), man["K"]))
            k = man["k"][i]
            p = D6.ic_path(k)
            @test isfile(p)
            d = load(p)
            @test d["k"] == k
            @test d["ordinal"] == i
            @test d["n_k"] == sel.n[i]
            @test d["t_k"] == sel.t[i]
            @test size(d["u"]) == (66, 66, 66, 3)
            @test !any(isnan, d["u"])
            @test size(d["dQ_warm"]) == (size(rec.dQ, 1), D6.N_WARM)
            @test d["dQ_warm"] == rec.dQ[:, D6.warmup_range(sel.n[i])]
            @test d["q_at_ic"] == rec.q[:, D6.ic_q_column(sel.n[i])]
            # The parameter subset is plain data and carries the forcing the replay needs.
            @test keys(d["params"]) == D6.PARAM_KEYS
            @test d["params"].ou_bodyforce.rng_seed == 333
            @test d["params"].Δt ≈ 2.5e-3          # R1 is Float64; the archive was Float32
        end
    end
end

@testitem "V29 the validation IC is R2's own inputs, and stays out of the scored set" default_imports = false setup = [D6, TrackedData] begin
    using Test
    using JLD2
    # 🔑 Ordinal 0 exists to check the D6 path against a run whose answer is already known: same
    # initial condition, so `n_k = 0` and `ou_advance = 0` -- the identity point of the replay --
    # and the driver's own model seeds. For that to mean anything the package must carry that run's
    # inputs exactly, and it must never leak into anything scored.
    #
    # 🔴 **The oracle moved from paper 2's archive to R2's own LinReg1 replica 1 on 2026-09-16.**
    # The archive is a different dynamical system (pre-`09954be1`, Float32, its own reference), so a
    # failed reproduction against it could not distinguish a driver bug from the system difference
    # -- which is why `validation_verdict` needed a `Z[16,32]` carve-out. R2's replica 1 shares the
    # record, the precision, the Nyquist convention and the solver, so the carve-out is gone too.
    @test D6.DRIVER_SEED_BASE == 236
    @test D6.DRIVER_SEED_BASE == 234 + 2          # `Xoshiro(seeds.to + i + 2)`, seeds.to = 234

    # It is not in the selection, at any K, and not in the manifest. Both must hold: `t_1 = 0` is
    # inside M0's fit window, and V28 needs the scored set disjoint from the online runs' own IC.
    for K in (5, 180, 336)
        @test !(1 in D6.select_ics(; K).k)
    end
    @test D6.validation_path() != D6.ic_path(1)     # separate filenames, not just separate indices

    p = D6.validation_path()
    rec = TrackedData.hitR1()
    if !isfile(p) || rec === nothing
        @test_skip "validation IC not built (analysis/build_d6_ics.jl validation)"
    else
        d = load(p)
        @test d["validation"] === true
        @test d["k"] == 1
        @test d["ordinal"] == 0
        @test d["n_k"] == 0
        @test d["t_k"] == 0.0
        @test d["driver_seed_base"] == D6.DRIVER_SEED_BASE
        @test size(d["u"]) == (66, 66, 66, 3)
        @test !any(isnan, d["u"])
        @test d["provenance"].nwarm == D6.N_WARM_DRIVER   # the driver's, not the scored set's
        @test d["provenance"].nlead == D6.N_LEAD

        # 🔴 Built from **R1**, the same record the scored ICs come from and the one R2's online
        # runs launched from. Neither archive record may appear here: `data_track2` is paper 2's
        # 100 TU record and `tsim10.0` its 10 TU one, and both are the pre-merge system.
        @test occursin("f64_lmwray3", d["provenance"].source)
        @test !occursin("data_track2", d["provenance"].source)
        @test !occursin("tsim10.0", d["provenance"].source)

        # And the two inputs R2's driver actually consumed, bit-for-bit.
        @test size(d["dQ_warm"]) == (size(rec.dQ, 1), D6.N_WARM_DRIVER)
        @test d["dQ_warm"] == rec.dQ[:, D6.warmup_range(0, D6.N_WARM_DRIVER)]
        @test d["dQ_warm"] == rec.dQ[:, 1:100]     # the online driver's literal slice
        @test d["q_at_ic"] == rec.q[:, D6.ic_q_column(0)]
        @test d["q_at_ic"] == rec.q[:, 1]
    end
end

@testitem "V29 the forecast length and the record's grid are what the plan says" default_imports = false setup = [D6] begin
    using Test
    # These constants are quoted in the handoff, in `metrics.md` section 5 and in the run driver, and
    # a change to any of them silently changes what D6 measures. Pin them here so the change has to
    # be deliberate.
    # 🔴 Reset 2026-09-15 from the LEVEL's decorrelation time, not the correction's (Rik). D6 scores
    # the forecast of the level, so the level's timescale is what has to saturate.
    @test D6.N_LEAD == 2172
    @test D6.N_LEAD * D6.FIELD_DT / D6.FIELD_STRIDE ≈ 5.43 atol = 1e-9   # 2172 steps = 5.43 TU
    @test D6.N_LEAD * 2.5e-3 ≈ 10 * D6.T_INT_MAX atol = 5e-3             # 10 x the slowest T(q)
    @test D6.T_INT_MAX ≈ 0.5430 atol = 1e-4                              # the LEVEL's, on R1
    @test D6.N_WARM == 220
    @test D6.N_WARM * 2.5e-3 ≈ D6.T_INT_MAX atol = 1e-2                  # ~1 x the slowest T(q)
    @test D6.N_WARM_DRIVER == 100                                        # what the drivers replay
    @test D6.N_REF == 40000
    @test D6.N_FIELDS == 401
    @test D6.FIELD_STRIDE == 100
    @test D6.FIELD_DT == 0.25
    @test D6.FIT_END_TU == 10.0

    # 🔑 The step size the OU replay depends on. `online_sgs` refuses to replay unless `tsim / Δt` is
    # integral, because `solve_unsteady` re-derives `Δt = (tend - tstart) / nstep` and stepping the
    # replay at a different Δt would reintroduce the misphase. Both the reference's 100 TU and D6's
    # 3.27 TU have to land on the same Float32.
    nsteps = D6.N_WARM + D6.N_LEAD
    @test nsteps == D6.N_WARM + D6.N_LEAD
    @test Float32(100.0) / 40000 === Float32(2.5e-3)
    @test Float32(nsteps * 2.5e-3) / nsteps === Float32(2.5e-3)
    @test round(Int, Float32(nsteps * 2.5e-3) / Float32(2.5e-3)) == nsteps
end
