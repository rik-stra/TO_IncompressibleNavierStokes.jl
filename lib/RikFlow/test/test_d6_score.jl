# V28 -- the lead-resolved scorer, and its synthetic control.
#
# P2c measures the surrogate, not the harness. Everything here is a case whose answer is known
# before the code runs: an ensemble drawn from the same law as the truth must be flat at *every*
# lead and read a corrected ratio of 1; a deliberately under-dispersed one must read below 1 with
# positive convexity; and the index arithmetic is checked by planting step numbers as values rather
# than by re-deriving the offsets in the test, which would only confirm whichever convention was
# assumed.
#
# ⚠️ `plan.md` section 14 files V28 under `test_rollout.jl`. That file does not exist -- V24, the
# rollout check it was to share, is still TODO -- so V28 lives here, next to the code it tests and
# beside V29's `test_d6_ics.jl`.

@testmodule D6Score begin
    const SRC = normpath(joinpath(@__DIR__, "..", "analysis", "score_d6.jl"))
    include(SRC)

    """
        exchangeable(K, nq, M, L; sigma_member = 1.0, rng)

    The V28 null: at every lead, the `M` members and the truth are `M+1` exchangeable draws around
    a common per-(IC, QoI, lead) signal. Ranks are then marginally uniform by construction and the
    corrected spread-skill ratio is 1 in expectation.

    `sigma_member` scales the member spread only, leaving the truth alone: below 1 gives a
    deliberately under-dispersed ensemble, above 1 an over-dispersed one.

    `signal_sd = 0` removes the shared signal, which makes the control **saturated** as well as
    calibrated: members and truth become independent draws from one marginal, so the skill sits
    exactly at the climatological level. With a signal present the control is calibrated and still
    skilful, which is the case that must report *no* saturation.
    """
    function exchangeable(K, nq, M, L; sigma_member = 1.0, signal_sd = 3.0,
                          rng = Random.default_rng())
        signal = randn(rng, K, nq, L) .* signal_sd
        fc = Array{Float64}(undef, K, nq, M, L)
        tr = Array{Float64}(undef, K, nq, L)
        for k in 1:K, i in 1:nq, j in 1:L
            for m in 1:M
                fc[k, i, m, j] = signal[k, i, j] + sigma_member * randn(rng)
            end
            tr[k, i, j] = signal[k, i, j] + randn(rng)
        end
        return fc, tr
    end

    """
        write_run(dir, k, n_k, member; nwarm, nt, nq = 6)

    One synthetic D6 output file whose `q` and `dQ` carry **step indices as values**, so that any
    misalignment in `assemble` shows up as a wrong number rather than as a plausible one.

    `q[:, c] = c - 1` (column `c` is run step `c-1`, the initial-state offset) and `dQ[:, c] = c`
    (no initial-state column).
    """
    function write_run(dir, k, n_k, member; nwarm, nt, nq = 6)
        q = Float32[c - 1 for _ in 1:nq, c in 1:(nt + 1)]
        dQ = Float32[c for _ in 1:nq, c in 1:nt]
        p = joinpath(dir, "d6_online_ic$(k)_m$(member).jld2")
        jldsave(p; q, dQ, tau = dQ, k, n_k, t_k = 0.25 * (k - 1), ordinal = 1, member,
                seed = UInt64(member), ou_advance = n_k, nwarm, nlead = nt - nwarm, M = 2)
        return p
    end

    "A reference whose columns also carry their own step index, for the same reason."
    planted_truth(nq, ncol) = (; q = Float64[c - 1 for _ in 1:nq, c in 1:ncol],
                               dQ = Float64[c for _ in 1:nq, c in 1:(ncol - 1)],
                               source = "planted")
end

@testitem "V28 truth alignment, by planted step indices" default_imports = false setup = [D6Score] begin
    using Test
    # Run step s is tracked step n_k + s; the forecast's t = 0 is at run step nwarm; and `q` carries
    # the initial state, so step m is column m+1. Composed: lead ℓ verifies against reference column
    # n_k + nwarm + ℓ + 1.
    @test D6Score.forecast_column(0; nwarm = 100) == 101
    @test D6Score.forecast_column(7; nwarm = 100) == 108
    @test D6Score.truth_column(4100, 0; nwarm = 100) == 4201
    @test D6Score.truth_column(4100, 7; nwarm = 100) == 4208

    # `dQ` has no initial-state column, so both offsets drop by one. Kept as separate functions
    # precisely so a shared helper cannot misalign the secondary metric silently.
    @test D6Score.forecast_column_dq(0; nwarm = 100) == 100
    @test D6Score.truth_column_dq(4100, 0; nwarm = 100) == 4200
    @test D6Score.truth_column(4100, 3) - D6Score.truth_column_dq(4100, 3) == 1
    @test D6Score.forecast_column(3) - D6Score.forecast_column_dq(3) == 1

    # Planted values: a record whose column c holds step c-1 must return the step itself.
    nq, ncol = 6, 40001
    ref = D6Score.planted_truth(nq, ncol)
    for (n_k, ℓ) in ((4100, 0), (4100, 1207), (38600, 33), (17900, 241))
        # The scored set's warm-up was reset from 100 to 220 when the forecast length moved to
        # the LEVEL's decorrelation time (2026-09-15), and these columns default to it. A literal
        # 100 here would test the alignment against a warm-up nothing runs.
        @test ref.q[1, D6Score.truth_column(n_k, ℓ)] == n_k + D6Score.N_WARM + ℓ
        @test ref.dQ[1, D6Score.truth_column_dq(n_k, ℓ)] == n_k + D6Score.N_WARM + ℓ
    end
end

@testitem "V28 assemble resolves both axes, and refuses to clip" default_imports = false setup = [D6Score] begin
    using Test
    using JLD2
    nq, nwarm, nlead = 6, 100, 1208
    nt = nwarm + nlead
    ref = D6Score.planted_truth(nq, 40001)
    grid = [0, 1, 33, 447, 1207]

    mktempdir() do dir
        for (k, n_k) in ((42, 4100), (44, 4300)), m in 1:2
            D6Score.write_run(dir, k, n_k, m; nwarm, nt, nq)
        end
        ens = D6Score.load_members(dir)
        @test ens.ks == [42, 44]
        @test ens.M == 2

        fc, tr = D6Score.assemble(ens, ref, grid)
        @test size(fc) == (2, nq, 2, length(grid))
        @test size(tr) == (2, nq, length(grid))
        # The forecast column carries the run step; the truth column carries the tracked step.
        for (j, ℓ) in pairs(grid)
            @test all(fc[:, :, :, j] .== nwarm + ℓ)
            @test tr[1, 1, j] == 4100 + nwarm + ℓ
            @test tr[2, 1, j] == 4300 + nwarm + ℓ
        end

        fcd, trd = D6Score.assemble(ens, ref, grid; level = false)
        for (j, ℓ) in pairs(grid)
            @test all(fcd[:, :, :, j] .== nwarm + ℓ)
            @test trd[1, 1, j] == 4100 + nwarm + ℓ
        end

        # A lead the run does not reach, and one the reference does not reach. Both must throw:
        # a clipped lead reads as a saturated one, which is the thing being measured.
        @test_throws ErrorException D6Score.assemble(ens, ref, [nlead + 1])
        short = (; q = ref.q[:, 1:4300], dQ = ref.dQ[:, 1:4300], source = "short")
        @test_throws ErrorException D6Score.assemble(ens, short, [1207])
    end
end

@testitem "V28 the burn-in offset must be identical across members of one IC" default_imports = false setup = [D6Score] begin
    using Test
    using JLD2
    nq = 6
    ref = D6Score.planted_truth(nq, 40001)
    mktempdir() do dir
        D6Score.write_run(dir, 42, 4100, 1; nwarm = 100, nt = 1308, nq)
        D6Score.write_run(dir, 42, 4100, 2; nwarm = 90, nt = 1308, nq)   # wrong on purpose
        ens = D6Score.load_members(dir)
        @test_throws ErrorException D6Score.assemble(ens, ref, [0, 5])
    end
    # And a disagreement on the initial condition itself.
    mktempdir() do dir
        D6Score.write_run(dir, 42, 4100, 1; nwarm = 100, nt = 1308, nq)
        D6Score.write_run(dir, 42, 4300, 2; nwarm = 100, nt = 1308, nq)
        @test_throws ErrorException D6Score.assemble(D6Score.load_members(dir), ref, [0])
    end
    # A ragged ensemble is refused at load: the finite-M correction is a function of M.
    mktempdir() do dir
        D6Score.write_run(dir, 42, 4100, 1; nwarm = 100, nt = 1308, nq)
        D6Score.write_run(dir, 42, 4100, 2; nwarm = 100, nt = 1308, nq)
        D6Score.write_run(dir, 44, 4300, 1; nwarm = 100, nt = 1308, nq)
        @test_throws ErrorException D6Score.load_members(dir)
    end
    @test D6Score.load_members(mktempdir()) === nothing

    # 🔑 The validation run (ordinal 0) must be invisible to the scorer's glob. It is the archived
    # runs' own IC, inside M0's fit window, so including it would put an IC the conditional mean has
    # seen into the rank histograms and break V28's disjointness. The separation is by filename --
    # `d6_valid_*` against `d6_online_*` -- so that no filtering step can be forgotten.
    mktempdir() do dir
        D6Score.write_run(dir, 42, 4100, 1; nwarm = 100, nt = 1308, nq = 6)
        D6Score.write_run(dir, 42, 4100, 2; nwarm = 100, nt = 1308, nq = 6)
        # a validation run beside them, named as `run_d6.jl` names it
        src = joinpath(dir, "d6_online_ic42_m1.jld2")
        cp(src, joinpath(dir, "d6_valid_ic1_m1.jld2"))
        cp(src, joinpath(dir, "d6_valid_ic1_m2.jld2"))
        ens = D6Score.load_members(dir)
        @test ens.ks == [42]          # not [1, 42]
        @test ens.M == 2              # and it did not become a ragged 4-member ensemble
    end
end

@testitem "V28 lead grids are per QoI, in physical time, and inside the run" default_imports = false setup = [D6Score] begin
    using Test
    leads = D6Score.lead_grid(D6Score.T_INT; dt = D6Score.DT, nlead = D6Score.N_LEAD)
    @test length(leads) == 6
    @test all(g -> issorted(g) && allunique(g) && all(>=(1), g), leads)
    @test all(g -> maximum(g) <= D6Score.N_LEAD, leads)

    # 🔴 The spread COLLAPSED when the grid moved from the correction's timescales to the level's
    # (2026-09-16). On `dQ` the six bands spanned a factor 36.8 and that was the whole argument for
    # a per-QoI grid; on `q` they span only 0.5430/0.2489 = 2.18. The per-QoI grid is kept because
    # it is still correct and costs nothing, but it is no longer load-bearing -- and if someone
    # later proposes one shared grid, this is the number that says it would now be defensible.
    @test maximum(leads[5]) == 2172                       # Z[16,32], 10 x 0.5430 TU = N_LEAD
    @test maximum(leads[6]) == 2158                       # E[16,32], 10 x 0.5395 TU
    @test maximum(leads[1]) == 996                        # Z[0,6],   10 x 0.2489 TU, the fastest
    @test maximum(leads[5]) / maximum(leads[1]) < 2.5     # was > 30 on the correction

    # The longest lead is what sets the forecast length: it fits, and one more multiple would not.
    @test D6Score.lead_grid([0.5430]; dt = D6Score.DT, nlead = 2172) isa Vector
    @test_throws ArgumentError D6Score.lead_grid([0.5430]; dt = D6Score.DT, nlead = 2171)
    @test_throws ArgumentError D6Score.lead_grid([0.5430]; dt = D6Score.DT, multipliers = (20,),
                                                 nlead = D6Score.N_LEAD)
    @test_throws ArgumentError D6Score.lead_grid([0.0]; dt = D6Score.DT, nlead = 100)

    # The union is what the forecast array is built on, and a lead off it is refused rather than
    # matched to a neighbour.
    g = D6Score.union_grid(leads)
    @test issorted(g) && allunique(g)
    @test all(gl -> all(in(g), gl), leads)
    @test D6Score.lead_positions(g, [[first(g), last(g)]]) == [[1, length(g)]]
    @test_throws ArgumentError D6Score.lead_positions(g, [[last(g) + 1]])
end

@testitem "V28 the finite-M correction reads 1 on the control and 0.953 without it" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    # A reliable ensemble satisfies E[RMSE^2] = ((M+1)/M) E[spread^2], so an **uncorrected** ratio
    # reads sqrt(M/(M+1)) on a perfect ensemble: 0.953 at M = 10. Both are asserted, so the
    # correction cannot be quietly dropped and the artefact mistaken for over-confidence.
    rng = Xoshiro(4242)
    M = 10
    fc, tr = D6Score.exchangeable(4000, 2, M, 2; rng)
    grid = [3, 11]

    on = D6Score.spread_skill_by_lead(fc, tr; grid)
    off = D6Score.spread_skill_by_lead(fc, tr; grid, correct = false)
    @test on.correction ≈ sqrt((M + 1) / M)
    @test off.correction == 1.0
    for i in 1:2, t in 1:2
        @test on.ratio[i][t] ≈ 1.0 atol = 0.05
        @test off.ratio[i][t] ≈ sqrt(M / (M + 1)) atol = 0.05
        @test on.ratio[i][t] / off.ratio[i][t] ≈ sqrt((M + 1) / M) rtol = 1e-12
    end
    # The dimensionless pooled ratio agrees with the per-QoI ones on an exchangeable control.
    @test all(r -> isapprox(r, 1.0; atol = 0.05), on.pooled_ratio)
end

@testitem "V28 under-dispersion reads below 1 with positive convexity, over-dispersion above" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    grid = [1, 9]
    for (sig, want_ratio_lt_1, want_convexity_positive) in ((0.5, true, true), (2.0, false, false))
        rng = Xoshiro(77)
        fc, tr = D6Score.exchangeable(600, 2, 10, 2; sigma_member = sig, rng)
        ss = D6Score.spread_skill_by_lead(fc, tr; grid)
        rh = D6Score.rank_histogram_by_lead(fc, tr; grid, rng = Xoshiro(78), nboot = 100)
        for i in 1:2, t in 1:2
            @test (ss.ratio[i][t] < 1) == want_ratio_lt_1
            @test (rh.hist[i][t].convexity > 0) == want_convexity_positive
        end
    end
end

@testitem "V28 the control is flat at every lead, as coverage over replications" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    # 🔑 Tested as coverage across replications, not on one seed. A 95% percentile interval misses
    # zero about 5% of the time by construction, so a single-draw assertion is a coin flip dressed
    # as a test -- an earlier round of this suite made exactly that mistake.
    R, K, M, nq = 30, 150, 10, 2
    grid = [2, 40, 400]
    cover_slope = zeros(Int, length(grid))
    cover_conv = zeros(Int, length(grid))
    for r in 1:R
        rng = Xoshiro(1000 + r)
        fc, tr = D6Score.exchangeable(K, nq, M, length(grid); rng)
        rh = D6Score.rank_histogram_by_lead(fc, tr; grid, rng, nboot = 150)
        for t in eachindex(grid)
            h = rh.hist[1][t]
            h.slope_ci[1] <= 0 <= h.slope_ci[2] && (cover_slope[t] += 1)
            h.convexity_ci[1] <= 0 <= h.convexity_ci[2] && (cover_conv[t] += 1)
        end
    end
    @info "V28 flatness coverage over $R replications" cover_slope = cover_slope ./ R cover_conv = cover_conv ./ R
    # Nominal is 0.95. The floor is loose because R = 30 gives a standard error of about 0.04, and
    # because the block bootstrap is conservative-to-anticonservative depending on the rank series.
    for t in eachindex(grid)
        @test cover_slope[t] / R >= 0.75
        @test cover_conv[t] / R >= 0.75
    end
end

@testitem "V28 saturation is reported, never extrapolated" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    # The level a calibrated ensemble converges to once members and truth are independent draws
    # from the same marginal: sigma * sqrt(1 + 1/M).
    rng = Xoshiro(9)
    x = randn(rng, 20000) .* 2.5
    @test D6Score.climatological_skill(x, 10) ≈ 2.5 * sqrt(1 + 1 / 10) rtol = 0.03
    @test isnan(D6Score.climatological_skill([1.0], 10))

    leads = [1, 10, 100, 1000]
    # Skill growing towards, and reaching, the climatological level.
    @test D6Score.saturation_lead(leads, [0.1, 0.5, 0.95, 1.0]; sat_level = 1.0) == 100
    @test D6Score.saturation_lead(leads, [0.1, 0.5, 0.90, 0.94]; sat_level = 1.0) === nothing
    @test D6Score.saturation_lead(leads, [0.1, 0.5, 0.90, 0.94]; sat_level = 1.0, frac = 0.9) == 100
    # 🔴 `nothing` is the answer when the grid never gets there. It is reported, and no curve is
    # fitted through the grid to read a crossing off it.
    @test D6Score.saturation_lead(leads, fill(0.1, 4); sat_level = 1.0) === nothing
    @test D6Score.saturation_lead(leads, fill(0.1, 4); sat_level = NaN) === nothing
    @test_throws DimensionMismatch D6Score.saturation_lead(leads, [0.1]; sat_level = 1.0)

    # Two controls, and the contrast between them is the test.
    #
    # `signal_sd = 0`: members and truth are independent draws from one marginal, so the ensemble is
    # saturated at every lead and every QoI must report the first lead on the grid.
    #
    # `signal_sd = 3`: the same construction with a shared signal is just as *calibrated* -- flat
    # ranks, ratio 1 -- but has real conditional skill, so it must report **no** saturation. A
    # scorer that confused calibration with saturation would pass the first and fail the second.
    grid = [1, 5, 25]
    nq, M = 2, 10
    pooled(tr) = reshape(permutedims(tr, (2, 1, 3)), size(tr, 2), size(tr, 1) * size(tr, 3))

    fc0, tr0 = D6Score.exchangeable(3000, nq, M, 3; signal_sd = 0.0, rng = Xoshiro(11))
    ss0 = D6Score.spread_skill_by_lead(fc0, tr0; grid)
    sat0, lev0 = D6Score.saturation(ss0, pooled(tr0), M)
    @test all(s -> s !== nothing, sat0)
    @test all(==(first(grid)), sat0)
    @test all(i -> isapprox(ss0.skill[i][1], lev0[i]; rtol = 0.05), 1:nq)

    fcs, trs = D6Score.exchangeable(3000, nq, M, 3; signal_sd = 3.0, rng = Xoshiro(12))
    sss = D6Score.spread_skill_by_lead(fcs, trs; grid)
    sats, levs = D6Score.saturation(sss, pooled(trs), M)
    @test all(s -> s === nothing, sats)
    @test all(i -> sss.skill[i][1] < 0.5 * levs[i], 1:nq)
end

@testitem "V28 ties and the clamp are counted exactly, not inferred" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    using JLD2

    # Trap 7. `metrics.md` #4 says the stabiliser "produces exact duplicates by construction", but
    # the clamp has been measured never to fire on HIT (min |q*| = 2.19e-2 against a 1e-2
    # threshold), so ties should be absent. Seeded tie-breaking stays as insurance; the count is
    # reported, because a non-zero count would mean that census is wrong.
    fc, tr = D6Score.exchangeable(200, 2, 5, 1; rng = Xoshiro(3))
    @test D6Score.tie_count(fc, tr, 1, 1) == 0            # continuous draws never tie
    fc[7, 1, 3, 1] = tr[7, 1, 1]                          # plant one
    fc[9, 1, 1, 1] = tr[9, 1, 1]
    @test D6Score.tie_count(fc, tr, 1, 1) == 2
    @test D6Score.tie_count(fc, tr, 2, 1) == 0            # and it is per QoI

    # The clamp fires by zeroing the whole `dQ` column, so a fired step is exactly an all-zero
    # column -- no `q_star` needed, which matters because `online_sgs` does not return one.
    # Warm-up columns are excluded: there `dQ` is replayed from the record, not predicted.
    mktempdir() do dir
        nwarm, nt = 100, 400
        for m in 1:2
            D6Score.write_run(dir, 42, 4100, m; nwarm, nt, nq = 6)
        end
        ens = D6Score.load_members(dir)
        cl = D6Score.clamp_report(ens)
        @test cl.nsteps == 2 * (nt - nwarm)
        @test cl.nfired == 0                              # `write_run` plants dQ[:, c] = c, never 0
        @test cl.rate == 0.0
        @test cl.per_run == [0, 0]
    end
    mktempdir() do dir
        nwarm, nt, nq = 100, 400, 6
        q = Float32[c - 1 for _ in 1:nq, c in 1:(nt + 1)]
        dQ = Float32[c for _ in 1:nq, c in 1:nt]
        dQ[:, 250] .= 0                                   # one fired step, after the warm-up
        dQ[:, 50] .= 0                                    # one inside it, which must NOT count
        jldsave(joinpath(dir, "d6_online_ic42_m1.jld2"); q, dQ, tau = dQ, k = 42, n_k = 4100,
                t_k = 10.25, ordinal = 1, member = 1, seed = UInt64(1), ou_advance = 4100,
                nwarm, nlead = nt - nwarm, M = 1)
        cl = D6Score.clamp_report(D6Score.load_members(dir))
        @test cl.nfired == 1
        @test cl.nsteps == nt - nwarm
    end
end

@testitem "V28 the whole driver runs, on synthetic members" default_imports = false setup = [D6Score, TrackedData] begin
    using Test
    using JLD2
    using Random
    # 🔑 Without this, `main` would first execute on pilot data -- i.e. after GPU time had been
    # spent. Everything here is synthetic and the numbers mean nothing; what is tested is that the
    # path from files on disk to `d6_scores.jld2` completes, on the real lead grid and the real
    # reference, with the real column arithmetic.
    if TrackedData.load_cache("data_track2") === nothing ||
       !isfile(joinpath(D6Score.HERE, "data", "hf_reference_tsim100.0_qois.jld2"))
        @test_skip "extracted records not present under analysis/data/"
    else
        rng = Xoshiro(31)
        nwarm, nlead, nq = D6Score.N_WARM, D6Score.N_LEAD, 6
        nt = nwarm + nlead
        sel = D6Score.select_ics(; K = 180)
        mktempdir() do dir
            for i in 1:6, m in 1:3
                k, n_k = sel.k[i], sel.n[i]
                q = randn(rng, Float32, nq, nt + 1) .* 10.0f0
                dQ = randn(rng, Float32, nq, nt)
                jldsave(joinpath(dir, "d6_online_ic$(k)_m$(m).jld2");
                        q, dQ, tau = dQ, k, n_k, t_k = sel.t[i], ordinal = i, member = m,
                        seed = UInt64(m), ou_advance = n_k, nwarm, nlead, M = 3)
            end
            mktempdir() do od
                out = D6Score.main(; dir, outdir = od, io = devnull)
                @test out !== nothing
                @test Set(keys(out)) == Set([:level, :correction])
                p = joinpath(od, "d6_scores.jld2")
                @test isfile(p)
                d = load(p)
                @test d["labels"] == D6Score.LABELS
                @test d["truth"] == "hf_reference"
                @test length(d["ics"]) == 6
                @test d["clamp_nfired"] == 0
                for tag in ("level", "correction")
                    r = d[tag]
                    @test r.K == 6 && r.M == 3
                    @test r.correction ≈ sqrt(4 / 3)
                    @test length(r.ratio) == nq
                    @test all(i -> length(r.ratio[i]) == length(r.leads[i]), 1:nq)
                    @test all(i -> length(r.counts[i]) == length(r.leads[i]), 1:nq)
                    @test all(c -> sum(c) == 6, r.counts[1])       # every IC lands in a bin
                    @test all(c -> length(c) == 4, r.counts[1])    # M + 1 bins at M = 3
                end
                # Members drawn independently of the truth have no skill, so nothing saturates
                # below the climatological level in the wrong direction: the reported saturation is
                # either a real lead or the -1 sentinel, never a silent zero.
                @test all(s -> s == -1 || s in D6Score.union_grid(D6Score.lead_grid(
                             D6Score.T_INT; dt = D6Score.DT, nlead = D6Score.N_LEAD)),
                          out[:level].saturation)
            end
        end
    end
end

@testitem "V28 the scorer refuses mismatched layouts" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    fc, tr = D6Score.exchangeable(50, 3, 4, 2; rng = Xoshiro(5))
    @test_throws DimensionMismatch D6Score.spread_skill_by_lead(fc, tr; grid = [1])
    @test_throws DimensionMismatch D6Score.spread_skill_by_lead(fc, tr[:, :, 1:1]; grid = [1, 2])
    @test_throws ArgumentError D6Score.spread_skill_by_lead(fc, tr; grid = [2, 1])
    @test_throws DimensionMismatch D6Score.spread_skill_by_lead(fc, tr; grid = [1, 2],
                                                                leads = [[1], [2]])
    @test_throws DimensionMismatch D6Score.rank_histogram_by_lead(fc, tr; grid = [1])
end

# V32 -- the validation criterion. Added 2026-09-11, after the first real validation run.
#
# The criterion this replaces was `max` per-QoI relative rms over all 1309 columns `< 1e-2`, and it
# reported a correct run as a failure. Two facts killed it, and both are asserted here so the
# criterion cannot drift back:
#
#   * that statistic **saturates**. Two draws from the same stationary law are √2 sd apart on it,
#     whatever the model does, and two archived LinReg1 replicas measured 1.00 (range 0.65-1.55).
#     A `1e-2` threshold on it therefore demands near-bit-identity from a chaotic, stochastic,
#     Float32 system.
#   * its premise -- "every input is the archive's, so the output must be too" -- is false after
#     commit `09954be1`: `∂` feeds `get_vi_functions`, so `tau` changed and the run is not the same
#     dynamical system as the archive (`claude_memory.md` gotchas #45, #46).
#
# The power that remains is in the replayed warm-up window, and the exact part of it -- `dQ`
# bit-identity -- was never asserted at all before this.
@testitem "V32 the validation gate is the replayed warm-up, and dQ there is exact" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    using Statistics

    nq, nwarm, n = 6, 100, 1309
    rng = Xoshiro(20260911)
    base = cumsum(randn(rng, nq, n); dims = 2) .+ 10 .* randn(rng, nq)
    dqa = randn(rng, nq, n - 1)
    sd = vec(std(base; dims = 2))

    # 1. A perfect reproduction passes and never diverges.
    v = D6Score.validation_verdict(copy(base), base, copy(dqa), dqa, nwarm)
    @test v.ok && v.dq_identical
    @test v.gate == 0
    @test v.diverge_col === nothing
    @test v.nwarm == nwarm && v.n == n

    # 2. dQ inside the warm-up is emitted verbatim from `spinnup_data`, so a single perturbed
    #    element there is a defect however small -- the gate on `q` cannot see it, and that is
    #    exactly the case the old criterion had no assertion for.
    dqbad = copy(dqa)
    dqbad[3, 42] = nextfloat(dqbad[3, 42])
    v = D6Score.validation_verdict(copy(base), base, dqbad, dqa, nwarm)
    @test !v.dq_identical
    @test !v.ok
    @test v.gate == 0            # q is still perfect; only the exact half fired

    # 3. A perturbation present from column 1 -- a wrong slice, a misphased chain, a seed mismatch --
    #    fails the gate.
    qbad = copy(base)
    qbad[1, :] .+= 0.1 * sd[1]
    v = D6Score.validation_verdict(qbad, base, copy(dqa), dqa, nwarm)
    @test !v.ok
    @test v.gate > 1e-2

    # 4. 🔴 REVERSED 2026-09-16. A perturbation confined to Z[16,32] must now FAIL.
    #
    #    It used to pass: the oracle was paper 2's archive, where Z[16,32] is a different quantity
    #    than it is today (gotcha #45), so the band had to be carved out of the verdict. The oracle
    #    is now R2's own LinReg1 replica 1 -- same record, same precision, same Nyquist convention,
    #    same solver -- so there is no code difference left to excuse and nothing is excluded.
    #    A carve-out that outlives its reason is a blind spot in exactly one band.
    qz = copy(base)
    qz[D6Score.IZ1632, :] .+= 0.1 * sd[D6Score.IZ1632]
    v = D6Score.validation_verdict(qz, base, copy(dqa), dqa, nwarm)
    @test !v.ok
    @test v.gate > 1e-2
    @test v.rel_full[D6Score.IZ1632] > 1e-2

    #    The carve-out is still reachable for a deliberate archive comparison, and then it passes.
    vx = D6Score.validation_verdict(qz, base, copy(dqa), dqa, nwarm; iexcl = D6Score.IZ1632)
    @test vx.ok

    # 5. Agreement over the warm-up followed by separation -- the real run's shape -- passes the
    #    gate and reports where it parted company, past `nwarm`.
    qdiv = copy(base)
    qdiv[:, (nwarm + 5):end] .+= 3 .* sd
    v = D6Score.validation_verdict(qdiv, base, copy(dqa), dqa, nwarm)
    @test v.ok
    @test v.diverge_col == nwarm + 5
    @test maximum(v.rel_full) > 1.0              # the full-window rms is saturated ...
    @test maximum(v.rel_warm) == 0               # ... while the window with an answer is exact
end

@testitem "V32 the full-window rms saturates, which is why it is not the criterion" default_imports = false setup = [D6Score] begin
    using Test
    using Random
    using Statistics

    # Two independent draws from one stationary law: no model, no bug, nothing shared but the law.
    # Their relative rms is √2 by construction, so any threshold below that on this statistic is
    # unreachable no matter how correct the code is.
    nq, n, rng = 6, 4000, Xoshiro(4242)
    a = randn(rng, nq, n) .* [1.0, 7.0, 0.3, 120.0, 2.0, 0.05] .+ [3.0, -1, 0, 50, 2, 0]
    b = randn(rng, nq, n) .* [1.0, 7.0, 0.3, 120.0, 2.0, 0.05] .+ [3.0, -1, 0, 50, 2, 0]
    sd = vec(std(a; dims = 2))
    rel = vec(sqrt.(mean(abs2, a .- b; dims = 2))) ./ sd
    @test all(x -> isapprox(x, sqrt(2); rtol = 0.05), rel)
    @test minimum(rel) > 1e-2 * 100              # the retired threshold is 4 orders of magnitude off

    # `replica_spread` reports that scale from the archive itself, so the report never shows a
    # saturated number without the yardstick beside it.
    ens = (; q = [a, b, a .+ 0.0])
    sp = D6Score.replica_spread(ens, n)
    @test sp.npairs == 3
    @test sp.lo == 0.0                            # the identical pair
    @test isapprox(sp.hi, sqrt(2); rtol = 0.05)

    # A single replica has no pair, and that must read as "no scale", not as agreement.
    sp1 = D6Score.replica_spread((; q = [a]), n)
    @test sp1.npairs == 0 && isnan(sp1.median)
end
