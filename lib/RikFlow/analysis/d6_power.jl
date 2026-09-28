# Mini-D6 power: how small a paired difference can a (K, M, N_LEAD) screen resolve?
# Plan "Start here" step 0(c) (2026-09-28).
#
#     julia --startup-file=no --project=analysis analysis/d6_power.jl                    # real D6 files
#     julia --startup-file=no --project=analysis analysis/d6_power.jl --synthetic        # validation
#     D6_POWER_A=<dir> D6_POWER_B=<dir> julia ... analysis/d6_power.jl                   # other pair
#
# Real mode reads `analysis/output/D6_LinReg1/` and `analysis/output/D6_LinReg7/` (override with
# D6_POWER_A / D6_POWER_B) -- `d6_online_ic<k>_m<member>.jld2` as `tools/run_d6.jl` writes them --
# and the regenerated HF reference `analysis/data/hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2`
# as truth. The assembled arrays are cached in `analysis/output/d6_power_cache.jld2`.
#
# What it computes, for A - B paired on the same ICs (A = LinReg1, B = LinReg7):
#   (a) the plan's declared PRIMARY score (§7, *Screening funnel*): ensemble fair CRPS
#       (`ts_score.jl` `crps_ensemble`, fair = true), each band standardised by the reference's sd,
#       averaged over the six bands and the leads <= 0.5 TU. Two lead sets: the D6 grid
#       {25, 50, 100, 200} steps, and every 5th step up to 200 ("dense").
#   (b) mean skill / climatology over the (band, lead) cells with lead <= N_LEAD, as §4c's table
#       (RMSE of the ensemble mean / (sd * sqrt(1 + 1/M))).
#   (c) the in-band spread-skill count, S7's [0.8, 1.25], finite-M corrected (`spread_skill`).
# for subsamples (K, M) in {30, 45, 90} x {5, 10}, K drawn as a CONTIGUOUS run of ICs or STRIDED
# evenly across the pool, from the full pool or from ICs with t >= 52 TU (the selection block's
# position), N_LEAD in {1000 (full grid), 400}.
#
# For each subsample: an IC moving-block bootstrap (`block_bootstrap_indices`) of the paired
# difference, both closures resampled on the SAME indices; the 90% CI half-width is the minimum
# detectable difference (MDD). Reported per configuration: the full-D6 difference, the median
# subsample point estimate and its sd across subsamples, the median MDD, and the fraction of
# subsamples whose 90% CI excludes 0 ("resolved").
#
# 🔑 Block length (gotchas #34, #67): the level's ACF falls to 0.1 at 0.43-0.60 TU but rings at
# +-0.2 with a ~1 TU period, so tau = 1 TU; b = ceil(tau / narrowest IC spacing in the subsample).
# The lag-1 ACF of the per-IC paired CRPS difference is printed as a check on that choice, and a
# b in {1, 2, 4} sensitivity line is printed for one configuration.
#
# ⚠️ LinReg1 and LinReg7 are the SAME sampler family with the same member seeds
# (`member_seed(k, member)` in run_d6.jl), so members 1..M are common random numbers across the two
# closures. A screen across sampler families (MVG vs an LSTM) pairs by IC only (plan §7). The
# "noCRN" rows break the CRN by scoring A's members 1-5 against B's members 6-10; use THOSE rows to
# size a cross-family screen.

using JLD2, Statistics, Printf, Random, LinearAlgebra

const HERE = @__DIR__
include(joinpath(HERE, "..", "src", "ts_score.jl"))   # crps_ensemble, block_bootstrap_indices

const DT = 2.5e-3
const NQ = 6
const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]
const GRID = [25, 50, 100, 200, 400, 1000]          # score_d6.jl LEADS
const PRIMARY_GRID = [25, 50, 100, 200]             # <= 0.5 TU
const DENSE = collect(5:5:200)
const STORE_LEADS = sort(unique(vcat(DENSE, GRID)))  # what the cache holds
const INBAND = (0.8, 1.25)
const TAU_BLOCK = parse(Float64, get(ENV, "D6_POWER_TAU", "1.0"))   # TU
const NBOOT = parse(Int, get(ENV, "D6_POWER_NBOOT", "2000"))
const NSUB = parse(Int, get(ENV, "D6_POWER_NSUB", "12"))           # subsamples per configuration
const T_SEL = 52.0                                                 # selection-block start, TU
const SEED = 20260928

# ---------------------------------------------------------------------------------------------
# loading (conventions of score_d6.jl, re-stated so a concurrent edit there cannot move them:
# forecast column nwarm + lead + 1 of the run's q, truth column n_k + nwarm + lead + 1)
# ---------------------------------------------------------------------------------------------

function scan_dir(dir)
    pat = r"^d6_online_ic(\d+)_m(\d+)\.jld2$"
    byic = Dict{Int,Dict{Int,String}}()
    for f in readdir(dir)
        m = match(pat, f)
        m === nothing && continue
        get!(byic, parse(Int, m[1]), Dict{Int,String}())[parse(Int, m[2])] = joinpath(dir, f)
    end
    return byic
end

"""
    load_pair(dirA, dirB, qref) -> (; fcA, fcB, tr, ks, tk, members)

`fc*` is `K x NQ x M x L` on `STORE_LEADS`, `tr` is `K x NQ x L`. Policy A of `results.md` §4c:
an IC is kept only if BOTH closures have every member and none diverged (so the pair is scored on
the intersection, which for LinReg1-LinReg7 is the published 87).
"""
function load_pair(dirA, dirB, qref)
    A, B = scan_dir(dirA), scan_dir(dirB)
    ks = sort(collect(intersect(keys(A), keys(B))))
    M = maximum(length(A[k]) for k in ks)
    keep = Int[]
    nk = Int[]
    data = Dict{Tuple{Int,Char},Array{Float64,3}}()
    L = length(STORE_LEADS)
    dropped = Int[]
    for k in ks
        ok = length(A[k]) == M && length(B[k]) == M && sort(collect(keys(A[k]))) == 1:M &&
             sort(collect(keys(B[k]))) == 1:M
        if !ok
            push!(dropped, k)
            continue
        end
        n_k0 = nothing
        arrs = Dict{Char,Array{Float64,3}}()
        bad = false
        for (tag, D) in (('A', A), ('B', B))
            x = zeros(NQ, M, L)
            for m in 1:M
                d = load(D[k][m])
                if get(d, "diverged", false) === true
                    bad = true
                    break
                end
                n_k0 === nothing && (n_k0 = (d["n_k"], d["nwarm"]))
                (d["n_k"], d["nwarm"]) == n_k0 || error("IC $k: members disagree on n_k/nwarm")
                q = d["q"]
                for (j, l) in pairs(STORE_LEADS)
                    x[:, m, j] = q[:, n_k0[2] + l + 1]
                end
            end
            bad && break
            arrs[tag] = x
        end
        if bad
            push!(dropped, k)
            continue
        end
        push!(keep, k)
        push!(nk, n_k0[1] + n_k0[2])
        data[(k, 'A')] = arrs['A']
        data[(k, 'B')] = arrs['B']
    end
    K = length(keep)
    fcA = zeros(K, NQ, M, L)
    fcB = similar(fcA)
    tr = zeros(K, NQ, L)
    for (i, k) in pairs(keep)
        fcA[i, :, :, :] = data[(k, 'A')]
        fcB[i, :, :, :] = data[(k, 'B')]
        for (j, l) in pairs(STORE_LEADS)
            tr[i, :, j] = qref[:, nk[i] + l + 1]
        end
    end
    # IC time = the package's own time n_k * dt (plan §7's partition is on this), not the origin
    tk = [(nk[i] - 100) * DT for i in 1:K]      # nwarm = 100 on every D6 run
    @printf("loaded %d paired ICs x M = %d (dropped %d: %s)\n", K, M, length(dropped),
            join(dropped, ", "))
    return (; fcA, fcB, tr, ks = keep, tk, M)
end

# ---------------------------------------------------------------------------------------------
# synthetic surrogate with a KNOWN difference
# ---------------------------------------------------------------------------------------------

"""
    ringing_ar2(n, rng; period, tdecay)

A stationary AR(2) whose ACF rings with `period` (TU) and decays on `tdecay` (TU), unit variance.
The level's ACF in the reference rings with a ~1 TU period (gotcha #67).
"""
function ringing_ar2(n, rng; period = 1.0, tdecay = 0.35)
    r = exp(-DT / tdecay)
    th = 2pi * DT / period
    p1, p2 = 2r * cos(th), -r^2
    x = zeros(n)
    for t in 3:n
        x[t] = p1 * x[t - 1] + p2 * x[t - 2] + randn(rng)
    end
    return x ./ std(x)
end

"""
    synthetic_pair(; K, M, delta, rng)

Truth: six ringing AR(2) bands. IC origins at D6's spacing (alternating 0.75 / 1.0 TU from 10.5 TU).
Each forecast member = truth + an error that starts at 0.02 sd and relaxes to climatology on 0.3 TU
(ensemble-mean part and member part, smooth in lead). Model B's error amplitude is (1 + delta)
times A's and B's spread is 0.8x (under-dispersed, like LinReg7). CRN: B reuses A's draws.
"""
function synthetic_pair(; K = 87, M = 10, delta = 0.03, rng = Xoshiro(SEED), nref = 40_000,
                        spreadB = 0.8)
    L = length(STORE_LEADS)
    truth = reduce(hcat, [ringing_ar2(nref, rng) for _ in 1:NQ])'   # NQ x nref
    starts = Int[]
    s = 4200
    while length(starts) < K
        push!(starts, s)
        s += isodd(length(starts)) ? 300 : 400
        s + 1100 > nref && (s = 4200 + rand(rng, 1:100))
    end
    fcA = zeros(K, NQ, M, L)
    fcB = similar(fcA)
    tr = zeros(K, NQ, L)
    lt = STORE_LEADS .* DT
    amp = @. sqrt(1 - exp(-2 * lt / 0.3)) + 0.02
    for k in 1:K, i in 1:NQ
        tr[k, i, :] = truth[i, starts[k] .+ STORE_LEADS]
        c = cumsum(randn(rng, L)) ./ sqrt.(1:L)       # smooth in lead
        c .*= 1 / std(c)
        for m in 1:M
            e = cumsum(randn(rng, L)) ./ sqrt.(1:L)
            e .*= 1 / std(e)
            ea = @. amp * (0.75 * c + 0.66 * e)
            fcA[k, i, m, :] = tr[k, i, :] .- ea
            eb = @. amp * (1 + delta) * (0.75 * c + spreadB * 0.66 * e)
            fcB[k, i, m, :] = tr[k, i, :] .- eb .+ 0.05 .* amp .* randn(rng, L)
        end
    end
    tk = starts .* DT
    return (; fcA, fcB, tr, ks = collect(1:K), tk, M, sdref = vec(std(truth; dims = 2)))
end

# ---------------------------------------------------------------------------------------------
# per-IC ingredients, then scores from index vectors (so the bootstrap is only index arithmetic)
# ---------------------------------------------------------------------------------------------

lidx(ls) = [findfirst(==(l), STORE_LEADS) for l in ls]

"""
    ingredients(fc, tr, sd, mem, nlead)

Per IC: the primary CRPS (grid and dense), and per (band, grid lead <= nlead) the squared error
of the ensemble mean and the member variance -- everything (a)-(c) need, for member subset `mem`.
"""
function ingredients(fc, tr, sd, mem, nlead)
    K = size(fc, 1)
    M = length(mem)
    jg, jd = lidx(PRIMARY_GRID), lidx(DENSE)
    cg = zeros(K)
    cd = zeros(K)
    for k in 1:K, i in 1:NQ
        for j in jd
            c = crps_ensemble(view(fc, k, i, mem, j), tr[k, i, j]; fair = true) / sd[i]
            cd[k] += c
            j in jg && (cg[k] += c)
        end
    end
    cg ./= NQ * length(jg)
    cd ./= NQ * length(jd)
    grid = filter(<=(nlead), GRID)
    jj = lidx(grid)
    e2 = zeros(K, NQ, length(jj))
    v = zeros(K, NQ, length(jj))
    for k in 1:K, i in 1:NQ, (a, j) in pairs(jj)
        x = view(fc, k, i, mem, j)
        mb = mean(x)
        e2[k, i, a] = (mb - tr[k, i, j])^2
        v[k, i, a] = sum(abs2, x .- mb) / (M - 1)
    end
    return (; cg, cd, e2, v, M, sd, grid)
end

function scores(g, idx)
    n = length(idx)
    f = sqrt((g.M + 1) / g.M)
    skill = 0.0
    inband = 0
    nc = size(g.e2, 2) * size(g.e2, 3)
    for i in axes(g.e2, 2), a in axes(g.e2, 3)
        se = 0.0
        sv = 0.0
        for k in idx
            se += g.e2[k, i, a]
            sv += g.v[k, i, a]
        end
        rmse = sqrt(se / n)
        skill += rmse / (g.sd[i] * sqrt(1 + 1 / g.M))
        r = f * sqrt(sv / n) / rmse
        inband += INBAND[1] <= r <= INBAND[2]
    end
    return (crps = mean(g.cg[idx]), crpsd = mean(g.cd[idx]), skill = skill / nc, inband)
end

const STATS = (:crps, :crpsd, :skill, :inband)

function paired(gA, gB, idx)
    a, b = scores(gA, idx), scores(gB, idx)
    return map(s -> getfield(a, s) - getfield(b, s), STATS)
end

"Block bootstrap of the paired differences over the positions `sub` (sorted by IC time)."
function boot(gA, gB, sub, blocklen, rng; nboot = NBOOT)
    K = length(sub)
    D = zeros(nboot, length(STATS))
    for r in 1:nboot
        ii = sub[block_bootstrap_indices(K, blocklen, rng)]
        D[r, :] .= paired(gA, gB, ii)
    end
    return D
end

# ---------------------------------------------------------------------------------------------
# subsampling
# ---------------------------------------------------------------------------------------------

function subsets(pool::Vector{Int}, K::Int, mode::Symbol, nsub::Int)
    N = length(pool)
    K >= N && return [pool]
    if mode === :contiguous
        starts = unique(round.(Int, range(1, N - K + 1; length = min(nsub, N - K + 1))))
        return [pool[s:(s + K - 1)] for s in starts]
    else
        step = N / K
        offs = range(0, step; length = nsub + 1)[1:nsub]
        out = [pool[unique(clamp.(floor.(Int, o .+ (0:(K - 1)) .* step) .+ 1, 1, N))] for o in offs]
        return unique(out)
    end
end

function blocklen_for(tk, sub; tau = TAU_BLOCK)
    length(sub) < 2 && return 1
    dmin = minimum(diff(tk[sub]))
    return max(1, ceil(Int, tau / dmin - 1e-9))
end

q05(x) = quantile(x, 0.05)
q95(x) = quantile(x, 0.95)

function run_power(P; io = stdout, label = "", truthdiff = nothing)
    rng = Xoshiro(SEED)
    sd = hasproperty(P, :sdref) ? P.sdref : P.sd
    Kall = size(P.fcA, 1)
    Mfull = P.M
    order = sortperm(P.tk)
    pools = Dict(:all => order, :late => filter(i -> P.tk[i] >= T_SEL, order))
    @printf(io, "\n==== %s: %d paired ICs, t = %.2f-%.2f TU; %d with t >= %.0f TU; M = %d\n", label,
            Kall, minimum(P.tk), maximum(P.tk), length(pools[:late]), T_SEL, Mfull)
    @printf(io, "IC spacing: min %.3f, median %.3f TU\n", minimum(diff(sort(P.tk))),
            median(diff(sort(P.tk))))

    # member designs: (name, members of A, members of B)
    mdes = Tuple{String,Vector{Int},Vector{Int}}[]
    Mfull >= 10 && push!(mdes, ("M10", collect(1:10), collect(1:10)))
    push!(mdes, ("M5", collect(1:5), collect(1:5)))
    Mfull >= 10 && push!(mdes, ("M5noCRN", collect(1:5), collect(6:10)))

    ing = Dict{Tuple{String,Int},Any}()
    for (name, ma, mb) in mdes, nl in (1000, 400)
        ing[(name, nl)] = (ingredients(P.fcA, P.tr, sd, ma, nl), ingredients(P.fcB, P.tr, sd, mb, nl))
    end

    # full-D6 reference and the IC-ACF check
    gA, gB = ing[("M10" in first.(mdes) ? "M10" : "M5", 1000)]
    full = paired(gA, gB, order)
    sA, sB = scores(gA, order), scores(gB, order)
    @printf(io, "full set (A, B, A-B): CRPS grid %.4f %.4f %+.4f | CRPS dense %.4f %.4f %+.4f | skill %.4f %.4f %+.4f | in-band %d %d %+d of %d\n",
            sA.crps, sB.crps, full[1], sA.crpsd, sB.crpsd, full[2], sA.skill, sB.skill, full[3],
            sA.inband, sB.inband, full[4], NQ * length(gA.grid))
    truthdiff === nothing ||
        @printf(io, "  KNOWN (large-K) difference: CRPS grid %+.4f, dense %+.4f, skill %+.4f, in-band %+.1f\n",
                truthdiff...)
    d = gA.cg[order] .- gB.cg[order]
    ac = autocorr(d, 3)
    @printf(io, "per-IC paired CRPS difference: ACF lag 1/2/3 over consecutive ICs %.2f %.2f %.2f (+-2/sqrt(K) = %.2f)\n", ac[2], ac[3], ac[4], 2 / sqrt(length(d)))
    for b in (1, 2, 4)
        D = boot(gA, gB, order, b, rng)
        @printf(io, "  block-length sensitivity, full set, b = %d: 90%% CI half-width CRPS %.4f, skill %.4f, in-band %.1f\n", b, (q95(D[:, 1]) - q05(D[:, 1])) / 2,
                (q95(D[:, 3]) - q05(D[:, 3])) / 2, (q95(D[:, 4]) - q05(D[:, 4])) / 2)
    end

    rows = []
    @printf(io, "\n%-5s %-10s %-4s %-8s %-5s %3s %2s | %-31s | %-31s | %-31s | %s\n", "pool", "mode", "K",
            "M", "NL", "n", "b", "CRPS grid: est sd MDD res", "CRPS dense: est sd MDD res",
            "skill: est sd MDD res", "in-band: est MDD res")
    for pool in (:all, :late), K in (30, 45, 90), mode in (:contiguous, :strided)
        pl = pools[pool]
        K > length(pl) + 3 && continue           # K = 90 from the 87 = the full set, once
        subs = subsets(pl, K, mode, NSUB)
        mode === :strided && K >= length(pl) && continue
        for (name, _, _) in mdes, nl in (1000, 400)
            g1, g2 = ing[(name, nl)]
            est = zeros(length(subs), 4)
            hw = zeros(length(subs), 4)
            res = falses(length(subs), 4)
            bls = Int[]
            for (s, sub) in pairs(subs)
                b = blocklen_for(P.tk, sub)
                push!(bls, b)
                est[s, :] .= paired(g1, g2, sub)
                D = boot(g1, g2, sub, b, rng)
                for c in 1:4
                    lo, hi = q05(D[:, c]), q95(D[:, c])
                    hw[s, c] = (hi - lo) / 2
                    res[s, c] = lo > 0 || hi < 0
                end
            end
            f(c) = @sprintf("%+.4f %.4f %.4f %3.0f%%", median(est[:, c]), length(subs) > 1 ? std(est[:, c]) : NaN,
                            median(hw[:, c]), 100 * mean(res[:, c]))
            @printf(io, "%-5s %-10s %-4d %-8s %-5d %3d %2d | %s | %s | %s | %+.1f %.1f %3.0f%%\n", pool, mode,
                    length(subs[1]), name, nl, length(subs), maximum(bls), f(1), f(2), f(3), median(est[:, 4]),
                    median(hw[:, 4]), 100 * mean(res[:, 4]))
            push!(rows, (; pool, mode, K = length(subs[1]), mdes = name, nl, nsub = length(subs),
                         b = maximum(bls), est_median = vec(median(est; dims = 1)),
                         est_sd = vec(std(est; dims = 1)), mdd = vec(median(hw; dims = 1)),
                         resolved = vec(mean(res; dims = 1))))
        end
    end
    return (; full, rows, sA, sB)
end

"Large-K estimate of the synthetic model's true difference (the target the CIs should cover)."
function synthetic_truth(; delta, spreadB = 0.8, M = 10)
    # skill and the in-band count depend on M (the ensemble mean's error and sqrt(1 + 1/M)), so the
    # target is computed at the M being tested; the fair CRPS is M-unbiased
    P = synthetic_pair(; K = 4000, M, delta, spreadB, rng = Xoshiro(SEED + 1), nref = 2_000_000)
    g = (ingredients(P.fcA, P.tr, P.sdref, 1:M, 1000), ingredients(P.fcB, P.tr, P.sdref, 1:M, 1000))
    return paired(g..., 1:4000)
end

"Coverage of the bootstrap 90% CI of the CRPS difference over independent synthetic D6 replicates."
function synthetic_coverage(; delta, nrep = 60, K = 45, M = 5, io = stdout)
    tru = synthetic_truth(; delta, M)
    rng = Xoshiro(SEED + 7)
    cov = zeros(Int, 4)
    for r in 1:nrep
        P = synthetic_pair(; K, M, delta, rng = Xoshiro(SEED + 100 + r))
        gA = ingredients(P.fcA, P.tr, P.sdref, 1:M, 1000)
        gB = ingredients(P.fcB, P.tr, P.sdref, 1:M, 1000)
        sub = sortperm(P.tk)
        D = boot(gA, gB, sub, blocklen_for(P.tk, sub), rng; nboot = 1000)
        for c in 1:3        # the in-band count's expectation is not the large-K count; skip it
            cov[c] += q05(D[:, c]) <= tru[c] <= q95(D[:, c])
        end
    end
    @printf(io, "synthetic CI coverage (K = %d, M = %d, %d replicates, nominal 90%%): CRPS grid %.0f%%, dense %.0f%%, skill %.0f%%\n", K, M, nrep, 100cov[1] / nrep, 100cov[2] / nrep, 100cov[3] / nrep)
    return cov ./ nrep
end

function main_synthetic(; io = stdout)
    delta = 0.03
    tru = synthetic_truth(; delta)
    # null check: identical closures -> every difference exactly 0
    P0 = synthetic_pair(; K = 30, M = 5, delta = 0.0, spreadB = 1.0)
    g0 = ingredients(P0.fcA, P0.tr, P0.sdref, 1:5, 1000)
    z = paired(g0, g0, 1:30)
    @printf(io, "null check (A vs itself): %s\n", join((@sprintf("%.1e", x) for x in z), " "))
    P = synthetic_pair(; K = 87, M = 10, delta)
    out = run_power(P; io, label = "SYNTHETIC ringing-AR(2) surrogate, B error x1.03, B spread x0.8",
                    truthdiff = tru)
    synthetic_coverage(; delta, io)
    return out
end

function main_real(; io = stdout)
    dA = get(ENV, "D6_POWER_A", joinpath(HERE, "output", "D6_LinReg1"))
    dB = get(ENV, "D6_POWER_B", joinpath(HERE, "output", "D6_LinReg7"))
    hf = get(ENV, "D6_POWER_HF", joinpath(HERE, "data", "hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2"))
    for p in (dA, dB)
        isdir(p) || error("no D6 directory at $p")
    end
    isfile(hf) || error("no HF reference cache at $hf")
    cache = joinpath(get(ENV, "D6_POWER_OUT", joinpath(HERE, "output")), "d6_power_cache_$(basename(dA))_$(basename(dB)).jld2")
    qref = load(hf, "q_ref")
    sd = vec(std(qref; dims = 2))
    P = if isfile(cache) && get(ENV, "D6_POWER_REBUILD", "0") != "1"
        c = load(cache)
        (; fcA = c["fcA"], fcB = c["fcB"], tr = c["tr"], ks = c["ks"], tk = c["tk"], M = c["M"])
    else
        p = load_pair(dA, dB, qref)
        jldsave(cache; p.fcA, p.fcB, p.tr, p.ks, p.tk, p.M, leads = STORE_LEADS)
        p
    end
    P = merge(P, (; sdref = sd))
    out = run_power(P; io, label = "REAL D6: A = $(basename(dA)), B = $(basename(dB))")
    jldsave(joinpath(get(ENV, "D6_POWER_OUT", joinpath(HERE, "output")), "d6_power_$(basename(dA))_$(basename(dB)).jld2");
            full = out.full, rows = out.rows, stats = collect(STATS))
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    "--synthetic" in ARGS ? main_synthetic() : main_real()
end
