# Window-averaged hindcast scores (2026-10-08): the measurements that settled the open choices of
# meta_files/handoff_window_scores_2026-10-08.md (Sec. 0). CRPS and normalized error averaged over
# every forecast step of a window (1-200, 1-1200) against the current lead-grid versions, for
# LinReg1 vs LinReg7 and LinReg1 vs the DDN; paired, lead-wise rule (score_d6.jl default), blocks
# of 2 ICs, 10 000 resamples. Also: every-5th-step vs every-step, the noCRN half-width (M = 5), the
# per-IC difference ACF and the block-length check. Uses score_d6.jl's loaders and `crps_by_ic`.
# A check, not the scorer: once score_d6.jl scores the windows (handoff step 2), it must reproduce
# these numbers. Prints only; writes nothing.
# 2026-10-08 (later): score_d6.jl now scores the windows (`paired_primary`, `window_delta`). This
# script stays the source of the paper's App. A.2 numbers (per-IC ACF, block length) and of the
# noCRN half-width (Sec. 4). delta_all is no longer a paper score (Rik): its line is gone.
#
#   cd lib/RikFlow && julia --startup-file=no --project=analysis analysis/window_scores.jl
include(joinpath(@__DIR__, "score_d6.jl"))
using Statistics, Printf, Random

const OUTD = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
truth = load_truth()
sc = primary_band_scale(truth.q)

function pair(da, db; nl = 1200)
    ea, eb = load_members(joinpath(OUTD, da)), load_members(joinpath(OUTD, db))
    ks = sort(intersect(ea.ks, eb.ks))
    a, b = restrict(ea, ks), restrict(eb, ks)
    grid = collect(1:min(nl, ea.nlead, eb.nlead))
    fa, tra = assemble(a, truth, grid)
    fb, trb = assemble(b, truth, grid)
    tra == trb || error("truth differs")
    mask_union!(fa, fb)
    return (; fa, fb, tr = tra, t = a.t, ks, grid)
end

hw(ci) = (ci.hi - ci.lo) / 2
win_crps(f, tr, cols) = vec(mean(crps_by_ic(f[:, :, :, cols], tr[:, :, cols], sc); dims = (2, 3)))

function report_crps(p, name, cols; label)
    sa = win_crps(p.fa, p.tr, cols)
    sb = win_crps(p.fb, p.tr, cols)
    ci = ic_block_bootstrap_ci(sa .- sb, p.t)
    @printf("  %-26s A %.4f  B %.4f  A-B %+.4f  CI [%+.4f, %+.4f]  half-width %.4f = %.1f%% of A  (%+.1f%% of B)\n",
            label, mean(sa), mean(sb), ci.mean, ci.lo, ci.hi, hw(ci), 100hw(ci) / mean(sa),
            100ci.mean / mean(sb))
    return (; sa, sb, ci)
end

# normalized error over a window: per QoI sqrt(sum_k sum_n E / sum_k sum_n W) / sigma, mean over QoIs
function delta_parts(f, tr, cols)
    m = cell_moments(f[:, :, :, cols], tr[:, :, cols])
    return (; S = dropdims(sum(m.E; dims = 3); dims = 3), Wt = dropdims(sum(m.W; dims = 3); dims = 3))
end
delta(P, idx) = mean(sqrt(sum(P.S[idx, i]) / sum(P.Wt[idx, i])) / sc[i] for i in axes(P.S, 2))

function report_delta(p, cols; label, nboot = 10_000, percell = false)
    # percell: the current definition, the mean of the per-cell delta over the 36 cells
    PA = percell ? [delta_parts(p.fa, p.tr, [c]) for c in cols] : [delta_parts(p.fa, p.tr, cols)]
    PB = percell ? [delta_parts(p.fb, p.tr, [c]) for c in cols] : [delta_parts(p.fb, p.tr, cols)]
    dl(P, idx) = mean(delta(x, idx) for x in P)
    o = sortperm(p.t)
    K = length(o)
    b = 2
    rng = Xoshiro(SEED)
    d = Vector{Float64}(undef, nboot)
    for r in 1:nboot
        idx = o[block_bootstrap_indices(K, b, rng)]
        d[r] = dl(PA, idx) - dl(PB, idx)
    end
    sort!(d)
    lo, hi = d[round(Int, 0.05nboot)], d[round(Int, 0.95nboot)]
    dA, dB = dl(PA, 1:K), dl(PB, 1:K)
    @printf("  %-26s A %.4f  B %.4f  A-B %+.4f  CI [%+.4f, %+.4f]  %s\n", label, dA, dB, dA - dB,
            lo, hi, hi < 0 ? "A lower, resolved" : lo > 0 ? "B lower, resolved" : "not resolved")
end

function noCRN(p, cols; label)
    # A's members 1-5 against B's members 6-10 (results_LSTMS.md Sec. 10a convention)
    sa = win_crps(p.fa[:, :, 1:5, :], p.tr, cols)
    sb = win_crps(p.fb[:, :, 6:10, :], p.tr, cols)
    ci = ic_block_bootstrap_ci(sa .- sb, p.t)
    @printf("  %-26s noCRN M = 5: A-B %+.4f  half-width %.4f = %.1f%% of A\n", label, ci.mean,
            hw(ci), 100hw(ci) / mean(sa))
end

acf(x, l) = (y = x .- mean(x); sum(y[1:end-l] .* y[1+l:end]) / sum(abs2, y))

for (da, db) in (("D6_LinReg1", "D6_LinReg7"), ("D6_LinReg1", "D6_DDN"))
    p = pair(da, db)
    @printf("\n=== A = %s, B = %s: K = %d ICs, steps 1..%d ===\n", da, db, length(p.ks), last(p.grid))
    println("CRPS (fair, sd units, mean over ICs and QoIs)")
    g = report_crps(p, "grid 25/50/100/200", [25, 50, 100, 200]; label = "grid 25/50/100/200")
    report_crps(p, "grid all six", [25, 50, 100, 200, 400, 1000]; label = "grid all six leads")
    w5 = report_crps(p, "", 5:5:200; label = "window 1-200, every 5th")
    w1 = report_crps(p, "", 1:200; label = "window 1-200, every step")
    report_crps(p, "", 1:1000; label = "window 1-1000")
    report_crps(p, "", 1:1200; label = "window 1-1200")
    @printf("  every-5th vs every-step per-IC CRPS_0.5: max |diff| A %.2e, corr of A-B %.5f\n",
            maximum(abs.(w5.sa .- w1.sa)), cor(w5.sa .- w5.sb, w1.sa .- w1.sb))
    d = w1.sa .- w1.sb
    @printf("  window 1-200: per-IC diff ACF lags 1-3: %.2f %.2f %.2f (±%.2f)\n", acf(d, 1), acf(d, 2),
            acf(d, 3), 2 / sqrt(length(d)))
    for bl in (1, 2, 4)
        ci = ic_block_bootstrap_ci(d, p.t; blocklen = bl)
        @printf("    block %d: half-width %.4f\n", bl, hw(ci))
    end
    da == "D6_LinReg1" && db == "D6_LinReg7" && (noCRN(p, [25, 50, 100, 200]; label = "grid 0.5"); noCRN(p, 1:200; label = "window 1-200"))
    println("Normalized error (paired, block 2, 10 000 resamples)")
    report_delta(p, [25, 50, 100, 200, 400, 1000]; label = "36-cell mean (current)", percell = true)
    report_delta(p, [25, 50, 100, 200]; label = "24-cell mean, leads <= 200", percell = true)
    report_delta(p, 1:200; label = "window 1-200")
end
