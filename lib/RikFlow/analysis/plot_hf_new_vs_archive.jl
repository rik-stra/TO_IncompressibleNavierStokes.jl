# The regenerated HF reference's QoI trajectories against paper 2's archived one.
#
# ⚠️ Renamed from `plot_hf_probe_vs_archive.jl` on 2026-09-14. Until then this script compared
# `hf_timing_probe.jl`'s ~1.3 TU window against the archive, because that was the only
# Float64/LMWray3 HF data that existed. The production run has since finished (100 TU, 20.72 h), so
# the probe is superseded as a *data* source and the script reads the full run. The probe's own job
# — projecting that cost — is done, and it was right to 1%. Figures written before the rename are
# named `hf_probe_vs_archive_*.png` and were deleted with it.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/plot_hf_new_vs_archive.jl
#
# Inputs, both cached QoI extracts written by `extract_archive.jl`:
#   the new run   analysis/data/hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2
#   the archive   analysis/data/hf_reference_tsim100.0_qois.jld2   (HF_REF_QOIS overrides)
# Either is extracted on demand from its ~1.4–2.6 GB source if the cache is absent.
#
# Writes three PNGs into analysis/figures/.
#
# ---------------------------------------------------------------------------------------------
# 🔴 **What this comparison can and cannot show — and the line moved on 2026-09-14.**
#
# The two runs are **different realisations**, not two computations of one trajectory. Three
# independent reasons, none of them a defect:
#
#   1. Precision. The OU chain draws `randn!` into a Float64 buffer here and a Float32 one in the
#      archive, which consumes the random stream differently. The forcing is a *different sample
#      path* from step one.
#   2. Stepper. LMWray3 against the archive's RK44.
#   3. `Z[16,32]` is computed under the corrected Nyquist convention of `09954be1`; the archive was
#      written under the old one (gotchas #45, #46).
#
# So **trajectory agreement is meaningless at every column but the first**, and that has not
# changed. Column 1 is the like-for-like check — both sides filter the *same* archived spin-up
# field with no time stepping — and it is where the Nyquist signature must appear and nothing else
# may.
#
# 🆕 **What HAS changed is that the DISTRIBUTIONAL comparison is now worth making.** Over the
# probe's 1.3 TU there were ~2 independent samples and the window mean said nothing — gotcha #52
# says so in as many words. Over 100 TU there are ~160–200, because `T_int` on the QoI *level* is
# 0.51–0.63 TU, five times the median `T_int` of the *correction* (gotcha #30), and the reason this
# record buys far less than its 40 001 columns suggest.
#
# 🔑 **Both distributional statistics here carry their own null, and the null is the point.**
# `report_marginals` puts a moving-block bootstrap interval on the difference of means, and
# `ks_block_test` puts a block-permutation p-value on the KS distance. Read the interval and the
# p-value, never the point estimate: at N_eff ≈ 170 a +0.1 sd offset and a KS of 0.09 are both
# ordinary excursions, and without their nulls beside them either one reads as a finding.
#
# ⚠️ **Six bands are not six tests.** They rise and fall together -- the trajectory figure shows
# it directly, and the numbers follow: the same sign of Δmean and p within 0.085-0.112 on all six.
# Treat the panel as ONE comparison reported six ways. "6/6 pass" is not six-fold evidence, and a
# p near 0.1 does not become stronger for being repeated.
#
# ⚠️ A tempting null that does **not** work, tried and rejected: the archive's own first 50 TU
# against its own second 50 TU. It is half the length, which pushes its KS *up*, and the two halves
# come from one realisation, which pushes it *down* — two biases of unknown relative size. The
# block permutation below has neither problem.
# ---------------------------------------------------------------------------------------------

using CairoMakie
using JLD2
using Printf
using Random
using Statistics

const HERE = @__DIR__
const FIGS = get(ENV, "HF_FIG_DIR", joinpath(HERE, "figures"))

# One owner for "where do the QoI caches come from": `extract_archive.jl` holds both extractors and
# both loaders, and its own main block does not run on include.
include(joinpath(HERE, "extract_archive.jl"))

const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

# 🔴 The archive's own sample spacing, fixed by the run that produced it: savefreq = 10 at
# Δt = 2.5e-4. It is a constant because the archive cannot be re-read cheaply to ask. The new run's
# spacing is NOT a constant — `extract_new_reference` stores it — and the two are checked against
# each other below rather than assumed equal.
const ARCH_DT = 2.5e-3

# Moving-average window for the trajectory figure, and the block length for both resampling tests.
# All in TU, and all must exceed `T_int` (0.51–0.63 TU on this record) or they do nothing useful: a
# smoother shorter than the correlation time does not smooth, and a resampling block shorter than it
# understates the uncertainty. Printed against the measured `T_int` so the margin stays visible.
const SMOOTH_TU = parse(Float64, get(ENV, "HF_SMOOTH_TU", "2"))
const BLOCK_TU = parse(Float64, get(ENV, "HF_BLOCK_TU", "2"))
const NBOOT = parse(Int, get(ENV, "HF_NBOOT", "4000"))
const NPERM = parse(Int, get(ENV, "HF_NPERM", "2000"))
# Columns shown in the deviation figure's left panel. The separation saturates by ~column 200, and
# past ~500 the six curves overplot into a block that hides the part worth seeing; the right panel
# carries the rest.
const ZOOM_COLS = parse(Int, get(ENV, "HF_ZOOM_COLS", "1000"))
const SEED = parse(Int, get(ENV, "HF_SEED", "20260914"))

CairoMakie.activate!(; type = "png", px_per_unit = 2)

# Same palette as `plot_validation.jl` and `plot_paper4.jl`: colour-blind safe, archive darker so it
# reads as the established record rather than as a competing model.
const C_ARCH = RGBf(0.15, 0.15, 0.18)
const C_NEW = RGBf(0.00, 0.45, 0.70)

# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

"""
    integrated_time(x)

The integrated autocorrelation time of `x`, **in samples**: `1 + 2 Σ_k ρ_k`, summed to the first
non-positive `ρ_k` (Sokal's window — the tail of `ρ_k` is noise, and summing it adds variance
without adding signal).

`length(x) / integrated_time(x)` is then the number of independent samples the record really holds.
On this data that is ~170 out of 40 001, and it is the single most important number for reading any
of the distributional statistics below.

Guarded against a constant series the way `ts_score.jl`'s `autocorr` is — **relative** to the
series' own magnitude, never absolutely (gotcha #32).
"""
function integrated_time(x)
    y = collect(float.(x))
    n = length(y)
    m = mean(y)
    y .-= m
    v = sum(abs2, y) / n
    scale = max(abs(m), maximum(abs, y))
    v <= (1e-12 * max(scale, eps()))^2 && return 1.0     # constant: no correlation to integrate
    τ = 1.0
    for k = 1:(n ÷ 4)
        ρ = sum(@views y[1:(n - k)] .* y[(k + 1):n]) / (n * v)
        ρ <= 0 && break
        τ += 2ρ * (1 - k / n)
    end
    return τ
end

"""
    boot_dmean(a, b, sa, block, nboot, rng)

Moving-block bootstrap 95% interval for `(mean(b) - mean(a)) / sa`, in units of `a`'s standard
deviation.

**Both** series are resampled: the question is whether two runs of this length can differ by what
these two differ by, so the null has to carry both runs' sampling variability, not one run's.

🔴 Blocks, not points. They preserve the serial correlation that puts `N_eff` two orders below the
column count; an i.i.d. bootstrap here returns an interval about `sqrt(T_int)` ≈ 15 times too
narrow, and would call an ordinary excursion a bias.
"""
function boot_dmean(a, b, sa, block, nboot, rng)
    n = length(a)
    nb = max(1, n ÷ block)
    out = Vector{Float64}(undef, nboot)
    for t = 1:nboot
        sb = 0.0
        sc = 0.0
        for _ = 1:nb
            i = rand(rng, 1:(n - block + 1))
            j = rand(rng, 1:(n - block + 1))
            sb += sum(@views b[i:(i + block - 1)])
            sc += sum(@views a[j:(j + block - 1)])
        end
        out[t] = (sb - sc) / (nb * block) / sa
    end
    return quantile(out, 0.025), quantile(out, 0.975)
end

"""
    ks_block_test(a, b, block, nperm, rng) -> (ks, p, q95)

Two-sample Kolmogorov–Smirnov distance between the marginals of `a` and `b`, **with a null**: a
block-permutation p-value and the null's 95th percentile.

Each record is cut into non-overlapping blocks of `block` samples; the pooled blocks are then
relabelled at random into two groups of the original sizes and the KS recomputed. Under the null
that both records are one stationary law, blocks are exchangeable between them, and permuting whole
blocks keeps the within-block serial correlation that a point-wise permutation would destroy.

🔑 `p` is the fraction of permutations reaching the observed KS or beyond. A large `p` means a
record of this length cannot tell the two marginals apart — which, for two independent realisations
of the same flow, is the outcome that says the regeneration is sound.

The pooled *multiset* of values does not change under relabelling, so it is sorted once and each
permutation is a single O(N) walk over that order. Ties are handled by only evaluating the gap where
the value actually changes.
"""
function ks_block_test(a, b, block, nperm, rng)
    nb = min(length(a), length(b)) ÷ block
    nb >= 4 || error("only $nb blocks of $block samples; the permutation null needs at least 4")
    vals = Vector{Float64}(undef, 2 * nb * block)
    bid = Vector{Int}(undef, 2 * nb * block)
    for k = 1:nb, (off, src) in ((0, a), (nb, b))
        dst = ((off + k - 1) * block + 1):((off + k) * block)
        vals[dst] .= @view src[((k - 1) * block + 1):(k * block)]
        bid[dst] .= off + k
    end
    N = length(vals)
    n1 = n2 = nb * block
    ord = sortperm(vals)
    bo = bid[ord]
    # Where the sorted value actually changes: the only places an empirical-CDF gap is defined.
    atend = [t == N || vals[ord[t]] != vals[ord[t + 1]] for t = 1:N]

    walk(grp) = begin
        c1 = 0
        c2 = 0
        d = 0.0
        @inbounds for t = 1:N
            grp[bo[t]] ? (c1 += 1) : (c2 += 1)
            atend[t] && (d = max(d, abs(c1 / n1 - c2 / n2)))
        end
        d
    end

    truth = [k <= nb for k = 1:(2 * nb)]
    ks = walk(truth)
    lab = copy(truth)
    hits = 0
    for _ = 1:nperm
        shuffle!(rng, lab)
        walk(lab) >= ks && (hits += 1)
    end
    # Null quantile, from a second pass — cheap next to the permutations themselves and worth having
    # for the figure, where a p-value alone does not convey how big a KS this length routinely gives.
    null = Vector{Float64}(undef, nperm)
    for t = 1:nperm
        shuffle!(rng, lab)
        null[t] = walk(lab)
    end
    return ks, (hits + 1) / (nperm + 1), quantile(null, 0.95)
end

"Moving average of `x` over `w` samples, centred, by cumulative sum."
function movmean(x, w)
    v = float.(collect(x))
    n = length(v)
    w = clamp(w, 1, n)
    c = cumsum(vcat(0.0, v))
    lo = max.(1, (1:n) .- w ÷ 2)
    hi = min.(n, lo .+ w .- 1)
    return (c[hi .+ 1] .- c[lo]) ./ (hi .- lo .+ 1)
end

# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

"""
Six panels, one per QoI band, both runs over their full 100 TU.

Each raw series is drawn faint with its `SMOOTH_TU` moving average solid on top. At 40 001 samples
per line the raw curve is a band rather than a trajectory — which is honest, because that band *is*
what the comparison is about — and the smoothed line is what lets the eye see that the two runs
wander over the same range by the same amount. Plotting only the smooth line would hide the spread;
only the raw would hide everything.

Each series is drawn on its **own** time axis, so the figure stays correct if the two are ever
sampled differently.
"""
function fig_trajectories(new, arc, tn, ta, w)
    fig = Figure(; size = (1100, 680))
    for i = 1:6
        r, c = fldmod1(i, 3)
        ax = Axis(fig[r, c];
            title = LABELS[i],
            xlabel = r == 2 ? "t [TU]" : "",
            ylabel = c == 1 ? "QoI" : "",
            # Linear, autoscaled per panel. The six bands are O(1e3) for Z and O(1) for E — same
            # order within each kind — so a log axis would compress the only thing worth seeing.
        )
        lines!(ax, ta, view(arc, i, :); color = (C_ARCH, 0.22), linewidth = 0.4)
        lines!(ax, tn, view(new, i, :); color = (C_NEW, 0.22), linewidth = 0.4)
        lines!(ax, ta, movmean(view(arc, i, :), w); color = C_ARCH, linewidth = 1.8)
        lines!(ax, tn, movmean(view(new, i, :), w); color = C_NEW, linewidth = 1.8)
    end
    Legend(fig[3, 1:3],
        [LineElement(color = C_ARCH, linewidth = 2), LineElement(color = C_NEW, linewidth = 2)],
        [@sprintf("archive, %.3g TU (Float32 / RK44, old Nyquist)", ta[end]),
         @sprintf("regenerated, %.3g TU (Float64 / LMWray3)", tn[end])];
        orientation = :horizontal, framevisible = false)
    rowsize!(fig.layout, 3, Relative(0.08))
    Label(fig[0, 1:3],
        @sprintf("HF QoI trajectories, full record — raw faint, %.3g TU moving average solid; different realisations, so the bands should overlap and the curves must not",
                 w * ARCH_DT);
        fontsize = 13, padding = (0, 0, 4, 0))
    return fig
end

"""
Relative deviation per band: log–log on the left, physical time on the right.

The left panel is the one that carries information, and it is **log in x on the column index**
rather than linear in `t`, for a specific reason: the whole separation happens in the first ~160
columns of 40 001, and on a linear time axis all of it lands in the first pixel. Column 1 is the
like-for-like comparison, so the curve *starts* at the filter-and-convention difference — five bands
at Float32 round-off and `Z[16,32]` four orders above them, alone, which is the Nyquist change
showing exactly where #45 and #46 predict it — and then grows to saturation near 1.

A start that is **not** small in the five other bands is the interesting outcome: it would mean the
filter, the masks or `compute_QoI` had changed. Saturation in the right panel is not a finding; it
is there so the saturated level is on the record beside the growth, and never mistaken for it.
"""
function fig_deviation(new, arc, t, n)
    fig = Figure(; size = (1150, 460))
    cols = Makie.wong_colors()
    dev(i, rng) = max.(abs.(view(new, i, rng) .- view(arc, i, rng)) ./ abs.(view(arc, i, rng)),
                       1e-16)           # a log axis cannot take an exact zero

    nz = min(n, ZOOM_COLS)
    axl = Axis(fig[1, 1];
        xlabel = "column  (1 = the shared field, t = (column − 1) × Δt_sample)",
        ylabel = "|new − archive| / |archive|",
        xscale = log10, yscale = log10,
        title = @sprintf("separation from the shared initial field (first %d columns)", nz))
    for i = 1:6
        lines!(axl, 1:nz, dev(i, 1:nz); color = cols[i], linewidth = 1.3, label = LABELS[i])
    end

    axr = Axis(fig[1, 2];
        xlabel = "t [TU]", yscale = log10, title = "full record in time — saturation, not a finding")
    for i = 1:6
        lines!(axr, t[1:n], dev(i, 1:n); color = (cols[i], 0.55), linewidth = 0.4)
    end

    Legend(fig[1, 3], axl; framevisible = false)
    return fig
end

"""
Empirical CDFs per band, both runs, full record.

The statistic the 100 TU record actually supports, and each panel carries its own null: `p` is the
block-permutation p-value for "these two marginals are one population" and `q95` is the KS that the
same permutation reaches 5% of the time. A KS below `q95` — equivalently `p` above 0.05 — is as much
agreement as two independent realisations of one flow can show at this record length.
"""
function fig_distributions(new, arc, ks, p, q95)
    fig = Figure(; size = (1150, 680))
    for i = 1:6
        r, c = fldmod1(i, 3)
        ax = Axis(fig[r, c];
            title = @sprintf("%s\nKS = %.4f   p = %.3f   (null 95%%: %.4f)", LABELS[i], ks[i], p[i],
                             q95[i]),
            titlesize = 11,
            xlabel = r == 2 ? "QoI" : "", ylabel = c == 1 ? "empirical CDF" : "")
        for (q, col) in ((arc, C_ARCH), (new, C_NEW))
            xs = sort(collect(view(q, i, :)))
            lines!(ax, xs, (1:length(xs)) ./ length(xs); color = col, linewidth = 1.6)
        end
    end
    Legend(fig[3, 1:3],
        [LineElement(color = C_ARCH, linewidth = 2), LineElement(color = C_NEW, linewidth = 2)],
        ["archive (Float32 / RK44)", "regenerated (Float64 / LMWray3)"];
        orientation = :horizontal, framevisible = false)
    rowsize!(fig.layout, 3, Relative(0.08))
    Label(fig[0, 1:3],
        "Marginal distributions over the full record — the comparison 100 TU makes possible, each with its block-permutation null";
        fontsize = 13, padding = (0, 0, 4, 0))
    return fig
end

# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------

"The like-for-like check: same field, no time stepping."
function report_column1(new, arc)
    println()
    println("column 1 — same archived spin-up field, no time stepping.")
    println("🔑 The ONLY like-for-like comparison here. Five bands must sit at Float32 round-off —")
    println("   the archive's own precision — and Z[16,32] alone must show the 09954be1 Nyquist")
    println("   correction at ~1.06e-3 (gotchas #45, #46). Anything else has moved.")
    println("  band          archive           regenerated       rel. diff")
    for i = 1:6
        a, b = arc[i, 1], new[i, 1]
        @printf("  %-12s  %.8e  %.8e  %.3e\n", LABELS[i], a, b, abs(b - a) / abs(a))
    end
end

"Where the two realisations part company, in physical time."
function report_divergence(new, arc, t, n)
    println()
    println("divergence — the first column at which each band's relative deviation crosses 0.1.")
    println("Expected and uninformative; printed so it is on the record as having happened at the")
    println("eddy-turnover scale rather than immediately or never.")
    for i = 1:6
        d = abs.(view(new, i, 1:n) .- view(arc, i, 1:n)) ./ abs.(view(arc, i, 1:n))
        k = findfirst(>=(0.1), d)
        if k === nothing
            @printf("  %-12s  never crosses 0.1 (max %.3e)\n", LABELS[i], maximum(d))
        else
            @printf("  %-12s  column %6d,  t = %7.3f TU\n", LABELS[i], k, t[k])
        end
    end
end

"""
Distributional statistics, each beside its own null. Returns `(mean_ok, ks_ok, ks, p, q95)`.

Prints `T_int` and `N_eff` in the first two columns, because every number after them is read against
`N_eff` and not against the 40 001 columns.
"""
function report_marginals(new, arc, dt)
    block = max(2, round(Int, BLOCK_TU / dt))
    rng = MersenneTwister(SEED)
    println()
    @printf("distributional statistics over the full record, block = %d samples (%.3g TU): %d bootstrap draws, %d permutations.\n",
            block, block * dt, NBOOT, NPERM)
    println("🔑 Read each statistic against its null. A CI containing 0 and a p above 0.05 mean")
    println("   this record length cannot tell the two apart — which is the expected outcome for")
    println("   two independent realisations of one flow.")
    println("  band          T_int[TU]  N_eff   Δmean/sd_a   95% CI                sd_n/sd_a      KS   null95      p")
    mean_ok = true
    ks_ok = true
    ks = zeros(6)
    pv = zeros(6)
    q95 = zeros(6)
    for i = 1:6
        a = collect(view(arc, i, :))
        b = collect(view(new, i, :))
        sa = std(a)
        τ = integrated_time(a)
        d0 = (mean(b) - mean(a)) / sa
        lo, hi = boot_dmean(a, b, sa, block, NBOOT, rng)
        ks[i], pv[i], q95[i] = ks_block_test(a, b, block, NPERM, rng)
        inside = lo < 0 < hi
        kok = pv[i] >= 0.05
        mean_ok &= inside
        ks_ok &= kok
        @printf("  %-12s %9.4f %6.0f   %+9.4f   [%+.4f, %+.4f]%s   %8.4f  %6.4f  %6.4f  %5.3f%s\n",
                LABELS[i], τ * dt, length(a) / τ, d0, lo, hi, inside ? " " : "*", std(b) / sa,
                ks[i], q95[i], pv[i], kok ? "" : " *")
        if block < 3 * τ
            @printf("     ⚠️ block %.3g TU is under 3·T_int = %.3g TU here; both nulls are optimistic\n",
                    block * dt, 3τ * dt)
        end
    end
    (mean_ok && ks_ok) ||
        println("  * marks a statistic outside its null — a difference this record length CAN resolve.")
    return mean_ok, ks_ok, ks, pv, q95
end

# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------

nr = load_new_reference()
new = Float64.(nr.q_ref)
arc = Float64.(let p = get(ENV, "HF_REF_QOIS", "")
    isempty(p) ? load_reference() : load(p, "q_ref")
end)

# 🔴 Refuse, do not warn. Every statistic below except the trajectory figure pairs column i against
# column i, so a spacing mismatch would not degrade the output — it would invalidate it silently.
# The new run carries its own spacing in its cache; the archive's is the constant above.
isapprox(nr.dt_sample, ARCH_DT; rtol = 1e-9) || error(
    "sample spacings differ — new $(nr.dt_sample), archive $ARCH_DT.\n" *
    "Every column-paired statistic, and both the deviation and distribution figures, would be " *
    "nonsense. Re-extract, or put the two on a common grid first.")

n = min(size(new, 2), size(arc, 2))
n >= 2 || error("only $n overlapping samples; nothing to plot")
tn = (0:size(new, 2)-1) .* nr.dt_sample
ta = (0:size(arc, 2)-1) .* ARCH_DT

mkpath(FIGS)

@printf("regenerated  %s\n", nr.path)
@printf("             %d samples every %.4g TU → %.4g TU; %d stored LES fields; solver wall time %.1f s = %.2f h\n",
        size(new, 2), nr.dt_sample, tn[end], nr.nfields, nr.comptime, nr.comptime / 3600)
@printf("             from %s\n", nr.source)
@printf("archive      %d samples every %.4g TU → %.4g TU\n", size(arc, 2), ARCH_DT, ta[end])
size(arc, 2) == size(new, 2) ||
    @warn "records differ in length; column-paired statistics use the overlap" new = size(new, 2) archive = size(arc, 2)
@printf("non-finite values: regenerated %s, archive %s\n", any(!isfinite, new), any(!isfinite, arc))

report_column1(new, arc)
report_divergence(new, arc, tn, n)
mean_ok, ks_ok, ks, pv, q95 = report_marginals(view(new, :, 1:n), view(arc, :, 1:n), nr.dt_sample)

w = max(2, round(Int, SMOOTH_TU / nr.dt_sample))
f1 = joinpath(FIGS, "hf_new_vs_archive_trajectories.png")
f2 = joinpath(FIGS, "hf_new_vs_archive_deviation.png")
f3 = joinpath(FIGS, "hf_new_vs_archive_distributions.png")
save(f1, fig_trajectories(new, arc, tn, ta, w))
save(f2, fig_deviation(new, arc, tn, n))
save(f3, fig_distributions(view(new, :, 1:n), view(arc, :, 1:n), ks, pv, q95))

println()
println("Written:")
for f in (f1, f2, f3)
    println("  $f")
end

println()
println("VERDICT")
println(mean_ok ? "  means  ✅ every band's 95% CI on the difference of means contains 0" :
                  "  means  🔴 at least one band's CI excludes 0 (marked *)")
println(ks_ok ? "  shapes ✅ every band's KS is inside its block-permutation null (p ≥ 0.05)" :
                "  shapes 🔴 at least one band's KS is outside its null (marked *)")
println("  ⚠️  The six bands move together, so the two lines above are ONE comparison reported six")
println("     ways, not six independent tests. Do not read \"6/6\" as six-fold evidence.")
println("  🔑 Column 1 is the check that actually constrains the code — read that first. The two")
println("     statistics above say only that 100 TU cannot separate two realisations of one flow,")
println("     which is what they are.")
