# Figures and scores for the REBASELINED online runs -- `results.md` §4b, P2r's R2.
#
# One figure per closure, each comparing that closure's online QoI trajectories against the
# regenerated HF reference, plus the KS tables §4b quotes. Before this existed, §4b's numbers came
# from an ad-hoc session and nothing in the repository reproduced them.
#
# 🔴 The reference here is the REGENERATED one (`load_new_reference`), never the archive's. The
# online runs are post-`09954be1` and the archive is not; scoring one against the other compares two
# different dynamical systems (memory #45, #46).
#
# ⚠️ **What this figure can and cannot show.** Regime C is free-running: nothing is replayed, the
# solver supplies `q*`, and two runs from the same initial field decorrelate within an eddy turnover
# (~0.3 TU). So *pointwise* agreement with the reference is not expected past the first few TU and
# its absence is not a defect -- the claim these runs make is distributional. The trajectory panel
# is therefore read for the envelope (does the closure stay in the right band of amplitudes, does it
# drift, does it collapse) and the marginal panel beside it carries the claim the KS statistic
# actually scores.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/extract_rebaseline.jl   # cache the QoIs
#   julia --startup-file=no --project=analysis analysis/plot_rebaseline.jl      # -> figures/*.png
#
# Environment:
#   REBASE_STRIDE   trajectory display stride, in samples (default 4 = 0.01 TU)
#   REBASE_FIG_DIR  where the PNGs go (default analysis/figures)

using CairoMakie
using JLD2
using Printf
using Statistics

const HERE = @__DIR__
const SRC = normpath(joinpath(HERE, "..", "src"))
const FIGS = get(ENV, "REBASE_FIG_DIR", joinpath(HERE, "figures"))

include(joinpath(SRC, "ts_score.jl"))            # ks_distance
include(joinpath(HERE, "extract_archive.jl"))    # load_new_reference
include(joinpath(HERE, "extract_rebaseline.jl")) # load_rebaseline, REBASE_MODELS

CairoMakie.activate!(; type = "png", px_per_unit = 2)

const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

# Display stride only. 0.01 TU against a measured `T_int` of 0.5-0.63 TU, so this cannot alias
# anything the eye is being asked to read; it exists because six 40 001-point lines per panel is
# 1.4 M vector segments in a PNG.
const STRIDE = parse(Int, get(ENV, "REBASE_STRIDE", "4"))

# The stabilizer's threshold, `time_series_methods.jl:162,190`: `any(abs.(q_star) .< 1e-2)` zeroes
# the whole `dQ` column. Drawn on the `E[16,32]` panel because that is the band that fires it
# (memory #59) -- on 736 of 736 fired steps in the worst replica, the other five never.
const CLAMP = 1e-2

# Same colour-blind-safe palette as `plot_paper4.jl` and `plot_hf_new_vs_archive.jl`: the reference
# is always the darker, thinner line so it reads as truth rather than as a fifth model.
const C_REF = RGBf(0.15, 0.15, 0.18)
const MODEL_COLOUR = Dict(
    "LinReg1" => RGBf(0.00, 0.45, 0.70),     # blue
    "LinReg5" => RGBf(0.00, 0.45, 0.70),     # blue
    "LinReg6" => RGBf(0.00, 0.45, 0.70),     # blue
    "DDN"     => RGBf(0.84, 0.37, 0.00),     # vermillion
    "nomodel" => RGBf(0.80, 0.47, 0.65),     # reddish purple
    "smag"    => RGBf(0.00, 0.62, 0.45),     # bluish green
)

# ---------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------

"""
    clamp_census(e)

Which steps the stabilizer fired on, and how close each closure came to firing it.

Returns one entry per replica: `fired` (the step indices whose `dQ` column is identically zero) and
`qmin` (the smallest reconstructed `|q*|` per band). A fired column is identically zero by
construction, so `q_star = q[:, 2:end] - dQ` is *exact* on exactly those steps -- which is why the
reconstruction is trustworthy here even though `online_sgs` never returns `q*` (memory #35).

🔴 **`fired` is meaningful only when `e.clampable`.** The stabilizer lives only in the `LinReg` path,
so the DDN's `dQ` columns are never identically zero **because nothing can zero them**, not because
the closure stayed clear of the threshold -- and its `qmin` on `E[16,32]` is **4.57e-4, a factor 22
below** the 1e-2 the LRS is held above, so the honest reading is the opposite of "0 fired". Callers
must branch on `clampable` before reporting a count; `qmin` is well defined for any closure carrying
a `dQ` stream and is what makes the asymmetry visible.

⚠️ Memory #59 records this as *"three orders below the threshold"*. The value it quotes (4.6e-4) is
right and the characterisation is not: 1e-2 / 4.57e-4 = 21.9.

Nothing at all is defined for the deterministic baselines: no `dQ`, no `q*`.
"""
function clamp_census(e)
    e.stochastic || return nothing
    out = NamedTuple[]
    for (dQ, qs) in zip(e.dQ, e.q_star)
        fired = e.clampable ? [j for j in axes(dQ, 2) if all(@view(dQ[:, j]) .== 0)] : Int[]
        # Which band was under the threshold on each fired step, and whether the events are spread
        # or clustered. Both change what the clamp is: a uniform 1% tax on every step is a different
        # object from one 0.8 TU excursion that the stabilizer sat on.
        by_band = [count(j -> abs(qs[i, j]) < CLAMP, fired) for i in axes(qs, 1)]
        bursts = isempty(fired) ? 0 : 1 + count(k -> fired[k] - fired[k - 1] > 1,
                                                2:length(fired))
        push!(out, (; fired, by_band, bursts, n = size(dQ, 2),
                    qmin = vec(minimum(abs, qs; dims = 2))))
    end
    return out
end

"""
    score_model(e, q_ref)

The regime-C marginal scores for one closure: summed KS per replica (#10) and the pooled ensemble
form (#11), both on the QoI **level**.

Summed over QoIs, **never over replicas** -- the two statistics answer different questions and are
never averaged together (`metrics.md`, and §3 of this report).
"""
function score_model(e, q_ref)
    nq = size(q_ref, 1)
    per_replica = [[ks_distance(@view(q_ref[i, :]), @view(q[i, :])) for i = 1:nq] for q in e.q]
    pooled = reduce(hcat, e.q)
    ensemble = [ks_distance(@view(q_ref[i, :]), @view(pooled[i, :])) for i = 1:nq]
    # 🔑 KS is a distance and carries no sign, so two closures that fail in opposite directions can
    # score alike. The mean ratio is one line and it separates them -- it is what says the TO
    # closures run LOW in every band while the deterministic baselines run high in [16,32].
    mean_ratio = [mean(@view(pooled[i, :])) / mean(@view(q_ref[i, :])) for i = 1:nq]
    return (; per_replica, summed = [sum(k) for k in per_replica],
            ensemble, ensemble_summed = sum(ensemble), mean_ratio,
            qmin = [minimum(@view(pooled[i, :])) for i = 1:nq],
            qmax = [maximum(@view(pooled[i, :])) for i = 1:nq], pooled)
end

# ---------------------------------------------------------------------------------------------
# Figure -- one per closure
# ---------------------------------------------------------------------------------------------

"""
    density_profile(x, edges)

A histogram of `x` on `edges`, normalised to a density. Stdlib only, like the rest of the metric
layer; returns `(centres, density)`. Values outside `edges` are dropped, and the caller sets the
edges from the combined reference-and-model range so that nothing is silently cut.
"""
function density_profile(x, edges)
    n = length(edges) - 1
    lo, hi = first(edges), last(edges)
    w = (hi - lo) / n
    c = zeros(Float64, n)
    for v in x
        (v < lo || v > hi) && continue
        j = min(n, max(1, Int(floor((v - lo) / w)) + 1))
        c[j] += 1
    end
    total = sum(c)
    total > 0 && (c ./= total * w)
    # Centres from the index, not from a float range: `lo + w/2 : w : hi - w/2` is one element
    # short whenever the accumulated step rounds past the stop, which it does on these ranges.
    return (lo .+ w .* ((1:n) .- 0.5), c)
end

"""
    fig_model(e, ref, sc, census)

The per-closure comparison figure: six bands, each a trajectory panel with the marginal beside it.

Layout is two bands per row, each band occupying a wide time axis and a narrow density axis that
shares its `y`. The two panels are deliberately adjacent rather than in separate figures: the
trajectory is what makes a stability or collapse failure visible, the marginal is what the KS
statistic actually scores, and reading either alone has misled this project before.
"""
function fig_model(e, ref, sc, census)
    q_ref = ref.q_ref
    dt = ref.dt_sample
    col = MODEL_COLOUR[e.key]
    nrep = length(e.q)
    t = (0:STRIDE:(size(q_ref, 2) - 1)) .* dt

    fig = Figure(size = (1450, 980))
    Label(fig[0, 1:4],
          "$(e.label) -- online QoI trajectories against the regenerated HF reference " *
          "(100 TU, regime C)", fontsize = 17, font = :bold)

    for i = 1:6
        row, blk = fldmod1(i, 2)
        ax = Axis(fig[row, 2blk - 1]; ylabel = LABELS[i],
                  xlabel = row == 3 ? "t [TU]" : "", xgridvisible = false, ygridvisible = false)
        axd = Axis(fig[row, 2blk]; xlabel = row == 3 ? "density" : "",
                   xgridvisible = false, ygridvisible = false, yticklabelsvisible = false,
                   xticklabelsvisible = false)

        for (r, q) in enumerate(e.q)
            lines!(ax, t, @view(q[i, 1:STRIDE:end]); color = (col, 0.55), linewidth = 0.5,
                   label = r == 1 ? (nrep > 1 ? "$nrep replicas" : "run") : nothing)
        end
        lines!(ax, t, @view(q_ref[i, 1:STRIDE:end]); color = C_REF, linewidth = 0.8,
               label = "HF reference")

        lo = min(minimum(@view q_ref[i, :]), minimum(sc.pooled[i, :]))
        hi = max(maximum(@view q_ref[i, :]), maximum(sc.pooled[i, :]))
        pad = 0.04 * (hi - lo)
        ylims!(ax, lo - pad, hi + pad)
        xlims!(ax, 0, t[end])

        edges = range(lo, hi; length = 81)
        cr, dr = density_profile(@view(q_ref[i, :]), edges)
        cm, dm = density_profile(@view(sc.pooled[i, :]), edges)
        band!(axd, cm, zeros(length(dm)), dm; color = (col, 0.35), direction = :y)
        lines!(axd, dm, cm; color = col, linewidth = 1.2)
        lines!(axd, dr, cr; color = C_REF, linewidth = 1.2)
        ylims!(axd, lo - pad, hi + pad)
        xlims!(axd, 0, nothing)
        text!(axd, 0.96, 0.98; text = @sprintf("KS %.4f", sc.ensemble[i]), space = :relative,
              align = (:right, :top), fontsize = 9)

        # The clamp lives on the `E[16,32]` band and only in the `LinReg` path. Marking it here is
        # the only place in this report where the stabilizer is visible rather than tabulated.
        #
        # ⚠️ Drawn only for the two TO closures. On a deterministic baseline there is no `dQ` to
        # zero, so a threshold line there annotates a mechanism that does not exist -- and at
        # `c_s = 0.07` it also falls outside the panel, which is how the mislabelling stayed
        # invisible the first time this figure was drawn.
        if LABELS[i] == "E[16,32]" && e.stochastic
            hlines!(ax, [CLAMP]; color = (:black, 0.7), linestyle = :dash, linewidth = 1)
            text!(ax, 0.01, 0.06; text = "dashed: clamp threshold 1e-2 (tested on q*, drawn on q)",
                  space = :relative, align = (:left, :bottom), fontsize = 8)
            if census !== nothing && e.clampable
                y = lo - pad / 2
                for c in census
                    isempty(c.fired) && continue
                    scatter!(ax, (c.fired .- 1) .* dt, fill(y, length(c.fired));
                             color = (:black, 0.35), marker = :vline, markersize = 5)
                end
                nfired = sum(length(c.fired) for c in census)
                nfired > 0 && text!(ax, 0.99, 0.97;
                                    text = @sprintf("clamp fired on %d of %d steps (all replicas)",
                                                    nfired, sum(c.n for c in census)),
                                    space = :relative, align = (:right, :top), fontsize = 8)
            end
        end

        i == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = 9)
        colsize!(fig.layout, 2blk, Relative(0.09))
    end

    # 🔴 Three cases, not two. "Cannot clamp" and "did not clamp" are different statements and the
    # figure has to make which one applies unmistakable -- that is the whole content of memory #59.
    cens = if census === nothing
        "This closure is deterministic: no dQ stream and no q*, so the stabilizer does not apply."
    elseif !e.clampable
        @sprintf("The stabilizer CANNOT fire here: it lives only in the LinReg path, and MVG_sampler never receives q*. Smallest reconstructed |q*| on E[16,32] is %.2e, %.0fx BELOW the 1e-2 threshold the LRS is clamped at -- so this closure is not treated like the LRS (memory #59).",
                 minimum(c.qmin[6] for c in census), CLAMP / minimum(c.qmin[6] for c in census))
    else
        n = sum(length(c.fired) for c in census)
        n == 0 ? "The stabilizer could fire here but never did on these runs." :
        @sprintf("The stabilizer fired on %d of %d steps across %d replica(s), always through E[16,32] -- so this closure's KS is model PLUS stabilizer (memory #59).",
                 n, sum(c.n for c in census), length(census))
    end
    Label(fig[4, 1:4],
          "Regime C is free-running: the runs share the reference's initial field but decorrelate " *
          "within ~0.3 TU, so the panels are read for the ENVELOPE, not for pointwise agreement.\n" *
          "The marginal beside each band is what the KS statistic scores; the number quoted is " *
          "the pooled ensemble KS (#11). Displayed at stride $(STRIDE) (= $(STRIDE * dt) TU).\n" *
          cens, fontsize = 11)

    out = joinpath(FIGS, "fig8_online_$(e.key).png")
    mkpath(FIGS)
    save(out, fig)
    @printf("  wrote %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)
    return out
end

# ---------------------------------------------------------------------------------------------

function main()
    ref = load_new_reference()
    @printf("reference: %s\n  q_ref = %s at dt_sample = %.4g, %d stored fields, %.2f h\n",
            basename(ref.source), size(ref.q_ref), ref.dt_sample, ref.nfields,
            ref.comptime / 3600)

    scored = NamedTuple[]
    for spec in REBASE_MODELS
        println()
        println("== ", spec.label)
        e = load_rebaseline(spec.key)
        sc = score_model(e, ref.q_ref)
        census = clamp_census(e)
        fig_model(e, ref, sc, census)
        push!(scored, (; e, sc, census))
    end

    # The tables `results.md` §4b quotes, printed in the form it quotes them so the two can be
    # diffed rather than retyped.
    println("\n\n### R2 -- summed KS on the QoI level, against the regenerated reference\n")
    println("| model | replicas | stable | summed KS, per replica | ensemble |")
    println("|---|---|---|---|---|")
    for s in scored
        rng = length(s.sc.summed) > 1 ?
              @sprintf("%.3f – %.3f", minimum(s.sc.summed), maximum(s.sc.summed)) : "—"
        @printf("| %s | %d | %d/%d | %s | %.3f |\n", s.e.label, s.e.nominal_replicas,
                length(s.e.q), s.e.nominal_replicas, rng, s.sc.ensemble_summed)
    end

    println("\n### Per-band KS, ensemble\n")
    println("| model | " * join(LABELS, " | ") * " |")
    println("|---|" * repeat("---|", 6))
    for s in scored
        @printf("| %s | %s |\n", s.e.label,
                join((@sprintf("%.4f", k) for k in s.sc.ensemble), " | "))
    end

    println("\n### Mean ratio to the reference, ensemble -- the direction KS discards\n")
    println("| model | " * join(LABELS, " | ") * " |")
    println("|---|" * repeat("---|", 6))
    for s in scored
        @printf("| %s | %s |\n", s.e.label,
                join((@sprintf("%.3f", r) for r in s.sc.mean_ratio), " | "))
    end

    println("\n### Range, ensemble, against the reference's own\n")
    @printf("%-28s %s\n", "reference",
            join((@sprintf("%s %.4g..%.4g", LABELS[i], minimum(@view(ref.q_ref[i, :])),
                           maximum(@view(ref.q_ref[i, :]))) for i in (5, 6)), "   "))
    for s in scored
        @printf("%-28s %s\n", s.e.label,
                join((@sprintf("%s %.4g..%.4g", LABELS[i], s.sc.qmin[i], s.sc.qmax[i])
                      for i in (5, 6)), "   "))
    end

    # 🔴 Three states, printed as three different sentences. "Cannot clamp" is not "clamped 0
    # times": the DDN's minima sit three orders BELOW the threshold the LRS is held above, so
    # collapsing the two into one count states the asymmetry backwards (memory #59).
    println("\n### Clamp census, per replica\n")
    for s in scored
        if s.census === nothing
            @printf("%-28s deterministic -- no dQ, no q*, stabilizer does not apply\n", s.e.label)
            continue
        end
        for (r, c) in zip(s.e.replica_index, s.census)
            fired = s.e.clampable ?
                    @sprintf("%4d / %d fired (%.2f%%) in %d burst(s), by band %s",
                             length(c.fired), c.n, 100 * length(c.fired) / c.n, c.bursts,
                             string(c.by_band)) :
                    "NOT CLAMPABLE (no q* in the MVG path)"
            @printf("%-28s replica %d: %s\n    min |q*| per band = %s\n",
                    s.e.label, r, fired, join((@sprintf("%.2e", v) for v in c.qmin), " "))
        end
    end
    println()
    return scored
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
