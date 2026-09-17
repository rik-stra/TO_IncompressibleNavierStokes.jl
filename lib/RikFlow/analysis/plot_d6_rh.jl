# RH-3: D6's lead-resolved rank histograms.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/plot_d6_rh.jl
#
# Reads the saved `d6_scores_D6_<model>.jld2` — nothing is recomputed here, so these figures cannot
# disagree with §4c's tables — and writes
#
#   figures/fig10_d6_rank_histograms_<model>.png   the 6 x 5 grid of histograms, one per closure
#   figures/fig10b_d6_rh_summary.png               convexity and slope against lead, all three
#
# ---------------------------------------------------------------------------------------------
# What these are, and the two ways to misread them
# ---------------------------------------------------------------------------------------------
#
# 🔑 **The verification axis is the initial condition, not time.** Each histogram is one (QoI, lead)
# cell, and its `K` entries are the `K` initial conditions: the rank of the truth among the `M`
# members at that lead. That is what D6 exists to supply — every archived online run is a single
# trajectory from a single IC, so each lead there has exactly one verification instance and a rank
# histogram over one sample is not a histogram (`score_d6.jl`'s header).
#
# ⚠️ **Flat is reliability, not skill.** A climatological ensemble is perfectly flat and has zero
# resolution, which is not hypothetical here: the DDN's regime-A histograms are near-flat while it
# scores the climatological CRPS exactly (§2). So these panels are read beside the skill row printed
# under each one, never alone.
#
# ⚠️ **`K/(M+1)` is 7.9 instances per bin at `K = 87, M = 10`** — comfortable for the Jolliffe-Primo
# contrasts, marginal for the omnibus chi-squared, which is the second reason the contrasts are
# primary and the reason no p-value is drawn on these panels.
#
# ⚠️ **The ICs are not independent.** D6's initial conditions are ~0.48 TU apart against level
# timescales of 0.25-0.54 TU, so the contrasts carry a block-bootstrap interval along the IC axis
# (`rank_histogram`'s own default) and the intervals drawn here are those, not multinomial ones.

using Statistics
using Printf
using JLD2
using CairoMakie

const HERE = @__DIR__
const OUT = joinpath(HERE, "output")
const FIGS = joinpath(HERE, "figures")

"The three closures D6 ran, in §4c's order, with the colours §4b and `plot_paper4.jl` already use."
const MODELS = [("LinReg1", RGBf(0.20, 0.40, 0.75)),
                ("LinReg7", RGBf(0.85, 0.45, 0.10)),
                ("DDN", RGBf(0.55, 0.15, 0.55))]

"S7's spread-skill acceptance band, drawn on the summary panel for the same reason §4c counts it."
const S7_BAND = (0.8, 1.25)

load_level(model) = load(joinpath(OUT, "d6_scores_D6_$(model).jld2"))

"""
    fig_histograms(model, colour)

The 6 x 5 grid for one closure: QoI down, lead across, one rank histogram per cell.

Each panel is normalised by its own flat expectation `K/(M+1)`, so **1.0 is flat** and the panels
are comparable across closures even though `K` is not part of the shape. The convexity is printed
in the corner with its sign convention spelled out in the caption, because the shape is what the
eye reads and the number is what the claim rests on.
"""
function fig_histograms(model, colour)
    d = load_level(model)
    lev, labels, T_int, dt = d["level"], d["labels"], d["T_int"], d["dt"]
    nq, nl = length(lev.counts), length(lev.counts[1])
    expected = lev.K / (lev.M + 1)

    fig = Figure(size = (1500, 1180))
    Label(fig[0, 1:nl],
          "RH-3, $model: rank of the truth among $(lev.M) members, over $(lev.K) initial " *
          "conditions, per (band, lead)", fontsize = 17, font = :bold)
    for i in 1:nq, t in 1:nl
        ℓ = lev.leads[i][t]
        ax = Axis(fig[i, t];
                  title = @sprintf("lead %d  (%.2f TU)", ℓ, ℓ * dt),
                  titlesize = 10,
                  xlabel = i == nq ? "rank bin" : "",
                  ylabel = t == 1 ? labels[i] : "",
                  ylabelsize = 11, xgridvisible = false, ygridvisible = false,
                  xticksvisible = false, xticklabelsvisible = false)
        barplot!(ax, 1:(lev.M + 1), lev.counts[i][t] ./ expected;
                 color = (colour, 0.8), gap = 0.0, strokewidth = 0)
        hlines!(ax, [1.0]; color = :black, linewidth = 1.0, linestyle = :dash)
        ylims!(ax, 0, nothing)
        # Skill beside shape, because a flat histogram from a climatological ensemble is flat for
        # the wrong reason. 1.0 is "no better than climatology".
        text!(ax, 0.5, 0.98;
              text = @sprintf("cvx %+.1f   slope %+.1f\nskill/clim %.2f",
                              lev.convexity[i][t], lev.slope[i][t],
                              lev.skill[i][t] / lev.clim_level[i]),
              space = :relative, fontsize = 8, align = (:center, :top))
    end
    Label(fig[nq + 1, 1:nl],
          "Dashed line = flat. Positive convexity = U = under-dispersed; negative = cap = " *
          "over-dispersed; slope = mean bias.\nLeads are per band, at " *
          "(0.25, 0.5, 1, 2, 5) x T_int(band). Flat is RELIABILITY, not skill — read it beside " *
          "the skill/clim printed in each panel, where 1.0 is no better than climatology.",
          fontsize = 11)
    p = joinpath(FIGS, "fig10_d6_rank_histograms_$(model).png")
    save(p, fig)
    return p
end

"""
    fig_summary()

Convexity and slope against lead, all three closures on one set of axes, one column per band.

🔑 This is the figure that answers RH-3 as a question rather than as thirty pictures: does the
dispersion defect grow, shrink or change sign with lead, and do the closures differ in it. The
histograms above are the evidence; this is the reading.
"""
function fig_summary()
    ds = [(m, c, load_level(m)) for (m, c) in MODELS]
    lev1 = ds[1][3]["level"]
    labels, dt = ds[1][3]["labels"], ds[1][3]["dt"]
    nq = length(lev1.counts)

    fig = Figure(size = (1500, 720))
    Label(fig[0, 1:(nq + 1)], "RH-3 contrasts against lead — the shape of the rank histogram, per band",
          fontsize = 17, font = :bold)
    for (row, (field, name)) in enumerate(((:convexity, "convexity  (+ = U = under-dispersed)"),
                                           (:slope, "slope  (mean bias)")))
        for i in 1:nq
            # Ticks at the lead values themselves. A log axis defaults to decade labels, which on a
            # grid spanning 0.06-2.7 TU prints exponents and hides where the five leads actually are.
            tv = lev1.leads[i] .* dt
            ax = Axis(fig[row, i];
                      title = row == 1 ? labels[i] : "", titlesize = 12,
                      xlabel = row == 2 ? "lead [TU]" : "",
                      ylabel = i == 1 ? name : "", ylabelsize = 11,
                      xscale = log10, xgridvisible = false, ygridvisible = false,
                      xticks = (tv, [@sprintf("%.2f", x) for x in tv]),
                      xticklabelsize = 8, xticklabelrotation = pi / 4)
            for (m, c, d) in ds
                lev = d["level"]
                x = lev.leads[i] .* dt
                y = getfield(lev, field)[i]
                ci = getfield(lev, field === :convexity ? :convexity_ci : :slope_ci)[i]
                band!(ax, x, first.(ci), last.(ci); color = (c, 0.13))
                lines!(ax, x, y; color = c, linewidth = 2, label = m)
                scatter!(ax, x, y; color = c, markersize = 7)
            end
            hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
        end
    end
    # The legend gets its own row: inside an axis it sat on top of the very curves it names.
    Legend(fig[1, nq + 1],
           [LineElement(color = c, linewidth = 3) for (_, c) in MODELS],
           [m for (m, _) in MODELS];
           framevisible = false, labelsize = 11)
    Label(fig[3, 1:(nq + 1)],
          "Shaded = 95% block-bootstrap interval along the INITIALISATION-TIME axis, which is the " *
          "dependence that matters here: D6's ICs are ~0.48 TU apart against level timescales of " *
          "0.25-0.54 TU.\nA contrast whose band contains 0 is not distinguishable from a flat " *
          "histogram at that lead. Read with §4c's spread-skill table: the two measure the same " *
          "defect, one by shape and one by variance.",
          fontsize = 11)
    p = joinpath(FIGS, "fig10b_d6_rh_summary.png")
    save(p, fig)
    return p
end

"""
    report(; io = stdout)

The RH-3 numbers §4c quotes, so the section can be diffed against a re-run rather than retyped.

Counts a cell as **flat** when both contrasts' 95% intervals contain 0 — the honest statement of
"calibrated at this lead, for this band, and nothing more".
"""
function report(; io = stdout)
    @printf(io, "\nRH-3 — D6's lead-resolved rank histograms, level q\n")
    @printf(io, "%-9s %8s %8s %10s %10s %12s %12s\n",
            "closure", "K", "bins", "flat", "U-shaped", "median cvx", "median slope")
    for (m, _) in MODELS
        lev = load_level(m)["level"]
        cvx = reduce(vcat, lev.convexity)
        slp = reduce(vcat, lev.slope)
        ccis = reduce(vcat, lev.convexity_ci)
        scis = reduce(vcat, lev.slope_ci)
        flat = count(j -> first(ccis[j]) <= 0 <= last(ccis[j]) &&
                          first(scis[j]) <= 0 <= last(scis[j]), eachindex(cvx))
        ushaped = count(j -> first(ccis[j]) > 0, eachindex(cvx))
        @printf(io, "%-9s %8d %8d %10s %10s %12s %12s\n", m, lev.K, lev.M + 1,
                "$flat/$(length(cvx))", "$ushaped/$(length(cvx))",
                @sprintf("%+.2f", median(cvx)), @sprintf("%+.2f", median(slp)))
    end
    println(io, "\nflat = both 95% intervals contain 0. U = convexity interval strictly above 0,")
    println(io, "i.e. under-dispersed at that (band, lead). Intervals are block-bootstrapped along")
    println(io, "the IC axis. ⚠️ The cells are not independent tests — the bands move together")
    println(io, "(§1: the six 1/e times span only 1.22x) and the leads are nested windows of one run.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(FIGS)
    for (m, c) in MODELS
        println("wrote ", fig_histograms(m, c))
    end
    println("wrote ", fig_summary())
    report()
end
