# Figures for the M0-vs-DDN scoring round.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/score_m0_ddn.jl   # writes the scores
#   julia --startup-file=no --project=analysis analysis/plot_paper4.jl    # writes the figures
#
# Reads `analysis/output/paper4_scores.jld2` and writes PNGs into `analysis/figures/`, which
# `analysis/results.md` embeds.

using CairoMakie
using JLD2
using Printf
using Statistics
using LinearAlgebra

const HERE = @__DIR__
const FIGS = joinpath(HERE, "figures")
# Which scoring run to plot. `RIKFLOW_DATASET=new` reads the rebaselined scores and writes
# `fig*_new.png`, so the two sets sit side by side and `results.md` can embed whichever a section
# is actually about. The archive figures are still needed: G1's reproduction and the
# five-configuration regime-C comparison exist only on paper 2's data.
const DATASET = Symbol(get(ENV, "RIKFLOW_DATASET", "archive"))
const SUFFIX = DATASET === :new ? "_new" : ""
const SCORES = joinpath(HERE, "output",
                        DATASET === :new ? "paper4_scores_new.jld2" : "paper4_scores.jld2")

CairoMakie.activate!(; type = "png", px_per_unit = 2)

# A colour-blind-safe pair for the two models, plus a neutral for the reference. The reference is
# always the darker, thinner line so it reads as truth rather than as a third model.
const C_REF = RGBf(0.15, 0.15, 0.18)
const C_M0 = RGBf(0.00, 0.45, 0.70)      # blue
const C_DDN = RGBf(0.84, 0.37, 0.00)     # vermillion
const C_TRAIN = RGBf(0.35, 0.70, 0.90)
const C_HELD = RGBf(0.80, 0.47, 0.65)

load_scores() = isfile(SCORES) ? load(SCORES) :
                error("no scores at $SCORES -- run analysis/score_m0_ddn.jl first")

# `REBASE_MODELS` is the one place the rebaselined closures' reporting order is curated (ascending
# lambda). Pulled in only for `:new`; the archive's order lives in `ARCHIVED_CONFIGS` and its
# figures are unchanged by any of this.
DATASET === :new && include(joinpath(HERE, "extract_rebaseline.jl"))

"""
    order_online(on)

The closure keys of a regime-C score dict, in reporting order.

⚠️ **Never `sort` these.** `"LinReg10" < "LinReg5"` lexicographically, so a sorted axis puts
lambda = 1e4 third along what is otherwise a lambda ladder — and a monotone trend read off such a
figure is an artefact of string collation, not of the sweep. Keys the score file carries but the
curated list does not mention are appended (sorted), so a new closure still shows up rather than
silently dropping out of the figure.
"""
function order_online(on)
    DATASET === :new || return sort(collect(keys(on)); by = n -> on[n].name)
    ranked = [m.key for m in REBASE_MODELS if haskey(on, m.key)]
    return vcat(ranked, sort([k for k in keys(on) if !(k in ranked)]))
end

"""
    shade_windows!(ax, train, heldout, dt, stride)

Mark the training and held-out ranges on a time axis.

Both are drawn because the single most common way to misread a score in this project is to forget
which window it came from: `plan.md` §7 fixes that **no time unit ever serves two roles**, and a
figure that does not show the split cannot be checked against that rule.
"""
function shade_windows!(ax, train, heldout, dt)
    vspan!(ax, train[1] * dt, train[2] * dt; color = (C_TRAIN, 0.18))
    vspan!(ax, heldout[1] * dt, heldout[2] * dt; color = (C_HELD, 0.12))
    vlines!(ax, [train[1] * dt, train[2] * dt]; color = C_TRAIN, linewidth = 1.2,
            linestyle = :dash)
end

# ---------------------------------------------------------------------------------------------
# figure 1 -- the records themselves
# ---------------------------------------------------------------------------------------------

function fig_trajectories(d)
    tr = d["trajectories"]
    dt0 = d["dt"]
    dt = dt0 * tr.stride
    labels = d["labels"]
    w = d["windows"]
    nq = size(tr.q_ref, 1)
    T = min(size(tr.q_ref, 2), size(tr.q_track, 2))
    t = (1:T) .* dt
    z0, z1 = tr.zoom_range
    tz = (z0:z1) .* dt0
    nz = min(size(tr.zoom_q_ref, 2), size(tr.zoom_q_track, 2))

    # Two columns per QoI, because one does not work. At 100 TU the tracked and reference curves
    # lie on top of each other -- tracking nudges the low-fidelity QoIs onto the reference, so
    # they agree to a small fraction of a standard deviation, and an overview plot of the two shows
    # a single line. The overview establishes the record; the full-resolution zoom straddling the
    # train/held-out boundary is what actually shows two curves and how close they are.
    fig = Figure(size = (1400, 1000))
    Label(fig[0, 1:2], "High-fidelity reference and tracked low-fidelity QoIs", fontsize = 17,
          font = :bold)
    Label(fig[1, 1], "100 TU overview (subsampled 1:$(tr.stride))", fontsize = 12,
          font = :bold, tellwidth = false)
    Label(fig[1, 2], @sprintf("full resolution, %.1f-%.1f TU", z0 * dt0, z1 * dt0),
          fontsize = 12, font = :bold, tellwidth = false)
    for i in 1:nq
        ax = Axis(fig[i + 1, 1]; xlabel = i == nq ? "t [TU]" : "", ylabel = labels[i],
                  xgridvisible = false, ygridvisible = false)
        shade_windows!(ax, w.train ./ tr.stride, w.heldout ./ tr.stride, dt)
        lines!(ax, t, Float64.(tr.q_ref[i, 1:T]); color = (C_REF, 0.55), linewidth = 1.6,
               label = "HF reference")
        lines!(ax, t, Float64.(tr.q_track[i, 1:T]); color = C_M0, linewidth = 0.6,
               label = "tracked LF")
        text!(ax, 0.99, 0.04;
              text = @sprintf("rms(track - ref) = %.3f sd", tr.track_vs_ref_rel_rms[i]),
              space = :relative, fontsize = 9, align = (:right, :bottom))
        i == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = 9,
                             patchsize = (14, 2), orientation = :horizontal)

        axz = Axis(fig[i + 1, 2]; xlabel = i == nq ? "t [TU]" : "", xgridvisible = false,
                   ygridvisible = false)
        vlines!(axz, [w.train[2] * dt0]; color = C_TRAIN, linewidth = 1.4, linestyle = :dash)
        lines!(axz, tz[1:nz], Float64.(tr.zoom_q_ref[i, 1:nz]); color = C_REF, linewidth = 1.6)
        lines!(axz, tz[1:nz], Float64.(tr.zoom_q_track[i, 1:nz]); color = C_M0, linewidth = 1.1)
        lines!(axz, tz[1:nz], Float64.(tr.zoom_q_star[i, 1:nz]); color = (C_DDN, 0.85),
               linewidth = 1.0, linestyle = :dash)
        i == 1 && text!(axz, 0.02, 0.95;
                        text = "black: HF reference\nblue: corrected q\norange dashed: predictor q*",
                        space = :relative, fontsize = 8.5, align = (:left, :top))
    end
    Label(fig[nq + 2, 1:2],
          "Blue band and dashed line: the training window, steps $(w.train[1])-$(w.train[2]) " *
          "= $(round(w.train[1]*dt0, digits=2))-$(round(w.train[2]*dt0, digits=2)) TU. " *
          "Pink band: the held-out window, $(round(w.heldout[1]*dt0, digits=2))-" *
          "$(round(w.heldout[2]*dt0, digits=2)) TU.\nThe gap between the blue and orange curves " *
          "in the right column is dQ, the correction the model emits; the curves themselves are " *
          "the level q,\nwhich is what the free-running scores are computed on.",
          fontsize = 11)
    save(joinpath(FIGS, "fig1_trajectories" * SUFFIX * ".png"), fig)
    return fig
end

function fig_dQ(d)
    tr = d["trajectories"]
    dt = d["dt"] * tr.stride
    labels = d["labels"]
    w = d["windows"]
    train = w.train ./ tr.stride
    held = w.heldout ./ tr.stride
    nq = size(tr.dQ_track, 1)
    T = size(tr.dQ_track, 2)
    t = (1:T) .* dt
    st = d["ref_dQ_stats"]

    fig = Figure(size = (1250, 780))
    Label(fig[0, 1:2],
          "The quantity every score is computed on: the SGS correction dQ = q - q*",
          fontsize = 17, font = :bold)
    for i in 1:nq
        r, c = fldmod1(i, 2)
        ax = Axis(fig[r, c]; xlabel = r == 3 ? "t [TU]" : "", ylabel = "dQ  " * labels[i],
                  xgridvisible = false, ygridvisible = false)
        shade_windows!(ax, train, held, dt)
        lines!(ax, t, Float64.(tr.dQ_track[i, :]); color = C_REF, linewidth = 0.6)
        hlines!(ax, [0.0]; color = (:black, 0.35), linewidth = 0.6)
        text!(ax, 0.02, 0.93; text = @sprintf("std %.3g   rho_1 %.4f", st.std[i], st.rho1[i]),
              space = :relative, fontsize = 10, align = (:left, :top))
    end
    Label(fig[4, 1:2],
          "dQ is the model's own output and the discriminating quantity WHERE THE PREDICTOR STREAM " *
          "IS REPLAYED: the level is then pinned
to the reference and cannot separate models. " *
          "Free-running, the level is free and is what the physical claims are about.",
          fontsize = 11)
    save(joinpath(FIGS, "fig2_dQ" * SUFFIX * ".png"), fig)
    return fig
end

# ---------------------------------------------------------------------------------------------
# figure 3 -- RH-1, the paper's problem statement
# ---------------------------------------------------------------------------------------------

function fig_rank_histograms(d)
    ra = d["regime_a"]
    labels = d["labels"]
    m0, ddn, m0in = ra["M0"], ra["DDN"], ra["M0_insample"]
    nq = m0.nq

    fig = Figure(size = (1320, 1080))
    Label(fig[0, 1:nq],
          "RH-1: rank of the truth among $(m0.M) draws from each fitted one-step density " *
          "(identical on q and on dQ)", fontsize = 17, font = :bold)

    # Three rows, because two would be misleading. The top row is the NULL: M0 scored on its own
    # training window, where the residual mean is 1e-4 standard deviations and the residual
    # standard deviation matches the fitted Sigma to 1.0001. It is still cap-shaped, because the
    # residual is leptokurtic and a Gaussian fitted to it is the wrong shape even at exactly the
    # right width. A held-out histogram has to be read against that, not against flatness.
    rows = ((1, m0in, RGBf(0.45, 0.45, 0.50), "M0 in-sample\n(the null)"),
            (2, m0, C_M0, "M0 held-out"),
            (3, ddn, C_DDN, "DDN held-out"))
    for (row, s, col, nm) in rows
        for i in 1:nq
            r = s.rh[i]
            ax = Axis(fig[row, i]; title = row == 1 ? labels[i] : "",
                      xlabel = row == 3 ? "rank bin" : "", ylabel = i == 1 ? nm : "",
                      ylabelsize = 11, xgridvisible = false, ygridvisible = false)
            barplot!(ax, 1:r.K, r.counts ./ r.expected; color = (col, 0.8), gap = 0.0,
                     strokewidth = 0)
            hlines!(ax, [1.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            ylims!(ax, 0, nothing)
            lab = row == 1 ? @sprintf("cvx %+.1f", r.convexity) :
                  @sprintf("cvx %+.1f (%+.1f vs null)", r.convexity,
                           r.convexity - m0in.rh[i].convexity)
            text!(ax, 0.5, 0.98; text = lab, space = :relative, fontsize = 8.5,
                  align = (:center, :top))
        end
    end

    # The pairing rule, made visible: reliability beside resolution. Normalised by the reference
    # correction's own standard deviation, because raw CRPS across these six QoIs spans four orders
    # of magnitude and a raw-unit panel is an enstrophy panel with the energy bands invisible.
    ax = Axis(fig[4, 1:nq]; ylabel = "mean CRPS / sd(dQ_ref)   [dimensionless]",
              xticks = (1:nq, labels), xgridvisible = false, ygridvisible = false)
    barplot!(ax, (1:nq) .- 0.2, m0.crps_norm; width = 0.38, color = (C_M0, 0.85),
             label = "M0 held-out")
    barplot!(ax, (1:nq) .+ 0.2, ddn.crps_norm; width = 0.38, color = (C_DDN, 0.85),
             label = "DDN held-out")
    hlines!(ax, [1 / sqrt(pi)]; color = :black, linestyle = :dash, linewidth = 1.2)
    text!(ax, 0.985, 1 / sqrt(pi); text = " climatological forecast: 1/sqrt(pi) = 0.564",
          space = :relative, fontsize = 9, align = (:right, :bottom))
    axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
    Label(fig[5, 1:nq],
          "Dashed line in the histograms = flat. Positive convexity = U = under-dispersed; " *
          "negative = cap = over-dispersed.\nThe histogram measures RELIABILITY only, which is " *
          "why CRPS sits beside it: DDN lands on the climatological CRPS in every band, so its " *
          "near-flat panels carry no skill.", fontsize = 11)
    save(joinpath(FIGS, "fig3_rank_histograms" * SUFFIX * ".png"), fig)
    return fig
end

# ---------------------------------------------------------------------------------------------
# figure 4 -- the trap, in one panel
# ---------------------------------------------------------------------------------------------

function fig_dynamics_vs_calibration(d)
    ra = d["regime_a"]
    labels = d["labels"]
    m0, ddn = ra["M0"], ra["DDN"]
    nq = m0.nq

    fig = Figure(size = (1200, 430))
    Label(fig[0, 1:2], "Flat is not skilful: calibration against dynamics", fontsize = 17,
          font = :bold)

    ax1 = Axis(fig[1, 1]; ylabel = "lag-1 autocorrelation of the predicted mean dQ",
               xticks = (1:nq, labels), xticklabelrotation = pi / 6,
               xgridvisible = false, ygridvisible = false)
    barplot!(ax1, (1:nq) .- 0.2, m0.rho1_mean; width = 0.38, color = (C_M0, 0.85), label = "M0")
    barplot!(ax1, (1:nq) .+ 0.2, ddn.rho1_mean; width = 0.38, color = (C_DDN, 0.85), label = "DDN")
    scatter!(ax1, 1:nq, m0.rho1_truth; color = C_REF, marker = :hline, markersize = 22,
             label = "realised dQ")
    axislegend(ax1; position = :rb, framevisible = false, labelsize = 10)

    ax2 = Axis(fig[1, 2]; ylabel = "convexity contrast (signed, N_eff-corrected)",
               xticks = (1:nq, labels), xticklabelrotation = pi / 6,
               xgridvisible = false, ygridvisible = false)
    for (s, col, off) in ((m0, C_M0, -0.15), (ddn, C_DDN, 0.15))
        xs = (1:nq) .+ off
        vals = [r.convexity for r in s.rh]
        los = [r.convexity - r.convexity_ci[1] for r in s.rh]
        his = [r.convexity_ci[2] - r.convexity for r in s.rh]
        scatter!(ax2, xs, vals; color = col, markersize = 11)
        errorbars!(ax2, xs, vals, los, his; color = col, whiskerwidth = 8)
    end
    hlines!(ax2, [0.0]; color = :black, linestyle = :dash, linewidth = 1)
    Label(fig[2, 1:2],
          "DDN's predictive mean is a constant, so its predicted dQ has zero autocorrelation " *
          "against a realised value near 0.94.\nA flat histogram beside that is the whole point " *
          "of the negative control.", fontsize = 11)
    save(joinpath(FIGS, "fig4_dynamics_vs_calibration" * SUFFIX * ".png"), fig)
    return fig
end

# ---------------------------------------------------------------------------------------------
# figure 5 -- the Gram spectrum, and what lambda acts on
# ---------------------------------------------------------------------------------------------

function fig_gram(d)
    g = d["gram"]
    fig = Figure(size = (1200, 460))
    Label(fig[0, 1:2], "Metric #23: what the ridge penalty is actually acting on", fontsize = 17,
          font = :bold)

    ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "singular-value index j",
               ylabel = "sigma_j^2", xgridvisible = false)
    for (nm, col) in (("harmonized", C_M0), ("faithful", C_DDN))
        s2 = g["$(nm)_lambda0.0"]["sigma2"]
        scatterlines!(ax1, 1:length(s2), s2; color = col, markersize = 5, linewidth = 1.2,
                      label = nm == "harmonized" ? "normal (paper 4)" : "standardise (paper 2)")
    end
    for (lam, ls) in ((0.01, :dash), (0.1, :dot), (1.0, :dashdot))
        hlines!(ax1, [lam]; color = (:black, 0.7), linestyle = ls, linewidth = 1,
                label = "lambda = $lam")
    end
    axislegend(ax1; position = :lb, framevisible = false, labelsize = 9)

    ax2 = Axis(fig[1, 2]; xlabel = "singular-value index j",
               ylabel = "alpha_j = sigma_j^2/(sigma_j^2+lambda)", xgridvisible = false)
    for (lam, ls) in ((0.01, :solid), (0.1, :dash), (1.0, :dot), (10.0, :dashdot))
        a = g["harmonized_lambda$(lam)"]["alpha"]
        lines!(ax2, 1:length(a), a; linestyle = ls, color = C_M0, linewidth = 1.5,
               label = "lambda = $lam")
    end
    hlines!(ax2, [0.5]; color = (:black, 0.4), linewidth = 0.8)
    ylims!(ax2, -0.02, 1.02)
    axislegend(ax2; position = :lb, framevisible = false, labelsize = 9)
    Label(fig[2, 1:2],
          "alpha_j near 1: the direction survives. alpha_j near 0: ridge has erased it. " *
          "Where lambda sits in this spectrum decides\nwhether the penalty is a " *
          "rank-deficiency fix, a shrink toward the marginal, or inactive.", fontsize = 11)
    save(joinpath(FIGS, "fig5_gram_spectrum" * SUFFIX * ".png"), fig)
    return fig
end

# ---------------------------------------------------------------------------------------------
# figure 6 -- the archived online ensembles
# ---------------------------------------------------------------------------------------------

function fig_online(d)
    on = d["online"]
    isempty(on) && return nothing
    # 🔴 Not `sort`. The keys are a lambda LADDER and a lexicographic sort files `LinReg10` between
    # `LinReg1` and `LinReg5`, so the x axis walks lambda as 0, 1e4, 1e-5, ... -- a monotone trend
    # read off this figure would be an artefact of string ordering. `REBASE_MODELS` is already in
    # ascending lambda (that is what its docstring promises), so use it and append anything the
    # score file carries that the list does not mention.
    names = order_online(on)
    floor_ = d["ks_floor"]

    fig = Figure(size = (1250, 470))
    Label(fig[0, 1:3],
          "Regime C: the $(DATASET === :new ? "rebaselined" : "archived") online ensembles, " *
          "scored on the QoI level q", fontsize = 17,
          font = :bold)

    # Column 1: the marginal distribution of the LEVEL, which is what the paper claims and what
    # paper 2's own compute_ks.jl measures. Column 2: the temporal structure, on both the level and
    # the correction, because the two carry very different amounts of signal.
    ax1 = Axis(fig[1, 1]; ylabel = "summed KS of q (per replica)",
               xticks = (1:length(names), names), xticklabelrotation = pi / 6,
               xgridvisible = false)
    for (i, n) in enumerate(names)
        s = on[n]
        scatter!(ax1, fill(i, length(s.ks_summed)), s.ks_summed; color = (C_M0, 0.8),
                 markersize = 9)
        scatter!(ax1, [i], [s.ks_ensemble]; color = C_DDN, marker = :diamond, markersize = 13)
        a = get(d["archived_ks"], n, nothing)
        a === nothing || rangebars!(ax1, [i + 0.28], [minimum(a.replicas)], [maximum(a.replicas)];
                                    color = (:black, 0.65), whiskerwidth = 8, linewidth = 1.2)
    end
    hlines!(ax1, [floor_.total]; color = :black, linestyle = :dash, linewidth = 1)
    text!(ax1, 0.02, 0.97;
          text = "blue: per replica (#10)\nred diamond: pooled (#11)\nblack bar: paper 2's own range\ndashed: D8 floor",
          space = :relative, fontsize = 8.5, align = (:left, :top))

    ax2 = Axis(fig[1, 2]; ylabel = "Delta rho", yscale = log10,
               xticks = (1:length(names), names), xticklabelrotation = pi / 6,
               xgridvisible = false)
    tiny = 1e-4
    for (i, n) in enumerate(names)
        s = on[n]
        scatter!(ax2, fill(i - 0.22, length(s.drho1)), max.(s.drho1, tiny); color = (C_M0, 0.85),
                 markersize = 8)
        scatter!(ax2, fill(i - 0.07, length(s.drho_tau)), max.(s.drho_tau, tiny);
                 color = (C_M0, 0.85), markersize = 8, marker = :rect)
        scatter!(ax2, fill(i + 0.10, length(s.drho1_dQ)), max.(s.drho1_dQ, tiny);
                 color = (C_DDN, 0.85), markersize = 8)
        scatter!(ax2, fill(i + 0.25, length(s.drho_tau_dQ)), max.(s.drho_tau_dQ, tiny);
                 color = (C_DDN, 0.85), markersize = 8, marker = :rect)
    end
    text!(ax2, 0.02, 0.97;
          text = "blue: level q     red: correction dQ\ncircle: lag 1 (#12)   square: lag $(on[names[1]].lag_tau) (#13)\nlog scale; floored at 1e-4",
          space = :relative, fontsize = 8.5, align = (:left, :top))

    ax3 = Axis(fig[1, 3]; ylabel = "stability fraction / spread-skill ratio",
               xticks = (1:length(names), names), xticklabelrotation = pi / 6,
               xgridvisible = false)
    barplot!(ax3, (1:length(names)) .- 0.2, [on[n].stability for n in names]; width = 0.38,
             color = (C_M0, 0.85), label = "stability (#16)")
    ss = [on[n].spread_skill === nothing ? NaN : on[n].spread_skill.ratio for n in names]
    barplot!(ax3, (1:length(names)) .+ 0.2, ss; width = 0.38, color = (C_DDN, 0.85),
             label = "clim. spread-skill (#18)")
    hlines!(ax3, [1.0]; color = :black, linestyle = :dash, linewidth = 1)
    axislegend(ax3; position = :lb, framevisible = false, labelsize = 9)
    Label(fig[2, 1:3],
          "Scored on the QoI LEVEL: regime C is genuinely free-running, so the level is not " *
          "pinned and it is what carries the invariant measure, the decorrelation\ntime and the " *
          "spread. Replicas are separate samples of the summed statistic and are never averaged " *
          "with the pooled one.\nThe spread-skill ratio carries its finite-M correction; at " *
          "M = 5 an uncorrected perfect ensemble would read 0.913.", fontsize = 11)
    save(joinpath(FIGS, "fig6_online" * SUFFIX * ".png"), fig)
    return fig
end

# ---------------------------------------------------------------------------------------------
# figure 7 -- the autocorrelation of dQ, model against reference
# ---------------------------------------------------------------------------------------------

function fig_autocorr(d)
    on = d["online"]
    isempty(on) && return nothing
    labels = d["labels"]
    dt = d["dt"]
    names = order_online(on)
    nq = length(labels)
    lag_tau = on[names[1]].lag_tau

    fig = Figure(size = (1250, 830))
    Label(fig[-1, 1:2],
          "Metrics #12-#15: is the ORDER right? Autocorrelation of the QoI level, first replica " *
          "of each configuration", fontsize = 17, font = :bold)
    cols = [C_M0, C_DDN, RGBf(0.0, 0.62, 0.45), RGBf(0.80, 0.47, 0.65), RGBf(0.34, 0.71, 0.91)]
    # Eight series is too many for an in-panel legend: it covered a quarter of panel 1 and
    # collided with the per-QoI annotation. One horizontal legend under the title instead.
    local leg_ax = nothing
    for i in 1:nq
        r, c = fldmod1(i, 2)
        ax = Axis(fig[r, c]; xlabel = r == 3 ? "lag [TU]" : "", ylabel = "rho  " * labels[i],
                  xgridvisible = false, ygridvisible = false)
        rref = on[names[1]].rho_ref[i]
        lag = (0:(length(rref) - 1)) .* dt
        lines!(ax, lag, rref; color = C_REF, linewidth = 2.4,
               label = i == 1 ? "reference q" : nothing)
        for (j, n) in enumerate(names)
            rm = on[n].rho_model[i]
            lines!(ax, (0:(length(rm) - 1)) .* dt, rm; color = (cols[mod1(j, length(cols))], 0.9),
                   linewidth = 1.1, label = i == 1 ? n : nothing)
        end
        # The corrections, dashed. They decorrelate far faster than the level, which is why the
        # level's Delta_rho has almost no dynamic range and the correction's has a great deal.
        rrefd = on[names[1]].rho_ref_dQ[i]
        lines!(ax, (0:(length(rrefd) - 1)) .* dt, rrefd; color = (C_REF, 0.45), linewidth = 1.4,
               linestyle = :dash, label = i == 1 ? "reference dQ" : nothing)

        # DDN, and the only place it can appear on this figure. It has no archived online runs, so
        # it has no free-running level; what it has is a fitted density, and one sampled path from
        # it is the honest way to show the correction it would emit. The line sits on zero from lag
        # 1 by construction -- state-independent draws -- against a reference correction that stays
        # correlated for a tenth of a time unit. That gap is the negative control, drawn.
        if haskey(d, "acf_regime_a")
            a = d["acf_regime_a"]
            lines!(ax, a.lags, a.ddn[i]; color = (C_DDN, 0.95), linewidth = 1.5,
                   linestyle = :dash, label = i == 1 ? "DDN dQ (fitted density)" : nothing)
            # Mid-right, where every curve has decayed below 0.2 by the end of the lag axis. The
            # legend moved to figure level precisely so all six panels can carry their own number.
            text!(ax, 0.985, 0.42;
                  text = @sprintf("DDN rho_1 = %+.4f\nref rho_1 = %+.4f", a.ddn[i][2],
                                  a.truth[i][2]),
                  space = :relative, fontsize = 8.5, align = (:right, :center))
        end
        vlines!(ax, [lag_tau * dt]; color = (:black, 0.45), linestyle = :dot, linewidth = 1)
        hlines!(ax, [0.0]; color = (:black, 0.3), linewidth = 0.6)
        i == 1 && (leg_ax = ax)
    end
    leg_ax === nothing ||
        Legend(fig[0, 1:2], leg_ax; orientation = :horizontal, framevisible = false,
               labelsize = 10, nbanks = 2, patchsize = (22, 2), colgap = 14)
    Label(fig[4, 1:2],
          "Solid: the QoI level, first replica of each configuration. Dashed grey: the " *
          "reference correction dQ. Dashed red: DDN's correction, one sampled path from " *
          "its fitted density.
Dotted vertical: the Delta_rho lag. DDN has no archived " *
          "online runs, so it cannot appear as a level; this is regime A on a regime-C " *
          "figure, deliberately.
The reference correction stays correlated for about a " *
          "tenth of a time unit; DDN sits on zero from lag 1, because its draws are " *
          "state-independent. That is the negative control.
On the level, lag 1 separates " *
          "the configurations by only 0.000-0.003 and the integral-timescale lag by 3x; on " *
          "the correction lag 1 spans 0.02-5.29 -- the level is the target the claims are " *
          "about, the correction is where the diagnostic signal is.", fontsize = 11)
    save(joinpath(FIGS, "fig7_autocorr" * SUFFIX * ".png"), fig)
    return fig
end

# ---------------------------------------------------------------------------------------------

function main()
    mkpath(FIGS)
    d = load_scores()
    @printf("plotting %s scores into %s
", DATASET, FIGS)
    for (nm, f) in (("fig1_trajectories", fig_trajectories),
                    ("fig2_dQ", fig_dQ),
                    ("fig3_rank_histograms", fig_rank_histograms),
                    ("fig4_dynamics_vs_calibration", fig_dynamics_vs_calibration),
                    ("fig5_gram_spectrum", fig_gram),
                    ("fig6_online", fig_online),
                    ("fig7_autocorr", fig_autocorr))
        print("  ", rpad(nm, 32))
        flush(stdout)
        try
            f(d)
            p = joinpath(FIGS, nm * SUFFIX * ".png")
            @printf("ok  (%.0f kB)\n", filesize(p) / 1024)
        catch err
            println("FAILED: ", err)
        end
        flush(stdout)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
