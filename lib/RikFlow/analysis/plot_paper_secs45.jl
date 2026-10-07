# Paper Secs. 4 and 5 (baseline.tex, colour.tex): the figures (2026-10-07).
#
#     PAPER_FIG_DIR=<paper>/figs julia --startup-file=no --project=lib/RikFlow/analysis \
#         lib/RikFlow/analysis/plot_paper_secs45.jl [fig ...]
#
# `fig` is any of fans, spread, ranks, step, resacf, dqacf (default: all). Vector PDFs sized for the
# text width (450 pt, 8 pt type) go into PAPER_FIG_DIR (default analysis/figures/paper, gitignored):
#
#   baseline_fans.pdf             LinReg1 and LinReg7 hindcast ensembles from the same ICs (Sec. 4.1)
#   baseline_spread_skill.pdf     spread-skill ratio per (QoI, lead) cell, and the long-run ratio
#   baseline_rank_histograms.pdf  rank histogram per (QoI, lead) cell, LinReg1 against LinReg7
#   baseline_step_response.pdf    the solver's response to a step in the correction (Sec. 4.3)
#   colour_residual_acf.pdf       residual ACF of LinReg1 and LinReg7, AR(1)/AR(2) fits (Sec. 5)
#   colour_dq_acf.pdf             ACF of dQ in the long runs against the tracked record (Sec. 5)
#
# Every number a figure shows comes from the code that made the paper's tables: the hindcast through
# score_d6.jl (assemble, cell_moments, cell_ratios, rank_histogram_by_lead; policy A, the 87 ICs all
# three baselines keep), the long runs and the dQ ACF as in baseline_extras.jl, the residual as in
# colour_tables.jl (deployed C, training rows 400-4000), the AR fits as lrs_ar_variant.jl's
# (`ar2_ls` on lags 1-20; AR(1) phi = lag-1 ACF). Each figure prints the numbers the text quotes, so
# a mismatch shows here and not in the PDF.
#
# The step response needs `exp_square_HIT/output/TO_LSTM/response/response_kernel.jld2`, written by
# m4_response.jl from the 38 replay runs (desktop only); without it that figure is skipped.
#
# Later rounds: PAPER45_DQ_EXTRA="Label=TO_LRS/<dir>,..." adds closures to the dQ ACF figure
# (e.g. the AR(2) closures once their long runs exist).

get!(ENV, "D6_EXCLUDE_ICS", "170,197,313")            # policy A: the 87 ICs all three keep
include(joinpath(@__DIR__, "score_d6.jl"))            # load_members, restrict, assemble, load_truth, ...
include(joinpath(SRC, "ts_history.jl"))               # HistorySpec, build_history
include(joinpath(SRC, "ts_scaling.jl"))               # scale_input
using CairoMakie

const OUTD = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const FIGDIR = get(ENV, "PAPER_FIG_DIR", joinpath(@__DIR__, "figures", "paper"))
const W = 450                                          # text width, pt (a4wide)
const NQ = 6
const QLAB = [L"Z_{[0,6]}", L"E_{[0,6]}", L"Z_{[7,15]}", L"E_{[7,15]}", L"Z_{[16,32]}", L"E_{[16,32]}"]
"Panel of QoI i in the 2 x 3 layout: Z in the top row, E below, one column per band."
pos(i) = (2 - isodd(i), cld(i, 2))
# Okabe-Ito, validated as a categorical set on white (dataviz validate_palette.js: all checks pass)
const COL = Dict("LinReg1" => to_color("#0072B2"), "LinReg7" => to_color("#D55E00"),
                 "DDN" => to_color("#009E73"))
const C_REF = to_color("#1a1a1a")
const CLOSURES = ("LinReg1", "LinReg7", "DDN")

set_theme!(merge(Theme(fontsize = 8, linewidth = 1.0,
                       Axis = (; xgridvisible = false, ygridvisible = false, spinewidth = 0.6,
                               xtickwidth = 0.6, ytickwidth = 0.6, xticksize = 2.5, yticksize = 2.5,
                               titlesize = 8, titlegap = 2, titlefont = :regular, xlabelpadding = 1, ylabelpadding = 2),
                       Legend = (; framevisible = false, patchsize = (14, 6), rowgap = 0, colgap = 8,
                                 padding = (0, 0, 0, 0))),
                 theme_latexfonts()))

function savefig(name, fig)
    mkpath(FIGDIR)
    p = joinpath(FIGDIR, name)
    save(p, fig; pt_per_unit = 1)
    @printf("wrote %s\n", p)
end

"A 2 x 3 grid of axes, one per QoI, laid out by `pos`."
function qoi_axes(fig; kw...)
    return [Axis(fig[pos(i)...]; title = QLAB[i], kw...) for i in 1:NQ]
end

# --------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------
const TRUTH = load_truth()
const REC = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))

"The three baselines on the ICs all three keep, assembled on the paper's lead grid."
function hindcast()
    ens = Dict(c => load_members(joinpath(OUTD, "D6_$c")) for c in CLOSURES)
    ks = sort(intersect((e.ks for e in values(ens))...))
    grid = filter(<=(minimum(e.nlead for e in values(ens))), LEADS)
    @printf("hindcast: K = %d ICs (excluded %s), leads %s\n", length(ks),
            join(sort(collect(EXCLUDE_ICS)), ","), join(grid, ","))
    out = Dict{String,Any}()
    for c in CLOSURES
        e = restrict(ens[c], ks)
        fc, tr = assemble(e, TRUTH, grid)
        out[c] = (; ens = e, fc, tr, mom = cell_moments(fc, tr))
    end
    return (; d = out, ks, grid, t = out["LinReg1"].ens.t)
end

"Long free runs (q, dQ per replica) of a closure directory under exp_square_HIT/output."
function longruns(dir; pat = r"^(DDN_)?data_online_tsim100\.0_replica\d\.jld2$")
    d = joinpath(OUTD, dir)
    fs = sort(filter(f -> occursin(pat, f), readdir(d)))
    return [jldopen(io -> (q = Float64.(io["data_online"].q), dQ = Float64.(io["data_online"].dQ)),
                    joinpath(d, f)) for f in fs]
end
const LONGDIR = Dict("LinReg1" => "TO_LRS/LinReg1", "LinReg7" => "TO_LRS/LinReg7", "DDN" => "TO_DDN")

# --------------------------------------------------------------------------------------------------
# Sec. 4.1: fans
# --------------------------------------------------------------------------------------------------
"""
Members of LinReg1 and LinReg7 from the same ICs (same member seeds), one QoI, four non-overlapping
ICs from the first scored one on (`plot_d6_fans.jl`'s stride rule). The QoI is chosen by a rule, not
by eye: the one whose LinReg7 median spread-skill ratio over the six leads is closest to the median
over all 36 cells.
"""
function fig_fans(H; nfans = 4)
    r7 = cell_ratios(H.d["LinReg7"].mom)
    med = median(vec(r7))
    qi = argmin(i -> abs(median(r7[i, :]) - med), 1:NQ)
    @printf("fans: LinReg7 median ratio %.2f overall; per QoI %s -> QoI %s\n", med,
            join((@sprintf("%.2f", median(r7[i, :])) for i in 1:NQ), " "), LABELS[qi])
    dt = DT
    files = Dict(c => H.d[c].ens.files for c in ("LinReg1", "LinReg7"))
    nk = Dict(k => jldopen(f -> f["n_k"], last(first(files["LinReg1"][k]))) for k in H.ks)
    ks = sort(H.ks; by = k -> nk[k])
    ncol = jldopen(f -> size(f["q"], 2), last(first(files["LinReg1"][ks[1]])))
    every = max(1, ceil(Int, (ncol - 1) * dt / minimum(diff([nk[k] for k in ks]) .* dt)))
    kept = ks[1:every:end][1:nfans]
    @printf("      ICs %s (every %d), t = %s TU\n", join(kept, ","), every,
            join((@sprintf("%.2f", nk[k] * dt) for k in kept), ", "))
    q_ref = TRUTH.q
    t0 = nk[kept[1]] * dt - 0.4
    t1 = (nk[kept[end]] + ncol - 1) * dt + 0.4
    cr = max(1, round(Int, t0 / dt)):min(size(q_ref, 2), round(Int, t1 / dt) + 1)
    fig = Figure(size = (W, 215), figure_padding = (2, 6, 2, 2))
    axs = Axis[]
    for (row, c) in enumerate(("LinReg1", "LinReg7"))
        ax = Axis(fig[row, 1]; ylabel = QLAB[qi], xlabel = row == 2 ? "t (TU)" : "",
                  xticklabelsvisible = row == 2)
        push!(axs, ax)
        for k in kept
            nwarm = jldopen(f -> f["nwarm"], last(first(files[c][k])))
            for (_, p) in files[c][k]
                q = jldopen(f -> Float64.(f["q"]), p)
                tt = ((nk[k] + nwarm):(nk[k] + size(q, 2) - 1)) .* dt
                lines!(ax, tt, q[qi, (nwarm + 1):end]; color = (COL[c], 0.45), linewidth = 0.5)
            end
            vlines!(ax, [(nk[k] + nwarm) * dt]; color = (:gray40, 0.8), linestyle = :dash,
                    linewidth = 0.5)
        end
        lines!(ax, (cr .- 1) .* dt, q_ref[qi, cr]; color = C_REF, linewidth = 0.9)
        text!(ax, 0.005, 0.97; text = c, space = :relative, align = (:left, :top),
              color = COL[c], font = :bold)
        xlims!(ax, t0, t1)
    end
    linkyaxes!(axs...)
    rowgap!(fig.layout, 4)
    savefig("baseline_fans.pdf", fig)
    return qi
end

# --------------------------------------------------------------------------------------------------
# Sec. 4.1: spread-skill ratio by lead
# --------------------------------------------------------------------------------------------------
"Per-cell ratio with a 90 % moving-block bootstrap interval over the ICs (Sec. 3.6's blocks)."
function ratio_ci(mom, t; nboot = 10_000, level = 0.90, rng = Xoshiro(SEED))
    o = sortperm(t)
    b = ic_blocklen(t)
    K = length(t)
    bs = Array{Float64}(undef, size(mom.V, 2), size(mom.V, 3), nboot)
    for r in 1:nboot
        bs[:, :, r] = cell_ratios(mom, o[block_bootstrap_indices(K, b, rng)])
    end
    a = (1 - level) / 2
    return mapslices(v -> quantile(v, a), bs; dims = 3)[:, :, 1],
           mapslices(v -> quantile(v, 1 - a), bs; dims = 3)[:, :, 1]
end

"Long-run spread-skill per QoI: the five replicas as members, the reference as truth, every step."
function longrun_ratio(c)
    rs = longruns(LONGDIR[c])
    K = min(minimum(size(r.q, 2) for r in rs), size(TRUTH.q, 2))
    arr = Array{Float64}(undef, K, NQ, length(rs))
    for (j, r) in enumerate(rs)
        arr[:, :, j] .= transpose(r.q[:, 1:K])
    end
    return spread_skill(arr, transpose(TRUTH.q[:, 1:K]))
end

function fig_spread(H)
    x = H.grid .* DT
    xl = 6.0                                     # where the long-run marker sits
    fig = Figure(size = (W, 250), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; xscale = log10, yscale = log10,
                   xticks = ([x[[1, 3, 5, 6]]; xl], [[@sprintf("%g", v) for v in x[[1, 3, 5, 6]]]; "long"]),
                   yticks = [0.1, 0.25, 0.5, 0.8, 1.25, 2], ytickformat = vs -> [@sprintf("%g", v) for v in vs])
    lr = Dict(c => longrun_ratio(c) for c in CLOSURES)
    for c in CLOSURES
        r = cell_ratios(H.d[c].mom)
        lo, hi = ratio_ci(H.d[c].mom, H.t)
        @printf("spread: %-8s calibrated %2d / 36, median %.2f; long run pooled %.2f, per QoI %s\n", c,
                calibrated_count(r), median(vec(r)), lr[c].ratio, join((@sprintf("%.2f", v) for v in lr[c].per_qoi), " "))
        for i in 1:NQ
            ax = axs[i]
            band!(ax, x, lo[i, :], hi[i, :]; color = (COL[c], 0.18))
            lines!(ax, x, r[i, :]; color = COL[c], linewidth = 1.0)
            scatter!(ax, x, r[i, :]; color = COL[c], markersize = 3.5)
            scatter!(ax, [xl], [lr[c].per_qoi[i]]; color = COL[c], marker = :diamond, markersize = 5)
        end
    end
    for (i, ax) in enumerate(axs)
        hspan!(ax, CAL_BAND...; color = (:gray50, 0.13))
        vlines!(ax, [sqrt(x[end] * xl)]; color = :gray60, linewidth = 0.5, linestyle = :dot)
        xlims!(ax, x[1] / 1.4, xl * 1.4)
        ylims!(ax, 0.09, 2.6)
        r, cc = pos(i)
        r == 2 && (ax.xlabel = "lead (TU)")
        r == 1 && (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "spread-skill ratio") : (ax.yticklabelsvisible = false)
    end
    Legend(fig[3, 1:3], [[LineElement(color = COL[c]), MarkerElement(color = COL[c], marker = :circle, markersize = 3.5)]
                         for c in CLOSURES], collect(CLOSURES); orientation = :horizontal, tellheight = true)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 4)
    savefig("baseline_spread_skill.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# Sec. 4.1: rank histograms
# --------------------------------------------------------------------------------------------------
function fig_ranks(H)
    rh = Dict(c => rank_histogram_by_lead(H.d[c].fc, H.d[c].tr; grid = H.grid, rng = Xoshiro(SEED),
                                          nboot = 1000) for c in ("LinReg1", "LinReg7"))
    K, M = length(H.ks), size(H.d["LinReg1"].fc, 3)
    expd = K / (M + 1)
    for c in ("LinReg1", "LinReg7")
        hs = collect(Iterators.flatten(rh[c].hist))
        @printf("ranks: %-8s flat %2d, U %2d, cap %2d, slope+ %2d, slope- %2d of 36\n", c,
                count(is_flat, hs), count(h -> h.convexity_ci[1] > 0, hs),
                count(h -> h.convexity_ci[2] < 0, hs), count(h -> h.slope_ci[1] > 0, hs),
                count(h -> h.slope_ci[2] < 0, hs))
    end
    L = length(H.grid)
    fig = Figure(size = (W, 330), figure_padding = (2, 6, 2, 2))
    ymax = 5.3
    for i in 1:NQ, j in 1:L
        ax = Axis(fig[i, j]; xticksvisible = false, xticklabelsvisible = false,
                  yticks = [0, 2, 4], yticklabelsvisible = j == 1, yticksvisible = j == 1,
                  title = i == 1 ? @sprintf("%g TU", H.grid[j] * DT) : "",
                  ylabel = j == 1 ? QLAB[i] : "")
        c1 = rh["LinReg1"].hist[i][j].counts ./ expd
        c7 = rh["LinReg7"].hist[i][j].counts ./ expd
        any(>(ymax), c7) && @printf("ranks: %s lead %d LinReg7 bin max %.2f clipped at %.1f\n",
                                    LABELS[i], H.grid[j], maximum(c7), ymax)
        barplot!(ax, 1:(M + 1), c1; color = (COL["LinReg1"], 0.55), gap = 0.12, strokewidth = 0)
        stairs!(ax, [0.5; (1:(M + 1)) .+ 0.5], [c7; c7[end]]; step = :pre, color = COL["LinReg7"],
                linewidth = 0.9)
        hlines!(ax, [1.0]; color = :gray30, linewidth = 0.5, linestyle = :dash)
        xlims!(ax, 0.5, M + 1.5); ylims!(ax, 0, ymax)
    end
    Label(fig[NQ + 1, 1:L], "rank of the truth among the 10 members (1 to 11), per lead";
          tellwidth = false, padding = (0, 0, 0, 0))
    Legend(fig[NQ + 2, 1:L], [PolyElement(color = (COL["LinReg1"], 0.55)), LineElement(color = COL["LinReg7"])],
           ["LinReg1", "LinReg7"]; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 3)
    savefig("baseline_rank_histograms.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# Sec. 4.3: step response
# --------------------------------------------------------------------------------------------------
function fig_step()
    f = joinpath(OUTD, "TO_LSTM", "response", "response_kernel.jld2")
    if !isfile(f)
        println("step: no $f -- skipped (run analysis/m4_response.jl on the desktop, or copy the file)")
        return nothing
    end
    S, ms, K = load(f, "S", "ms", "K")          # S[:, j, k, im]: q*_{m+k} per unit step in j
    k = 1:K
    fig = Figure(size = (W, 230), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig)
    for i in 1:NQ
        s = S[i, i, :, :]
        sm = vec(mean(s; dims = 2))
        @printf("step: %-8s S_k at k = 1, 10, 25, 50, 100, 200: %s\n", LABELS[i],
                join((@sprintf("%.2f", sm[kk]) for kk in (1, 10, 25, 50, 100, 200) if kk <= K), " "))
        ax = axs[i]
        lines!(ax, [0, K], [0, K]; color = :gray55, linestyle = :dash, linewidth = 0.6)
        band!(ax, k, vec(minimum(s; dims = 2)), vec(maximum(s; dims = 2)); color = (C_REF, 0.15))
        lines!(ax, k, sm; color = C_REF)
        r, cc = pos(i)
        r == 2 && (ax.xlabel = "steps after the step")
        cc == 1 && (ax.ylabel = "response per unit step")
        xlims!(ax, 0, K)
    end
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("baseline_step_response.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# Sec. 5: residual ACF with AR fits
# --------------------------------------------------------------------------------------------------
const LAGAX = (; xscale = log10, xticks = [1, 2, 5, 10, 20, 50, 100, 200])

# AR(2) fitted to an ACF, verbatim from m0c_checks.jl (`ar2_acf`, `ar2_stationary`, `ar2_ls`), which
# lrs_ar_variant.jl's `fit_ar` calls on lags 1-20; copied because that file needs RikFlow.
function ar2_acf(p1, p2, L)
    r = zeros(L + 1)
    r[1] = 1
    r[2] = p1 / (1 - p2)
    for k in 2:L
        r[k + 1] = p1 * r[k] + p2 * r[k - 1]
    end
    return r
end
ar2_stationary(p1, p2) = abs(p2) < 1 && p2 + p1 < 1 && p2 - p1 < 1
function ar2_ls(racf; L = 20)
    best = (Inf, 0.0, 0.0)
    for p1 in range(-1.99, 1.99; length = 399), p2 in range(-0.99, 0.99; length = 199)
        ar2_stationary(p1, p2) || continue
        r = ar2_acf(p1, p2, L)
        e = sum(abs2, r[2:end] .- racf[2:(L + 1)])
        e < best[1] && (best = (e, p1, p2))
    end
    _, p1, p2 = best
    for s in (0.005, 0.001, 0.0002), _ in 1:3
        for d1 in (-2s, -s, 0, s, 2s), d2 in (-2s, -s, 0, s, 2s)
            a, b = p1 + d1, p2 + d2
            ar2_stationary(a, b) || continue
            e = sum(abs2, ar2_acf(a, b, L)[2:end] .- racf[2:(L + 1)])
            e < best[1] && (best = (e, a, b))
        end
        _, p1, p2 = best
    end
    return p1, p2
end

"The deployed model's residual on the training rows (steps 400-4000), as colour_tables.jl builds it."
function train_residual(name)
    m = load(joinpath(OUTD, "TO_LRS", name, "LinReg.jld2"))
    C = permutedims(Matrix{Float64}(m["c"]))
    s = m["scaling"].in_scaling
    hist = HistorySpec(; h = m["hist_len"], n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    X, Y, _ = build_history(hist, scale_input(REC["q_star"][:, 400:3999], s), scale_input(REC["q"][:, 400:4000], s))
    return Y .- X * C
end

function fig_resacf(; L = 200)
    lags = 1:L                                   # log lag axis: lag 0 (= 1) is not drawn
    R = Dict(c => train_residual(c) for c in ("LinReg1", "LinReg7"))
    fig = Figure(size = (W, 240), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; LAGAX...)
    for i in 1:NQ
        ax = axs[i]
        hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
        a1 = autocorr(R["LinReg1"][:, i], L)
        a7 = autocorr(R["LinReg7"][:, i], L)
        p1, p2 = ar2_ls(a7; L = 20)
        f2 = ar2_acf(p1, p2, L)
        f1 = a7[2] .^ (0:L)
        q1, q2 = ar2_ls(a1; L = 20)
        @printf("resacf: %-8s rho1 LinReg1 %.2f, LinReg7 %.2f | LinReg7 AR(2) phi = (%.3f, %.3f), fitted rho1 %.2f, max |fit - data| lags 1-20 %.3f; min ACF lags 1-40 data %+.3f, AR(2) %+.3f; LRV data %.1f AR2 %.1f AR1 %.1f | LinReg1 AR(2) phi = (%.3f, %.3f)\n",
                LABELS[i], a1[2], a7[2], p1, p2, f2[2], maximum(abs, f2[2:21] .- a7[2:21]),
                minimum(a7[2:41]), minimum(f2[2:41]), 1 + 2sum(a7[2:end]), 1 + 2sum(f2[2:end]),
                1 + 2sum(f1[2:end]), q1, q2)
        lines!(ax, lags, a1[2:end]; color = COL["LinReg1"], linewidth = 1.1)
        lines!(ax, lags, a7[2:end]; color = COL["LinReg7"], linewidth = 1.1)
        lines!(ax, lags, f2[2:end]; color = C_REF, linewidth = 0.8, linestyle = :dash)
        lines!(ax, lags, f1[2:end]; color = C_REF, linewidth = 0.8, linestyle = :dot)
        r, cc = pos(i)
        r == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
        xlims!(ax, 1, L); ylims!(ax, -0.3, 1.02)
    end
    Legend(fig[3, 1:3], [LineElement(color = COL["LinReg1"]), LineElement(color = COL["LinReg7"]),
                         LineElement(color = C_REF, linestyle = :dash), LineElement(color = C_REF, linestyle = :dot)],
           ["LinReg1", "LinReg7", "AR(2) fit to LinReg7", "AR(1) fit to LinReg7"]; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("colour_residual_acf.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# Sec. 5: ACF of dQ in the long runs against the tracked record
# --------------------------------------------------------------------------------------------------
function fig_dqacf(; L = 200)
    lags = 1:L                                   # log lag axis: lag 0 (= 1) is not drawn
    sets = [("LinReg1", LONGDIR["LinReg1"], COL["LinReg1"]), ("LinReg7", LONGDIR["LinReg7"], COL["LinReg7"])]
    extra = get(ENV, "PAPER45_DQ_EXTRA", "")
    extracol = [to_color("#009E73"), to_color("#CC79A7"), to_color("#56B4E9")]
    for (j, s) in enumerate(filter(!isempty, split(extra, ",")))
        lab, dir = split(s, "=")
        push!(sets, (String(lab), String(dir), extracol[j]))
    end
    dqref = REC["q"][:, 2:end] .- REC["q_star"]
    fig = Figure(size = (W, 240), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; LAGAX...)
    runs = Dict(dir => longruns(dir) for (_, dir, _) in sets)
    for i in 1:NQ
        ar = autocorr(collect(dqref[i, :]), L)
        ax = axs[i]
        hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
        for (lab, dir, col) in sets
            # one line per run, not a mean: a single run's burst (LinReg1 replica 4, E[0,6], 30-40 TU:
            # dQ alternating in sign step to step) would otherwise move the mean and stay invisible
            A = reduce(hcat, [autocorr(r.dQ[i, 101:end], L) for r in runs[dir]])
            am = vec(mean(A; dims = 2))
            @printf("dqacf: %-8s %-8s lag1 %.2f (record %.2f; runs %s), max |mean - record| %.2f, per run %s\n",
                    LABELS[i], lab, am[2], ar[2], join((@sprintf("%.2f", v) for v in A[2, :]), " "),
                    maximum(abs, am[2:end] .- ar[2:end]),
                    join((@sprintf("%.2f", maximum(abs, A[2:end, j] .- ar[2:end])) for j in axes(A, 2)), " "))
            for j in axes(A, 2)
                lines!(ax, lags, A[2:end, j]; color = (col, 0.7), linewidth = 0.6)
            end
        end
        lines!(ax, lags, ar[2:end]; color = C_REF, linewidth = 1.1, linestyle = :dash)
        r, cc = pos(i)
        r == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
        xlims!(ax, 1, L); ylims!(ax, -0.3, 1.02)
    end
    Legend(fig[3, 1:3], [[LineElement(color = C_REF, linestyle = :dash)];
                         [LineElement(color = col) for (_, _, col) in sets]],
           ["tracked record"; [lab * ", long runs" for (lab, _, _) in sets]]; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("colour_dq_acf.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    want = isempty(ARGS) ? ["fans", "spread", "ranks", "step", "resacf", "dqacf"] : ARGS
    if any(in(want), ("fans", "spread", "ranks"))
        H = hindcast()
        "fans" in want && fig_fans(H)
        "spread" in want && fig_spread(H)
        "ranks" in want && fig_ranks(H)
    end
    "step" in want && fig_step()
    "resacf" in want && fig_resacf()
    "dqacf" in want && fig_dqacf()
end
