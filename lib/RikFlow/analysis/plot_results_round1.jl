# Paper Sec. 7.1 (results, step 1), first closure: S1.1 = LinReg7 + AR(2) (Rik, 2026-10-15). Three figures:
#
#   results_s11_fans.pdf      held-out hindcast fans, 6 QoIs x {LinReg1, LinReg7, S1.1}, the four ICs of Fig. 2
#   results_s11_longrun.pdf   long-run QoI distributions (5 runs x 100 TU pooled) against the reference,
#                             with the pooled KS distance per QoI and closure
#   results_s11_dqacf.pdf     ACF of the correction in the long runs (gated steps and the h after them left
#                             out, as Fig. 6), S1.1, LinReg7 + AR(1) (`LinReg7_ar1ls`, Sec. 7.1.2, added
#                             2026-10-15) and LinReg7 against the tracked record
#
# Usage (from lib/RikFlow): julia --project=analysis analysis/plot_results_round1.jl [--step s2] [fans] [longrun] [dqacf] [traj]
#
# 🔑 `--step <tag>` (2026-10-16) picks the closure in the third column and the file names
# results_<tag>_*.pdf: s11 (default) = LinReg7 + AR(2); s2 = LinReg1 + power-law scale (Sec. 7.2).
# `--step s1` (2026-10-21) draws the three figures of Sec. 7.1 as a whole instead (fig_s1_*).

include(joinpath(@__DIR__, "plot_paper_secs45.jl"))

const STEPS = Dict("s11" => ("LinReg7_ar2n", "LinReg7 + AR(2)"),
                   "s1ar1" => ("LinReg7_ar1ls", "LinReg7 + AR(1)"),     # Sec. 7.1.1 since 2026-10-20 (AR(1) first)
                   "s1" => ("Splice1_E0x7_ar2n", "LinReg^E + AR(2)"),   # step 1 as a whole, Figs 13-15 (below)
                   "s2all" => ("Splice1_E0x7_ar2n_pl", "LinReg^E + AR(2) + power-law scale"),   # step 2, Figs 17-18
                   "s2" => ("LinReg1_pl", "LinReg1 + power-law scale"))
const STEP = let i = findfirst(==("--step"), ARGS)
    i === nothing ? "s11" : ARGS[i + 1]
end
haskey(STEPS, STEP) || error("--step $STEP: known steps are $(join(sort(collect(keys(STEPS))), ", "))")
const S11, SLAB = STEPS[STEP]           # (S11 kept as the name: the closure of the third column)
const S11_AR1 = "LinReg7_ar1ls"          # LinReg7 + AR(1) by `ar1_ls` (2026-10-15), on S1.1's ACF figure
# the closures added to the step's correction-ACF figure (PAPER45_DQ_EXTRA)
const DQ_EXTRA = STEP == "s11" ? SLAB * "=TO_LRS/" * S11 * ",LinReg7 + AR(1)=TO_LRS/" * S11_AR1 :
                 SLAB * "=TO_LRS/" * S11
const R1_SETS = (("LinReg1", "TO_LRS/LinReg1", COL["LinReg1"]), ("LinReg7", "TO_LRS/LinReg7", COL["LinReg7"]),
                 (SLAB, "TO_LRS/" * S11, to_color("#009E73")))

fig_r1_fans(; tics = (10.25, 15.0, 20.0, 24.75)) =
    fans_figure(("LinReg1", "LinReg7", S11), ("LinReg1", "LinReg7", SLAB), [s[3] for s in R1_SETS],
                "results_$(STEP)_fans.pdf"; tics)

"Hindcast fans, one column per closure (`names` = D6 directories), from the four ICs of Fig. 2."
function fans_figure(names, labels, cols, file; tics = (10.25, 15.0, 20.0, 24.75))
    fig = Figure(size = (W, 560), figure_padding = (2, 8, 2, 2))
    q_ref = TRUTH.q
    t0, t1 = first(tics) - 0.4, last(tics) + 3.4
    cr = round(Int, t0 / DT):round(Int, t1 / DT)
    for (c, name) in enumerate(names)
        ens = load_members(joinpath(OUTD, "D6_" * name))
        col = cols[c]
        ks = [k for k in ens.ks if any(t -> isapprox(0.25 * (k - 1), t; atol = 1e-6), tics)]
        for i in 1:NQ
            ax = Axis(fig[i, c]; xticklabelsvisible = i == NQ, yticklabelsvisible = c == 1,
                      xticklabelsize = 7, yticklabelsize = 7, xlabelsize = 9, ylabelsize = 9)
            i == 1 && (ax.title = labels[c]; ax.titlesize = 9)
            c == 1 && (ax.ylabel = QLAB[i])
            i == NQ && (ax.xlabel = "t (TU)")
            for k in ks
                nk = jldopen(f -> f["n_k"], last(first(ens.files[k])))
                nwarm = jldopen(f -> f["nwarm"], last(first(ens.files[k])))
                for (_, p) in ens.files[k]
                    q = jldopen(f -> Float64.(f["q"]), p)
                    tt = ((nk + nwarm):(nk + size(q, 2) - 1)) .* DT
                    lines!(ax, tt, q[i, (nwarm + 1):end]; color = (col, 0.5), linewidth = 0.45)
                end
                vlines!(ax, [(nk + nwarm) * DT]; color = (:gray40, 0.8), linestyle = :dash, linewidth = 0.5)
            end
            lines!(ax, (cr .- 1) .* DT, q_ref[i, cr]; color = C_REF, linewidth = 0.8)
            xlims!(ax, t0, t1)
        end
    end
    for i in 1:NQ
        linkyaxes!(contents(fig[i, :])...)
    end
    rowgap!(fig.layout, 2); colgap!(fig.layout, 4)
    savefig(file, fig)
end

"Long-run distributions, pooled over the 5 runs and every step, against the reference; KS per QoI."
function fig_r1_longrun(; nbin = 60)
    ref = TRUTH.q
    runs = Dict(lab => [Float64.(r.q) for r in longruns(dir)] for (lab, dir, _) in R1_SETS)
    fig = Figure(size = (W, 310), figure_padding = (2, 8, 2, 2))
    axs = qoi_axes(fig; yticklabelsvisible = false, xticklabelsize = 7, yticklabelsize = 7)
    for i in 1:NQ
        ax = axs[i]
        lo, hi = quantile(ref[i, :], 0.0005), quantile(ref[i, :], 0.9995)
        all_x = vcat(ref[i, :], (r[i, :] for (lab, _, _) in R1_SETS for r in runs[lab])...)
        lo = min(lo, quantile(all_x, 0.001)); hi = max(hi, quantile(all_x, 0.999))
        edges = range(lo, hi; length = nbin + 1)
        mids = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
        dens(x) = [count(v -> a <= v < b, x) for (a, b) in zip(edges[1:(end - 1)], edges[2:end])] ./ (length(x) * step(edges))
        txt = String[]
        for (lab, _, col) in R1_SETS
            x = reduce(vcat, (r[i, :] for r in runs[lab]))
            lines!(ax, mids, dens(x); color = col, linewidth = 1.0)
            ks = ensemble_ks(runs[lab], ref).per_qoi[i]
            push!(txt, @sprintf("%.2f", ks))
        end
        lines!(ax, mids, dens(ref[i, :]); color = C_REF, linewidth = 1.1, linestyle = :dash)
        text!(ax, 0.02, 0.97; text = "KS
" * join(txt, "
"), space = :relative, align = (:left, :top),
              fontsize = 6.5)
        rr, cc = pos(i)
        rr == 2 && (ax.xlabel = "level")
        cc == 1 && (ax.ylabel = "density")
    end
    tot = [@sprintf("%s, KS %.2f", lab, ensemble_ks(runs[lab], ref).total) for (lab, _, _) in R1_SETS]
    Legend(fig[3, 1:3], [[LineElement(color = col) for (_, _, col) in R1_SETS];
                         [LineElement(color = C_REF, linestyle = :dash)]],
           [tot; "reference"]; orientation = :horizontal, labelsize = 8, nbanks = 2)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    for (lab, _, _) in R1_SETS
        e = ensemble_ks(runs[lab], ref)
        @printf("longrun: %-16s pooled KS per QoI %s, summed %.2f\n", lab,
                join((@sprintf("%.2f", v) for v in e.per_qoi), " "), e.total)
    end
    savefig("results_$(STEP)_longrun.pdf", fig)
end

"""
Long runs as in results.md's `fig8_online_*` (Rik, 2026-10-15): per QoI a wide panel with the five
100 TU trajectories (thin, rasterized) and the reference (black), and beside it, on the same y axis,
the long-term distribution of the pooled runs (filled) against the reference's (black), with the
pooled KS of this closure. `others` overlays further closures' distributions as lines.
"""
function fig_r1_traj(lab; others = (), stride = 20, file = nothing, table = R1_SETS, disp = identity,
                     runs_label = nothing, styles = Dict{String,Symbol}(), nbanks = 1)
    sets = Dict(l => (d, c) for (l, d, c) in table)
    dir, col = sets[lab]
    runs = [Float64.(r.q) for r in longruns(dir)]
    oth = [(l, [Float64.(r.q) for r in longruns(sets[l][1])], sets[l][2]) for l in others]
    ref = TRUTH.q
    t = (0:stride:(size(ref, 2) - 1)) .* DT
    ks = ensemble_ks(runs, ref).per_qoi
    fig = Figure(size = (W, 520), figure_padding = (2, 8, 2, 2))
    for i in 1:NQ
        ax = Axis(fig[i, 1]; ylabel = QLAB[i], xticklabelsvisible = i == NQ, xlabel = i == NQ ? "t (TU)" : "",
                  xticklabelsize = 7, yticklabelsize = 7, ylabelsize = 9, xlabelsize = 9)
        axd = Axis(fig[i, 2]; yticklabelsvisible = false, xticklabelsvisible = false,
                   xlabel = i == NQ ? "density" : "", xlabelsize = 9)
        for q in runs
            lines!(ax, t, q[i, 1:stride:length(t) * stride]; color = (col, 0.5), linewidth = 0.4, rasterize = 3)
        end
        lines!(ax, t, ref[i, 1:stride:end]; color = C_REF, linewidth = 0.6, rasterize = 3)
        x = reduce(vcat, (q[i, :] for q in runs))
        lo = min(minimum(ref[i, :]), minimum(x)); hi = max(maximum(ref[i, :]), maximum(x))
        pad = 0.04 * (hi - lo)
        edges = range(lo, hi; length = 61)
        mids = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
        dens(v) = [count(z -> a <= z < b, v) for (a, b) in zip(edges[1:(end - 1)], edges[2:end])] ./ (length(v) * step(edges))
        d = dens(x)
        band!(axd, mids, zeros(length(d)), d; color = (col, 0.35), direction = :y)
        lines!(axd, d, mids; color = col, linewidth = 1.0)
        for (l, ro, co) in oth
            lines!(axd, dens(reduce(vcat, (q[i, :] for q in ro))), mids; color = co, linewidth = 0.8,
                   linestyle = get(styles, l, :solid))
        end
        lines!(axd, dens(ref[i, :]), mids; color = C_REF, linewidth = 1.0)
        text!(axd, 0.95, 0.97; text = @sprintf("KS %.2f", ks[i]), space = :relative, align = (:right, :top),
              fontsize = 7)
        for a in (ax, axd)
            ylims!(a, lo - pad, hi + pad)
        end
        xlims!(ax, 0, t[end]); xlims!(axd, 0, nothing)
        colsize!(fig.layout, 2, Relative(0.16))
    end
    el = [[LineElement(color = col, linewidth = 2.5)];
          [LineElement(color = co, linewidth = 2.5, linestyle = get(styles, l, :solid)) for (l, _, co) in oth];
          [LineElement(color = C_REF, linewidth = 2.5)]]
    Legend(fig[NQ + 1, 1:2], el, [something(runs_label, lab * " (5 runs)"); [disp(l) for (l, _, _) in oth]; "reference"];
           orientation = :horizontal, labelsize = 8, nbanks)
    @printf("traj: %-16s pooled KS per QoI %s, summed %.2f\n", lab, join((@sprintf("%.2f", v) for v in ks), " "), sum(ks))
    for (l, ro, _) in oth
        e = ensemble_ks(ro, ref)
        @printf("traj: %-16s pooled KS per QoI %s, summed %.2f\n", l, join((@sprintf("%.2f", v) for v in e.per_qoi), " "), e.total)
    end
    rowgap!(fig.layout, 3); colgap!(fig.layout, 3)
    savefig(something(file, "results_traj_" * replace(lab, " + " => "_", "(" => "", ")" => "") * ".pdf"), fig)
end

# --------------------------------------------------------------------------------------------------
# 🔑 `--step s1` (Rik, 2026-10-21): Sec. 7.1 told feature by feature ("what each additional feature
# adds"), with one subset of closures per figure and one colour per closure in all three (Okabe-Ito
# plus Fig. 6's violet; validate_palette.js, all pairs: CVD and normal-vision floors pass; orange, pink
# and sky blue below 3:1 on white, relieved by the titles and legends). LinReg^E at lambda = 1 (Rik,
# 2026-10-21: "please use linregE with lambda=1", replacing lambda = 10), the closure that passes clause
# S; its violet is Fig. 6's for lambda = 1.
#   results_s1_fans.pdf    hindcast fans: LinReg7 | LinReg7 + AR(1) | LinReg^E | LinReg^E + AR(2)
#   results_s1_dqacf.pdf   correction ACF in the long runs, mean of the five runs per closure
#   results_s1_traj.pdf    long runs of LinReg^E + AR(2), distributions of LinReg1, LinReg7, LinReg7 + AR(2)
# --------------------------------------------------------------------------------------------------
const S1_SETS = (("LinReg1", "TO_LRS/LinReg1", COL["LinReg1"]),
                 ("LinReg7", "TO_LRS/LinReg7", COL["LinReg7"]),
                 ("LinReg7 + AR(1)", "TO_LRS/LinReg7_ar1ls", to_color("#E69F00")),
                 ("LinReg7 + AR(2)", "TO_LRS/LinReg7_ar2n", to_color("#CC79A7")),
                 ("LinReg^E", "TO_LRS/Splice1_E0x7", to_color("#5B3A9E")),
                 ("LinReg^E + AR(2)", "TO_LRS/Splice1_E0x7_ar2n", to_color("#56B4E9")))
const S1_DISP = Dict("LinReg^E" => L"\mathrm{LinReg}^E", "LinReg^E + AR(2)" => L"\mathrm{LinReg}^E + \mathrm{AR(2)}")
s1disp(k) = get(S1_DISP, k, k)
s1col(k) = only(c for (l, _, c) in S1_SETS if l == k)

function fig_s1_fans()
    cl = ("LinReg7", "LinReg7 + AR(1)", "LinReg^E", "LinReg^E + AR(2)")
    fans_figure(("LinReg7", "LinReg7_ar1ls", "Splice1_E0x7", "Splice1_E0x7_ar2n"), map(s1disp, cl),
                map(s1col, cl), "results_s1_fans.pdf")
end

"The correction's ACF in the long runs, as Fig. 6 but the mean of the five runs per closure."
function fig_s1_dqacf(; L = 200, cl = ("LinReg7", "LinReg7 + AR(1)", "LinReg7 + AR(2)", "LinReg^E",
                                          "LinReg^E + AR(2)"))
    lags = 1:L
    sets = [s for k in cl for s in S1_SETS if s[1] == k]
    dqref = REC["q"][:, 2:end] .- REC["q_star"]
    A = Dict{String,Any}()
    for (k, dir, _) in sets
        rs = longruns(dir)
        kp = [gate_keep(dq_of(r), hist_len_of(dir))[101:end] for r in rs]      # as Fig. 6
        A[k] = [reduce(hcat, [autocorr_masked(dq_of(r)[i, 101:end], m, L) for (r, m) in zip(rs, kp)]) for i in 1:NQ]
        @printf("s1 dqacf: %-16s steps left out per run (firing + %d after): %s\n", k, hist_len_of(dir),
                join((count(!, m) for m in kp), " "))
    end
    fig = Figure(size = (W, 250), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; LAGAX...)
    for i in 1:NQ
        ar = autocorr(collect(dqref[i, :]), L)
        ax = axs[i]
        hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
        for (k, _, col) in sets
            am = vec(mean(A[k][i]; dims = 2))
            @printf("s1 dqacf: %-8s %-16s lag1 %.2f (record %.2f), lag 10 / 50 %.2f / %.2f (record %.2f / %.2f), max |mean - record| %.2f, max |run - mean| %.2f\n",
                    LABELS[i], k, am[2], ar[2], am[11], am[51], ar[11], ar[51], maximum(abs, am[2:end] .- ar[2:end]),
                    maximum(abs, A[k][i][2:end, :] .- am[2:end]))
            lines!(ax, lags, am[2:end]; color = col, linewidth = 1.0)
        end
        lines!(ax, lags, ar[2:end]; color = C_REF, linewidth = 1.1, linestyle = :dash)
        r, cc = pos(i)
        r == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
        xlims!(ax, 1, L); ylims!(ax, -0.3, 1.02)
    end
    Legend(fig[3, 1:3], [[LineElement(color = C_REF, linestyle = :dash)]; [LineElement(color = c) for (_, _, c) in sets]],
           ["tracked record"; [s1disp(k) for (k, _, _) in sets]]; orientation = :horizontal, nbanks = 2)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("results_s1_dqacf.pdf", fig)
end

function fig_s1_traj()
    fig_r1_traj("LinReg^E + AR(2)"; others = ("LinReg1", "LinReg7", "LinReg7 + AR(2)"),
                file = "results_s1_traj.pdf", table = S1_SETS, disp = s1disp,
                runs_label = L"\mathrm{LinReg}^E + \mathrm{AR(2)}\ \mathrm{(5\ runs)}")
    # Fig. res-s11-longrun (Sec. 7.1.2) in the colours of the other three
    fig_r1_traj("LinReg7 + AR(2)"; others = ("LinReg1", "LinReg7"), file = "results_s1_traj_l7ar2.pdf",
                table = S1_SETS)
end

# --------------------------------------------------------------------------------------------------
# 🔑 `--step s2all` (Rik, 2026-10-22: "rework the step-2 figures to show the scale on each base, as
# for step 1 (one subset of closures per figure, the same colours)"; Fig. 19, the correction's ACF,
# dropped: the scale leaves the long-run colour within 0.06 of its base's on every base). The step-1
# closures keep their colours; LinReg1 + scale keeps the green of the earlier step-2 figures, LinReg^E +
# AR(2) + scale is wine. A ninth hue distinct from all eight cannot be had (validate_palette.js, all
# pairs), so LinReg7 + scale is LinReg7's vermillion, dashed (it never shares a figure with LinReg7).
#   results_s2all_fans.pdf   hindcast fans: LinReg1 | + scale | LinReg^E + AR(2) | + scale
#   results_s2all_traj.pdf   long runs of LinReg^E + AR(2) + scale; distributions of its base, LinReg1
#                            and LinReg7 + scale
# --------------------------------------------------------------------------------------------------
const S2_SETS = (S1_SETS...,
                 ("LinReg1 + power-law scale", "TO_LRS/LinReg1_pl", to_color("#009E73")),
                 ("LinReg7 + power-law scale", "TO_LRS/LinReg7_pl", COL["LinReg7"]),
                 ("LinReg^E + AR(2) + power-law scale", "TO_LRS/Splice1_E0x7_ar2n_pl", to_color("#882255")))
const S2_STYLE = Dict("LinReg7 + power-law scale" => :dash)
s2disp(k) = k == "LinReg^E + AR(2) + power-law scale" ? rich("LinReg", superscript("E"), " + AR(2) + power-law scale") :
            s1disp(k)
s2col(k) = only(c for (l, _, c) in S2_SETS if l == k)

fig_s2_fans() = fans_figure(("LinReg1", "LinReg1_pl", "Splice1_E0x7_ar2n", "Splice1_E0x7_ar2n_pl"),
                            ("LinReg1", "+ power-law scale", s1disp("LinReg^E + AR(2)"), "+ power-law scale"),
                            map(s2col, ("LinReg1", "LinReg1 + power-law scale", "LinReg^E + AR(2)",
                                        "LinReg^E + AR(2) + power-law scale")), "results_s2all_fans.pdf")

fig_s2_traj() = fig_r1_traj("LinReg^E + AR(2) + power-law scale";
                            others = ("LinReg^E + AR(2)", "LinReg1", "LinReg7 + power-law scale"),
                            file = "results_s2all_traj.pdf", table = S2_SETS, disp = s2disp,
                            runs_label = rich("LinReg", superscript("E"), " + AR(2) + power-law scale (5 runs)"),
                            styles = S2_STYLE, nbanks = 2)

"""
The correction's ACF in the long runs, as Fig. 6, for LinReg1, LinReg7 and the step's closure against the
record (s11: also LinReg7 + AR(1)).
"""
function fig_r1_dqacf(; L = 200)
    withenv("PAPER45_DQ_EXTRA" => DQ_EXTRA) do
        fig_dqacf(; L)
    end
    mv(joinpath(FIGDIR, "colour_dq_acf.pdf"), joinpath(FIGDIR, "results_$(STEP)_dqacf.pdf"); force = true)
    fig_dqacf(; L, linrege = true)       # restore Fig. 6 (colour_dq_acf.pdf) as the paper has it
end

if abspath(PROGRAM_FILE) == @__FILE__
    i = findfirst(==("--step"), ARGS)
    args = i === nothing ? ARGS : ARGS[setdiff(eachindex(ARGS), (i, i + 1))]
    if STEP == "s1"
        want = isempty(args) ? ["fans", "dqacf", "traj"] : args
        "fans" in want && fig_s1_fans()
        "dqacf" in want && fig_s1_dqacf()
        "traj" in want && fig_s1_traj()
        exit()
    end
    if STEP == "s2all"
        want = isempty(args) ? ["fans", "traj"] : args
        "fans" in want && fig_s2_fans()
        "traj" in want && fig_s2_traj()
        exit()
    end
    want = isempty(args) ? ["fans", "longrun", "dqacf"] : args
    "fans" in want && fig_r1_fans()
    "longrun" in want && fig_r1_longrun()
    "dqacf" in want && fig_r1_dqacf()
    if "traj" in want
        fig_r1_traj(SLAB; others = ("LinReg1", "LinReg7"), file = "results_$(STEP)_traj.pdf")
        if STEP == "s11"                 # the baselines' own panels, once
            fig_r1_traj("LinReg1"; file = "results_traj_LinReg1.pdf")
            fig_r1_traj("LinReg7"; file = "results_traj_LinReg7.pdf")
        end
    end
end
