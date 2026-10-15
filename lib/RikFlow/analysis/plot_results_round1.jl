# Paper Sec. 7.1 (results, step 1), first closure: S1.1 = LinReg7 + AR(2) (Rik, 2026-10-15). Three figures:
#
#   results_s11_fans.pdf      held-out hindcast fans, 6 QoIs x {LinReg1, LinReg7, S1.1}, the four ICs of Fig. 2
#   results_s11_longrun.pdf   long-run QoI distributions (5 runs x 100 TU pooled) against the reference,
#                             with the pooled KS distance per QoI and closure
#   results_s11_dqacf.pdf     ACF of the correction in the long runs (gated steps and the h after them left
#                             out, as Fig. 6), S1.1 and LinReg7 against the tracked record
#
# Usage (from lib/RikFlow): julia --project=analysis analysis/plot_results_round1.jl [fans] [longrun] [dqacf]

include(joinpath(@__DIR__, "plot_paper_secs45.jl"))

const S11 = "LinReg7_ar2n"
const R1_SETS = (("LinReg1", "TO_LRS/LinReg1", COL["LinReg1"]), ("LinReg7", "TO_LRS/LinReg7", COL["LinReg7"]),
                 ("LinReg7 + AR(2)", "TO_LRS/" * S11, to_color("#009E73")))

function fig_r1_fans(; tics = (10.25, 15.0, 20.0, 24.75))
    names = ("LinReg1", "LinReg7", S11)
    labels = ("LinReg1", "LinReg7", "LinReg7 + AR(2)")
    fig = Figure(size = (W, 560), figure_padding = (2, 8, 2, 2))
    q_ref = TRUTH.q
    t0, t1 = first(tics) - 0.4, last(tics) + 3.4
    cr = round(Int, t0 / DT):round(Int, t1 / DT)
    for (c, name) in enumerate(names)
        ens = load_members(joinpath(OUTD, "D6_" * name))
        col = R1_SETS[c][3]
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
    savefig("results_s11_fans.pdf", fig)
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
    savefig("results_s11_longrun.pdf", fig)
end

"""
Long runs as in results.md's `fig8_online_*` (Rik, 2026-10-15): per QoI a wide panel with the five
100 TU trajectories (thin, rasterized) and the reference (black), and beside it, on the same y axis,
the long-term distribution of the pooled runs (filled) against the reference's (black), with the
pooled KS of this closure. `others` overlays further closures' distributions as lines.
"""
function fig_r1_traj(lab; others = (), stride = 20, file = nothing)
    sets = Dict(l => (d, c) for (l, d, c) in R1_SETS)
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
        for (_, ro, co) in oth
            lines!(axd, dens(reduce(vcat, (q[i, :] for q in ro))), mids; color = co, linewidth = 0.8)
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
    el = [[LineElement(color = col, linewidth = 2.5)]; [LineElement(color = co, linewidth = 2.5) for (_, _, co) in oth];
          [LineElement(color = C_REF, linewidth = 2.5)]]
    Legend(fig[NQ + 1, 1:2], el, [lab * " (5 runs)"; [l for (l, _, _) in oth]; "reference"];
           orientation = :horizontal, labelsize = 8)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 3)
    savefig(something(file, "results_traj_" * replace(lab, " + " => "_", "(" => "", ")" => "") * ".pdf"), fig)
end

"The correction's ACF in the long runs, as Fig. 6, for LinReg7 and S1.1 against the record."
function fig_r1_dqacf(; L = 200)
    withenv("PAPER45_DQ_EXTRA" => "LinReg7 + AR(2)=TO_LRS/" * S11) do
        fig_dqacf(; L)
    end
    mv(joinpath(FIGDIR, "colour_dq_acf.pdf"), joinpath(FIGDIR, "results_s11_dqacf.pdf"); force = true)
    fig_dqacf(; L)                       # restore Fig. 6 (colour_dq_acf.pdf) as the paper has it
end

if abspath(PROGRAM_FILE) == @__FILE__
    want = isempty(ARGS) ? ["fans", "longrun", "dqacf"] : ARGS
    "fans" in want && fig_r1_fans()
    "longrun" in want && fig_r1_longrun()
    "dqacf" in want && fig_r1_dqacf()
    if "traj" in want
        fig_r1_traj("LinReg7 + AR(2)"; others = ("LinReg1", "LinReg7"), file = "results_s11_traj.pdf")
        fig_r1_traj("LinReg1"; file = "results_traj_LinReg1.pdf")
        fig_r1_traj("LinReg7"; file = "results_traj_LinReg7.pdf")
    end
end
