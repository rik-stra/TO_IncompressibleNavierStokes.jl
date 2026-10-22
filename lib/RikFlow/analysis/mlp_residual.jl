# Paper Sec. 6 (mlp_offline.tex): the residual of the network mean of step 3a (MLP + MVG), offline, by
# the measures of Sec. 5 (colour.tex), against LinReg1's (2026-10-22, Rik: "first fit the MLP offline
# and inspect the residual as is done in section 5").
#
#     PAPER_FIG_DIR=<paper>/figs julia --startup-file=no --project=lib/RikFlow/analysis \
#         lib/RikFlow/analysis/mlp_residual.jl [dir]
#
# Reads `exp_square_HIT/output/TO_LSTM/<dir>/{pred,fits}.jld2` (default dir `step3/MLP_h5_d4_dqinc`), written by
# `tools/mlp_mean_fit.jl`: the predicted correction of every seed on every step of the record, raw
# units. 🔑 The residual is that of the LEVEL, q^{n+1} - (q^{n*} + dQhat), as Sec. 5 defines it for
# LinReg (`full_residual`), so the two closures are compared on one quantity; the network is trained
# on that same difference (mlp_mean_fit.jl), so least squares on its input is LinReg1 exactly.
#
# Prints every number the section quotes; writes
#   mlp_residual_acf.pdf   ACF on 10-100 TU: the correction, LinReg1, the MLP (three seeds)
#   mlp_scale.pdf          spread about the bin mean against the level, as colour_scale.pdf
#   mlp_shape.pdf          distribution under the constant and the power-law scale, as colour_shape_*.pdf
# The functions of plot_paper_secs45.jl (`full_residual`, `qoi_axes`, `savefig`, the theme) are
# reused by inclusion, so LinReg1's numbers here are Sec. 5's.

include(joinpath(@__DIR__, "plot_paper_secs45.jl"))

const MLPDIR = isempty(ARGS) ? "step3/MLP_h5_d4_dqinc" : ARGS[1]
const C_MLP = to_color("#CC79A7")          # Okabe-Ito; validate_palette.js with LinReg1/LinReg7: pass (contrast 2.98: legend kept)
const KMAX = 200

const PRED = load(joinpath(OUTD, "TO_LSTM", MLPDIR, "pred.jld2"))
const FITS = load(joinpath(OUTD, "TO_LSTM", MLPDIR, "fits.jld2"))
const SEEDS = PRED["seeds"]

"An MLP's residual on the record, in `full_residual`'s layout: rows = steps n, q^{n+1} - q^{n*} - dQhat."
function mlp_residual(pred)
    n = PRED["cols"]
    dq = permutedims(REC["q"][:, n .+ 1] .- REC["q_star"][:, n])
    tr = findall(k -> 400 <= k <= 4000, n)
    ho = findall(k -> 10 < k * DT <= 100, n)
    return (; R = dq .- permutedims(pred), qs = permutedims(REC["q_star"][:, n]), tr, ho, n)
end

# LinReg1's residual in raw units (`full_residual` gives it in LinReg's scaled level); its 10-100 TU
# rows are the steps n = 4001-40000, as the MLP's
const RL1 = let r = full_residual("LinReg1"),
                s = load(joinpath(OUTD, "TO_LRS", "LinReg1", "LinReg.jld2"), "scaling").in_scaling
    (; r..., R = r.R .* vec(s.sigma)')
end
const RM = Dict(s => mlp_residual(PRED["pred3"][s]) for s in SEEDS)       # deployed: stage 3 + refitted head
const RLS = mlp_residual(PRED["predls"])                                   # least squares, same regressor = LinReg1

# --------------------------------------------------------------------------------------------------
# 0. the fit: stage 1 against the LS floor (closures.tex, offline check), the choice, the regressor
# --------------------------------------------------------------------------------------------------
function report_fit()
    st = FITS["stage1"]
    @printf("fit: LS floor on 8.2-10 TU (fit 1-8.2 TU): NLL %.3f, 0.5 SSE %.4f per step\n", FITS["ls_val"], FITS["ls_sse"])
    for (k, v) in sort(collect(st); by = first)
        @printf("fit: stage 1 %d x %3d (%6d weights) wd %.0e seed %d: val NLL %.3f, 0.5 SSE %.4f, best @ %d of %d updates (%s)\n",
                FITS["nl"], k[1], v.nparams, k[2], k[3], v.val, v.half_sse, v.best_update, v.h.updates, v.h.stop_reason)
    end
    @printf("fit: chosen width %d, weight decay %.0e (mean validation NLL %s)\n", FITS["H_star"], FITS["wd_star"],
            join((@sprintf("%d/%.0e: %.3f", k[1], k[2], m) for (k, m) in sort(collect(FITS["meanval"]))), ", "))
    # the regressor and target are LinReg1's: least squares on 1-10 TU must reproduce LinReg1
    length(RLS.ho) == length(RL1.ho) || error("10-100 TU rows differ: $(length(RLS.ho)) vs $(length(RL1.ho))")
    d = RLS.R[RLS.ho, :] .- RL1.R[RL1.ho, :]
    @printf("fit: LS (same regressor, 1-10 TU) vs LinReg1 residual on 10-100 TU, rms difference / sd: %s\n",
            join((@sprintf("%.3f", sqrt(mean(abs2, d[:, i])) / std(RL1.R[RL1.ho, i])) for i in 1:NQ), " "))
end

# --------------------------------------------------------------------------------------------------
# 1. one-step error and the nonlinearity used
# --------------------------------------------------------------------------------------------------
"RMS of the residual on 10-100 TU over sd(dQ) there, per QoI (colour.tex's one-step error)."
function onestep(r)
    n = r.n[r.ho]
    sdq = vec(std(REC["q"][:, n .+ 1] .- REC["q_star"][:, n]; dims = 2))
    return vec(sqrt.(mean(abs2, r.R[r.ho, :]; dims = 1))) ./ sdq
end

function report_error()
    # LinReg1 on the same steps n = 4001-40000
    sdq(rows) = vec(std(REC["q"][:, rows .+ 1] .- REC["q_star"][:, rows]; dims = 2))
    e1 = vec(sqrt.(mean(abs2, RL1.R[RL1.ho, :]; dims = 1))) ./ sdq(RM[first(SEEDS)].n[RM[first(SEEDS)].ho])
    @printf("error: one-step RMS / sd(dQ), 10-100 TU: LinReg1 %s\n", join((@sprintf("%.3f", x) for x in e1), " "))
    @printf("error: LS (same regressor)        %s\n", join((@sprintf("%.3f", x) for x in onestep(RLS)), " "))
    for s in SEEDS
        @printf("error: MLP seed %d                %s  (ratio to LinReg1 %s)\n", s,
                join((@sprintf("%.3f", x) for x in onestep(RM[s])), " "),
                join((@sprintf("%.2f", x) for x in onestep(RM[s]) ./ e1), " "))
    end
    # the training window (1-10 TU) too: the network's fit there against LinReg1's
    for s in SEEDS
        tr1 = vec(sqrt.(mean(abs2, RL1.R[RL1.tr, :]; dims = 1)))
        trm = vec(sqrt.(mean(abs2, RM[s].R[RM[s].tr, :]; dims = 1)))
        @printf("error: 1-10 TU, MLP seed %d / LinReg1: %s\n", s, join((@sprintf("%.2f", x) for x in trm ./ tr1), " "))
    end
    # how far the network's correction lies from the linear map's, relative to the linear residual
    for s in SEEDS
        d = PRED["pred3"][s] .- PRED["predls"]
        ho = RM[s].ho
        @printf("nonlin: seed %d, rms(MLP - LS) / rms(LS residual), 10-100 TU: %s\n", s,
                join((@sprintf("%.2f", sqrt(mean(abs2, d[i, ho])) / sqrt(mean(abs2, RLS.R[ho, i]))) for i in 1:NQ), " "))
    end
end

# --------------------------------------------------------------------------------------------------
# 2. colour (colour.tex Sec. 5.1-5.2): ACF on 10-100 TU, lag one and the long-run variance ratio
# --------------------------------------------------------------------------------------------------
acf_ho(r, i) = autocorr(collect(r.R[r.ho, i]), KMAX)       # ts_score.jl, as colour_tables.jl (RF.autocorr)
lrv(a) = 1 + 2sum(a[2:end])

function fig_mlp_acf()
    dqr = (; R = permutedims(REC["q"][:, 2:end] .- REC["q_star"]), ho = findall(k -> 10 < k * DT <= 100, 1:size(REC["q_star"], 2)))
    fig = Figure(size = (W, 260), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; LAGAX...)
    lags = 1:KMAX
    for i in 1:NQ
        ax = axs[i]
        hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
        lines!(ax, lags, acf_ho(dqr, i)[2:end]; color = C_REF, linestyle = :dash, linewidth = 1.0)
        a1 = acf_ho(RL1, i)
        lines!(ax, lags, a1[2:end]; color = COL["LinReg1"], linewidth = 1.0)
        for (k, s) in enumerate(SEEDS)
            am = acf_ho(RM[s], i)
            lines!(ax, lags, am[2:end]; color = C_MLP, linewidth = k == 1 ? 1.1 : 0.6)
            @printf("acf: %-8s MLP seed %d rho1 %.2f, ratio %.1f | LinReg1 rho1 %.2f, ratio %.1f | max |MLP - LinReg1| lags 1-200 %.2f\n",
                    LABELS[i], s, am[2], lrv(am), a1[2], lrv(a1), maximum(abs, am[2:end] .- a1[2:end]))
        end
        rr, cc = pos(i)
        rr == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
        xlims!(ax, 1, KMAX); ylims!(ax, -0.3, 1.02)
    end
    Legend(fig[3, 1:3], [LineElement(color = C_REF, linestyle = :dash), LineElement(color = COL["LinReg1"]),
                         LineElement(color = C_MLP, linewidth = 1.1), LineElement(color = C_MLP, linewidth = 0.6)],
           [L"dQ", "LinReg1", "MLP, seed 1", "MLP, seeds 2, 3"]; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("mlp_residual_acf.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# 3. scale (colour.tex Sec. 5.4): spread about the bin mean against the level; the power law
# --------------------------------------------------------------------------------------------------
"Spread about the mean and the mean, per bin of `x[rows]` cut at `edges` (as `fig_scale`)."
function binned(x, e, rows, edges)
    b = searchsortedfirst.(Ref(edges), x[rows])
    nb = length(edges) + 1
    xm = [exp(mean(log.(x[rows][b .== k]))) for k in 1:nb]
    ym = [std(e[rows[b .== k]]) for k in 1:nb]
    mu = [mean(e[rows[b .== k]]) for k in 1:nb]
    return xm, ym, mu
end

function fig_mlp_scale(; nbin = 10)
    fig = Figure(size = (W, 250), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; xscale = log10, yscale = log10, xticks = [0.5, 1, 1.5],
                   yticks = [0.3, 0.5, 0.7, 1, 1.4, 2], xtickformat = vs -> [@sprintf("%g", v) for v in vs],
                   ytickformat = vs -> [@sprintf("%g", v) for v in vs])
    for i in 1:NQ
        hlines!(axs[i], [1.0]; color = :gray55, linestyle = :dash, linewidth = 0.6)
    end
    for (name, r, col) in (("LinReg1", RL1, COL["LinReg1"]), ("MLP s$(first(SEEDS))", RM[first(SEEDS)], C_MLP))
        sd = vec(std(r.R[r.tr, :]; dims = 1))
        E = r.R ./ sd'
        for i in 1:NQ
            ax = axs[i]
            p = fit_powerlaw_scale(E[r.tr, i], r.qs[r.tr, i])
            x = r.qs[:, i] ./ p.qref
            edges = quantile(x[r.tr], range(0, 1; length = nbin + 1)[2:(end - 1)])
            lo, hi = p.qclip ./ p.qref
            xx = exp.(range(log(0.3), log(2.1); length = 200))
            lines!(ax, xx, exp(p.logvar / 2) .* clamp.(xx, lo, hi) .^ p.beta; color = col, linewidth = 1.1)
            for (rows, filled) in ((r.tr, true), (r.ho, false))
                xm, ym, _ = binned(x, E[:, i], rows, edges)
                scatter!(ax, xm, ym; color = filled ? col : :white, strokecolor = col, strokewidth = 0.7,
                         markersize = 4.5)
            end
            nllc(rows) = mean(0.5 .* (log(2pi) .+ log(mean(abs2, E[r.tr, i])) .+ E[rows, i] .^ 2 ./ mean(abs2, E[r.tr, i])))
            lv(rows) = p.logvar .+ 2p.beta .* log.(clamp.(r.qs[rows, i], p.qclip...) ./ p.qref)
            nllp(rows) = mean(0.5 .* (log(2pi) .+ lv(rows) .+ E[rows, i] .^ 2 .* exp.(-lv(rows))))
            _, str_, mtr = binned(x, E[:, i], r.tr, edges)
            _, sho, mho = binned(x, E[:, i], r.ho, edges)
            @printf("scale: %-8s %-8s beta %.2f, held-out NLL gain %.3f; lowest/highest tenth sd %.2f/%.2f (1-10), %.2f/%.2f (10-100); bin means 1-10 max |.| %.2f, 10-100 %s\n",
                    name, LABELS[i], p.beta, nllc(r.ho) - nllp(r.ho), str_[1], str_[end],
                    sho[1], sho[end], maximum(abs, mtr), join((@sprintf("%+.2f", v) for v in mho), " "))
        end
    end
    for i in 1:NQ
        rr, cc = pos(i)
        ax = axs[i]
        rr == 2 ? (ax.xlabel = L"q^{n*}_i / q_{\mathrm{ref},i}") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "spread / constant scale") : (ax.yticklabelsvisible = false)
        xlims!(ax, 0.3, 2.1); ylims!(ax, 0.25, 2.1)
    end
    mk(c, filled) = MarkerElement(color = filled ? c : :white, strokecolor = c, strokewidth = 0.7,
                                  marker = :circle, markersize = 4.5)
    Legend(fig[3, 1:3],
           Any[[mk(COL["LinReg1"], true), LineElement(color = COL["LinReg1"])],
               [mk(C_MLP, true), LineElement(color = C_MLP)],
               mk(:gray35, true), mk(:gray35, false), LineElement(color = :gray55, linestyle = :dash)],
           ["LinReg1", "MLP", "1–10 TU", "10–100 TU", "constant scale"]; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("mlp_scale.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# 4. shape (colour.tex Sec. 5.4): the MLP's residual under the constant and the power-law scale
# --------------------------------------------------------------------------------------------------
function fig_mlp_shape(; zmax = 6.0, dz = 0.25)
    kurt(x) = (y = x .- mean(x); mean(y .^ 4) / mean(y .^ 2)^2 - 3)
    skw(x) = (y = x .- mean(x); mean(y .^ 3) / mean(y .^ 2)^1.5)
    tail3(x) = mean(abs.((x .- mean(x)) ./ std(x)) .> 3) / 0.0026998
    r = RM[first(SEEDS)]
    E = r.R ./ vec(std(r.R[r.tr, :]; dims = 1))'
    S = map(1:NQ) do i
        p = fit_powerlaw_scale(E[r.tr, i], r.qs[r.tr, i])
        s = exp.(p.logvar / 2 .+ p.beta .* log.(clamp.(r.qs[:, i], p.qclip...) ./ p.qref))
        (E[:, i], E[:, i] ./ s)
    end
    for i in 1:NQ, (lab, x) in (("constant", S[i][1]), ("power law", S[i][2]))
        @printf("shape: MLP %-8s %-9s kurt %5.2f / %5.2f, skew %+.2f / %+.2f, |z|>3 %.1f / %.1f x Gaussian (1-10 / 10-100 TU)\n",
                LABELS[i], lab, kurt(x[r.tr]), kurt(x[r.ho]), skw(x[r.tr]), skw(x[r.ho]), tail3(x[r.tr]), tail3(x[r.ho]))
    end
    edges = collect(-zmax:dz:zmax)
    mids = edges[1:(end - 1)] .+ dz / 2
    fig = Figure(size = (W, 250), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; yscale = log10, yticks = [1e-4, 1e-3, 1e-2, 1e-1, 1],
                   ytickformat = vs -> [@sprintf("%g", v) for v in vs], xticks = -6:2:6)
    zz = range(-zmax, zmax; length = 400)
    for i in 1:NQ
        ax = axs[i]
        lines!(ax, zz, exp.(-zz .^ 2 ./ 2) ./ sqrt(2pi); color = C_REF, linestyle = :dash, linewidth = 0.8)
        for (x, col) in ((S[i][1], :gray55), (S[i][2], C_MLP))
            z = x[r.ho]
            z = (z .- mean(z)) ./ std(z)
            h = [count(e -> lo <= e < lo + dz, z) for lo in edges[1:(end - 1)]] ./ (length(z) * dz)
            keep = h .> 0
            lines!(ax, mids[keep], h[keep]; color = col, linewidth = 1.0)
        end
        rr, cc = pos(i)
        rr == 2 ? (ax.xlabel = "standardized residual") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "density") : (ax.yticklabelsvisible = false)
        xlims!(ax, -zmax, zmax); ylims!(ax, 1e-4, 1)
    end
    Legend(fig[3, 1:3], [LineElement(color = :gray55), LineElement(color = C_MLP),
                         LineElement(color = C_REF, linestyle = :dash)],
           ["÷ constant scale", "÷ power-law scale", L"\mathcal{N}(0,1)"]; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("mlp_shape.pdf", fig)
end

report_fit()
report_error()
fig_mlp_acf()
fig_mlp_scale()
fig_mlp_shape()
