# Paper Sec. 6 (mlp_offline.tex): the residual of the step-3 network means, offline, by the measures of
# Sec. 5 (colour.tex), against LinReg1's -- for any number of networks side by side (2026-10-23, Rik: "take
# both losses forward", the likelihood- and CRPS-trained CNN). Generalises `mlp_residual.jl`.
#
#     PAPER_FIG_DIR=<dir> julia --startup-file=no --project=lib/RikFlow/analysis \
#         lib/RikFlow/analysis/net_residual.jl NLL=step3/CNN_step3_nll CRPS=step3/CNN_step3_crps
#
# Each `Label=dir[,dir...]` names `exp_square_HIT/output/TO_LSTM/<dir>/pred.jld2`, written by `cnn_mean_fit.jl` (or
# `mlp_mean_fit.jl`) with the refit: `pred3[seed]`, the network trained again on all of 1-10 TU (the
# protocol's network), is analysed; `predls`, least squares on 1-10 TU, must reproduce LinReg1 (checked).
# The residual is that of the LEVEL, q^{n+1} - (q^{n*} + dQhat), as Sec. 5 defines it (`full_residual`);
# the networks are trained on that same difference. Prints every number; writes
#   net_residual_acf.pdf   ACF on 10-100 TU: the correction, LinReg1, each network (first seed bold)
#   net_scale.pdf          spread about the bin mean against the level (first seed), as colour_scale.pdf
#   net_shape_<label>.pdf  distribution under the constant and the power-law scale (first seed)
# The functions of plot_paper_secs45.jl (`full_residual`, `qoi_axes`, `savefig`, the theme) are reused by
# inclusion, so LinReg1's numbers are Sec. 5's.

include(joinpath(@__DIR__, "plot_paper_secs45.jl"))

const KMAX = 200
const NETCOL = [to_color("#CC79A7"), to_color("#E69F00"), to_color("#009E73")]   # validate_palette.js with LinReg1: pass
const NET_MAIN = abspath(PROGRAM_FILE) == @__FILE__        # false when net_sweep.jl includes this file
NET_MAIN && isempty(ARGS) && error("usage: net_residual.jl Label=dir [Label=dir ...]")
const NETS = NET_MAIN ? [(; label = String(a[1]), dir = String(a[2])) for a in split.(ARGS, "=")] : []

"A network's residual on the record, in `full_residual`'s layout: rows = steps n, q^{n+1} - q^{n*} - dQhat."
function net_residual(cols, pred)
    dq = permutedims(REC["q"][:, cols .+ 1] .- REC["q_star"][:, cols])
    tr = findall(k -> 400 <= k <= 4000, cols)
    ho = findall(k -> 10 < k * DT <= 100, cols)
    return (; R = dq .- permutedims(pred), qs = permutedims(REC["q_star"][:, cols]), tr, ho, n = cols)
end

# LinReg1's residual in raw units (`full_residual` gives it in LinReg's scaled level); its 10-100 TU rows
# are the steps n = 4001-40000, as every network's at h <= 5... and checked below for each network
const RL1 = let r = full_residual("LinReg1"),
                s = load(joinpath(OUTD, "TO_LRS", "LinReg1", "LinReg.jld2"), "scaling").in_scaling
    (; r..., R = r.R .* vec(s.sigma)')
end

# `Label=dirA,dirB,...`: the seeds of one network fitted in separate runs (one seed per process), merged
"The refitted network(s) of `net.dir` (comma-separated runs), their residual per seed, checked against LinReg1."
function load_net(net)
    r, seeds, meta = Dict{Int,Any}(), Int[], nothing
    for dir in split(net.dir, ",")
        P = load(joinpath(OUTD, "TO_LSTM", dir, "pred.jld2"))
        cols = P["cols"]
        rls = net_residual(cols, P["predls"])
        length(rls.ho) == length(RL1.ho) || error("$(net.label): 10-100 TU rows differ from LinReg1's")
        d = maximum(abs, rls.R[rls.ho, :] .- RL1.R[RL1.ho, :]) / maximum(abs, RL1.R[RL1.ho, :])
        @printf("check %s (%s): least squares on 1-10 TU against LinReg1's residual on 10-100 TU, max rel diff %.1e (h = %d)\n",
                net.label, dir, d, P["h"])
        for s in P["seeds"]
            haskey(r, s) && error("$(net.label): seed $s appears twice")
            r[s] = net_residual(cols, P["pred3"][s]); push!(seeds, s)
        end
        meta = P
    end
    return (; net..., seeds = sort(seeds), r, meta)
end
const NETR = map(load_net, NETS)

pr(v; f = "%.2f") = join((Printf.format(Printf.Format(f), x) for x in v), " ")
rmsq(R, rows) = vec(sqrt.(mean(abs2, R[rows, :]; dims = 1)))

# --------------------------------------------------------------------------------------------------
# 1. one-step error against LinReg1, 10-100 TU and 1-10 TU
# --------------------------------------------------------------------------------------------------
function report_error()
    r0 = NETR[1].r[NETR[1].seeds[1]]
    n = r0.n[r0.ho]                                     # steps 4001-40000, LinReg1's 10-100 TU rows too
    sdq = vec(std(REC["q"][:, n .+ 1] .- REC["q_star"][:, n]; dims = 2))
    @printf("error: one-step rms / sd(dQ), 10-100 TU: LinReg1 %s\n", pr(rmsq(RL1.R, RL1.ho) ./ sdq; f = "%.3f"))
    for net in NETR, s in net.seeds
        r = net.r[s]
        @printf("error: %-5s seed %d: rms / LinReg1, 10-100 TU %s | 1-10 TU %s\n", net.label, s,
                pr(rmsq(r.R, r.ho) ./ rmsq(RL1.R, RL1.ho)), pr(rmsq(r.R, r.tr) ./ rmsq(RL1.R, RL1.tr)))
    end
end

"""
The steps of 10-100 TU at which at least one predicted level q^{n*}_j lies outside its range on 1-10 TU
(the range Sec. 5 clips the power law to): their share; the one-step error inside and outside that range
against LinReg1's on the same steps; the share of the network's excess squared error over LinReg1's that
the outside steps carry; the residual's mean outside, in units of LinReg1's rms. Seed means. The rows of
every network and of LinReg1 are the same steps (checked in `load_net`).
"""
function range_split(net)
    r0 = net.r[net.seeds[1]]
    lo, hi = vec(minimum(r0.qs[r0.tr, :]; dims = 1)), vec(maximum(r0.qs[r0.tr, :]; dims = 1))
    out = [any((r0.qs[k, :] .< lo) .| (r0.qs[k, :] .> hi)) for k in r0.ho]
    io, oo = r0.ho[.!out], r0.ho[out]                 # the network's rows
    i1, o1 = RL1.ho[.!out], RL1.ho[out]               # LinReg1's rows, the same steps
    ns = length(net.seeds)
    sse(R, rows) = vec(sum(abs2, R[rows, :]; dims = 1))
    rin = sum(rmsq(net.r[s].R, io) ./ rmsq(RL1.R, i1) for s in net.seeds) / ns
    rout = sum(rmsq(net.r[s].R, oo) ./ rmsq(RL1.R, o1) for s in net.seeds) / ns
    share = sum((sse(net.r[s].R, oo) .- sse(RL1.R, o1)) ./ (sse(net.r[s].R, r0.ho) .- sse(RL1.R, RL1.ho)) for s in net.seeds) / ns
    bias = sum(vec(mean(net.r[s].R[oo, :]; dims = 1)) for s in net.seeds) / ns ./ rmsq(RL1.R, RL1.ho)
    @printf("range: %-12s %.1f %% of the 10-100 TU steps have a level outside its 1-10 TU range\n", net.label, 100count(out) / length(out))
    @printf("range: %-12s rms / LinReg1 inside %s | outside %s | outside share of the excess SSE %s | mean outside / LinReg1 rms %s\n",
            net.label, pr(rin), pr(rout), pr(share), pr(bias; f = "%+.2f"))
    # the colour without the outside steps: lag pairs with either step outside left out (autocorr_masked)
    keep = .!out
    am = [sum(autocorr_masked(collect(net.r[s].R[r0.ho, i]), keep, KMAX) for s in net.seeds) / ns for i in 1:NQ]
    a1 = [autocorr_masked(collect(RL1.R[RL1.ho, i]), keep, KMAX) for i in 1:NQ]
    @printf("range: %-12s ACF 10-100 TU inside only: rho1 %s | ratio %s (LinReg1 rho1 %s | ratio %s)\n", net.label,
            pr([a[2] for a in am]), pr(lrv.(am); f = "%.1f"), pr([a[2] for a in a1]), pr(lrv.(a1); f = "%.1f"))
end

"""
The network's residual on 10-100 TU as LinReg1's plus the difference of the two predictions,
R_net = R_LR1 + d with d = dQhat_LR1 - dQhat_net (2026-10-23, Rik: "critically analyze ... the weight
decay has a non monotone effect on the ACF"): per QoI, seed means of the share of d in the variance
of R_net, the correlation of d with R_LR1, d's lag-one ACF and long-run variance ratio, and the
lag-one ACF and ratio of R_net against what the two parts give if uncorrelated,
rho_net = (1 - f) rho_LR1 + f rho_d with f = var(d) / var(R_net) (and the same for the ratio).
Also the long-run variance ratio of R_net per seed.
"""
function decompose(net)
    ns = length(net.seeds)
    st = map(net.seeds) do s
        r = net.r[s]
        map(1:NQ) do i
            rn, rl = collect(r.R[r.ho, i]), collect(RL1.R[RL1.ho, i])
            d = rn .- rl
            an, al, ad = autocorr(rn, KMAX), autocorr(rl, KMAX), autocorr(d, KMAX)
            f = var(d) / var(rn)
            (; f, c = cor(d, rl), rho_d = ad[2], L_d = lrv(ad), rho = an[2], L = lrv(an),
             rho_pred = (1 - f) * al[2] + f * ad[2], L_pred = (1 - f) * lrv(al) + f * lrv(ad))
        end
    end
    m(k) = [sum(st[j][i][k] for j in 1:ns) / ns for i in 1:NQ]
    @printf("decomp: %-14s var share of d %s | cor(d, R_LR1) %s | d: rho1 %s, ratio %s\n", net.label,
            pr(m(:f)), pr(m(:c); f = "%+.2f"), pr(m(:rho_d)), pr(m(:L_d); f = "%.0f"))
    @printf("decomp: %-14s rho1 %s (from the parts %s) | ratio %s (from the parts %s) | ratio per seed %s\n",
            net.label, pr(m(:rho)), pr(m(:rho_pred)), pr(m(:L); f = "%.1f"), pr(m(:L_pred); f = "%.1f"),
            join((pr([st[j][i].L for i in 1:NQ]; f = "%.0f") for j in 1:ns), " / "))
end

# --------------------------------------------------------------------------------------------------
# 2. colour: ACF on 10-100 TU, lag one and the long-run variance ratio (colour.tex Sec. 5.1-5.2)
# --------------------------------------------------------------------------------------------------
acf_ho(r, i) = autocorr(collect(r.R[r.ho, i]), KMAX)       # ts_score.jl, as colour_tables.jl
lrv(a) = 1 + 2sum(a[2:end])

function fig_acf()
    dqr = (; R = permutedims(REC["q"][:, 2:end] .- REC["q_star"]), ho = findall(k -> 10 < k * DT <= 100, 1:size(REC["q_star"], 2)))
    fig = Figure(size = (W, 270), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; LAGAX...)
    lags = 1:KMAX
    for i in 1:NQ
        ax = axs[i]
        hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
        lines!(ax, lags, acf_ho(dqr, i)[2:end]; color = C_REF, linestyle = :dash, linewidth = 1.0)
        a1 = acf_ho(RL1, i)
        lines!(ax, lags, a1[2:end]; color = COL["LinReg1"], linewidth = 1.0)
        @printf("acf: %-8s LinReg1 rho1 %.2f, ratio %.1f\n", LABELS[i], a1[2], lrv(a1))
        for (j, net) in enumerate(NETR), (k, s) in enumerate(net.seeds)
            am = acf_ho(net.r[s], i)
            lines!(ax, lags, am[2:end]; color = NETCOL[j], linewidth = k == 1 ? 1.1 : 0.5)
            @printf("acf: %-8s %-5s seed %d rho1 %.2f, ratio %.1f | max |net - LinReg1| lags 1-200 %.2f\n",
                    LABELS[i], net.label, s, am[2], lrv(am), maximum(abs, am[2:end] .- a1[2:end]))
        end
        rr, cc = pos(i)
        rr == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
        xlims!(ax, 1, KMAX); ylims!(ax, -0.3, 1.02)
    end
    els = Any[LineElement(color = C_REF, linestyle = :dash), LineElement(color = COL["LinReg1"])]
    labs = Any[L"dQ", "LinReg1"]
    for (j, net) in enumerate(NETR)
        push!(els, LineElement(color = NETCOL[j], linewidth = 1.1)); push!(labs, "CNN, $(net.label)")
    end
    Legend(fig[3, 1:3], els, labs; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("net_residual_acf.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# 3. scale: spread about the bin mean against the level, the power law (colour.tex Sec. 5.4)
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

function fig_scale(; nbin = 10)
    fig = Figure(size = (W, 255), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; xscale = log10, yscale = log10, xticks = [0.5, 1, 1.5],
                   yticks = [0.3, 0.5, 0.7, 1, 1.4, 2, 3], xtickformat = vs -> [@sprintf("%g", v) for v in vs],
                   ytickformat = vs -> [@sprintf("%g", v) for v in vs])
    for i in 1:NQ
        hlines!(axs[i], [1.0]; color = :gray55, linestyle = :dash, linewidth = 0.6)
    end
    sets = [("LinReg1", RL1, COL["LinReg1"]); [(net.label, net.r[net.seeds[1]], NETCOL[j]) for (j, net) in enumerate(NETR)]]
    for (name, r, col) in sets
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
                scatter!(ax, xm, ym; color = filled ? col : :white, strokecolor = col, strokewidth = 0.7, markersize = 4.5)
            end
            nllc(rows) = mean(0.5 .* (log(2pi) .+ log(mean(abs2, E[r.tr, i])) .+ E[rows, i] .^ 2 ./ mean(abs2, E[r.tr, i])))
            lv(rows) = p.logvar .+ 2p.beta .* log.(clamp.(r.qs[rows, i], p.qclip...) ./ p.qref)
            nllp(rows) = mean(0.5 .* (log(2pi) .+ lv(rows) .+ E[rows, i] .^ 2 .* exp.(-lv(rows))))
            _, str_, mtr = binned(x, E[:, i], r.tr, edges)
            _, sho, mho = binned(x, E[:, i], r.ho, edges)
            @printf("scale: %-8s %-8s beta %.2f, held-out NLL gain %.3f; lowest/highest tenth sd %.2f/%.2f (1-10), %.2f/%.2f (10-100); bin means 1-10 max |.| %.2f, 10-100 %s\n",
                    name, LABELS[i], p.beta, nllc(r.ho) - nllp(r.ho), str_[1], str_[end], sho[1], sho[end],
                    maximum(abs, mtr), pr(mho; f = "%+.2f"))
        end
    end
    for i in 1:NQ
        rr, cc = pos(i)
        ax = axs[i]
        rr == 2 ? (ax.xlabel = L"q^{n*}_i / q_{\mathrm{ref},i}") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "spread / constant scale") : (ax.yticklabelsvisible = false)
        xlims!(ax, 0.3, 2.1); ylims!(ax, 0.25, 3.2)
    end
    mk(c, filled) = MarkerElement(color = filled ? c : :white, strokecolor = c, strokewidth = 0.7, marker = :circle, markersize = 4.5)
    els = Any[[mk(col, true), LineElement(color = col)] for (_, _, col) in sets]
    labs = Any[name == "LinReg1" ? name : "CNN, $name" for (name, _, _) in sets]
    append!(els, [mk(:gray35, true), mk(:gray35, false), LineElement(color = :gray55, linestyle = :dash)])
    append!(labs, ["1–10 TU", "10–100 TU", "constant scale"])
    Legend(fig[3, 1:3], els, labs; orientation = :horizontal, nbanks = 2)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("net_scale.pdf", fig)
end

# --------------------------------------------------------------------------------------------------
# 4. shape: the residual under the constant and the power-law scale (colour.tex Sec. 5.4)
# --------------------------------------------------------------------------------------------------
function fig_shape(net, col; zmax = 6.0, dz = 0.25)
    kurt(x) = (y = x .- mean(x); mean(y .^ 4) / mean(y .^ 2)^2 - 3)
    skw(x) = (y = x .- mean(x); mean(y .^ 3) / mean(y .^ 2)^1.5)
    tail3(x) = mean(abs.((x .- mean(x)) ./ std(x)) .> 3) / 0.0026998
    r = net.r[net.seeds[1]]
    E = r.R ./ vec(std(r.R[r.tr, :]; dims = 1))'
    S = map(1:NQ) do i
        p = fit_powerlaw_scale(E[r.tr, i], r.qs[r.tr, i])
        s = exp.(p.logvar / 2 .+ p.beta .* log.(clamp.(r.qs[:, i], p.qclip...) ./ p.qref))
        (E[:, i], E[:, i] ./ s)
    end
    for i in 1:NQ, (lab, x) in (("constant", S[i][1]), ("power law", S[i][2]))
        @printf("shape: %-5s %-8s %-9s kurt %5.2f / %5.2f, skew %+.2f / %+.2f, |z|>3 %.1f / %.1f x Gaussian (1-10 / 10-100 TU)\n",
                net.label, LABELS[i], lab, kurt(x[r.tr]), kurt(x[r.ho]), skw(x[r.tr]), skw(x[r.ho]), tail3(x[r.tr]), tail3(x[r.ho]))
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
        for (x, c) in ((S[i][1], :gray55), (S[i][2], col))
            z = x[r.ho]
            z = (z .- mean(z)) ./ std(z)
            h = [count(e -> lo <= e < lo + dz, z) for lo in edges[1:(end - 1)]] ./ (length(z) * dz)
            keep = h .> 0
            lines!(ax, mids[keep], h[keep]; color = c, linewidth = 1.0)
        end
        rr, cc = pos(i)
        rr == 2 ? (ax.xlabel = "standardized residual") : (ax.xticklabelsvisible = false)
        cc == 1 ? (ax.ylabel = "density") : (ax.yticklabelsvisible = false)
        xlims!(ax, -zmax, zmax); ylims!(ax, 1e-4, 1)
    end
    Legend(fig[3, 1:3], [LineElement(color = :gray55), LineElement(color = col), LineElement(color = C_REF, linestyle = :dash)],
           ["÷ constant scale", "÷ power-law scale", L"\mathcal{N}(0,1)"]; orientation = :horizontal)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("net_shape_$(net.label).pdf", fig)
end

if NET_MAIN
    report_error()
    foreach(range_split, NETR)
    fig_acf()
    fig_scale()
    for (j, net) in enumerate(NETR)
        fig_shape(net, NETCOL[j])
    end
end
