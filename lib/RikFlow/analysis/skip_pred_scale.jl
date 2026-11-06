# Step 5, card item (c) (`analysis/cards/step5.md`, 2026-11-06): does the spread of LinReg1's residual
# depend on its PREDICTED CORRECTION at a given level? If it does, the step-5 network also reads the
# skip's prediction W_s z (with or without chosen by the seed-mean validation CRPS); if not, it does
# not, and the methods sentence on it (closures.tex sec:lever-stoch) is cut.
#
#     julia --startup-file=no --project=analysis analysis/skip_pred_scale.jl [LinReg1] [--fig]
#
# The residual as `plot_paper_secs45.jl` `full_residual` (deployed C, the whole tracked record, level
# target), standardised per QoI by its 1-10 TU sd, as Fig. `colour-scale`. The predicted correction is
# the deployed prediction minus q*^n, c = (X C)_i - q*_i (scaled units), standardised by its 1-10 TU sd.
# Two readings, both fitted on 1-10 TU (steps 400-4000) and read on 10-100 TU:
#
#   1. Binned: level quintiles (edges from 1-10 TU) x quintiles of c WITHIN each level quintile (edges
#      from that bin's 1-10 TU rows). Per cell the spread about the cell mean, over the spread about
#      the level bin's mean (the same rows): 1 everywhere = no dependence at a given level.
#   2. Held out NLL (nats per step), Gaussian about zero with log-variance
#        L: a + 2b log(q*/q_ref)                      (step 2's power law, unclipped)
#        LC: L + g1 c + g2 c^2                        (+ the predicted correction, sign and size)
#      and the constant variance K; gain = NLL(K) - NLL(L), NLL(L) - NLL(LC) on 10-100 TU.
#
#   3. The residual over the power law's sd (L), per decile of c on 10-100 TU: its spread over the overall
#      spread; 1 everywhere = the level, through the power law, carries all of c's information on spread.
#
# --fig writes analysis/figures/skip_pred_scale_<name>.png (one panel per QoI: reading 1 on 10-100 TU
# against the c quintile, one line per level quintile; reading 3 in red).

using LinearAlgebra
using Statistics
using Printf
using JLD2

const SRC = normpath(joinpath(@__DIR__, "..", "src"))
include(joinpath(SRC, "ts_history.jl"))
include(joinpath(SRC, "ts_scaling.jl"))

const OUTD = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const REC = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const DT = 2.5e-3
const NQ = 6
const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

"Residual, level (physical) and predicted correction of a deployed LinReg on the whole record."
function residual_and_prediction(name)
    m = load(joinpath(OUTD, "TO_LRS", name, "LinReg.jld2"))
    C = permutedims(Matrix{Float64}(m["c"]))
    s = m["scaling"].in_scaling
    hist = HistorySpec(; h = m["hist_len"], n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    X, Y, st = build_history(hist, scale_input(REC["q_star"], s), scale_input(REC["q"], s))
    P = X * C
    tr = findall(n -> 400 <= n <= 4000, st)
    ho = findall(n -> 10 < n * DT <= 100, st)
    R = Y .- P
    c = P .- X[:, 1:NQ]                                  # predicted correction (scaled units)
    E = R ./ vec(std(R[tr, :]; dims = 1))'
    Cn = c ./ vec(std(c[tr, :]; dims = 1))'
    qs = X[:, 1:NQ] .* vec(s.sigma)' .+ vec(s.mu)'
    return (; E, Cn, qs, tr, ho)
end

"Gaussian log-variance regression about zero, log var = F w, by Newton (as `fit_powerlaw_scale`)."
function fit_logvar(F, e; iters = 200)
    e2 = e .^ 2
    w = zeros(size(F, 2)); w[1] = log(mean(e2))
    L(w) = 0.5 * sum(F * w .+ e2 .* exp.(-(F * w)))
    for _ in 1:iters
        u = e2 .* exp.(-(F * w))
        d = (0.5 .* (F' * (F .* u))) \ (0.5 .* (F' * (1 .- u)))
        t = 1.0
        while L(w - t * d) > L(w) && t > 1e-8
            t /= 2
        end
        w -= t * d
        maximum(abs, t * d) < 1e-12 && break
    end
    return w
end
nll(F, w, e) = mean(0.5 .* (log(2pi) .+ F * w .+ e .^ 2 .* exp.(-(F * w))))

"Level quintile, then the quintile of c within it; edges from the 1-10 TU rows."
function cells(x, c, tr; nl = 5, nc = 5)
    le = quantile(x[tr], range(0, 1; length = nl + 1)[2:(end - 1)])
    lb = searchsortedfirst.(Ref(le), x)
    cb = zeros(Int, length(x))
    for k in 1:nl
        rows = findall(==(k), lb)
        ce = quantile(c[intersect(rows, tr)], range(0, 1; length = nc + 1)[2:(end - 1)])
        cb[rows] = searchsortedfirst.(Ref(ce), c[rows])
    end
    return lb, cb
end

function main(args)
    name = something(findfirst(a -> !startswith(a, "--"), args) |> (i -> i === nothing ? nothing : args[i]), "LinReg1")
    dofig = "--fig" in args
    r = residual_and_prediction(name)
    nl = nc = 5
    @printf("%s: residual spread against the predicted correction c at a given level (card step5 (c))\n", name)
    @printf("rows 1-10 TU %d, 10-100 TU %d; c and the residual standardised by their 1-10 TU sd\n\n",
            length(r.tr), length(r.ho))
    ratios = Dict{Int, Matrix{Float64}}()
    dec = Dict{Int, Vector{Float64}}()
    tot = zeros(3)
    for i in 1:NQ
        e, c, q = r.E[:, i], r.Cn[:, i], r.qs[:, i]
        lb, cb = cells(q, c, r.tr; nl, nc)
        @printf("%s  corr(c, log q*) on 1-10 TU %+.2f\n", LABELS[i], cor(c[r.tr], log.(q[r.tr])))
        for (rows, lab) in ((r.tr, "1-10"), (r.ho, "10-100"))
            M = fill(NaN, nl, nc); Mu = fill(NaN, nl, nc)
            for k in 1:nl
                rk = intersect(findall(==(k), lb), rows)
                sk = std(e[rk])
                for j in 1:nc
                    rkj = rk[cb[rk] .== j]
                    length(rkj) < 20 && continue
                    M[k, j] = std(e[rkj]) / sk
                    Mu[k, j] = mean(e[rkj])
                end
            end
            lab == "10-100" && (ratios[i] = M)
            @printf("  %-6s TU  spread / level-bin spread, rows = level quintile (low->high), cols = c quintile (neg->pos)\n", lab)
            for k in 1:nl
                @printf("      L%d  %s    cell means %s\n", k, join((@sprintf("%5.2f", v) for v in M[k, :]), " "),
                        join((@sprintf("%+5.2f", v) for v in Mu[k, :]), " "))
            end
            @printf("      mean over level bins: %s\n", join((@sprintf("%5.2f", v) for v in vec(mean(M; dims = 1))), " "))
        end
        # held-out NLL: constant, power law, power law + c terms
        qref = exp(mean(log.(q[r.tr])))
        lq = 2 .* log.(q ./ qref)
        FK = ones(length(e), 1)
        FL = hcat(ones(length(e)), lq)
        FC = hcat(FL, c, c .^ 2)
        wK = fit_logvar(FK[r.tr, :], e[r.tr]); wL = fit_logvar(FL[r.tr, :], e[r.tr]); wC = fit_logvar(FC[r.tr, :], e[r.tr])
        h = r.ho
        gL = nll(FK[h, :], wK, e[h]) - nll(FL[h, :], wL, e[h])
        gC = nll(FL[h, :], wL, e[h]) - nll(FC[h, :], wC, e[h])
        gLi = nll(FK[r.tr, :], wK, e[r.tr]) - nll(FL[r.tr, :], wL, e[r.tr])
        gCi = nll(FL[r.tr, :], wL, e[r.tr]) - nll(FC[r.tr, :], wC, e[r.tr])
        tot .+= (gL, gC, gCi)
        # 3. at a given level through the power law: the residual over the power law's sd, per decile of c
        z = e ./ exp.(FL * wL ./ 2)
        de = quantile(c[r.tr], range(0, 1; length = 11)[2:(end - 1)])
        db = searchsortedfirst.(Ref(de), c)
        zr = [std(z[h[db[h] .== j]]) / std(z[h]) for j in 1:10]
        dec[i] = zr
        @printf("  residual / power-law sd, 10-100 TU, spread per decile of c (neg->pos) over its overall spread: %s
",
                join((@sprintf("%.2f", v) for v in zr), " "))
        @printf("  NLL gain (nats/step): power law over constant %.3f held out (%.3f in sample), beta %.2f; + c terms over power law %.3f held out (%.3f in sample), g1 %+.3f g2 %+.3f\n\n",
                gL, gLi, wL[2], gC, gCi, wC[3], wC[4])
    end
    @printf("summed over the QoIs: power law over constant %.3f held out; + c over power law %.3f held out (%.3f in sample)\n",
            tot[1], tot[2], tot[3])
    if dofig
        @eval using CairoMakie
        Base.invokelatest(makefig, ratios, dec, name)
    end
end

function makefig(ratios, dec, name)
    fig = Figure(size = (700, 420))
    for i in 1:NQ
        rr, cc = 2 - isodd(i), cld(i, 2)
        ax = Axis(fig[rr, cc]; title = LABELS[i], xticks = 1:5,
                  xlabel = rr == 2 ? "quintile of c within the level quintile" : "",
                  ylabel = cc == 1 ? "spread / level-bin spread" : "")
        lines!(ax, range(0.6, 5.4; length = 10), dec[i]; color = :red, linewidth = 1.5, label = "÷ power law, deciles of c")
        hlines!(ax, [1.0]; color = :gray60, linestyle = :dash)
        M = ratios[i]
        for k in 1:size(M, 1)
            scatterlines!(ax, 1:size(M, 2), M[k, :]; label = "level Q$k", markersize = 5,
                          color = k, colormap = :viridis, colorrange = (1, size(M, 1)))
        end
        ylims!(ax, 0.4, 1.8)
        i == 1 && axislegend(ax; position = :ct, labelsize = 7, nbanks = 2, framevisible = false)
    end
    Label(fig[0, :], "$(name): residual spread against the predicted correction c, 10-100 TU (fits on 1-10 TU)"; fontsize = 11)
    d = joinpath(@__DIR__, "figures"); mkpath(d)
    p = joinpath(d, "skip_pred_scale_$(name).png")
    save(p, fig; px_per_unit = 2)
    println("wrote ", p)
end

main(ARGS)
