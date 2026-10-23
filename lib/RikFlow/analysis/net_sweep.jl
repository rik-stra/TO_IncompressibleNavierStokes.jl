# Paper Sec. 6 (mlp_offline.tex): the step-3 CNN trained on each loss -- the Gaussian likelihood, the
# CRPS and the per-QoI RMSE -- at each weight decay, as Sec. 5 sweeps the ridge strength of the linear
# mean (2026-10-23, Rik: "tabulate the best RMSE, likelihood and CRPS results. Then ... plots like the
# ones in section 5, where we slowly increase regularization (weight decay)").
#
#     PAPER_FIG_DIR=<dir> julia --startup-file=no --project=lib/RikFlow/analysis \
#         lib/RikFlow/analysis/net_sweep.jl
#
# Reads `exp_square_HIT/output/TO_LSTM/step3/grid_sw_<loss>_wd<wd>_s<seed>/{fits,pred}.jld2`, one seed
# per run, each written by `cnn_mean_fit.jl` with RIKFLOW_CNN_REFIT=1 (stage 1 on 1-8.2 TU with early
# stopping on 8.2-10 TU, then the refit on 1-10 TU). Every loss is scored by the same measures on the
# validation window, 8.2-10 TU, in the standardised target: half the squared error summed over the
# QoIs, the Gaussian NLL with the covariance of the residuals on the fit rows, and the CRPS with their
# standard deviations; least squares on the same rows (LinReg1 fitted on 1-8.2 TU) is scored alike.
# The weight decay of each loss is the one with the lowest validation value of its own loss, averaged
# over the seeds, as the closures' protocol chooses it. On 10-100 TU, the refitted networks: one-step
# error against LinReg1's, and the residual's ACF (seed mean in the figure). Prints every number; writes
#   net_sweep_acf.pdf   ACF on 10-100 TU per weight decay, one block per loss (NETSWEEP_FIG, default
#                       "nll,crps"), as colour_sweep_acf.pdf

include(joinpath(@__DIR__, "net_residual.jl"))           # plot_paper_secs45.jl, LinReg1's residual, ACF tools

const SW_LOSSES = [("nll", "likelihood"), ("crps", "CRPS"), ("rmse", "RMSE")]
const SW_WDS = ["0", "0.01", "0.03", "0.1", "0.3", "1"]
const SW_SEEDS = 1:3
# NETSWEEP_PREFIX: step3/grid_cap_ is the sweep with the fits that reached the 40k cap redone at 200k
# (2026-10-23, Rik), the others linked
const SW_PREFIX = get(ENV, "NETSWEEP_PREFIX", "step3/grid_sw_")
swdir(loss, wd, s) = "$(SW_PREFIX)$(loss)_wd$(wd)_s$(s)"
# one sequential hue per loss, light to dark with the weight decay (ColorBrewer RdPu / YlOrBr; the
# lightness is monotone, which is what a sequential ramp needs); LinReg1 keeps its blue
const SW_RAMP = Dict("nll" => ["#FA9FB5", "#F768A1", "#DD3497", "#AE017E", "#7A0177", "#49006A"],
                     "crps" => ["#FEC44F", "#FE9929", "#EC7014", "#CC4C02", "#993404", "#662506"],
                     "rmse" => ["#A1D99B", "#74C476", "#41AB5D", "#238B45", "#006D2C", "#00441B"])

mean_(v) = sum(v) / length(v)
rng_(v; f = "%.4f") = @sprintf("%s (%s..%s)", Printf.format(Printf.Format(f), mean_(v)),
                              Printf.format(Printf.Format(f), minimum(v)), Printf.format(Printf.Format(f), maximum(v)))

"Every (loss, wd): the fits' validation scores per seed and the refitted networks' residuals."
function load_sweep()
    S = Dict{Tuple{String,String},Any}()
    ls = nothing
    for (loss, _) in SW_LOSSES, wd in SW_WDS
        dirs = [swdir(loss, wd, s) for s in SW_SEEDS]
        all(d -> isfile(joinpath(OUTD, "TO_LSTM", d, "pred.jld2")), dirs) || (@printf("sweep: %s wd %s missing\n", loss, wd); continue)
        fits = map(dirs) do d
            F = load(joinpath(OUTD, "TO_LSTM", d, "fits.jld2"))
            ls = (; nll = F["ls_val"], sse = F["ls_sse"], crps = F["ls_crps"])
            only(values(F["res"]))
        end
        S[(loss, wd)] = (; fits, net = load_net((; label = "$loss wd $wd", dir = join(dirs, ","))))
    end
    return S, ls
end

"Seed mean of the ACF on 10-100 TU (rows: lags 0-KMAX, columns: QoIs)."
acf_mean(net) = mean_([hcat((acf_ho(net.r[s], i) for i in 1:NQ)...) for s in net.seeds])
acf_tr(r, i) = autocorr(collect(r.R[r.tr, i]), KMAX)

function report(S, ls)
    @printf("table: LinReg1 (least squares on 1-8.2 TU), validation: 0.5 SSE %.4f | NLL %.3f | CRPS %.4f\n", ls.sse, ls.nll, ls.crps)
    best = Dict{String,String}()
    for (loss, name) in SW_LOSSES
        wds = [wd for wd in SW_WDS if haskey(S, (loss, wd))]
        isempty(wds) && continue
        own = Dict(wd => mean_([f.val_loss for f in S[(loss, wd)].fits]) for wd in wds)
        best[loss] = wds[argmin([own[wd] for wd in wds])]
        for wd in wds
            f, net = S[(loss, wd)].fits, S[(loss, wd)].net
            rs = [net.r[s] for s in net.seeds]
            ho = mean_([rmsq(r.R, r.ho) ./ rmsq(RL1.R, RL1.ho) for r in rs])
            tr = mean_([rmsq(r.R, r.tr) ./ rmsq(RL1.R, RL1.tr) for r in rs])
            a = acf_mean(net)
            @printf("sweep: %-10s wd %-4s%s own val loss %s | val 0.5 SSE %s | NLL %s | CRPS %s | best update %s\n",
                    name, wd, wd == best[loss] ? "*" : " ", rng_([x.val_loss for x in f]), rng_([x.val_half_sse for x in f]),
                    rng_([x.nll_refit for x in f]; f = "%.3f"), rng_([x.crps_refit for x in f]),
                    join((string(x.best_update) for x in f), "/"))
            @printf("       rms / LinReg1, seed mean: 10-100 TU %s | 1-10 TU %s\n", pr(ho), pr(tr))
            @printf("       ACF 10-100 TU, seed mean: rho1 %s | ratio %s\n", pr(a[2, :]),
                    pr([lrv(a[:, i]) for i in 1:NQ]; f = "%.1f"))
            at = mean_([hcat((acf_tr(r, i) for i in 1:NQ)...) for r in rs])
            @printf("       ACF 1-10 TU,   seed mean: rho1 %s | ratio %s\n", pr(at[2, :]),
                    pr([lrv(at[:, i]) for i in 1:NQ]; f = "%.1f"))
        end
    end
    a1 = hcat((acf_ho(RL1, i) for i in 1:NQ)...)
    @printf("sweep: LinReg1 ACF 10-100 TU: rho1 %s | ratio %s\n", pr(a1[2, :]), pr([lrv(a1[:, i]) for i in 1:NQ]; f = "%.1f"))
    # the table of the paper: each loss at its chosen weight decay
    for (loss, name) in SW_LOSSES
        haskey(best, loss) || continue
        f, net = S[(loss, best[loss])].fits, S[(loss, best[loss])].net
        ho = mean_([rmsq(net.r[s].R, net.r[s].ho) ./ rmsq(RL1.R, RL1.ho) for s in net.seeds])
        @printf("latex: %s & %s & %.4f & %.2f & %.4f & %s \\\\\n", name, best[loss], mean_([x.val_half_sse for x in f]),
                mean_([x.nll_refit for x in f]), mean_([x.crps_refit for x in f]),
                join((@sprintf("%.2f / %.2f", ho[i], ho[i + 1]) for i in 1:2:NQ), " & "))
    end
    @printf("latex: LinReg1 & -- & %.4f & %.2f & %.4f & %s \\\\\n", ls.sse, ls.nll, ls.crps,
            join(("1 / 1" for _ in 1:3), " & "))
    for (loss, _) in SW_LOSSES, wd in SW_WDS
        haskey(S, (loss, wd)) && range_split(S[(loss, wd)].net)
    end
    for (loss, _) in SW_LOSSES, wd in SW_WDS
        haskey(S, (loss, wd)) && decompose(S[(loss, wd)].net)
    end
    return best
end

function fig_sweep_net(S, best)
    want = split(get(ENV, "NETSWEEP_FIG", "nll,crps"), ",")
    blocks = [(loss, name) for (loss, name) in SW_LOSSES if loss in want]
    dqr = (; R = permutedims(REC["q"][:, 2:end] .- REC["q_star"]), ho = findall(k -> 10 < k * DT <= 100, 1:size(REC["q_star"], 2)))
    A0 = hcat((acf_ho(dqr, i) for i in 1:NQ)...)
    A1 = hcat((acf_ho(RL1, i) for i in 1:NQ)...)
    lags = 1:KMAX
    fig = Figure(size = (W, 235 * length(blocks)), figure_padding = (2, 6, 2, 2))
    for (b, (loss, name)) in enumerate(blocks)
        wds = [wd for wd in SW_WDS if haskey(S, (loss, wd))]
        curves = Any[(A0, L"dQ", C_REF, :dash, 1.0), (A1, "LinReg1", COL["LinReg1"], :solid, 1.0)]
        for (k, wd) in enumerate(wds)
            push!(curves, (acf_mean(S[(loss, wd)].net), wd == best[loss] ? "$wd (chosen)" : wd,
                           to_color(SW_RAMP[loss][k]), :solid, wd == best[loss] ? 1.5 : 0.9))
        end
        g = fig[2b - 1, 1] = GridLayout()
        Label(g[0, 1:3], "CNN, $name loss: weight decay"; fontsize = 8, halign = :left, tellwidth = false)
        axs = [Axis(g[pos(i)...]; title = QLAB[i], LAGAX...) for i in 1:NQ]
        for i in 1:NQ
            ax = axs[i]
            hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
            for (a, _, col, ls, lw) in curves
                lines!(ax, lags, a[2:end, i]; color = col, linewidth = lw, linestyle = ls)
            end
            r, cc = pos(i)
            r == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
            cc == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
            xlims!(ax, 1, KMAX); ylims!(ax, -0.3, 1.02)
        end
        Legend(fig[2b, 1], [LineElement(color = c, linestyle = ls, linewidth = lw) for (_, _, c, ls, lw) in curves],
               [lab for (_, lab, _, _, _) in curves]; orientation = :horizontal, nbanks = 1)
        rowgap!(g, 3); colgap!(g, 10)
    end
    rowgap!(fig.layout, 4)
    savefig("net_sweep_acf.pdf", fig)
end

S, LS = load_sweep()
BEST = report(S, LS)
fig_sweep_net(S, BEST)
