# Input-noise training of the step-3 CNN, the loss and the weight decay (results_LSTMS.md Sec. 36; 2026-10-26), in the
# style of Sec. 6's sweep (`net_sweep.jl`): each (loss, noise mode) at weight decays {0, .01, .03, .1, .3, 1} x seeds
# {1, 2, 3}, every fit with the window noise of a closure on its inputs (tools/step3_innoise.jl).
#
#     PAPER_FIG_DIR=<dir> julia --startup-file=no --project=lib/RikFlow/analysis lib/RikFlow/analysis/net_noise_sweep.jl
#
# Reads `TO_LSTM/step3/<prefix><loss>_wd<wd>_s<seed>/{fits,pred}.jld2` (cnn_mean_fit.jl, RIKFLOW_CNN_REFIT=1, one seed
# per run) and their score files (`cnn_mean_fit.jl` RIKFLOW_CNN_SCORE, `TO_LSTM/step3/innoise_scores/`). Modes:
#   fixed: Σ = the round-5 fixed point of the RMSE network (`step3/cnn_nt_rmse_wd0.3_s1_r5`, its S_deploy), prefix ntsw_
#   head:  Σ = the learned head (stop-gradient), started at that Σ (HEAD = inj), prefix ntswh_ (likelihood and CRPS)
# 🔒 The weight decay of each (loss, mode) is the one with the lowest seed-mean value of its OWN validation loss on
# 8.2-10 TU WITH the noise (`val_loss`, the early-stopping value), and nothing else: never the 10-100 TU columns, never
# an online result. Every fit is scored on the validation window clean and noisy (½SE, NLL and CRPS with the constant
# Gaussian refitted to the fit rows' residuals, noisy for noisy), and its refitted network on 10-100 TU (clean and
# deployed residual, gain) against LinReg1's clean one-step error. Prints every number; writes
#   nt_sweep_acf.pdf   ACF of the DEPLOYED residual on 10-100 TU per weight decay, one block per (loss, mode)
#                      (NTSWEEP_FIG, default "rmse/fixed,nll/head"), as net_sweep_acf.pdf

include(joinpath(@__DIR__, "net_noise_residual.jl"))    # load_cnn, load_lrs, the ACF and the plot tools

const NT_CONF = [("rmse", "fixed", "ntsw_", "RMSE"), ("crps", "fixed", "ntsw_", "CRPS"), ("nll", "fixed", "ntsw_", "likelihood"),
                 ("crps", "head", "ntswh_", "CRPS"), ("nll", "head", "ntswh_", "likelihood")]
const NT_WDS = ["0", "0.01", "0.03", "0.1", "0.3", "1"]
const NT_SEEDS = 1:3
const NT_SIGMA = "step3/cnn_nt_rmse_wd0.3_s1_r5"
ntdir(pre, loss, wd, s) = "step3/$(pre)$(loss)_wd$(wd)_s$(s)"
mean_(v) = sum(v) / length(v)
rng_(v; f = "%.4f") = @sprintf("%s (%s..%s)", Printf.format(Printf.Format(f), mean_(v)),
                              Printf.format(Printf.Format(f), minimum(v)), Printf.format(Printf.Format(f), maximum(v)))

"Every (loss, mode, wd) with all seeds present: the stage-1 fits and the scored, refitted networks."
function load_ntsweep()
    S = Dict{Tuple{String,String,String},Any}()
    for (loss, mode, pre, _) in NT_CONF, wd in NT_WDS
        dirs = [ntdir(pre, loss, wd, s) for s in NT_SEEDS]
        ok = all(d -> isfile(joinpath(OUTD, "TO_LSTM", d, "pred.jld2")) &&
                      isfile(joinpath(OUTD, "TO_LSTM", NN_SCORES, basename(d) * ".jld2")), dirs)
        ok || (@printf("sweep: %s/%s wd %s incomplete\n", loss, mode, wd); continue)
        fits = map(dirs) do d
            F = load(joinpath(OUTD, "TO_LSTM", d, "fits.jld2"))
            (F["loss"], F["innoise"]) == (Symbol(loss), Symbol(mode)) || error("$d: loss/mode $(F["loss"]) / $(F["innoise"])")
            mode == "fixed" && F["innoise_sigma"] != NT_SIGMA && error("$d: injected Σ from $(F["innoise_sigma"])")
            P = load(joinpath(OUTD, "TO_LSTM", d, "pred.jld2"))
            (; r = only(values(F["res"])), Sdep = only(values(P["S_deploy"])))
        end
        S[(loss, mode, wd)] = (; fits, net = load_cnn("$loss/$mode wd $wd", join(dirs, ",")))
    end
    return S
end

function report(S, L1)
    tr, ho = rows_tr(L1), rows_ho(L1)
    b = first_seed(L1).R
    rtr, rho = rmsq(b, tr), rmsq(b, ho)
    Sr5 = load(joinpath(OUTD, "TO_LSTM", NT_SIGMA, "pred.jld2"), "S_deploy")[1]
    best = Dict{Tuple{String,String},String}()
    for (loss, mode, _, name) in NT_CONF
        wds = [wd for wd in NT_WDS if haskey(S, (loss, mode, wd))]
        isempty(wds) && continue
        own = [mean_([f.r.val_loss for f in S[(loss, mode, wd)].fits]) for wd in wds]
        best[(loss, mode)] = wds[argmin(own)]
        for wd in wds
            f, net = S[(loss, mode, wd)].fits, S[(loss, mode, wd)].net
            vs = [net.seeds[s].val for s in sort(collect(keys(net.seeds)))]
            @printf("sweep: %-10s %-5s wd %-4s%s own val loss %s | best update %s\n", name, mode, wd, wd == best[(loss, mode)] ? "*" : " ",
                    rng_([x.r.val_loss for x in f]), join((string(x.r.best_update) for x in f), "/"))
            @printf("       val noisy: 0.5 SE %s | NLL %s | CRPS %s\n", rng_([v.half_sse_n for v in vs]),
                    rng_([v.nll_n for v in vs]; f = "%.3f"), rng_([v.crps_n for v in vs]))
            @printf("       val clean: 0.5 SE %s | NLL %s | CRPS %s\n", rng_([v.half_sse for v in vs]),
                    rng_([v.nll for v in vs]; f = "%.3f"), rng_([v.crps for v in vs]))
            @printf("       val common Σ: 0.5 SE %s | NLL %s | CRPS %s\n", rng_([v.half_sse_c for v in vs]),
                    rng_([v.nll_c for v in vs]; f = "%.3f"), rng_([v.crps_c for v in vs]))
            @printf("       / LinReg1 clean rms, seed mean: 10-100 TU clean %s | deployed %s | 1-10 TU clean %s | deployed %s\n",
                    pr(smean(v -> rmsq(v.R, ho) ./ rho, net)), pr(smean(v -> rmsq(v.Rn, ho) ./ rho, net)),
                    pr(smean(v -> rmsq(v.R, tr) ./ rtr, net)), pr(smean(v -> rmsq(v.Rn, tr) ./ rtr, net)))
            @printf("       gain 1-10 TU %s | 10-100 TU %s | deployed sd / round-5 sd %s | S_deploy sd / injected sd %s\n",
                    pr(smean(v -> rmsq(v.G, tr) ./ v.seta, net)), pr(smean(v -> rmsq(v.G, ho) ./ v.seta, net)),
                    pr(mean_([sqrt.(diag(x.Sdep) ./ diag(Sr5)) for x in f]); f = "%.3f"),
                    pr(mean_([sqrt.(diag(x.Sdep) ./ diag(x.r.noisy.Sinj)) for x in f]); f = "%.3f"))
            for (nm, fl) in (("clean", :R), ("deployed", :Rn))
                a = acf_mean_(net, fl, ho)
                @printf("       ACF 10-100 TU %-8s rho1 %s | ratio %s\n", nm, pr(a[2, :]), pr([lrv(a[:, i]) for i in 1:NQ]; f = "%.1f"))
            end
        end
    end
    # the table: each (loss, mode) at its chosen weight decay, seed means
    for (loss, mode, _, name) in NT_CONF
        haskey(best, (loss, mode)) || continue
        wd = best[(loss, mode)]
        net = S[(loss, mode, wd)].net
        vs = [v.val for v in values(net.seeds)]
        ho_ = smean(v -> rmsq(v.R, ho) ./ rho, net)
        hn_ = smean(v -> rmsq(v.Rn, ho) ./ rho, net)
        # the validation columns under the COMMON Σ (the round-5 fixed point): like for like across modes
        @printf("latex: %s, %s & %s & %.4f & %.2f & %.4f & %s & %s \\\\\n", name, mode, wd, mean_([v.half_sse_c for v in vs]),
                mean_([v.nll_c for v in vs]), mean_([v.crps_c for v in vs]),
                join((@sprintf("%.2f / %.2f", ho_[i], ho_[i + 1]) for i in 1:2:NQ), " & "),
                join((@sprintf("%.2f / %.2f", hn_[i], hn_[i + 1]) for i in 1:2:NQ), " & "))
    end
    return best
end

function fig_sweep(S, best)
    want = split(get(ENV, "NTSWEEP_FIG", "rmse/fixed,nll/head"), ",")
    blocks = [(loss, mode, name) for (loss, mode, _, name) in NT_CONF if "$loss/$mode" in want && haskey(best, (loss, mode))]
    ramp = Dict("rmse" => ["#A1D99B", "#74C476", "#41AB5D", "#238B45", "#006D2C", "#00441B"],    # as net_sweep.jl
                "crps" => ["#FEC44F", "#FE9929", "#EC7014", "#CC4C02", "#993404", "#662506"],
                "nll" => ["#FA9FB5", "#F768A1", "#DD3497", "#AE017E", "#7A0177", "#49006A"])
    L1 = MODELS[1]
    A1 = acf_mean_(L1, :Rn, rows_ho(L1))
    lags = 1:KMAX
    fig = Figure(size = (W, 235 * length(blocks)), figure_padding = (2, 6, 2, 2))
    for (bi, (loss, mode, name)) in enumerate(blocks)
        wds = [wd for wd in NT_WDS if haskey(S, (loss, mode, wd))]
        curves = Any[(A1, "LinReg1", COL["LinReg1"], 1.0)]
        for (k, wd) in enumerate(wds)
            net = S[(loss, mode, wd)].net
            push!(curves, (acf_mean_(net, :Rn, rows_ho(net)), wd == best[(loss, mode)] ? "$wd (chosen)" : wd, to_color(ramp[loss][k]),
                           wd == best[(loss, mode)] ? 1.5 : 0.9))
        end
        g = fig[2bi - 1, 1] = GridLayout()
        Label(g[0, 1:3], "CNN, $name loss, noise $(mode == "head" ? "of the learned head" : "of fixed covariance"): weight decay";
              fontsize = 8, halign = :left, tellwidth = false)
        axs = [Axis(g[pos(i)...]; title = QLAB[i], LAGAX...) for i in 1:NQ]
        for i in 1:NQ
            ax = axs[i]
            hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
            for (a, _, col, lw) in curves
                lines!(ax, lags, a[2:end, i]; color = col, linewidth = lw)
            end
            r, c = pos(i)
            r == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
            c == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
            xlims!(ax, 1, KMAX); ylims!(ax, -0.55, 1.02)
        end
        Legend(fig[2bi, 1], [LineElement(color = c, linewidth = lw) for (_, _, c, lw) in curves], [lab for (_, lab, _, _) in curves];
               orientation = :horizontal, nbanks = 1)
        rowgap!(g, 3); colgap!(g, 10)
    end
    rowgap!(fig.layout, 4)
    savefig("nt_sweep_acf.pdf", fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    S = load_ntsweep()
    # LinReg1 on the sweep's steps, its noise path from the first fit of seed 1 (common random numbers)
    c1 = first(v.net for (k, v) in S if k[1] == "rmse")
    empty!(MODELS)
    push!(MODELS, load_lrs("LinReg1", "LinReg1", c1.n, c1.seeds[1].E, c1.m0))
    BEST = report(S, MODELS[1])
    fig_sweep(S, BEST)
end
