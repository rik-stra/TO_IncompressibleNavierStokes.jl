# Step 5 (2026-11-06; card analysis/cards/step5.md, "Offline check and what is reported beside it"): the samples of
# the stochastic skip network against the level, against their error, and the share of their variance from the
# network, from the `pred.jld2` that `cnn_mean_fit.jl` writes with RIKFLOW_CNN_LOSS = crps_gen and REFIT = 1.
#
#     julia --startup-file=no --project=analysis analysis/gen_offline.jl <run dir under TO_LSTM> [<run> ...] [--fig]
#
# Per run and seed, in the target's units (what LinReg1 misses, / its 1-10 TU sd), by deciles of q*/q_ref (edges
# from 1-10 TU, per QoI, as Fig. colour-scale / `plot_paper_secs45.jl` `fig_scale`):
#   - the samples' sd (rms over the bin's steps) over the constant sd of LinReg1 + MVG (sqrt diag Σ), for the
#     stage-1 network on 8.2-10 TU and the refitted one on 10-100 TU; beside it LinReg1's residual's spread about
#     the bin mean (the figure's markers) and step 2's power law fitted on it on 1-10 TU, clipped as deployed;
#   - the error of the samples' mean (rms) over their sd, per bin (1 = spread matches error);
#   - the share of the sample variance from the network's part (sd of g(., ε) squared over the samples' variance).
# --fig writes analysis/figures/gen_offline_<run>.png (seed 1: spread per decile against the level).

using LinearAlgebra
using Statistics
using Printf
using JLD2

const SRC = normpath(joinpath(@__DIR__, "..", "src"))
include(joinpath(SRC, "ts_scale.jl"))                 # fit_powerlaw_scale (step 2's fit)

const TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
const REC = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const DT = 2.5e-3
const NQ = 6
const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]
fmt(v; d = 2) = join((@sprintf("%.*f", d, x) for x in v), " ")

function main(args)
    runs = filter(a -> !startswith(a, "--"), args)
    isempty(runs) && error("usage: gen_offline.jl <run> [<run> ...] [--fig]")
    dofig = "--fig" in args
    for run in runs
        P = load(joinpath(TO, run, "pred.jld2"))
        P["loss"] === :crps_gen || error("$run is not a crps_gen run")
        cols = P["cols"]
        so, mo = vec(P["scaling"].out_scaling.sigma), vec(P["scaling"].out_scaling.mu)
        S = P["Sigma"]
        sd0 = sqrt.(diag(S))
        # the target on every step: the skip LinReg's residual, centred at its MVG mean, / so (in the units of
        # Fig. colour-scale up to sd / so ~ 1); raw dq - sample mean = so * (Y - mu)
        Y = P["ytarget"]
        dq = REC["q"][:, cols .+ 1] .- REC["q_star"][:, cols]
        qs = REC["q_star"][:, cols]
        tr = findall(n -> 400 <= n <= 4000, cols)
        va = findall(n -> 0.8 * 3600 + 400 < n <= 4000, cols)          # stage 1's validation rows, ~8.2-10 TU
        ho = findall(n -> 10 < n * DT <= 100, cols)
        @printf("\n%s (skip %s, Mrec %d): spread per decile of q*/q_ref, over LinReg1 + MVG's constant sd\n", run, P["skip"], P["Mrec"])
        figdat = Dict{Int,Any}()
        for seed in sort(collect(keys(P["gen"])))
            g = P["gen"][seed]
            for (stage, mean_raw, sd, sdg, rows, lab) in ((1, g.mean1, g.sd1, g.sdg1, va, "stage 1, 8.2-10 TU"),
                                                          (3, g.mean3, g.sd3, g.sdg3, ho, "refit, 10-100 TU"))
                mu = (mean_raw .- dq) ./ so .+ Y                         # the sample mean, target units
                err = Y .- mu
                @printf("  seed %d, %s:\n", seed, lab)
                for i in 1:NQ
                    p = fit_powerlaw_scale(Y[i, tr], qs[i, tr])          # step 2's fit, on LinReg1's residual
                    x = qs[i, :] ./ p.qref
                    edges = quantile(x[tr], range(0, 1; length = 11)[2:(end - 1)])
                    b = searchsortedfirst.(Ref(edges), x)
                    lo, hi = p.qclip ./ p.qref
                    spr, pl, emp, eos, shr = (zeros(10) for _ in 1:5)
                    for k in 1:10
                        r = rows[b[rows] .== k]
                        isempty(r) && (spr[k] = pl[k] = emp[k] = eos[k] = shr[k] = NaN; continue)
                        spr[k] = sqrt(mean(abs2, sd[i, r])) / sd0[i]
                        xm = exp(mean(log.(x[r])))
                        pl[k] = exp(p.logvar / 2) * clamp(xm, lo, hi)^p.beta / sd0[i]
                        emp[k] = std(Y[i, r]) / sd0[i]                  # LinReg1's residual about the bin mean
                        eos[k] = sqrt(mean(abs2, err[i, r])) / sqrt(mean(abs2, sd[i, r]))
                        shr[k] = mean(abs2, sdg[i, r]) / mean(abs2, sd[i, r])
                    end
                    @printf("    %-8s spread %s | power law %s | LinReg1 resid. spread %s\n", LABELS[i], fmt(spr), fmt(pl), fmt(emp))
                    @printf("    %-8s error/sd %s | net share %s\n", "", fmt(eos), fmt(shr))
                    seed == first(sort(collect(keys(P["gen"])))) && stage == 3 && (figdat[i] = (; spr, pl, emp))
                end
                @printf("    overall: error / sd %s | net share %s\n",
                        fmt(sqrt.(vec(mean(abs2, err[:, rows]; dims = 2))) ./ sqrt.(vec(mean(abs2, sd[:, rows]; dims = 2)))),
                        fmt(vec(mean(abs2, sdg[:, rows]; dims = 2)) ./ vec(mean(abs2, sd[:, rows]; dims = 2))))
            end
            @printf("  seed %d: samples' correlation, max |. - Sigma's|: 1-10 TU %.3f, 10-100 TU %.3f; L L' %.3f\n", seed,
                    maximum(abs, g.corr10 .- S ./ (sd0 * sd0')), maximum(abs, g.corrho .- S ./ (sd0 * sd0')),
                    maximum(abs, (g.L3 * g.L3') ./ sqrt.(diag(g.L3 * g.L3') * diag(g.L3 * g.L3')') .- S ./ (sd0 * sd0')))
            # the matrices: lower triangle the samples' (10-100 TU), upper triangle Sigma's (LinReg1 + MVG)
            Rs, R0 = g.corrho, S ./ (sd0 * sd0')
            println("    correlation, lower: samples (refit, 10-100 TU) | upper: LinReg1's Sigma; QoIs ", join(LABELS, " "))
            for i in 1:NQ
                println("      ", join((@sprintf("%+.2f", j < i ? Rs[i, j] : j > i ? R0[i, j] : 1.0) for j in 1:NQ), " "))
            end
        end
        dofig && makefig(figdat, run)
    end
end

function makefig(fd, run)
    @eval using CairoMakie
    Base.invokelatest() do
        fig = Figure(size = (700, 400))
        for i in 1:NQ
            rr, cc = 2 - isodd(i), cld(i, 2)
            ax = Axis(fig[rr, cc]; title = LABELS[i], xlabel = rr == 2 ? "decile of q*/q_ref (1-10 TU edges)" : "",
                      ylabel = cc == 1 ? "spread / constant" : "")
            lines!(ax, 1:10, fd[i].pl; color = :gray40, label = "power law (step 2)")
            scatter!(ax, 1:10, fd[i].emp; color = :white, strokecolor = :black, strokewidth = 0.8, label = "LinReg1 residual spread")
            scatterlines!(ax, 1:10, fd[i].spr; color = :red, label = "generator samples")
            hlines!(ax, [1.0]; color = :gray70, linestyle = :dash)
            i == 1 && axislegend(ax; position = :lt, labelsize = 7, framevisible = false)
        end
        Label(fig[0, :], "$run, seed 1 refit, 10-100 TU"; fontsize = 11)
        d = joinpath(@__DIR__, "figures"); mkpath(d)
        p = joinpath(d, "gen_offline_$(replace(run, '/' => '_')).png")
        save(p, fig; px_per_unit = 2)
        println("wrote ", p)
    end
end

main(ARGS)
