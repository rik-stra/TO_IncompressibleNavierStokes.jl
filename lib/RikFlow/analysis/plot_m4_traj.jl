# Held-out trajectories for every trained M4 model -- `fig12`, `fig12b` of `results_LSTMS.md`
# §6.4.
#
#     julia --startup-file=no --project=analysis lib/RikFlow/analysis/plot_m4_traj.jl
#
# One teacher-forced trajectory per trained M4 model, on the held-out selection window.
#
# Reads the trajectories computed under the training project (the conversion from trained `ps` to
# deployed `LSTMWeights` needs the Lux extension, which `analysis/` deliberately does not carry)
# and draws them against the recorded truth.
#
# 🔴 **These are SINGLE realisations, not ensembles.** One latent draw per model, so the spread
# between a model and the truth mixes "the model is wrong" with "this draw was unlucky". Read the
# shape and the envelope here; the calibration statement is CRPS and the rank histogram.
using Statistics, Printf, JLD2, CairoMakie

const TRAJ = get(ENV, "TRAJ_FILE", joinpath(@__DIR__, "output", "m4_traj_heldout.jld2"))
const FIGS = get(ENV, "FIG_DIR", joinpath(@__DIR__, "figures"))
isfile(TRAJ) || error("no trajectories at $TRAJ -- run analysis/m4_traj_heldout.jl first")
mkpath(FIGS)
d = load(TRAJ)
labels, groups = d["labels"], d["groups"]
steps, preds, truth = d["steps"], d["preds"], d["truth"]
mu, sg = d["mu"], d["sg"]
nq = size(truth, 2)

# Back to physical QoI units. `out_scaling === in_scaling` in the training driver, so one pair does
# both directions.
unscale(z, i) = z .* sg[i] .+ mu[i]

const STRIDE_COLS = [RGBf(0.20, 0.40, 0.75), RGBf(0.80, 0.15, 0.15), RGBf(0.90, 0.55, 0.10),
                     RGBf(0.15, 0.60, 0.35), RGBf(0.55, 0.15, 0.55)]
const CELL_COLS = [RGBf(0.20, 0.40, 0.75), RGBf(0.85, 0.45, 0.10), RGBf(0.35, 0.35, 0.35)]

"""
    fig_traj(idx, cols, title, path; nzoom)

Six rows, one per QoI. Left the residual `prediction - truth` over the whole scored window, right
a `nzoom`-step zoom of the trajectory itself.

🔑 **The trajectory panel alone cannot separate these models.** Teacher-forced one-step prediction
on a record this smooth puts every line on top of the truth at full-window scale, so the left
column plots what actually differs -- the residual, on a common axis per QoI, where a wider band
is a worse model and a band that drifts is one whose error is not stationary across the window.
"""
function fig_traj(idx, cols, ttl, path; nzoom = 500)
    fig = Figure(; size = (1500, 1850))
    Label(fig[0, 1:2], ttl; fontsize = 20, font = :bold)
    zr = 1:min(nzoom, length(steps))
    for i in 1:nq
        axl = Axis(fig[i, 1]; ylabel = "q$i",
                   xlabel = i == nq ? "step" : "", xticklabelsvisible = i == nq)
        axr = Axis(fig[i, 2]; xlabel = i == nq ? "step" : "", xticklabelsvisible = i == nq)
        hlines!(axl, [0.0]; color = :black, linewidth = 1.5)
        lines!(axr, steps[zr], unscale(truth[zr, i], i); color = :black, linewidth = 2.5,
               label = "truth")
        for (k, m) in enumerate(idx)
            c = cols[mod1(k, length(cols))]
            ls = occursin("batch 2", labels[m]) ? :dash : :solid
            res = unscale(preds[:, i, m], i) .- unscale(truth[:, i], i)
            lines!(axl, steps, res; color = (c, 0.55), linewidth = 0.6, linestyle = ls)
            lines!(axr, steps[zr], unscale(preds[zr, i, m], i); color = (c, 0.9), linewidth = 1.4,
                   linestyle = ls, label = labels[m])
        end
        i == 1 && (axl.title = "residual, whole scored window ($(length(steps)) steps)";
                   axr.title = "trajectory, first $(length(zr)) steps")
    end
    Legend(fig[nq + 1, 1:2], contents(fig[1, 2])[1]; orientation = :horizontal,
           framevisible = false, nbanks = 2)
    save(path, fig)
    println("wrote ", path)
end

si = findall(==("stride scan"), groups)
ci = findall(==("cells"), groups)
fig_traj(si, STRIDE_COLS,
         "M4 stride scan — one teacher-forced trajectory per point, held-out window " *
         "$(d["eval_window"])  (cell StochLSTM2, :storn, seed 1, member 1)",
         joinpath(FIGS, "fig12_lstm_traj_stride.png"))
fig_traj(ci, CELL_COLS,
         "M4 cells — one teacher-forced trajectory each, held-out window $(d["eval_window"])" *
         "  (seed 1, member 1)",
         joinpath(FIGS, "fig12b_lstm_traj_cells.png"))

# The per-QoI single-draw RMSE, in physical units, so the figure has a number beside it.
println()
@printf("%-28s %s\n", "model", join((@sprintf("%9s", "q$i") for i in 1:nq), " "))
for m in 1:length(labels)
    e = [sqrt(mean((unscale(preds[:, i, m], i) .- unscale(truth[:, i], i)) .^ 2)) for i in 1:nq]
    @printf("%-28s %s\n", labels[m], join((@sprintf("%9.4g", x) for x in e), " "))
end
