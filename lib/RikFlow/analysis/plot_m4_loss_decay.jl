# Loss decay for every trained M4 model on a common axis -- `fig13` of `results_LSTMS.md` §6.4.
#
#     julia --startup-file=no --project=analysis lib/RikFlow/analysis/plot_m4_loss_decay.jl
#
# Loss decay for every trained M4 model, on a COMMON axis.
#
# 🔴 **The x axis is OPTIMISER UPDATES, not epochs, and that is the whole point of the figure.**
# An epoch is a different amount of work in every one of these runs: the three cell fits predate
# the row-based split and ran at 1 update/epoch, while the stride-scan points run at 2, 4, 2, 3
# and 5. Drawing these against `epoch` would put the shortest stride five times further right than
# it belongs and invert the comparison. `results_LSTMS.md` §6 states the rule; this is it applied.
#
# ⚠️ **The `:lstm` control is on its own panel** -- Gaussian log-density against sum of squares
# (§2), so it shares no vertical axis with the latent cells. Same trap as fig11.
using Statistics, Printf, JLD2, CairoMakie

const TO = get(ENV, "TO_LSTM_DIR",
               normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM")))
const FIGS = get(ENV, "FIG_DIR", joinpath(@__DIR__, "figures"))
mkpath(FIGS)

"A curve on the common axis: `(label, arch, updates, train, val, lr, best_val, best_upd)`."
curves = Any[]

# --- the stride scan: updates = epoch x updates/epoch ---------------------------------------------
sc = load(joinpath(TO, "stride_scan_StochLSTM2_seed1.jld2"))
for r in sc["results"]
    u = (1:length(r.val)) .* r.upd_per_epoch
    push!(curves, (; label = "stride $(r.stride), batch $(r.batch)", arch = :storn, group = :stride,
                   u, train = r.train, val = r.val, lr = r.lrhist,
                   best_val = r.best_val, best_upd = r.best_epoch * r.upd_per_epoch))
end

# --- the cell fits: pre-split, 1 update/epoch, so updates == epochs -------------------------------
for (cell, arch) in [(2, :storn), (5, :vrnn), (7, :lstm)]
    p = joinpath(TO, "StochLSTM$cell", "StochLSTM_seed1.jld2")
    isfile(p) || continue
    lo = load(p)["extras"].losses
    push!(curves, (; label = "StochLSTM$cell :$arch (pre-split)", arch, group = :cell,
                   u = collect(1:length(lo.val)), train = lo.train, val = lo.val, lr = lo.lr,
                   best_val = lo.best_val, best_upd = lo.best_epoch))
end

const COL = Dict("stride 400, batch 32" => RGBf(0.20, 0.40, 0.75),
                 "stride 400, batch 2"  => RGBf(0.80, 0.15, 0.15),
                 "stride 100, batch 32" => RGBf(0.90, 0.55, 0.10),
                 "stride 50, batch 32"  => RGBf(0.15, 0.60, 0.35),
                 "stride 20, batch 32"  => RGBf(0.55, 0.15, 0.55))
const CELLCOL = Dict(:storn => RGBf(0.0, 0.0, 0.0), :vrnn => RGBf(0.45, 0.45, 0.45),
                     :lstm => RGBf(0.05, 0.55, 0.60))
colour(c) = get(COL, c.label, CELLCOL[c.arch])
style(c) = occursin("batch 2", c.label) ? :dash : :solid
width(c) = c.group === :cell ? 2.8 : 1.4

lat = [c for c in curves if c.arch !== :lstm]
ctl = [c for c in curves if c.arch === :lstm]

fig = Figure(; size = (1800, 1000))
Label(fig[0, 1:3], "M4 loss decay, all trained models — x axis is OPTIMISER UPDATES, not epochs";
      fontsize = 20, font = :bold)

ax1 = Axis(fig[1, 1]; xlabel = "optimiser updates", ylabel = "validation loss per scored step",
           title = "validation  (emission = :none)", xscale = log10, yscale = log10)
ax2 = Axis(fig[1, 2]; xlabel = "optimiser updates", ylabel = "training loss per scored step",
           title = "training  (emission = :none)", xscale = log10, yscale = log10)
ax3 = Axis(fig[1, 3]; xlabel = "optimiser updates", ylabel = "learning rate",
           title = "the decay-on-plateau schedule (all 8, control included)",
           xscale = log10, yscale = log10)
for c in lat
    lines!(ax1, c.u, c.val; color = colour(c), linewidth = width(c), linestyle = style(c),
           label = c.label)
    scatter!(ax1, [c.best_upd], [c.best_val]; color = colour(c), markersize = 11)
    lines!(ax2, c.u, c.train; color = colour(c), linewidth = width(c), linestyle = style(c))
end
for c in curves
    lines!(ax3, c.u, c.lr; color = colour(c), linewidth = width(c), linestyle = style(c))
end
Legend(fig[2, 1:3], ax1; orientation = :horizontal, framevisible = false, nbanks = 2)

# 🔑 The last decade is where the points actually separate; the log-log panel compresses it to
# nothing. Linear y, and the floor each run reached is the number §6.2 is read off.
ax4 = Axis(fig[3, 1]; xlabel = "optimiser updates", ylabel = "validation loss",
           title = "the last two thirds, linear scale")
for c in lat
    k = findall(>(1000), c.u)
    lines!(ax4, c.u[k], c.val[k]; color = colour(c), linewidth = width(c), linestyle = style(c))
end
ylims!(ax4, 0.0008, 0.004)

ax5 = Axis(fig[3, 2]; xlabel = "optimiser updates", ylabel = "loss (Gaussian log-density)",
           title = "the :lstm control — DIFFERENT UNITS, own axis", xscale = log10)
for c in ctl
    lines!(ax5, c.u, c.train; color = (colour(c), 0.45), linestyle = :dash)
    lines!(ax5, c.u, c.val; color = colour(c), linewidth = 2.5, label = c.label)
end
axislegend(ax5; position = :rt, framevisible = false)

# §6.1's learning-rate scan, for completeness: the other six fits that exist.
ax6 = Axis(fig[3, 3]; xlabel = "optimiser updates (= epochs, pre-split)",
           ylabel = "validation loss", title = "§6.1 lr scan (CLOSED) — the other six fits",
           xscale = log10, yscale = log10)
lrp = joinpath(TO, "lr_scan_StochLSTM2_seed1.jld2")
if isfile(lrp)
    for (k, r) in enumerate(load(lrp)["results"])
        v = replace(r.val, NaN => Inf)          # a diverged point stops, it does not plot as zero
        good = findall(isfinite, v)
        lines!(ax6, good, v[good]; linewidth = 1.8, label = @sprintf("lr = %g", r.lr))
    end
    axislegend(ax6; position = :lb, framevisible = false, nbanks = 2, labelsize = 11)
end

path = joinpath(FIGS, "fig13_lstm_loss_decay.png")
save(path, fig)
println("wrote ", path)

@printf("\n%-30s %8s %10s %12s %10s %8s\n", "model", "updates", "best val", "best @ upd",
        "final lr", "decays")
for c in curves
    d = count(i -> c.lr[i] < c.lr[i-1] * 0.999, 2:length(c.lr))
    @printf("%-30s %8d %10.5g %12d %10.3g %8d\n", c.label, c.u[end], c.best_val, c.best_upd,
            c.lr[end], d)
end
