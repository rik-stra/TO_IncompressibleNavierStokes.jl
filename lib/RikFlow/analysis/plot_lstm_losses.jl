# M4's training and validation loss curves.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/plot_lstm_losses.jl [cell ...]
#
# Reads the saved fits — `output/TO_LSTM/StochLSTM<i>/StochLSTM_seed<s>.jld2`, whose `extras.losses`
# carries the whole curve — and writes
#
#   figures/fig11_lstm_losses.png   train and validation against epoch, plus the lr schedule
#
# Nothing is recomputed, so this figure cannot disagree with what the driver printed.
#
# ---------------------------------------------------------------------------------------------
# What these curves are, and the one way to misread them
# ---------------------------------------------------------------------------------------------
#
# 🔴 **The vertical scale depends on `emission` and the cells are NOT on a common one.** With
# `emission = :none` the reconstruction term is a plain sum of squares; with `:constant` it is a
# Gaussian log-density, which can go negative. So the `:lstm` control shares no axis with the
# latent cells and is drawn on its own panel. This is the same trap as `results_LSTMS.md` §2's
# "the ELBO is not the NLL", one level down: two numbers with the same name and different units.
#
# ⚠️ **An epoch here is only a handful of gradient steps.** At `L = 500` on a 3599-row record there
# are 9 segments, 7 of them training, and `batch = 8`, so one epoch is ONE update. The x axis is
# therefore very nearly a count of optimiser steps, which is why the curves run for thousands of
# epochs and why "300 epochs" was not a long fit. The panel prints the step count so the reading
# does not have to be inferred.
#
# ⚠️ **Validation here is an inner split for early stopping**, the trailing `val_frac` of segments —
# not the selection window. Selection is on a window disjoint from both training and the online
# evaluation, which is the protocol's job and not this figure's.

using Statistics
using Printf
using JLD2
using CairoMakie

const HERE = @__DIR__
const OUT = normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "TO_LSTM"))
const FIGS = joinpath(HERE, "figures")

"The cells this figure is about, with the colours the rest of the M4 section uses."
const DEFAULT_CELLS = [2, 5, 7]
const COLOURS = Dict(:storn => RGBf(0.20, 0.40, 0.75),
                     :vrnn => RGBf(0.85, 0.45, 0.10),
                     :lstm => RGBf(0.35, 0.35, 0.35),
                     :vaernn => RGBf(0.55, 0.15, 0.55))

"""
    load_cell(i; seed)

The saved fit for cell `i`, as `(; name, arch, emission, beta, losses, best_epoch, best_val)`.

The configuration is read back from the fit's own `extras.cfg`, not from `inputs_lstm.jld2`: the
table is regenerated per checkout and could have moved since the fit was made, and what this
figure is about is the run that happened.
"""
function load_cell(i::Integer; seed::Integer = 1)
    path = joinpath(OUT, "StochLSTM$(i)", "StochLSTM_seed$(seed).jld2")
    isfile(path) || error("no fit at $path -- run exp_square_HIT/11_train_StochLSTM.jl $i $seed")
    d = load(path)
    e = d["extras"]
    cfg = e.cfg
    return (; i, name = cfg.name, arch = cfg.arch, emission = get(cfg, :emission, :none),
            beta = cfg.beta, L = cfg.L, burn = cfg.burn, lr = cfg.lr,
            losses = e.losses, best_epoch = e.losses.best_epoch, best_val = e.losses.best_val)
end

label(c) = @sprintf("%s  beta=%g  (%s)", c.arch, c.beta, c.emission)

"Draw one cell's train and validation curves onto `ax`: train dashed, validation solid."
function curves!(ax, c)
    col = get(COLOURS, c.arch, RGBf(0.4, 0.4, 0.4))
    ep = 1:length(c.losses.train)
    lines!(ax, ep, c.losses.train; color = (col, 0.45), linestyle = :dash)
    lines!(ax, ep, c.losses.val; color = col, linewidth = 2, label = label(c))
    scatter!(ax, [c.best_epoch], [c.best_val]; color = col, markersize = 10)
    return ax
end

"""
    fig_losses(cells)

One row: the `emission = :none` cells on a shared log axis, the `:constant` control on its own,
and the learning-rate schedule beside them.

🔑 **The last panel is why the schedule is worth drawing rather than assuming.** A flat lr line
means the decay-on-plateau rule never fired — validation improved at every epoch — which says the
fit was still in its monotone descent phase and the training length, not the learning rate, is
what was binding.
"""
function fig_losses(cells)
    shared = [c for c in cells if c.emission === :none]
    own = [c for c in cells if c.emission !== :none]
    fig = Figure(; size = (1500, 480))

    ax1 = Axis(fig[1, 1]; xlabel = "epoch (= 1 optimiser step here)", ylabel = "loss per scored step",
               title = "emission = :none  (sum of squares)", yscale = log10, xscale = log10)
    for c in shared
        curves!(ax1, c)
    end
    axislegend(ax1; position = :lb, framevisible = false)

    ax2 = Axis(fig[1, 2]; xlabel = "epoch", ylabel = "negative ELBO per scored step",
               title = "emission = :constant  (Gaussian log-density -- a DIFFERENT scale)",
               xscale = log10)
    for c in own
        curves!(ax2, c)
    end
    isempty(own) || axislegend(ax2; position = :rt, framevisible = false)

    ax3 = Axis(fig[1, 3]; xlabel = "epoch", ylabel = "learning rate", title = "lr schedule",
               yscale = log10, xscale = log10)
    for c in cells
        lines!(ax3, 1:length(c.losses.lr), c.losses.lr;
               color = get(COLOURS, c.arch, RGBf(0.4, 0.4, 0.4)), linewidth = 2)
    end

    Label(fig[0, :], "M4 training curves -- dashed train, solid validation, dot = best iterate";
          fontsize = 16, font = :bold)
    return fig
end

"The table the figure is read with: where the fit got to, and whether it had stopped moving."
function report(cells)
    @printf("%-14s %-8s %-9s %-8s %7s %10s %10s %10s %9s\n",
            "cell", "arch", "emission", "beta", "epochs", "best val", "final val",
            "final trn", "best ep")
    println("-"^94)
    for c in cells
        n = length(c.losses.val)
        @printf("%-14s %-8s %-9s %-8g %7d %10.4g %10.4g %10.4g %9d\n",
                c.name, c.arch, c.emission, c.beta, n, c.best_val, c.losses.val[end],
                c.losses.train[end], c.best_epoch)
    end
    println()
    for c in cells
        n = length(c.losses.val)
        # the last decade of training, as a fraction: a curve that has flattened moves little over it
        lo = max(1, n ÷ 10)
        drop = c.losses.val[lo] / c.losses.val[n]
        # 🔴 ONE string literal. A `"..." * "..."` format parses fine and fails at MACROEXPANSION,
        # which for a documented function happens while the DOCSTRING is processed -- so Julia
        # reports the error at the docstring, not here. Gotcha #47; V31 is the guard, and this
        # file is in `analysis/`, which V31 does not scan.
        sched = length(unique(c.losses.lr)) == 1 ? "never decayed" :
                @sprintf("decayed %g -> %g", c.losses.lr[1], c.losses.lr[end])
        @printf("  %-14s val fell %.2fx over the last decade of epochs (%d -> %d); lr %s\n",
                c.name, drop, lo, n, sched)
    end
end

cells = [load_cell(parse(Int, a)) for a in (isempty(ARGS) ? string.(DEFAULT_CELLS) : ARGS)]
report(cells)
mkpath(FIGS)
out = joinpath(FIGS, "fig11_lstm_losses.png")
save(out, fig_losses(cells); px_per_unit = 2)
println("\nwrote $out")
