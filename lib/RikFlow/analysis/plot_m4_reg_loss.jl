# Loss decay of the regularisation sweep: does a regulariser delay the overfitting?
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/plot_m4_reg_loss.jl
#
# Reads `output/TO_LSTM/explore/<tag>/StochLSTM_seed1.jld2` for every tag matching `REG_TAGS`
# (a regex, default `^reg_|^dq_vrnn_b1e-4$`) and draws training and validation loss against the
# optimiser update, one panel per fit, with the returned (best-validation) iterate marked and the
# held-out R^2 of the correction in the title. Environment: `REG_TAGS`, `FIG_DIR`.
#
# 🔑 **What to read**: the `dQ`-target fits overfit within tens of updates (validation minimum at
# update 26-70, then rising while training falls). A regulariser that works moves the minimum later
# and lowers the train/val gap -- and must show it on the held-out R^2, which the validation curve
# (it is also the stopping set) cannot.
# ⚠️ Each panel has its own y-axis: the loss is the ELBO at the fit's OWN beta (at beta = 1 the KL
# term is a large share of it), and a longer training range has a different scaling. Compare shapes,
# not levels, across panels.

using JLD2, Printf, CairoMakie

const EXPL = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM", "explore"))
const FIGS = get(ENV, "FIG_DIR", joinpath(@__DIR__, "figures"))
const PAT = Regex(get(ENV, "REG_TAGS", "^reg_|^dq_vrnn_b1e-4\$"))

tags = sort(filter(t -> occursin(PAT, t) && isfile(joinpath(EXPL, t, "StochLSTM_seed1.jld2")),
                   readdir(EXPL)))
isempty(tags) && error("no fits matching $(PAT.pattern) under $EXPL")
# the unregularised reference first
tags = vcat(filter(==("dq_vrnn_b1e-4"), tags), filter(!=("dq_vrnn_b1e-4"), tags))

# reference palette slots 1-2, validated; each series is also named in the legend
const C_VAL, C_TRAIN = colorant"#2a78d6", colorant"#eb6834"
const INK, INK2, GRID, BG = colorant"#0b0b0b", colorant"#52514e", colorant"#e6e5e1", colorant"#fcfcfb"
set_theme!(Theme(fontsize = 12, textcolor = INK,
                 Axis = (xgridcolor = GRID, ygridcolor = GRID, topspinevisible = false,
                         rightspinevisible = false, leftspinecolor = INK2, bottomspinecolor = INK2,
                         xticklabelcolor = INK2, yticklabelcolor = INK2, backgroundcolor = BG)))

ncol = 4
nrow = cld(length(tags), ncol)
fig = Figure(size = (1500, 300 * nrow + 90), backgroundcolor = BG)
Label(fig[0, 1:ncol], "M4, `dQ` target — loss decay per fit (train: mean since the previous validation; " *
      "dashed: returned iterate; title: held-out R² of the correction, mean over QoIs)";
      fontsize = 14, halign = :left, tellwidth = false)
println(rpad("fit", 26), "updates  best@   val@best   train@best  val@end  held-out R2 (mean | per QoI)")
for (i, tag) in enumerate(tags)
    x = load(joinpath(EXPL, tag, "StochLSTM_seed1.jld2"), "extras")
    l = x.losses
    u, v, t = l.update, l.val, l.train
    ok = isfinite.(t)                            # a warm start's update-0 row has train = NaN
    r2 = hasproperty(x, :heldout) ? x.heldout.r2mean : NaN
    ttl = @sprintf("%s\nbest @ %d / %d, held-out R² %s", tag, l.best_update, l.updates,
                   isnan(r2) ? "—" : @sprintf("%.3f", r2))
    ax = Axis(fig[cld(i, ncol), mod1(i, ncol)]; title = ttl, titlesize = 11, titlealign = :left,
              xlabel = cld(i, ncol) == nrow ? "optimiser update" : "", yscale = log10,
              ytickformat = vs -> [@sprintf("%.3g", v) for v in vs])
    lines!(ax, u[ok], t[ok]; color = C_TRAIN, linewidth = 1.5)
    lines!(ax, u, v; color = C_VAL, linewidth = 2)
    vlines!(ax, [l.best_update]; color = INK2, linestyle = :dash, linewidth = 1)
    ib = l.best_index
    @printf("%-26s %6d  %5d   %8.4g   %8.4g   %7.4g  %s\n", tag, l.updates, l.best_update, v[ib],
            ok[ib] ? t[ib] : NaN, v[end],
            hasproperty(x, :heldout) ? (@sprintf("%.3f | ", r2) * join((@sprintf("%.2f", r) for r in x.heldout.r2), " ")) : "—")
end
Legend(fig[nrow + 1, 1:ncol], [LineElement(color = C_VAL, linewidth = 2), LineElement(color = C_TRAIN, linewidth = 2),
                               LineElement(color = INK2, linestyle = :dash)],
       ["validation (inner split)", "training", "returned iterate"]; orientation = :horizontal,
       framevisible = false, tellheight = true)
mkpath(FIGS)
path = joinpath(FIGS, "fig17_lstm_reg_loss.png")
save(path, fig; px_per_unit = 2)
println("wrote $path")
