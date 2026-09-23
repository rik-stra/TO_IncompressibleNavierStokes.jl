# The long re-run of the `(400, batch = 2)` control, against the main scan's full-batch points.
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/plot_m4_long_run.jl
#
# Environment: `TO_LSTM_DIR` (default `exp_square_HIT/output/TO_LSTM`), `FIG_DIR` (default
# `analysis/figures`), `LONG_RUN_FILE` (default `stride_scan_StochLSTM2_seed1_points2_cap10000.jld2`).
#
# 🔑 **What the figure is for: showing that the run did not stop because it never stopped
# improving**, not because the stop rule was too loose. The top panel is validation loss on a
# linear update axis, so the slow descent at `min_lr` reads at its true length; the bottom panel is
# the learning rate, a separate axis rather than a second y-scale on the first. The two full-batch
# points from the main scan are there for scale and end at their 3000-update cap.

using Printf, JLD2, CairoMakie

const TO = get(ENV, "TO_LSTM_DIR",
               normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM")))
const FIGS = get(ENV, "FIG_DIR", joinpath(@__DIR__, "figures"))
const LONG = get(ENV, "LONG_RUN_FILE", "stride_scan_StochLSTM2_seed1_points2_cap10000.jld2")

# Reference palette slots 1-3, validated (light surface): aqua is under 3:1 contrast, so every
# line is direct-labelled rather than identified by colour alone.
const C_LONG, C_400, C_100 = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a"
const INK, INK2, GRID = colorant"#0b0b0b", colorant"#52514e", colorant"#e6e5e1"

long = only(load(joinpath(TO, LONG))["results"])
main = load(joinpath(TO, "stride_scan_StochLSTM2_seed1.jld2"))["results"]
pick(s, b) = only(r for r in main if r.stride == s && r.batch == b)
r400, r100 = pick(400, 32), pick(100, 32)

bestsofar(v) = accumulate(min, v)
floor_at(r) = (i = findfirst(<=(1.0001e-5), r.lrhist); i === nothing ? nothing : r.upd_axis[i])
uf = floor_at(long)

set_theme!(Theme(fontsize = 15, textcolor = INK,
                 Axis = (xgridcolor = GRID, ygridcolor = GRID, xminorgridvisible = false,
                         leftspinecolor = INK2, bottomspinecolor = INK2,
                         topspinevisible = false, rightspinevisible = false,
                         xtickcolor = INK2, ytickcolor = INK2,
                         xticklabelcolor = INK2, yticklabelcolor = INK2)))

fig = Figure(size = (1100, 760), backgroundcolor = colorant"#fcfcfb")
Label(fig[0, 1], "M4 StochLSTM2, stride 400 batch 2, run to a 10 000-epoch cap ($(long.updates) updates)";
      fontsize = 17, halign = :left, tellwidth = false)

ax = Axis(fig[1, 1]; yscale = log10, ylabel = "validation loss (inner split)",
          backgroundcolor = colorant"#fcfcfb", xticklabelsvisible = false)
ax2 = Axis(fig[2, 1]; yscale = log10, xlabel = "optimiser update", ylabel = "learning rate",
           backgroundcolor = colorant"#fcfcfb")
linkxaxes!(ax, ax2)
rowsize!(fig.layout, 2, Relative(0.28))

# raw validation of the noisy run, recessive; its best-so-far carries the reading
lines!(ax, long.upd_axis, long.val; color = (C_LONG, 0.30), linewidth = 1)
lines!(ax, long.upd_axis, bestsofar(long.val); color = C_LONG, linewidth = 2)
for (r, c) in ((r400, C_400), (r100, C_100))
    lines!(ax, r.upd_axis, bestsofar(r.val); color = c, linewidth = 2)
end
ylims!(ax, 4e-4, 5e-2)
ax.yticks = ([5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2], ["5e-4", "1e-3", "2e-3", "5e-3", "1e-2", "2e-2"])
ax2.yticks = ([1e-5, 1e-4, 1e-3, 1e-2], ["1e-5", "1e-4", "1e-3", "1e-2"])
ax.xticks = ax2.xticks = (0:5000:30000, string.(0:5000:30000))

if uf !== nothing
    for a in (ax, ax2)
        vlines!(a, [uf]; color = INK2, linestyle = :dash, linewidth = 1)
    end
    text!(ax, uf, 3.2e-2; text = "  lr reaches 1e-5 at update $uf", color = INK2,
          align = (:left, :center), fontsize = 13)
end

# direct labels at each line's end, in text ink with a coloured key mark
function endlabel!(a, x, y, c, s)
    scatter!(a, [x], [y]; color = c, markersize = 9, strokecolor = colorant"#fcfcfb",
             strokewidth = 2)
    text!(a, x, y; text = "  " * s, color = INK, align = (:left, :center), fontsize = 13)
end
bl = bestsofar(long.val)
endlabel!(ax, long.upd_axis[end], bl[end], C_LONG, @sprintf("b2 control  %.2e", bl[end]))
endlabel!(ax, r400.upd_axis[end], minimum(r400.val), C_400,
          @sprintf("stride 400 b32  %.2e  (cap)", minimum(r400.val)))
endlabel!(ax, r100.upd_axis[end], minimum(r100.val), C_100,
          @sprintf("stride 100 b32  %.2e  (cap)", minimum(r100.val)))

# a few best-so-far readings along the long run, so the slope at min_lr is legible as numbers
for U in (3000, 10000, 20000)
    j = findlast(<=(U), long.upd_axis)
    scatter!(ax, [long.upd_axis[j]], [bl[j]]; color = C_LONG, markersize = 8,
             strokecolor = colorant"#fcfcfb", strokewidth = 2)
    text!(ax, long.upd_axis[j], bl[j]; text = @sprintf("%.2e", bl[j]), color = INK2,
          align = (:center, :bottom), offset = (0, 8), fontsize = 12)
end

for (r, c) in ((long, C_LONG), (r400, C_400), (r100, C_100))
    stairs!(ax2, r.upd_axis, r.lrhist; color = c, linewidth = 2, step = :post)
end
ylims!(ax2, 5e-6, 2e-2)
xlims!(ax, 0, long.updates * 1.22)            # room for the end labels

mkpath(FIGS)
path = joinpath(FIGS, "fig14_lstm_b2_long_run.png")
save(path, fig; px_per_unit = 2)
println("wrote $path")
