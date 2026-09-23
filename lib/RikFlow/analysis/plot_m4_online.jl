# M4 in the solver: a short online run against the reference and the other online closures.
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/plot_m4_online.jl
#
# Environment: `M4_ONLINE_FILE` (default: the stride-100 export's `data_online_tsim10.0_replica1`),
# `QOI_CACHE` (default: R1's extracted cache in `analysis/data/`), `FIG_DIR`.
#
# 🔑 **Everything is compared on the SAME window length** -- the M4 run's `tsim` -- so every
# closure's marginals are estimated from the same number of steps. The reference marginal is the
# full 100 TU record, and 🔴 **the KS column is only readable beside its own noise floor**: the
# reference's own `tsim`-long windows scored against its 100 TU marginal (#61 -- summed KS on one
# draw swings 2.7x with where the record is cut). A closure inside that band is not distinguishable
# from the truth by this statistic at this length.
#
# ⚠️ The correction statistics exclude the first `nwarm` columns: the closure REPLAYS the reference
# `dQ` there, so including them would score the reference against itself.

using Statistics, Printf, JLD2, CairoMakie

const OUT = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const M4F = get(ENV, "M4_ONLINE_FILE",
                joinpath(OUT, "TO_LSTM", "StochLSTM2_s100b32_points3_cap10000",
                         "data_online_tsim10.0_replica1.jld2"))
const CACHE = get(ENV, "QOI_CACHE",
                  joinpath(@__DIR__, "data",
                           "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const FIGS = get(ENV, "FIG_DIR", joinpath(@__DIR__, "figures"))
const QNAMES = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

ref = load(CACHE)
m4 = load(M4F)
nwarm = m4["nwarm"]
dt = m4["params"].Δt
ncol = size(m4["data_online"].q, 2)          # tsim / dt + 1
nstep = ncol - 1
@printf("M4 online: %d steps (%.1f TU), nwarm %d, model %s\n", nstep, nstep * dt, nwarm,
        basename(dirname(m4["model_file"])))

# --- the runs, all cut to the same first `ncol` columns -------------------------------------------
runs = Pair{String,NamedTuple}[]
push!(runs, "M4 stride 100" => (; q = m4["data_online"].q, dQ = m4["data_online"].dQ))
function add!(label, path; key = "data_online")
    isfile(path) || (println("missing $path -- skipped"); return)
    d = load(path)[key]
    size(d.q, 2) >= ncol || (println("$label shorter than the M4 run -- skipped"); return)
    push!(runs, label => (; q = d.q[:, 1:ncol],
                          dQ = hasproperty(d, :dQ) ? d.dQ[:, 1:nstep] : nothing))
end
add!("LinReg1", joinpath(OUT, "TO_LRS", "LinReg1", "data_online_tsim100.0_replica1.jld2"))
add!("DDN", joinpath(OUT, "TO_DDN", "DDN_data_online_tsim100.0_replica1.jld2"))
for (label, dir) in (("no model", "no_model"), ("Smagorinsky 0.07", "smag"))
    fs = filter(f -> endswith(f, ".jld2"), readdir(joinpath(OUT, dir)))
    isempty(fs) || add!(label, joinpath(OUT, dir, only(fs)); key = only(filter(k -> k != "params",
                                                                          keys(load(joinpath(OUT, dir, only(fs)))))))
end

qref, dQref = ref["q"], ref["dQ"]
sdq = vec(std(qref; dims = 2))

"Two-sample Kolmogorov-Smirnov distance."
function ks(a, b)
    a, b = sort(vec(a)), sort(vec(b))
    na, nb = length(a), length(b)
    i = j = 0; dmax = 0.0
    while i < na && j < nb
        x = min(a[i + 1], b[j + 1])
        while i < na && a[i + 1] <= x; i += 1; end
        while j < nb && b[j + 1] <= x; j += 1; end
        dmax = max(dmax, abs(i / na - j / nb))
    end
    return dmax
end
summed_ks(q) = sum(ks(q[k, :], qref[k, :]) for k in 1:size(q, 1))
lag1(x) = (y = x .- mean(x); sum(y[1:(end - 1)] .* y[2:end]) / sum(abs2, y))

# the yardstick: the reference's own non-overlapping windows of the same length
floor_ks = [summed_ks(qref[:, s:(s + ncol - 1)]) for s in 1:ncol:(size(qref, 2) - ncol + 1)]
@printf("\nsummed KS vs the reference's 100 TU marginal, %.1f TU windows\n", nstep * dt)
@printf("  reference's own windows (n = %d): min %.3f  median %.3f  max %.3f\n",
        length(floor_ks), minimum(floor_ks), median(floor_ks), maximum(floor_ks))
println("\n", rpad("run", 18), rpad("summed KS", 11), "per-QoI mean offset (ref sd)  |  sd ratio")
for (label, r) in runs
    off = (vec(mean(r.q; dims = 2)) .- vec(mean(qref; dims = 2))) ./ sdq
    sdr = vec(std(r.q; dims = 2)) ./ sdq
    @printf("%-18s %-10.3f %s  | %s\n", label, summed_ks(r.q),
            join((@sprintf("%+.2f", x) for x in off), " "), join((@sprintf("%.2f", x) for x in sdr), " "))
end

# how long each run stays on the reference trajectory (same IC, same forcing realisation)
println("\nfirst time (TU) the level departs the reference by > 0.5 ref sd, per QoI")
for (label, r) in runs
    tdep = [(i = findfirst(>(0.5), abs.(r.q[k, :] .- qref[k, 1:ncol]) ./ sdq[k]);
             i === nothing ? NaN : (i - 1) * dt) for k in 1:6]
    @printf("  %-18s %s\n", label, join((@sprintf("%5.2f", x) for x in tdep), " "))
end

# the correction the closure actually produced, warm-up excluded
cols = (nwarm + 1):nstep
dsd = vec(std(dQref; dims = 2))
println("\ncorrection dQ after the warm-up: sd ratio vs the reference dQ | lag-1 autocorr (ref) | zero columns")
for (label, r) in runs
    r.dQ === nothing && continue
    all(iszero, r.dQ) && continue
    x = r.dQ[:, cols]
    @printf("  %-18s %s | %s | %d\n", label,
            join((@sprintf("%.2f", std(x[k, :]) / dsd[k]) for k in 1:6), " "),
            join((@sprintf("%.3f(%.3f)", lag1(x[k, :]), lag1(dQref[k, cols])) for k in 1:6), " "),
            count(j -> all(iszero, x[:, j]), axes(x, 2)))
end

# --- figure --------------------------------------------------------------------------------------
const C = Dict("M4 stride 100" => colorant"#2a78d6", "LinReg1" => colorant"#eb6834",
               "DDN" => colorant"#1baf7a", "no model" => colorant"#eda100",
               "Smagorinsky 0.07" => colorant"#e87ba4")
const INK, INK2, GRID, BG = colorant"#0b0b0b", colorant"#52514e", colorant"#e6e5e1", colorant"#fcfcfb"
set_theme!(Theme(fontsize = 13, textcolor = INK,
                 Axis = (xgridcolor = GRID, ygridcolor = GRID, topspinevisible = false,
                         rightspinevisible = false, leftspinecolor = INK2, bottomspinecolor = INK2,
                         xticklabelcolor = INK2, yticklabelcolor = INK2, backgroundcolor = BG)))

t = (0:(ncol - 1)) .* dt
fig = Figure(size = (1250, 1150), backgroundcolor = BG)
Label(fig[0, 1:3], @sprintf("M4 stride 100 in the solver, %.0f TU, replica 1 — against the reference and the other closures", nstep * dt);
      fontsize = 16, halign = :left, tellwidth = false)
for k in 1:6
    a1 = Axis(fig[k, 1]; ylabel = QNAMES[k], xticklabelsvisible = k == 6,
              xlabel = k == 6 ? "t (TU)" : "")
    lines!(a1, t, qref[k, 1:ncol]; color = INK, linewidth = 1.2)
    lines!(a1, t, runs[1].second.q[k, :]; color = C["M4 stride 100"], linewidth = 1.2)
    # ECDF over the window, against the 100 TU reference marginal
    a2 = Axis(fig[k, 2]; xticklabelsvisible = true, yticks = [0, 0.5, 1])
    xs = sort(qref[k, :]); lines!(a2, xs, (1:length(xs)) ./ length(xs); color = INK, linewidth = 2)
    for (label, r) in runs
        xs = sort(r.q[k, :])
        lines!(a2, xs, (1:length(xs)) ./ length(xs); color = C[label], linewidth = 1.5)
    end
    # the correction: reference vs M4, after the warm-up
    a3 = Axis(fig[k, 3]; xticklabelsvisible = k == 6, xlabel = k == 6 ? "t (TU)" : "")
    tc = (1:nstep) .* dt
    lines!(a3, tc, dQref[k, 1:nstep]; color = INK, linewidth = 0.8)
    lines!(a3, tc, runs[1].second.dQ[k, :]; color = (C["M4 stride 100"], 0.8), linewidth = 0.8)
    vlines!(a3, [nwarm * dt]; color = INK2, linestyle = :dash, linewidth = 1)
    k == 1 && (a1.title = "level q(t): reference vs M4"; a2.title = "ECDF over the window (reference: 100 TU)";
               a3.title = "correction dQ(t): reference vs M4 (dashed: end of replay)")
end
# legend: identity is never colour-alone -- every series is named here
elems = [LineElement(color = INK, linewidth = 2); [LineElement(color = C[l], linewidth = 2) for (l, _) in runs]]
Legend(fig[7, 1:3], elems, ["reference (R1)"; first.(runs)]; orientation = :horizontal,
       framevisible = false, tellheight = true)
mkpath(FIGS)
path = joinpath(FIGS, "fig15_lstm_online_s100.png")
save(path, fig; px_per_unit = 2)
println("\nwrote $path")
