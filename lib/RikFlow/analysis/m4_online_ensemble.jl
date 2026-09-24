# M4 in the solver, 100 TU ensembles: two M4 fits against the reference and the linear closures.
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/m4_online_ensemble.jl
#
# Environment: `M4_ENSEMBLES` -- `label=dir;label=dir` under `output/TO_LSTM/` (default: the two
# stride-100 exports, beta 1e-4 and 1e-2), `QOI_CACHE`, `FIG_DIR`.
#
# 🔑 **The replicas are PAIRED across closures**: replica i starts from the same field under the same
# OU forcing, and M4's latent stream is seeded `Xoshiro(234 + i + 2)` in every model. Differences at
# a fixed i are the model's, not the draw's -- but only in the first ~0.5 TU; after that the runs
# are independent realisations and are compared as distributions.
#
# 🔴 **The plateau diagnostic is the one this was written for.** The 10 TU run of the beta = 1e-4
# fit (fig15) sat ~4 TU in a flat, low-variability state that summed KS scored inside the
# reference's own spread. A window is "flat" when its 0.5 TU (200-step) standard deviation is below
# 30% of the reference's MEDIAN 0.5 TU sd for that QoI; the reference's own flat fraction over its
# 100 TU is the null.
#
# ⚠️ Summed KS here is ONE draw per replica against the reference's single 100 TU record: #61 put
# its noise floor at 0.33-0.89 for the LRS cells, so read it as a ranking aid, never a verdict.

using Statistics, Printf, JLD2, CairoMakie

const OUT = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const CACHE = get(ENV, "QOI_CACHE",
                  joinpath(@__DIR__, "data",
                           "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const FIGS = get(ENV, "FIG_DIR", joinpath(@__DIR__, "figures"))
const QNAMES = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]
const W = 200               # 0.5 TU at dt = 2.5e-3
const FLAT = 0.3

ens_spec = get(ENV, "M4_ENSEMBLES",
               "M4 beta 1e-4=StochLSTM2_s100b32_points3_cap10000;M4 beta 1e-2=StochLSTM3_s100b32_points3_cap10000")
ENS = [(; label = strip(first(split(s, "="))), dir = joinpath(OUT, "TO_LSTM", strip(last(split(s, "=")))))
       for s in split(ens_spec, ";")]

ref = load(CACHE)
qref, dQref = ref["q"], ref["dQ"]
ncol = size(qref, 2)
sdq, mq = vec(std(qref; dims = 2)), vec(mean(qref; dims = 2))
dsd = vec(std(dQref; dims = 2))

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
lag1(x) = (y = x .- mean(x); sum(y[1:(end - 1)] .* y[2:end]) / sum(abs2, y))
rollsd(x) = [std(view(x, i:(i + W - 1))) for i in 1:(length(x) - W + 1)]
const REFSD = [median(rollsd(qref[k, :])) for k in 1:6]
flat(x, k) = rollsd(x) .< FLAT * REFSD[k]
"Contiguous true runs, as (start, length) in windows."
function episodes(b)
    out = Tuple{Int,Int}[]; i = 1
    while i <= length(b)
        if b[i]; j = i; while j < length(b) && b[j + 1]; j += 1; end; push!(out, (i, j - i + 1)); i = j + 1
        else i += 1 end
    end
    return out
end

# --- load every run's q and dQ, cut to the reference length ----------------------------------------
function load_run(path; key = "data_online")
    d = load(path)
    o = d[key]
    nw = get(d, "nwarm", 100)
    q = o.q
    return (; q, dQ = hasproperty(o, :dQ) ? o.dQ : nothing, nwarm = nw,
            complete = size(q, 2) == ncol, finite = all(isfinite, q))
end
groups = Pair{String,Vector}[]
for e in ENS
    fs = sort(filter(f -> occursin(r"^data_online_tsim100\.0_replica\d+\.jld2$", f), readdir(e.dir)))
    push!(groups, e.label => [load_run(joinpath(e.dir, f)) for f in fs])
end
for (label, dir, pat) in (("LinReg1", joinpath(OUT, "TO_LRS", "LinReg1"), r"^data_online_tsim100\.0_replica\d+\.jld2$"),
                          ("DDN", joinpath(OUT, "TO_DDN"), r"^DDN_data_online_tsim100\.0_replica\d+\.jld2$"))
    isdir(dir) || continue
    fs = sort(filter(f -> occursin(pat, f), readdir(dir)))
    push!(groups, label => [load_run(joinpath(dir, f)) for f in fs])
end

# --- the fits themselves, for context -----------------------------------------------------------
println("fits deployed:")
for e in ENS
    f = load(joinpath(e.dir, "StochLSTM_seed1.jld2"))
    x = f["extras"]; l = x.losses
    @printf("  %-14s beta %.0e  best val %.4g at update %d of %d  (%s)\n", e.label, x.cfg.beta,
            l.best_val, l.best_update, l.updates, l.stopped_early ? "stopped early" : "hit the cap")
end

# --- per-replica table -------------------------------------------------------------------------
reff = [mean(flat(qref[k, :], k)) for k in 1:6]
@printf("\nflat-window fraction of the REFERENCE (null), per QoI: %s\n",
        join((@sprintf("%.3f", x) for x in reff), " "))
println("\n", rpad("run", 16), "rep  ok   sumKS   flat frac per QoI                     longest flat (TU)  ",
        "Z[0,6] off   sd ratio(q) Z16/E16   dQ sd ratio Z16  dQ lag1 E[0,6] (ref ", @sprintf("%.3f", lag1(dQref[2, :])), ")")
summary = Dict{String,Any}()
for (label, reps) in groups
    rows = []
    for (i, r) in enumerate(reps)
        ok = r.complete && r.finite
        fr = [mean(flat(r.q[k, :], k)) for k in 1:6]
        lng = maximum((l for k in 1:6 for (_, l) in episodes(flat(r.q[k, :], k))); init = 0) * 2.5e-3
        sks = sum(ks(r.q[k, :], qref[k, :]) for k in 1:6)
        off1 = (mean(r.q[1, :]) - mq[1]) / sdq[1]
        sdr5, sdr6 = std(r.q[5, :]) / sdq[5], std(r.q[6, :]) / sdq[6]
        cols = (r.nwarm + 1):size(r.dQ, 2)
        dsr5 = r.dQ === nothing ? NaN : std(r.dQ[5, cols]) / dsd[5]
        dl1 = r.dQ === nothing ? NaN : lag1(r.dQ[2, cols])
        push!(rows, (; fr, lng, sks, ok))
        @printf("%-16s %2d  %-4s %6.3f  %s   %6.2f          %+6.2f      %5.2f %5.2f        %5.2f            %6.3f\n",
                label, i, ok ? "yes" : "NO", sks, join((@sprintf("%.3f", x) for x in fr), " "), lng,
                off1, sdr5, sdr6, dsr5, dl1)
    end
    summary[label] = rows
end

# the ceiling: fig16 showed every M4 fit capped near Z[16,32] ~ 2800 where the reference reaches 3800
q999(x) = sort(vec(x))[ceil(Int, 0.999 * length(x))]
@printf("
Z[16,32] 99.9th percentile / max -- reference %.0f / %.0f
", q999(qref[5, :]), maximum(qref[5, :]))
for (label, reps) in groups
    @printf("  %-18s %s
", label, join((@sprintf("%.0f/%.0f", q999(r.q[5, :]), maximum(r.q[5, :])) for r in reps), "  "))
end

# --- figure: Z[16,32] per replica, flat windows shaded ------------------------------------------
const INK, INK2, GRID, BG = colorant"#0b0b0b", colorant"#52514e", colorant"#e6e5e1", colorant"#fcfcfb"
# reference palette slots 1-4, validated on the light surface; every panel is titled with its
# ensemble, so identity never rests on colour (aqua is under 3:1 contrast)
const CM = [colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#4a3aa7"]
const SHADE = (colorant"#eda100", 0.25)
set_theme!(Theme(fontsize = 12, textcolor = INK,
                 Axis = (xgridcolor = GRID, ygridcolor = GRID, topspinevisible = false,
                         rightspinevisible = false, leftspinecolor = INK2, bottomspinecolor = INK2,
                         xticklabelcolor = INK2, yticklabelcolor = INK2, backgroundcolor = BG)))
const KF = 5          # the QoI the figure shows: Z[16,32]
t = (0:(ncol - 1)) .* 2.5e-3
# the M4 ensembles are the ones named in M4_ENSEMBLES -- the first `length(ENS)` groups, whatever
# their labels (a prefix test silently dropped every ensemble not labelled "M4 ...")
m4groups = groups[1:length(ENS)]
nrow = 1 + maximum(length(g.second) for g in m4groups)
length(m4groups) <= length(CM) ||
    error("$(length(m4groups)) M4 ensembles but only $(length(CM)) colours -- add a slot or split the figure")
fig = Figure(size = (max(1500, 420 * length(m4groups)), 170 * nrow + 120), backgroundcolor = BG)
Label(fig[0, 1:length(m4groups)], "$(QNAMES[KF]) online, 100 TU per replica — shaded: 0.5 TU windows with sd < $(Int(100FLAT))% of the reference's median";
      fontsize = 15, halign = :left, tellwidth = false)
yl = (0, 1.25 * maximum(qref[KF, :]))
function panel!(pos, x, col, title)
    a = Axis(pos; title, titlealign = :left, titlesize = 12, xticklabelsvisible = false)
    for (s, l) in episodes(flat(x, KF))
        vspan!(a, t[s + W ÷ 2 - 1], t[min(s + l - 1 + W ÷ 2, length(t))]; color = SHADE)
    end
    lines!(a, t[1:length(x)], x; color = col, linewidth = 0.7)
    ylims!(a, yl...)
    return a
end
for (c, (label, reps)) in enumerate(m4groups)
    panel!(fig[1, c], qref[KF, :], INK, "reference (R1)" * (c == 1 ? "" : "  (same)"))
    for (i, r) in enumerate(reps)
        a = panel!(fig[i + 1, c], r.q[KF, :], CM[c], @sprintf("%s — replica %d  (flat %.0f%%)", label, i,
                                                          100 * mean(flat(r.q[KF, :], KF))))
        i == length(reps) && (a.xticklabelsvisible = true; a.xlabel = "t (TU)")
    end
end
mkpath(FIGS)
path = joinpath(FIGS, "fig16_lstm_online_ensembles.png")
save(path, fig; px_per_unit = 2)
println("\nwrote $path")
