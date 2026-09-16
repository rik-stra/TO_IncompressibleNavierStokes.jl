# D6's ensembles drawn on the reference trajectory: one fan per initial condition.
#
# The figure `results.md` has no equivalent of. `plot_rebaseline.jl` shows free-running 100 TU
# trajectories, where model and reference decorrelate within an eddy turnover and the panels can
# only be read for their envelope. D6 is the opposite experiment -- every member starts *on* the
# reference at a known time -- so the thing worth seeing is each fan leaving the truth it started
# from, and how fast.
#
# 🔴 **The first `nwarm` steps are NOT a forecast.** There the sampler replays the recorded `dQ`
# verbatim, so all `M` members are the same run and sit on the reference. Spread can only begin at
# the warm-up boundary, which is drawn on every fan. Reading the fan's width before that line as
# "the ensemble is confident" is reading the replay.
#
# Two figures, because one cannot do both jobs. Each fan runs `nlead * dt` -- 5.43 TU for the pilot
# on disk, 3.00 TU after `N_LEAD` was cut on 2026-09-16 -- while the ICs
# are only 0.25-0.50 TU apart, so on a shared axis the fans overlap almost completely and only the
# first turnover of each is legible:
#
#   fig9_d6_fans.png        all five fans on one reference. Shows where the ICs sit relative to the
#                           reference's own features, and that they cover a common window.
#   fig9b_d6_fans_by_ic.png one panel per (band, IC) over that fan's own window. This is the one to
#                           read for growth rate; it also carries the ensemble mean.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/plot_d6_fans.jl
#
# Environment:
#   D6_OUT         where the run outputs are (default exp_square_HIT/output/D6)
#   D6_FAN_STRIDE  display stride in steps (default 2 = 0.005 TU)
#   D6_FAN_PAD     TU of reference drawn before the first IC and after the last forecast (default 1)
#   D6_FAN_MAX     draw only the first N initial conditions (default 0 = all)
#   D6_FAN_TAG     suffix for the figure names and titles, e.g. the closure

using CairoMakie
using JLD2
using Printf
using Statistics

const HERE = @__DIR__
const FIGS = get(ENV, "REBASE_FIG_DIR", joinpath(HERE, "figures"))
const D6_DIR = get(ENV, "D6_OUT",
                   normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "D6")))
const STRIDE = parse(Int, get(ENV, "D6_FAN_STRIDE", "2"))
const PAD_TU = parse(Float64, get(ENV, "D6_FAN_PAD", "1.0"))
"How many initial conditions to draw, after striding. 0 means all of them."
const FAN_MAX = parse(Int, get(ENV, "D6_FAN_MAX", "0"))
"""
    FAN_EVERY

Draw every Nth initial condition. **0 (the default) derives the smallest N whose fans do not
overlap**, from the fans actually loaded.

🔑 At K = 90 the ICs sit 0.97 TU apart while a fan runs `nlead * dt` = 3.25 TU, so consecutive fans
cover each other more than three deep and only the first turnover of each is ever legible. Every
3rd still overlaps, by 0.33 TU; every 4th clears it with 0.64 TU to spare. That arithmetic moves
with `N_LEAD` and with K, so it is computed rather than written down.
"""
const FAN_EVERY = parse(Int, get(ENV, "D6_FAN_EVERY", "0"))
"Columns in the per-IC figure. 23 non-overlapping fans would make 138 unreadable panels."
const FAN_PANELS = parse(Int, get(ENV, "D6_FAN_PANELS", "8"))
"Tag put in the figure names and titles, so two closures do not overwrite each other's output."
const TAG = get(ENV, "D6_FAN_TAG", "")

# The reference carries its own sampling interval (`extract_new_reference` stores `dt_sample`
# from the run's `savefreq * dt` rather than assuming it), so this is a CHECK, not the source of
# truth: `main` asserts the loaded reference agrees. A run sampled differently must not land
# silently on this axis.
const DT_EXPECTED = 2.5e-3

include(joinpath(HERE, "extract_archive.jl"))    # load_new_reference

CairoMakie.activate!(; type = "png", px_per_unit = 2)

const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]
const C_REF = RGBf(0.10, 0.10, 0.12)

# One colour per initial condition, colour-blind safe (Okabe-Ito). Members of one IC share it, so a
# fan reads as one object; the eye separates ICs by hue and members by overlap.
const C_IC = [RGBf(0.00, 0.45, 0.70), RGBf(0.84, 0.37, 0.00), RGBf(0.00, 0.62, 0.45),
              RGBf(0.80, 0.47, 0.65), RGBf(0.34, 0.71, 0.91), RGBf(0.94, 0.89, 0.26),
              RGBf(0.35, 0.35, 0.35)]

"""
    fan_colors(n)

One colour per initial condition.

Okabe-Ito while it lasts, because up to seven fans the eye separates them best by hue. Past seven
a recycled palette would give two fans the same colour, so it switches to a sequential ramp -- which
also happens to encode what the categorical palette cannot, that the ICs are ordered in time.
"""
fan_colors(n::Integer) = n <= length(C_IC) ? C_IC[1:n] :
                         [get(cgrad(:viridis), x) for x in range(0.0, 0.88; length = n)]

"""
    load_fans(dir)

Every scored D6 output in `dir`, grouped by initial condition.

Returns a vector of `(; k, n_k, nwarm, q)` sorted by `n_k`, with `q` a vector of `N_Q x (nt+1)`
matrices, one per member. Validation files (`d6_valid_*`) are skipped: ordinal 0 is not a scored IC
and sits inside the fit window.
"""
function load_fans(dir = D6_DIR)
    isdir(dir) || error("no D6 output directory at $dir (set D6_OUT)")
    pat = r"^d6_online_ic(\d+)_m(\d+)\.jld2$"
    by_k = Dict{Int,Vector{Tuple{Int,String}}}()
    for f in readdir(dir)
        m = match(pat, f)
        m === nothing && continue
        push!(get!(by_k, parse(Int, m.captures[1]), Tuple{Int,String}[]),
              (parse(Int, m.captures[2]), joinpath(dir, f)))
    end
    isempty(by_k) && error("no d6_online_ic*_m*.jld2 in $dir")
    fans = NamedTuple[]
    for k in sort(collect(keys(by_k)))
        entries = sort(by_k[k]; by = first)
        qs, n_k, nwarm = Matrix{Float64}[], -1, -1
        for (_, p) in entries
            d = load(p)
            n_k < 0 && (n_k = d["n_k"]; nwarm = d["nwarm"])
            d["n_k"] == n_k || error("$p disagrees on n_k")
            d["nwarm"] == nwarm || error("$p disagrees on nwarm")
            push!(qs, Float64.(d["q"]))
        end
        push!(fans, (; k, n_k, nwarm, q = qs))
    end
    return sort(fans; by = f -> f.n_k)
end

"""
    stride_fans(fans, dt)

Thin `fans` so the drawn ones do not cover each other, and say what was chosen.

Returns `(kept, every)`. With `FAN_EVERY = 0` the stride is the smallest that separates consecutive
start points by at least one fan length; a fan length is read off the data rather than assumed, so
this stays right when `N_LEAD` changes.
"""
function stride_fans(fans, dt)
    length(fans) < 2 && return fans, 1
    span = (size(first(first(fans).q), 2) - 1) * dt            # one fan, in TU
    gaps = diff([f.n_k for f in fans]) .* dt
    every = FAN_EVERY > 0 ? FAN_EVERY : max(1, ceil(Int, span / minimum(gaps)))
    kept = fans[1:every:end]
    FAN_MAX > 0 && (kept = kept[1:min(FAN_MAX, length(kept))])
    @printf("fan length %.2f TU, IC spacing %.2f TU (min %.2f) -> every %d%s
",
            span, mean(gaps), minimum(gaps), every,
            FAN_EVERY > 0 ? " (forced by D6_FAN_EVERY)" : " (derived: smallest non-overlapping)")
    @printf("  %d of %d fans drawn; consecutive start points %.2f TU apart, fan length %.2f TU%s
",
            length(kept), length(fans), every * minimum(gaps), span,
            every * minimum(gaps) >= span ? " -- no overlap" : " -- ⚠️ STILL OVERLAPPING")
    return kept, every
end

"""
    ref_columns(fan)

The reference columns member column `c` corresponds to: `n_k + c`.

Checked rather than assumed -- `plot_d6_fans` asserts the member's own first column matches the
reference there, which is the same alignment `score_d6.jl`'s `truth_column` uses and the one thing
that would silently shift every fan sideways.
"""
ref_columns(fan) = (fan.n_k + 1):(fan.n_k + size(first(fan.q), 2))

function main()
    ref = load_new_reference()
    q_ref = Float64.(ref.q_ref)
    dt = Float64(ref.dt_sample)
    isapprox(dt, DT_EXPECTED; rtol = 1e-9) ||
        error("reference sampled at dt = $dt, not the expected $DT_EXPECTED")
    fans, every = stride_fans(load_fans(), dt)
    nq = size(q_ref, 1)

    @printf("%d initial conditions from %s\n", length(fans), D6_DIR)
    for f in fans
        @printf("  k = %3d  n_k = %5d  t = %6.2f TU  M = %2d  nwarm = %d  nt+1 = %d\n",
                f.k, f.n_k, f.n_k * dt, length(f.q), f.nwarm, size(first(f.q), 2))
    end

    # 🔴 The alignment check. A fan drawn one column out would look like a model that starts wrong.
    for f in fans
        cols = ref_columns(f)
        last(cols) <= size(q_ref, 2) ||
            error("fan k = $(f.k) runs past the reference: needs column $(last(cols))")
        q0 = first(f.q)[:, 1]
        rel = maximum(abs.(q0 .- q_ref[:, first(cols)]) ./ abs.(q_ref[:, first(cols)]))
        rel < 1e-2 || error("fan k = $(f.k) does not start on the reference: rel $(rel)")
    end
    println("  alignment: every fan's first column sits on the reference ✓")

    t0 = first(fans).n_k * dt - PAD_TU
    t1 = (last(fans).n_k + size(first(last(fans).q), 2)) * dt + PAD_TU
    c0 = max(1, round(Int, t0 / dt))
    c1 = min(size(q_ref, 2), round(Int, t1 / dt))
    tref = ((c0:STRIDE:c1) .- 1) .* dt

    fancols = fan_colors(length(fans))
    M = length(first(fans).q)

    # Non-overlapping fans span the whole record, so a fixed width would squeeze each one to a
    # few pixels. Budget width per fan instead, and cap it at what a PNG viewer will take.
    fig = Figure(size = (clamp(150 * length(fans) + 350, 1500, 5200), 1500))
    Label(fig[0, 1], "D6$(isempty(TAG) ? "" : " — " * TAG): the reference trajectory with each " *
          "initial condition's $(M)-member ensemble on top", fontsize = 18, font = :bold, tellwidth = false)

    for i = 1:nq
        ax = Axis(fig[i, 1]; ylabel = LABELS[i], xlabel = i == nq ? "t [TU]" : "",
                  xgridvisible = false, ygridvisible = false)
        for (j, f) in enumerate(fans)
            col = fancols[j]
            cols = ref_columns(f)
            t = ((cols[1]:STRIDE:cols[end]) .- 1) .* dt
            for q in f.q
                lines!(ax, t, @view(q[i, 1:STRIDE:end]); color = (col, 0.30), linewidth = 0.6)
            end
            # Where the replay stops and the forecast starts. Before it the members are one run.
            tw = (f.n_k + f.nwarm) * dt
            vlines!(ax, [tw]; color = (col, 0.85), linewidth = 1.0, linestyle = :dash)
            scatter!(ax, [f.n_k * dt], [q_ref[i, f.n_k + 1]]; color = col, markersize = 7)
        end
        lines!(ax, tref, @view(q_ref[i, c0:STRIDE:c1]); color = C_REF, linewidth = 1.3,
               label = i == 1 ? "HF reference" : nothing)
        xlims!(ax, t0, t1)
        i == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
    end

    Label(fig[nq + 1, 1],
          "Dots mark each IC on the reference; the dashed line of the same colour is where its " *
          "replayed warm-up ends and the forecast begins.\nBefore that line the $(first(fans).nwarm) " *
          "warm-up steps emit the recorded dQ verbatim, so all $(M) members are the same run and " *
          "lie on the reference —\nspread before it would be a bug, not confidence. Displayed at " *
          "stride $(STRIDE) (= $(STRIDE * dt) TU).", fontsize = 11, tellwidth = false)

    mkpath(FIGS)
    out = joinpath(FIGS, "fig9_d6_fans$(isempty(TAG) ? "" : "_" * TAG).png")
    save(out, fig)
    @printf("\nwrote %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)

    out2 = by_ic_figure(q_ref, fans, dt)
    @printf("wrote %s (%.2f MB)\n", basename(out2), filesize(out2) / 2^20)
    return (out, out2)
end

"""
    by_ic_figure(q_ref, fans, dt)

One panel per (band, initial condition), each over that fan's own window.

The overlay figure answers "where are the ICs"; this one answers "how fast does a fan open".
Panels in a row share a y-axis so growth is comparable across ICs at a glance, and each carries the
**ensemble mean** -- the quantity `score_d6.jl`'s skill term is built from, and the one whose
departure from the reference is error rather than spread.
"""
function by_ic_figure(q_ref, fans_all, dt)
    nq = size(q_ref, 1)
    fans = fans_all[1:min(FAN_PANELS, length(fans_all))]
    nic = length(fans)
    fancols = fan_colors(nic)
    M = length(first(fans).q)
    fig = Figure(size = (max(2000, 260 * nic + 220), 1700))
    Label(fig[0, 1:nic],
          "D6$(isempty(TAG) ? "" : " — " * TAG): each initial condition's ensemble against the " *
          "reference, over its own lead window" *
          (nic < length(fans_all) ? "  (first $nic of $(length(fans_all)) drawn)" : ""),
          fontsize = 18, font = :bold, tellwidth = false)

    rows = [Axis[] for _ = 1:nq]
    for i = 1:nq, (j, f) in enumerate(fans)
        col = fancols[j]
        cols = ref_columns(f)
        t = ((cols[1]:STRIDE:cols[end]) .- 1) .* dt
        ax = Axis(fig[i, j];
                  ylabel = j == 1 ? LABELS[i] : "",
                  xlabel = i == nq ? "t [TU]" : "",
                  title = i == 1 ? @sprintf("IC %d  (t = %.2f TU)", f.k, f.n_k * dt) : "",
                  titlesize = 13, xgridvisible = false, ygridvisible = false,
                  yticklabelsvisible = j == 1, xticklabelsvisible = i == nq)
        push!(rows[i], ax)
        for q in f.q
            lines!(ax, t, @view(q[i, 1:STRIDE:end]); color = (col, 0.45), linewidth = 0.7)
        end
        # The ensemble mean: skill, as against the spread the individual members show.
        qbar = vec(mean(reduce(hcat, [q[i, 1:STRIDE:end] for q in f.q]); dims = 2))
        lines!(ax, t, qbar; color = col, linewidth = 1.8)
        lines!(ax, t, @view(q_ref[i, cols[1]:STRIDE:cols[end]]); color = C_REF, linewidth = 1.4)
        vlines!(ax, [(f.n_k + f.nwarm) * dt]; color = (C_REF, 0.5), linewidth = 1.0,
                linestyle = :dash)
        xlims!(ax, (cols[1] - 1) * dt, (cols[end] - 1) * dt)
    end
    for r in rows
        linkyaxes!(r...)
    end

    nstep = size(first(first(fans).q), 2) - 1
    Label(fig[nq + 1, 1:nic],
          "Thin lines: the $(M) members. Thick coloured line: the ensemble mean. Black: the HF " *
          "reference. Dashed: the end of the replayed warm-up, where the forecast starts.\n" *
          "Panels in a row share a y-axis. Each window is $(nstep) steps = " *
          "$(round(nstep * dt; digits = 2)) TU, of which the first $(first(fans).nwarm) are replay.",
          fontsize = 11, tellwidth = false)

    out = joinpath(FIGS, "fig9b_d6_fans_by_ic$(isempty(TAG) ? "" : "_" * TAG).png")
    save(out, fig)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
