# The autocorrelation functions behind §1's timescale table, and behind D6's lead grid.
#
# §1 reports `T_int` as six numbers per series. A number is not enough to size a forecast window:
# `T_int = dt (1/2 + sum rho_k)` truncated at Sokal's window is an *integral*, and an oscillatory
# ACF can integrate to a small number while staying visibly non-zero out to many multiples of it --
# or cross zero long before it. Only the curve says which.
#
# 🔴 This figure is why `N_LEAD` changed. It fixed `10 * max(T_int) / dt = 2172` steps = 5.43 TU
# until 2026-09-16; the curves show the reference's autocorrelation is inside its own Bartlett band
# from `2 x T_int` onward, so the `5x` and `10x` leads were measuring climatology. `N_LEAD` is now
# 1200 steps = 3.00 TU and `score_d6.jl`'s `MULTIPLIERS` stops at `5x`. Both grids are drawn: the
# solid vertical is the kept maximum, the faint one past it the dropped `10x`.
#
# Left column: the LEVEL `q`, out to 10 TU. Right column: the CORRECTION `dQ`, out to 1 TU -- two
# different series with timescales differing by 3-60x (§1), plotted on their own axes because a
# shared one shows one of them as a spike.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/plot_acf.jl

using CairoMakie
using JLD2
using Printf
using Statistics

const HERE = @__DIR__
const SRC = normpath(joinpath(HERE, "..", "src"))
const FIGS = get(ENV, "REBASE_FIG_DIR", joinpath(HERE, "figures"))

for f in ("ts_scaling", "ts_history", "ts_models", "ts_fit", "ts_score")
    include(joinpath(SRC, "$f.jl"))
end
include(joinpath(HERE, "extract_qois.jl"))
include(joinpath(HERE, "extract_archive.jl"))

CairoMakie.activate!(; type = "png", px_per_unit = 2)

const DT = 2.5e-3
const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

"R1's tracking record, cached QoIs. Same constant as `score_m0_ddn.jl`'s `NEW_TRACK_QOIS`."
const NEW_TRACK = joinpath(HERE, "data",
    "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")
"Paper 2's tracking record. Kept only to show the level's 1.6x slowdown as a curve, not a ratio."
const ARCHIVE_TRACK = joinpath(HERE, "data",
    "data_track2_dns512_les64_Re2000.0_tsim100.0_qois.jld2")

# `score_d6.jl`'s constant, repeated rather than included: `score_d6.jl` pulls in the whole D6
# scoring layer. 🔴 If that table changes this one must follow -- there is no check tying them.
const T_INT_LEVEL = [0.2489, 0.4732, 0.4893, 0.4742, 0.5430, 0.5395]
const MULTIPLIERS = (0.25, 0.5, 1, 2, 5, 10)

const C_NEW_REF = RGBf(0.10, 0.10, 0.12)
const C_NEW_TRK = RGBf(0.00, 0.45, 0.70)
const C_ARCHIVE = RGBf(0.62, 0.62, 0.66)
const C_LEAD = RGBf(0.84, 0.37, 0.00)

"""
    first_below(r, level)

Index of the first lag whose autocorrelation drops to `level` or below, as a **lag** (`r[1]` is
lag 0), or `nothing` if it never does inside the computed window.

Reported rather than interpolated: at 2.5e-3 TU per lag the difference between adjacent lags is
far below anything this figure is read for, and an interpolated crossing would imply a precision
the estimator does not have.
"""
function first_below(r::AbstractVector, level::Real)
    k = findfirst(<=(level), r)
    return k === nothing ? nothing : k - 1
end

fmt_lag(k, dt) = k === nothing ? "     --" : @sprintf("%7.3f", k * dt)

"""
    bartlett_se(r, n; q)

Large-lag standard error of a sample autocorrelation, `sqrt((1 + 2 sum_{j=1}^{q} r_j^2) / n)`.

Bartlett's formula under the null that the series is uncorrelated beyond lag `q`. This is what the
figure is actually read for: without it "rho = 0.08 at the 10x lead" is an eyeball judgement, and
with it the question "does the longest lead measure anything the reference has not already
forgotten?" has a number attached.

`q` defaults to the first lag at which `r` falls below 0.1 -- the same crossing the table reports,
so the band and the crossing are not two independent conventions.
"""
function bartlett_se(r::AbstractVector, n::Integer; q = nothing)
    qq = max(q === nothing ? something(first_below(r, 0.1), length(r) - 1) : q, 1)
    return sqrt((1 + 2 * sum(abs2, view(r, 2:(qq + 1)))) / n)
end

"""
    acf_panel!(ax, series, maxlag; dt)

Draw one ACF panel and return the crossings it is read for.

`series` is a vector of `(label, vector, colour, linestyle)`. The first entry is the one the
crossings are reported from -- the reference, in both columns -- because the grid is sized off the
reference, not off any particular model run.
"""
function acf_panel!(ax, series, maxlag::Int; dt = DT)
    stats = nothing
    for (j, (lab, x, col, ls)) in enumerate(series)
        r = autocorr(collect(Float64.(x)), maxlag)
        lines!(ax, (0:maxlag) .* dt, r; color = col, linewidth = j == 1 ? 1.6 : 1.2,
               linestyle = ls, label = lab)
        if j == 1
            se = bartlett_se(r, length(x))
            stats = (; zero_lag = first_below(r, 0.0), e_lag = first_below(r, exp(-1)),
                     tenth = first_below(r, 0.1), se = se, r = r)
            # Everything inside this band is indistinguishable from an uncorrelated series.
            band!(ax, [0.0, maxlag * dt], [-2se, -2se], [2se, 2se];
                  color = (RGBf(0.45, 0.45, 0.50), 0.22))
        end
    end
    hlines!(ax, [0.0]; color = (:black, 0.35), linewidth = 0.8)
    hlines!(ax, [exp(-1)]; color = (:black, 0.25), linewidth = 0.8, linestyle = :dot)
    if stats !== nothing
        stats.e_lag === nothing ||
            scatter!(ax, [stats.e_lag * dt], [exp(-1)]; color = RGBf(0.00, 0.62, 0.45),
                     markersize = 10, marker = :dtriangle)
        stats.tenth === nothing ||
            scatter!(ax, [stats.tenth * dt], [0.1]; color = RGBf(0.80, 0.47, 0.65),
                     markersize = 10, marker = :dtriangle)
    end
    return stats
end

function main()
    ref = load_new_reference()
    isapprox(Float64(ref.dt_sample), DT; rtol = 1e-9) ||
        error("reference sampled at $(ref.dt_sample), not $DT")
    q_ref = Float64.(ref.q_ref)
    trk = load_qois(NEW_TRACK)
    q_trk, dQ_trk = Float64.(trk.q), Float64.(trk.dQ)
    q_arc = Float64.(load_reference())
    dQ_arc = Float64.(load_qois(ARCHIVE_TRACK).dQ)

    nq = size(q_ref, 1)
    maxlag_q, maxlag_dq = 4000, 400          # 10 TU and 1 TU

    fig = Figure(size = (1600, 1950))
    Label(fig[0, 1:2], "Autocorrelation of the QoIs: the level (left) and the correction (right)",
          fontsize = 19, font = :bold)
    Label(fig[1, 1], "level  q  —  0 to 10 TU", fontsize = 14, font = :bold)
    Label(fig[1, 2], "correction  dQ  —  0 to 1 TU", fontsize = 14, font = :bold)

    rows = NamedTuple[]
    for i = 1:nq
        axl = Axis(fig[i + 1, 1]; ylabel = LABELS[i], xlabel = i == nq ? "lag [TU]" : "",
                   xgridvisible = false, ygridvisible = false)
        sl = acf_panel!(axl, [("HF reference, new", view(q_ref, i, :), C_NEW_REF, :solid),
                              ("R1 tracked", view(q_trk, i, :), C_NEW_TRK, :solid),
                              ("HF reference, archive", view(q_arc, i, :), C_ARCHIVE, :dash)],
                        maxlag_q)

        # D6's lead grid for this band, old and new: solid marks the kept maximum (5x).
        for m in MULTIPLIERS
            t = m * T_INT_LEVEL[i]
            kept = m <= 5                                  # `MULTIPLIERS` after 2026-09-16
            vlines!(axl, [t]; color = (C_LEAD, m == 5 ? 0.95 : (kept ? 0.35 : 0.20)),
                    linewidth = m == 5 ? 1.6 : 0.8, linestyle = m == 5 ? :solid : :dot)
        end
        xlims!(axl, 0, maxlag_q * DT)
        ylims!(axl, -0.45, 1.05)
        i == 1 && axislegend(axl; position = :rt, framevisible = false, labelsize = 10)

        axr = Axis(fig[i + 1, 2]; xlabel = i == nq ? "lag [TU]" : "",
                   xgridvisible = false, ygridvisible = false)
        sr = acf_panel!(axr, [("R1 tracked", view(dQ_trk, i, :), C_NEW_TRK, :solid),
                              ("archive tracked", view(dQ_arc, i, :), C_ARCHIVE, :dash)],
                        maxlag_dq)
        xlims!(axr, 0, maxlag_dq * DT)
        ylims!(axr, -0.3, 1.05)
        i == 1 && axislegend(axr; position = :rt, framevisible = false, labelsize = 10)

        push!(rows, (; i, sl, sr))
    end

    Label(fig[nq + 2, 1:2],
          "Left: orange verticals are D6's lead grid for that band, (0.25, 0.5, 1, 2, 5, 10) × T_int; " *
          "the solid one is 5×T, the longest lead kept after 2026-09-16 (N_LEAD = 1200 steps = 3.00 TU).\n" *
          "Dotted horizontal is 1/e, solid horizontal is zero, and the grey band is ±2 Bartlett " *
          "standard errors — inside it the autocorrelation is indistinguishable from zero.\n" *
          "Green triangle marks the 1/e crossing, pink the 0.1 crossing, both read off the first " *
          "curve in each panel (the reference).", fontsize = 11)

    mkpath(FIGS)
    out = joinpath(FIGS, "fig1b_acf.png")
    save(out, fig)

    println()
    println("Level q -- where the REFERENCE actually decorrelates, against the lead grid [TU]")
    println("  QoI          T_int   lag@1/e   lag@0.1   lag@0    | 10xT_int  10xT/lag@0.1     2*se")
    for r in rows
        rat = r.sl.tenth === nothing ? NaN : (10 * T_INT_LEVEL[r.i]) / (r.sl.tenth * DT)
        @printf("  %-10s %7.4f %s %s %s |  %7.3f    %8s   %6.3f\n", LABELS[r.i], T_INT_LEVEL[r.i],
                fmt_lag(r.sl.e_lag, DT), fmt_lag(r.sl.tenth, DT), fmt_lag(r.sl.zero_lag, DT),
                10 * T_INT_LEVEL[r.i], isnan(rat) ? "--" : @sprintf("%.1fx", rat), 2 * r.sl.se)
    end

    println()
    println("Correction dQ -- R1 tracked [TU]")
    println("  QoI        lag@1/e   lag@0.1   lag@0")
    for r in rows
        @printf("  %-10s %s %s %s\n", LABELS[r.i], fmt_lag(r.sr.e_lag, DT),
                fmt_lag(r.sr.tenth, DT), fmt_lag(r.sr.zero_lag, DT))
    end

    println()
    println("Level q -- residual |rho| at each lead-grid multiplier, on the new reference")
    print("  QoI       ")
    for m in MULTIPLIERS
        @printf("%8s", @sprintf("%gxT", m))
    end
    println()
    for r in rows
        @printf("  %-10s", LABELS[r.i])
        for m in MULTIPLIERS
            k = round(Int, m * T_INT_LEVEL[r.i] / DT)
            @printf("%8.3f", k + 1 <= length(r.sl.r) ? r.sl.r[k + 1] : NaN)
        end
        println()
    end

    @printf("\nwrote %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
