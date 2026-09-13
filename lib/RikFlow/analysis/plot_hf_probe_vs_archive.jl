# The timing probe's QoI trajectories against the opening of paper 2's archived HF reference.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/plot_hf_probe_vs_archive.jl
#
# Inputs, both overridable by environment variable:
#   HFT_FILE      exp_square_HIT/output/hf_timing_probe_512_f64_lmwray3.jld2   (the new run)
#   HF_REF_QOIS   analysis/data/hf_reference_tsim100.0_qois.jld2               (the archive)
#   HF_REF_RAW    a full data_train_*.jld2, read directly if the cache is absent
#
# Writes two PNGs into analysis/figures/.
#
# 🔴 **The two runs are not the same realisation, and no amount of agreement or disagreement in the
# body of these curves means anything.** Three independent reasons:
#
#   1. Precision. The OU chain draws `randn!` into a Float64 buffer here and a Float32 one in the
#      archive, which consumes the random stream differently. The forcing is a *different sample
#      path* from step one, not a more accurate version of the same one.
#   2. Stepper. LMWray3 against the archive's RK44.
#   3. `Z[16,32]` and `E[16,32]` are computed under the corrected Nyquist convention of `09954be1`;
#      the archive was written under the old one (gotchas #45, #46).
#
# Add chaos to that and the trajectories must separate. **The comparison that does carry
# information is the first column**, where both sides filter the *same* archived spin-up field and
# no time stepping has happened yet: the six values should agree to a few digits, and the top two
# bands are where the Nyquist change shows. That number is printed, not plotted.
#
# Everything after column 1 is a distributional check at best — over the probe's ~0.8 TU, which is
# well under one eddy turnover, even that is weak. The window mean and standard deviation per band
# are printed for what they are worth; read them as "the new run is in the same regime", never as
# "the new run reproduces the archive".

using CairoMakie
using JLD2
using Printf
using Statistics

const HERE = @__DIR__
const FIGS = get(ENV, "HF_FIG_DIR", joinpath(HERE, "figures"))
const DATA = joinpath(HERE, "data")

const PROBE = get(
    ENV,
    "HFT_FILE",
    normpath(joinpath(HERE, "..", "exp_square_HIT", "output",
                      "hf_timing_probe_512_f64_lmwray3.jld2")),
)
const REF = get(ENV, "HF_REF_QOIS", joinpath(DATA, "hf_reference_tsim100.0_qois.jld2"))
const REF_RAW = get(ENV, "HF_REF_RAW", "")

const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

# 🔴 The archive's own sample spacing, fixed by the run that produced it: savefreq = 10 at
# Δt = 2.5e-4. It is *not* read from the probe. Using the probe's spacing for both series is right
# only while the probe runs production sampling, and silently wrong the moment it does not —
# `HFT_SAVEFREQ=2` would put the archive on a five-times-compressed axis with nothing to show for
# it. Each series gets its own axis below, and `commensurate` decides what may be compared
# column-by-column.
const ARCH_DT = 2.5e-3

# How much of the archive to show behind the probe window. The probe covers well under one eddy
# turnover, so on its own axis there is no way to tell a difference from an ordinary excursion;
# this is the context that makes that judgeable.
const CONTEXT_TU = parse(Float64, get(ENV, "HF_CONTEXT_TU", "10"))

CairoMakie.activate!(; type = "png", px_per_unit = 2)

# Same palette as `plot_validation.jl` and `plot_paper4.jl`: colour-blind safe, archive darker and
# thinner so it reads as the established record rather than as a competing model.
const C_ARCH = RGBf(0.15, 0.15, 0.18)
const C_NEW = RGBf(0.00, 0.45, 0.70)
const C_WARM = RGBf(0.93, 0.93, 0.90)

# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

"""
    load_probe(path)

The new run's QoIs on one absolute time axis, plus the sample index where the measured block
starts.

🔴 The probe runs **two** solves and each restarts its step counter at zero, so the measured
block's first sample is the same physical state as the warm-up's last. Concatenating naively
duplicates that column and shifts everything after it by one sample — a quarter of a percent of the
window, invisible on a plot and wrong in a difference. It is dropped here.
"""
function load_probe(path)
    isfile(path) || error(
        "no probe output at:\n    $path\n\n" *
        "Run exp_square_HIT/batch_scripts/run_hf_timing_probe.sh first, or point HFT_FILE at it.",
    )
    d = load(path)
    p = d["params"]
    warm = Float64.(d["qoi_warm"])
    meas = Float64.(d["qoi_hist"])
    q = hcat(warm, meas[:, 2:end])          # see the docstring: drop the repeated state
    dt_sample = Float64(p.savefreq) * Float64(p.Δt)
    t = (0:size(q, 2)-1) .* dt_sample
    (; q, t, dt_sample, nwarm = size(warm, 2), params = p)
end

"""
    load_archive(ncols)

The archive's first `ncols` QoI samples. Prefers the extracted cache; falls back to reading a full
`data_train_*.jld2` when `HF_REF_RAW` names one.
"""
function load_archive(ncols)
    if isfile(REF)
        q = Float64.(load(REF, "q_ref"))
        return (; q, src = REF)
    elseif !isempty(REF_RAW) && isfile(REF_RAW)
        @info "reading the full archive (slow — the QoIs are ~1 MB of a ~1.4 GB file)" REF_RAW
        q = jldopen(REF_RAW, "r") do io
            Float64.(stack(io["data_train"].data[1].qoi_hist))
        end
        return (; q, src = REF_RAW)
    end
    error(
        "no archived HF reference QoIs.\n" *
        "  looked for the cache at: $REF\n" *
        "  and HF_REF_RAW was " * (isempty(REF_RAW) ? "unset" : "set to a missing file: $REF_RAW") *
        "\n\nBuild the cache with\n" *
        "    julia --project=analysis analysis/extract_archive.jl\n" *
        "(set RIKFLOW_ARCHIVE if the frozen archive is not at its default path), or point\n" *
        "HF_REF_RAW at data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2 directly.",
    )
end

# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

"""
Six panels, one per QoI band.

The archive runs out to `CONTEXT_TU`; the probe covers whatever it covers, shaded. Showing only
the overlap would be misleading in the one direction that matters — over 0.8 TU any two turbulent
trajectories look similar, and there would be nothing to say whether a gap is a difference or an
ordinary excursion. The archive's own wander over 10 TU is that yardstick.

Each series is drawn on its **own** time axis (`ARCH_DT` for the archive, the probe's measured
spacing for the probe), so the figure stays correct even when the probe is run with non-production
sampling.
"""
function fig_trajectories(pr, ar, nctx, n)
    fig = Figure(; size = (1000, 620))
    tarc = (0:nctx-1) .* ARCH_DT
    tnew = pr.t[1:n]
    for i = 1:6
        r, c = fldmod1(i, 3)
        ax = Axis(fig[r, c];
            title = LABELS[i],
            xlabel = r == 2 ? "t" : "",
            ylabel = c == 1 ? "QoI" : "",
            # Linear, autoscaled per panel. On the archive's opening the six bands are
            # O(1e3) for Z and O(1) for E — same order within each kind — so a log axis would
            # compress the only thing worth seeing, which is how the two lines separate.
        )
        # The probe's whole window, not just its warm-up: at 10 TU the warm-up is a hairline and
        # what the eye needs is how little of the record the new run covers.
        vspan!(ax, 0, tnew[end]; color = C_WARM)
        lines!(ax, tarc, view(ar.q, i, 1:nctx); color = C_ARCH, linewidth = 1.2)
        lines!(ax, tnew, view(pr.q, i, 1:n); color = C_NEW, linewidth = 1.8)
    end
    Legend(fig[3, 1:3],
        [LineElement(color = C_ARCH, linewidth = 2), LineElement(color = C_NEW, linewidth = 2),
         PolyElement(color = C_WARM)],
        [@sprintf("archive, %.3g TU (Float32 / RK44, old Nyquist)", tarc[end]),
         @sprintf("new, %.3g TU (Float64 / LMWray3)", tnew[end]),
         "probe window"];
        orientation = :horizontal, framevisible = false)
    rowsize!(fig.layout, 3, Relative(0.08))
    Label(fig[0, 1:3],
        "HF QoI trajectories: timing probe against the archive's first " *
        @sprintf("%.3g TU", tarc[end]) *
        " — different realisations, divergence is expected";
        fontsize = 13, padding = (0, 0, 4, 0))
    fig
end

"""
Relative deviation per band, log axis.

This is the figure that says something. The first column is the like-for-like comparison — same
field, no stepping — so the curve starts at the filter-and-convention difference and grows from
there at whatever rate the two realisations separate. A start that is *not* small is the
interesting outcome: it would mean the filter, the masks or `compute_QoI` changed, not the physics.
"""
function fig_deviation(pr, ar, n)
    fig = Figure(; size = (900, 480))
    ax = Axis(fig[1, 1];
        xlabel = "t", ylabel = "|new - archive| / |archive|", yscale = log10,
        title = "Relative deviation by band")
    vspan!(ax, 0, pr.t[pr.nwarm]; color = C_WARM)
    cols = Makie.wong_colors()
    for i = 1:6
        d = abs.(view(pr.q, i, 1:n) .- view(ar.q, i, 1:n)) ./ abs.(view(ar.q, i, 1:n))
        d = max.(d, 1e-16)              # a log axis cannot take an exact zero
        lines!(ax, pr.t[1:n], d; color = cols[i], linewidth = 1.5, label = LABELS[i])
    end
    # Outside the axis, not `axislegend`: the deviation grows left to right, so any in-axis corner
    # is where the curves end up.
    Legend(fig[1, 2], ax; framevisible = false)
    fig
end

# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------

pr = load_probe(PROBE)
ar = load_archive(size(pr.q, 2))

# Context window for the archive, and the column-paired overlap for everything else.
nctx = min(size(ar.q, 2), round(Int, CONTEXT_TU / ARCH_DT) + 1)
n = min(size(pr.q, 2), size(ar.q, 2))
n >= 2 || error("only $n overlapping samples; nothing to plot")

# 🔴 Column-by-column comparison is only meaningful when both series are sampled at the same
# spacing. The trajectory figure is safe either way — each series has its own axis — but the
# deviation curve and the window statistics pair column i against column i, which is nonsense
# across different Δt. Production sampling gives exactly ARCH_DT, so this only fires on a probe run
# with HFT_SAVEFREQ / Δt overridden.
commensurate = isapprox(pr.dt_sample, ARCH_DT; rtol = 1e-9)

mkpath(FIGS)

@printf("probe   %s\n", PROBE)
@printf("        %d samples every %.4g t  (%d warm-up + %d measured), t = 0 .. %.4g\n",
    size(pr.q, 2), pr.dt_sample, pr.nwarm, size(pr.q, 2) - pr.nwarm, pr.t[end])
@printf("archive %s\n        %d samples every %.4g t; showing %d (%.3g TU) as context\n",
    ar.src, size(ar.q, 2), ARCH_DT, nctx, (nctx - 1) * ARCH_DT)
if size(ar.q, 2) < size(pr.q, 2)
    @warn "the archive is shorter than the probe window; comparing over the overlap only"
end
if !commensurate
    @warn "probe and archive are sampled at different spacings — the trajectory figure is still " *
          "correct (each series has its own axis) but column-paired comparisons are not" probe =
        pr.dt_sample archive = ARCH_DT
end

println()
println("column 1 — same field, no time stepping. This is the only like-for-like comparison.")
println("  band          archive         new             rel. diff")
for i = 1:6
    a, b = ar.q[i, 1], pr.q[i, 1]
    @printf("  %-12s  %.6e    %.6e    %.2e\n", LABELS[i], a, b, abs(b - a) / abs(a))
end

println()
@printf("window statistics. The probe's %.3g TU against the same window of the archive, and\n",
    pr.t[n])
@printf("against the archive's full %.3g TU — the second is the yardstick: a new-run mean that\n",
    (nctx - 1) * ARCH_DT)
println("sits inside the archive's own wander is as much agreement as this window can show.")
println("  band          new (probe window)        archive (same window)     archive (context)")
for i = 1:6
    b = view(pr.q, i, 1:n)
    a = view(ar.q, i, 1:n)
    c = view(ar.q, i, 1:nctx)
    @printf("  %-12s  %.3e ± %.1e    %.3e ± %.1e    %.3e ± %.1e\n",
        LABELS[i], mean(b), std(b), mean(a), std(a), mean(c), std(c))
end

f1 = joinpath(FIGS, "hf_probe_vs_archive_trajectories.png")
save(f1, fig_trajectories(pr, ar, nctx, n))
println()
println("Written:")
println("  $f1")
if commensurate
    f2 = joinpath(FIGS, "hf_probe_vs_archive_deviation.png")
    save(f2, fig_deviation(pr, ar, n))
    println("  $f2")
else
    println("  (deviation figure skipped: sample spacings differ, see the warning above)")
end
