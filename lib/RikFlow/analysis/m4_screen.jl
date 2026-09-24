# Screen short online runs of exploratory M4 fits against the reference's own short windows.
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/m4_screen.jl
#
# Reads every `output/TO_LSTM/explore/<tag>/data_online_tsim*_replica*.jld2` (CPU `_cpu` and GPU runs), plus -- as
# baselines -- the first `TSCREEN` TU of the cluster replicas of the two teacher-forced stride-100
# fits. Environment: `TSCREEN` (TU per run scored, default 10), `QOI_CACHE`.
#
# 🔑 **The null is the reference cut into windows of the SAME length** as the screened runs: a 10 TU
# window of the true flow has its own flat fraction, exceedance and KS, and a candidate is judged
# against that band, not against the 100 TU numbers.
#
# Metrics, all on Z[16,32] (the band where the ceiling and the flat state are clearest) unless named:
#   flat    fraction of 0.5 TU windows with sd < 30% of the reference's median 0.5 TU sd
#   >2900   fraction of time above 2900 -- every M4 run on the cluster had 0; the reference ~5%
#   max/min largest / smallest value reached -- a :dQ-target fit drained Z[16,32] to 4 once
#           (reference minimum 721), so both tails are screened
#   clamp   steps on which the |q*| < 1e-2 gate zeroed dQ (an alarm on HIT, #65)
#   sdr     sd ratio against the reference's 100 TU sd
#   KS      summed KS over the six QoIs against the reference's 100 TU marginal (ranking aid, #61)
#   lag1    lag-1 autocorrelation of dQ on E[0,6], warm-up excluded (reference 0.743)

using Statistics, Printf, JLD2

const OUT = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const TO = joinpath(OUT, "TO_LSTM")
const CACHE = get(ENV, "QOI_CACHE",
                  joinpath(@__DIR__, "data",
                           "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const TSCREEN = parse(Float64, get(ENV, "TSCREEN", "10"))
const DT, W, FLAT, THR = 2.5e-3, 200, 0.3, 2900.0
const n = round(Int, TSCREEN / DT) + 1

ref = load(CACHE)
qref, dQref = ref["q"], ref["dQ"]
const REFSD = [median(std(view(qref, k, i:(i + W - 1))) for i in 1:(size(qref, 2) - W + 1)) for k in 1:6]
sd100 = vec(std(qref; dims = 2))

function ks(a, b)
    a, b = sort(vec(a)), sort(vec(b)); na, nb = length(a), length(b); i = j = 0; d = 0.0
    while i < na && j < nb
        x = min(a[i + 1], b[j + 1])
        while i < na && a[i + 1] <= x; i += 1; end
        while j < nb && b[j + 1] <= x; j += 1; end
        d = max(d, abs(i / na - j / nb))
    end
    return d
end
lag1(x) = (y = x .- mean(x); sum(y[1:(end - 1)] .* y[2:end]) / sum(abs2, y))
flatfrac(x, k) = mean(std(view(x, i:(i + W - 1))) < FLAT * REFSD[k] for i in 1:(length(x) - W + 1))

function metrics(q, dQ; nwarm = 100)
    z = q[5, :]
    (; flat = flatfrac(z, 5), exc = mean(z .> THR), zmax = maximum(z), zmin = minimum(z),
     clamp = dQ === nothing ? 0 : count(j -> all(iszero, view(dQ, :, j)), (nwarm + 1):size(dQ, 2)),
     sdr = std(z) / sd100[5],
     KS = sum(ks(q[k, :], qref[k, :]) for k in 1:6),
     lag = dQ === nothing ? NaN : lag1(dQ[2, (nwarm + 1):end]), ok = all(isfinite, q))
end
fmt(m) = @sprintf("%5.1f%%  %5.1f%%  %5.0f  %5.0f  %5.2f  %5.2f  %6.3f  %4d  %s", 100m.flat, 100m.exc, m.zmax,
                  m.zmin, m.sdr, m.KS, m.lag, m.clamp, m.ok ? "" : "NON-FINITE")

println(@sprintf("screening window %.0f TU; columns: flat  >%.0f  max  min  sdr  KS  lag1(dQ E06)  clamp", TSCREEN, THR))
nulls = [metrics(qref[:, s:(s + n - 1)], dQref[:, s:(s + n - 2)]; nwarm = 0)
         for s in 1:(n - 1):(size(qref, 2) - n + 1)]
for (nm, f) in (("flat", m -> 100m.flat), (">2900", m -> 100m.exc), ("max", m -> m.zmax), ("min", m -> m.zmin),
                ("sdr", m -> m.sdr), ("KS", m -> m.KS), ("lag1", m -> m.lag))
    v = f.(nulls)
    @printf("  reference %-6s over %d windows: min %.3g  median %.3g  max %.3g\n", nm, length(v),
            minimum(v), median(v), maximum(v))
end

cut(d) = (; q = d.q[:, 1:min(n, size(d.q, 2))], dQ = d.dQ[:, 1:min(n - 1, size(d.dQ, 2))])
println()
for (label, dir) in (("baseline beta1e-4 (GPU)", "StochLSTM2_s100b32_points3_cap10000"),
                     ("baseline beta1e-2 (GPU)", "StochLSTM3_s100b32_points3_cap10000"),
                     ("baseline b1e-4 roll (GPU)", "StochLSTM2_s100b32_points3_cap10000_roll100_L200"))
    p = joinpath(TO, dir)
    isdir(p) || continue
    for f in sort(filter(x -> occursin(r"^data_online_tsim100\.0_replica[12]\.jld2$", x), readdir(p)))
        println(rpad(label, 34), rpad(replace(f, "data_online_" => ""), 26), fmt(metrics(cut(load(joinpath(p, f), "data_online"))...)))
    end
end
edir = joinpath(TO, "explore")
for tag in (isdir(edir) ? sort(readdir(edir)) : String[])
    p = joinpath(edir, tag)
    # both local CPU runs (`_cpu`) and GPU runs
    fs = sort(filter(x -> occursin(r"^data_online_tsim.*_replica\d+(_cpu)?\.jld2$", x), readdir(p)))
    isempty(fs) && continue
    x = load(joinpath(p, "StochLSTM_seed1.jld2"), "extras")
    for f in fs
        d = load(joinpath(p, f))
        o = d["data_online"]
        size(o.q, 2) < n && println(rpad(tag, 34), rpad(f, 26), "shorter than the screening window ($(size(o.q, 2)) cols)")
        size(o.q, 2) < n && continue
        println(rpad(tag, 34), rpad(replace(f, "data_online_" => "", "_cpu.jld2" => ""), 26),
                fmt(metrics(cut(o)...; nwarm = get(d, "nwarm", 100))),
                @sprintf("   [val TF %.3g, ro %.3g]", x.val.tf, x.val.ro))
    end
end
