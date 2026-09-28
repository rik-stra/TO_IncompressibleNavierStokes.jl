# Score 100 TU online runs: each 20 TU chunk with `m4_screen.jl`'s metrics (the reference band is its
# own five 20 TU windows), plus the whole run against the whole 100 TU reference.
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/m4_screen_long.jl fitdir...
using Statistics, Printf, JLD2
const TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
ref = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
qref, dQref = ref["q"], ref["dQ"]
const DT, W, FLAT, THR = 2.5e-3, 200, 0.3, 2900.0
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
metrics(q, dQ) = (z = q[5, :]; (; flat = flatfrac(z, 5), exc = mean(z .> THR), zmax = maximum(z), zmin = minimum(z),
    clamp = count(j -> all(iszero, view(dQ, :, j)), axes(dQ, 2)), sdr = std(z) / sd100[5],
    KS = sum(ks(q[k, :], qref[k, :]) for k in 1:6), lag = lag1(dQ[2, :])))
fmt(m) = @sprintf("%5.1f%%  %5.1f%%  %5.0f  %5.0f  %5.2f  %5.2f  %6.3f  %4d", 100m.flat, 100m.exc, m.zmax,
                  m.zmin, m.sdr, m.KS, m.lag, m.clamp)
println("columns: flat  >2900  max  min  sdr  KS  lag1(dQ E[0,6])  clamp")
n = round(Int, 20 / DT) + 1
nul = [metrics(qref[:, s:(s + n - 1)], dQref[:, s:(s + n - 2)]) for s in 1:(n - 1):(size(qref, 2) - n + 1)]
for (nm, f) in (("flat", m -> 100m.flat), (">2900", m -> 100m.exc), ("min", m -> m.zmin), ("sdr", m -> m.sdr),
                ("KS", m -> m.KS), ("lag1", m -> m.lag))
    v = f.(nul); @printf("  reference 20 TU %-6s min %.3g  max %.3g\n", nm, minimum(v), maximum(v))
end
for d in ARGS
    for x in sort(filter(x -> occursin(r"^data_online_tsim100\.0_replica\d\.jld2$", x), readdir(joinpath(TO, d))))
        o = jldopen(joinpath(TO, d, x), "r") do fh; fh["data_online"]; end
        N = size(o.dQ, 2)
        N < 40000 && (@printf("%-24s r%s  stopped at step %d (%.1f TU)\n", basename(d), x[end-5], N, N * DT))
        for c in 1:5
            s = (c - 1) * (n - 1) + 1
            s + n - 1 <= size(o.q, 2) || break
            dq = o.dQ[:, max(s, 101):(s + n - 2)]
            @printf("%-24s r%s  %3d-%3d TU  %s\n", basename(d), x[end-5], 20(c - 1), 20c, fmt(metrics(o.q[:, s:(s + n - 1)], dq)))
        end
        size(o.q, 2) >= 40001 &&
            @printf("%-24s r%s  whole 100 TU  KS %.2f  sd ratio %.2f  mean offset %s\n", basename(d), x[end-5],
                    sum(ks(o.q[k, :], qref[k, :]) for k in 1:6), std(o.q[5, :]) / sd100[5],
                    join((@sprintf("%5.2f", (mean(o.q[k, :]) - mean(qref[k, :])) / sd100[k]) for k in 1:6), ""))
    end
end
