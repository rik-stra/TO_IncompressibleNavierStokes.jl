# Per-QoI moments of 20 TU online M4 runs against the 100 TU reference (`results_LSTMS.md` §7d).
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/m4_online_moments.jl diag/<tag> explore/<tag> ...
#
# Arguments are fit directories under `output/TO_LSTM/`. Per replica: the mean offset in reference sd
# (null: the reference's own 20 TU windows, printed first), the sd ratio, and `sd(dQ)` against the
# reference correction's (warm-up excluded). A shifted mean with a normal sd ratio is a BIAS, which
# `m4_screen.jl`'s >2900 and KS columns show only indirectly.
using JLD2, Statistics, Printf
ref = load(get(ENV, "QOI_CACHE", joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")))
qr = ref["q"]
# `REF_TU_MAX` (2026-09-28): the reference only up to this time (plan step 1: 74, keeping the 76-97
# TU confirmation block out of every score); unset = the whole record
haskey(ENV, "REF_TU_MAX") && (qr = qr[:, 1:(round(Int, parse(Float64, ENV["REF_TU_MAX"]) / 2.5e-3) + 1)]; println("reference truncated to t <= $(ENV["REF_TU_MAX"]) TU"))
mu = vec(mean(qr; dims = 2)); sd = vec(std(qr; dims = 2)); dsd = vec(std(ref["dQ"][:, 1:(size(qr, 2) - 1)]; dims = 2))
# null: 20 TU windows of the reference itself
n = 8001
nul = [((vec(mean(qr[:, s:s+n-1]; dims = 2)) .- mu) ./ sd) for s in 1:(n-1):(size(qr, 2) - n + 1)]
@printf("%-34s %s\n", "reference 20 TU windows, dmean range", join((@sprintf("%5.2f..%5.2f ", minimum(getindex.(nul, k)), maximum(getindex.(nul, k))) for k in 1:6), ""))
TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
for d in ARGS
    for f in sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d\.jld2$", x), readdir(joinpath(TO, d))))
        o = load(joinpath(TO, d, f), "data_online")
        size(o.q, 2) < n && continue
        q = o.q[:, 1:n]; dQ = o.dQ[:, 101:(n - 1)]
        dm = (vec(mean(q; dims = 2)) .- mu) ./ sd
        sr = vec(std(q; dims = 2)) ./ sd
        dr = vec(std(dQ; dims = 2)) ./ dsd
        @printf("%-22s r%s  dmean %s | sdr %s | sd(dQ)/ref %s\n", basename(d), f[end-5], join((@sprintf("%6.2f", x) for x in dm), ""),
                join((@sprintf("%5.2f", x) for x in sr), ""), join((@sprintf("%5.2f", x) for x in dr), ""))
    end
end
