# Closed-loop calibration of a per-QoI correction offset against the TRAINING period's level
# (2026-09-27, Rik: "try all 3").
#
#     julia --project=lib/RikFlow/training m4_calibrate.jl <base fit dir> <J prefix> <delta> <out dir> [mu] [c0...]
#
# Target: the level's mean over 1-10 TU of the record (rows 400-4000) -- a training-only statistic.
# The error of a run is its 20 TU mean level minus that target, per QoI, in the record's sd.
# `<J prefix>`: runs `<J prefix><j>p` / `<J prefix><j>m` with offsets +-delta (dQ sd) in QoI j
# (replica 1, the baseline's seed); J[:, j] = (e(+) - e(-)) / (2 delta), and the symmetric part
# (e(+) + e(-) - 2 e(base)) / 2 is printed as a linearity check. Newton step, damped:
#     c = c0 - (J'J + mu I) \ (J' e(base))           (c in dQ sd; e(base) averaged over replicas)
# and the calibrated variant of the base fit is written to `<out dir>`.

using RikFlow, JLD2, Statistics, Printf, LinearAlgebra
const RF = RikFlow
base, jpre, delta, outdir = ARGS[1], ARGS[2], parse(Float64, ARGS[3]), ARGS[4]
mu_damp = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.0
TO = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
ref = load(normpath(joinpath(@__DIR__, "..", "..", "analysis", "data",
                             "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")))
qr = ref["q"]; sdq = vec(std(qr; dims = 2)); tgt = vec(mean(qr[:, 400:4000]; dims = 2))
f(v; d = 3) = join((@sprintf("%8.*f", d, x) for x in v), "")

function level_err(dir; reps = nothing)
    fs = sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d\.jld2$", x), readdir(joinpath(TO, dir))))
    reps === nothing || (fs = filter(x -> parse(Int, x[end-5]) in reps, fs))
    es = Vector{Vector{Float64}}()
    for x in fs
        o = jldopen(joinpath(TO, dir, x), "r") do fh; fh["data_online"]; end
        size(o.q, 2) < 8001 && (println("  $dir/$x: short run ($(size(o.q, 2)) cols) -- skipped"); continue)
        push!(es, (vec(mean(o.q[:, 1:8001]; dims = 2)) .- tgt) ./ sdq)
    end
    return es
end

eb_all = level_err(base)
eb = mean(eb_all)
eb1 = only(level_err(base; reps = (1,)))
@printf("base %s: level error vs 1-10 TU (q sd), mean of %d replicas %s ; replica 1 %s\n", base, length(eb_all), f(eb), f(eb1))
nq = length(eb)
J = zeros(nq, nq)
for j in 1:nq
    ep = level_err("$(jpre)$(j)p"); em = level_err("$(jpre)$(j)m")
    # a side that diverged gives no level; fall back to a one-sided difference against the base
    if !isempty(ep) && !isempty(em)
        J[:, j] = (only(ep) .- only(em)) ./ (2delta)
        @printf("  J[:, %d] %s   symmetric part / antisymmetric %.3f\n", j, f(J[:, j]),
                norm((only(ep) .+ only(em)) ./ 2 .- eb1) / max(norm((only(ep) .- only(em)) ./ 2), 1e-12))
    elseif !isempty(ep)
        J[:, j] = (only(ep) .- eb1) ./ delta
        @printf("  J[:, %d] %s   (one-sided, +delta; the -delta run diverged)\n", j, f(J[:, j]))
    elseif !isempty(em)
        J[:, j] = (eb1 .- only(em)) ./ delta
        @printf("  J[:, %d] %s   (one-sided, -delta; the +delta run diverged)\n", j, f(J[:, j]))
    else
        error("both offset runs for QoI $j diverged")
    end
end
@printf("cond(J) %.1f, singular values %s\n", cond(J), f(svdvals(J)))
c0 = length(ARGS) >= 6 ? parse.(Float64, ARGS[6:end]) : zeros(nq)
step = (J' * J + mu_damp * I) \ (J' * eb)
c = c0 .- step
@printf("Newton step (dQ sd) %s ; predicted residual %s\n", f(c), f(eb .- J * step))
cs = join((@sprintf("%.5f", x) for x in c), ",")
run(`julia --startup-file=no --project=$(normpath(joinpath(@__DIR__, "..", "..", "training"))) $(joinpath(@__DIR__, "m4_offset_variant.jl")) $base $outdir $cs`)
