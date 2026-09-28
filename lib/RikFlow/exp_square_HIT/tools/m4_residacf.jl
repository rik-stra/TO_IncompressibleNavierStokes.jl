# Per-QoI lag-1/2/5 ACF of the skip-only (M0) residual on the held-out 52-74 TU block, teacher-forced,
# for the M0@50 fits. usage: julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/tools/m4_residacf.jl <fit dirs under p4>
using RikFlow, Lux, Optimisers, Zygote, JLD2, Statistics, LinearAlgebra, Printf
const RF = RikFlow
R = normpath(joinpath(@__DIR__, ".."))
include(joinpath(R, "tools", "m4_data.jl"))
rec = load_m4_qois(; track_file = "none")
acf(x, k) = (y = x .- mean(x); sum(y[1:(end - k)] .* y[(k + 1):end]) / sum(abs2, y))
names = ("Z06", "E06", "Z715", "E715", "Z1632", "E1632")
for d in ARGS
    fit = RF.load_stochlstm(joinpath(R, "output", "TO_LSTM", "p4", d, "StochLSTM_seed1.jld2"))
    cfg, spec, Ws = fit.extras.cfg, fit.spec, Float64.(fit.weights.Ws)
    dat = m4_training_data(rec, cfg, spec.hist; target = :dQ)
    a, b = cfg.train_range[1], size(rec.q_star, 2)
    qs = RF.scale_input(rec.q[:, a:(b + 1)], dat.scaling.in_scaling)
    qss = RF.scale_input(rec.q_star[:, a:b], dat.scaling.in_scaling)
    Xf, _, st = RF.build_history(spec.hist, qss, qs); cols = (a - 1) .+ st
    H = findall(c -> 20800 <= c <= 29600, cols)
    all(diff(cols[H]) .== 1) || error("held-out block not contiguous")
    r = RF.scale_input(rec.dQ[:, cols[H]], dat.scaling.out_scaling) .- Ws * permutedims(Xf)[:, H]
    rs = vec(std(r; dims = 2))
    println(rpad(d, 22), join((@sprintf("%s %.2f/%.2f/%.2f (sd %.3f)  ", names[k], acf(r[k, :], 1), acf(r[k, :], 2), acf(r[k, :], 5), rs[k]) for k in 1:6), ""))
end
