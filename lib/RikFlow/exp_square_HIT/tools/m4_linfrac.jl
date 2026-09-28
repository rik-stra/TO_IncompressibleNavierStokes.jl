# How much of an M3f fit's network output g(x) = y - Ws x is LINEAR in x: regress g on x over the
# training rows, report held-out (52-74 TU) R^2 of that linear fit, and the held-out loss of
# skip + (linear part of g) vs skip + g.  usage: julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/tools/m4_linfrac.jl <fit dirs under p4>
using RikFlow, Lux, Optimisers, Zygote, JLD2, Statistics, LinearAlgebra, Printf
const RF = RikFlow
R = normpath(joinpath(@__DIR__, ".."))
include(joinpath(R, "tools", "m4_data.jl"))
rec = load_m4_qois(; track_file = "none")
for d in ARGS
    fit = RF.load_stochlstm(joinpath(R, "output", "TO_LSTM", "p4", d, "StochLSTM_seed1.jld2"))
    ps, cfg, spec = fit.extras.ps, fit.extras.cfg, fit.spec
    dat = m4_training_data(rec, cfg, spec.hist; target = :dQ)
    ntr = floor(Int, 0.8 * size(dat.Xc, 2))
    a, b = cfg.train_range[1], size(rec.q_star, 2)
    qs = RF.scale_input(rec.q[:, a:(b + 1)], dat.scaling.in_scaling)
    qss = RF.scale_input(rec.q_star[:, a:b], dat.scaling.in_scaling)
    Xf, _, st = RF.build_history(spec.hist, qss, qs); cols = (a - 1) .+ st
    Xf = permutedims(Xf); Yf = RF.scale_input(rec.dQ[:, cols], dat.scaling.out_scaling)
    H = findall(c -> 20800 <= c <= 29600, cols)
    net(X) = (o = RF.lstm_forward(spec, ps, Float32.(reshape(X, size(X, 1), 1, :)), zeros(Float32, 0, 1, size(X, 2)));
              Float64.(o.Y[:, 1, :]) .- Float64.(ps.Ws) * X)
    Xt = Float64.(dat.Xc[:, 1:ntr]); gt = net(Xt)
    Cg = Xt' \ gt'                                  # linear part of g, fitted on training rows
    Xh = Float64.(Xf[:, H]); gh = net(Xh); Yh = Yf[:, H]; sx = Float64.(ps.Ws) * Xh
    r2 = 1 - sum(abs2, gh .- Cg' * Xh) / sum(abs2, gh .- mean(gh; dims = 2))
    l(y) = 0.5 * sum(abs2, Yh .- y) / size(Yh, 2)
    @printf("%-22s linear share of g: held-out R2 %.3f | held loss: skip %.4f, skip+lin(g) %.4f, skip+g %.4f\n",
            d, r2, l(sx), l(sx .+ Cg' * Xh), l(sx .+ gh))
end
