# R2 offline (results_LSTMS.md §14): is the deployed LinReg1 the exact Float64 fit (R2-1), and how
# much do the fit window (1-8.2 TU, as m4_linear_eta.jl fitted, vs 1-10 TU) and Float32 evaluation
# (as the network path deploys) move the ONE-STEP correction? Teacher-forced on the tracked record,
# differences in sd(dQ) on 1-10 TU. One-step only: it cannot predict a closed-loop bias.
#
#     julia --startup-file=no --project=lib/RikFlow lib/RikFlow/analysis/r2_offline.jl
using RikFlow, JLD2, Statistics, LinearAlgebra, Printf
const RF = RikFlow
root = normpath(joinpath(@__DIR__, ".."))
rec = load(joinpath(root, "analysis/data/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
q, qs, dQ = rec["q"], rec["q_star"], rec["dQ"]
lrs = load(joinpath(root, "exp_square_HIT/output/TO_LRS/LinReg1/LinReg.jld2"))
C0 = Matrix{Float64}(lrs["c"]); C = size(C0, 1) == 6 ? permutedims(C0) : C0; sc = lrs["scaling"].in_scaling
mu, sg = vec(sc.mu), vec(sc.sigma)
hist = RF.HistorySpec(; h = 5, n_qoi = 6, hist_var = :q_star_q, include_predictor = true)
# whole record in LinReg1's scaling; row r <-> record step n = steps[r]; target = q[:, n+1]
X, Y, steps = RF.build_history(hist, RF.scale_input(qs, sc), RF.scale_input(q, sc))
@assert size(X, 2) == size(C, 1)
DT = 2.5e-3
# training rows exactly as 5_train_LinReg.jl: data[:, 400:3999] -> steps of that sub-record
a, b = 400, 4000
Xt, Yt, st = RF.build_history(hist, RF.scale_input(qs[:, a:(b - 1)], sc), RF.scale_input(q[:, a:b], sc))
C10 = Xt \ Yt
ntr = floor(Int, 0.8 * size(Xt, 1))
C82 = Xt[1:ntr, :] \ Yt[1:ntr, :]
@printf("rows %d (1-10 TU), 80%% = %d rows, last train step at %.2f TU\n", size(Xt, 1), ntr, (a - 1 + st[ntr]) * DT)
@printf("cond(Xt) %.3e\n", cond(Xt))
# R2-1: deployed vs Float64 refit
r_dep = Yt .- Xt * C; r_10 = Yt .- Xt * C10
@printf("R2-1  coef max rel |C - C10|/max|C| %.2e | train SSE deployed - refit %.2e (rel)\n",
        maximum(abs, C .- C10) / maximum(abs, C), (sum(abs2, r_dep) - sum(abs2, r_10)) / sum(abs2, r_10))
# one-step predicted correction dQhat = qhat - q*, raw units, on a window of record steps
sdd = vec(std(dQ[:, 400:4000]; dims = 2))
pred(Cm, Xm) = (Xm * Cm)' .* sg .+ mu                      # raw q^{n+1}, 6 x N
dqhat(Cm, Xm, stp) = pred(Cm, Xm) .- qs[:, stp]
names = ["Z06", "E06", "Z715", "E715", "Z1632", "E1632"]
f(x) = join((@sprintf("%9.2e", v) for v in x), "")
function compare(lab, d1, d2, rows)
    D = (d1[:, rows] .- d2[:, rows]) ./ sdd
    @printf("  %-34s mean %s\n  %-34s rms  %s\n", lab, f(vec(mean(D; dims = 2))), "", f(vec(sqrt.(mean(abs2, D; dims = 2)))))
end
println("QoIs: ", join(names, "  "), "   (differences in sd(dQ) on 1-10 TU)")
for (wl, lo, hi) in [("1-10 TU", 1.0, 10.0), ("10-100 TU", 10.0, 100.0)]
    rows = findall(n -> lo <= n * DT <= hi, steps)
    stp = steps[rows]
    Xw = X[rows, :]
    d_dep = dqhat(C, Xw, stp); d_82 = dqhat(C82, Xw, stp)
    println("window $wl ($(length(rows)) steps)")
    compare("fit 1-8.2 TU minus deployed (F64)", d_82, d_dep, axes(d_dep, 2))
    # Float32, network-path style: x cast to F32, coefficients in F32, product in F32, back up
    d_32 = (Float64.(Float32.(C)' * Float32.(Xw')) .* sg .+ mu) .- qs[:, stp]
    compare("deployed F32 level-target minus F64", d_32, d_dep, axes(d_dep, 2))
    # dQ target as the network path: y = (dQ - mo)/so = Cd' x; Cd = translation of C (exact affine)
    #   dQhat = (C'x) sg + mu - q*,  q* = x[1:6] sg + mu  =>  dQhat = sg .* (C'x - x[1:6])
    E = zeros(size(C)); E[1:6, 1:6] .= I(6)
    mo = vec(mean(dQ[:, 400:4000]; dims = 2)); so = sdd
    Cd = (C .- E) .* (sg ./ so)' ; Cd[end, :] .-= mo ./ so      # y = Cd' x (bias row last)
    d_64d = (Cd' * Xw') .* so .+ mo
    d_32d = Float64.(Float32.(Cd)' * Float32.(Xw')) .* so .+ mo
    compare("dQ-param F64 minus deployed (check)", d_64d, d_dep, axes(d_dep, 2))
    compare("dQ-param F32 minus F64", d_32d, d_64d, axes(d_dep, 2))
    d_32c = Float64.(Float32.(Cd))' * Xw' .* so .+ mo
    compare("  of which coef rounding only", d_32c, d_64d, axes(d_dep, 2))
end
@printf("max|C| deployed %.1f, C82 %.1f; |C82 - C| / |C| (Frobenius) %.3f\n", maximum(abs, C), maximum(abs, C82), norm(C82 .- C) / norm(C))
