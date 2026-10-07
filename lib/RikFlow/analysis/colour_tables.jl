# Paper Sec. 5 (colour.tex): the colour of the residual of every mean model in the two tables, the
# h = 7 one-step check, and LinReg7's residual size against LinReg1's (results_LSTMS.md §15).
#
#     julia --startup-file=no --project=lib/RikFlow lib/RikFlow/analysis/colour_tables.jl
#
# Every fit is the closures' own: h lags of q and q* (`:q_star_q`, predictor included), the level as
# target, QoIs centred and scaled on the training window (`_normalise`, `:normal`), exact ridge as the
# augmented least-squares solve with the bias unpenalized (`5_train_LinReg.jl`), one λ per QoI
# allowed (paper 3's rule, LinReg^E's splice). Training window 1-10 TU = steps 400-4000 of R1's
# tracked record. 🔑 Refits are checked against the deployed `TO_LRS/LinReg<n>` files present here.
#
# Per QoI: the lag-one autocorrelation ρ1 and the long-run variance ratio 1 + 2 Σ_{k=1}^{200} ρ_k of
# the residual (scaled units), on the training rows and, teacher-forced with the same C, on 10-100 TU
# (no fit uses it).
using RikFlow, JLD2, Statistics, LinearAlgebra, Printf
const RF = RikFlow
const DT = 2.5e-3
const NQ = 6
const LAB = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const A, B = 400, 4000                     # training window: steps 400-4000 (1-10 TU)
const KMAX = 200                           # lags in the long-run variance ratio (Sec. 4.3's window)

rec = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
q, qs = rec["q"], rec["q_star"]
_, insc = RF._normalise(q[:, A:(B - 1)]; normalization = :normal)
sc(x) = RF.scale_input(x, insc)

"Training design and target (rows = steps), exactly as `5_train_LinReg.jl` builds them."
function train_design(h)
    hist = RF.HistorySpec(; h, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    X, Y, _ = RF.build_history(hist, sc(qs[:, A:(B - 1)]), sc(q[:, A:B]))
    return X, Y
end
"Whole-record design; `steps` = record step n of each row (target q[:, n+1])."
function full_design(h)
    hist = RF.HistorySpec(; h, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    return RF.build_history(hist, sc(qs), sc(q))
end

"Exact ridge, bias unpenalized, one λ per output column (a scalar λ means the same for all)."
function ridge(X, Y, lam)
    m = size(X, 2)
    lams = lam isa Number ? fill(float(lam), size(Y, 2)) : collect(float.(lam))
    C = zeros(m, size(Y, 2))
    for j in axes(Y, 2)
        if lams[j] == 0
            C[:, j] = X \ Y[:, j]
        else
            P = Matrix{Float64}(I, m, m) * sqrt(lams[j]); P[m, m] = 0
            C[:, j] = [X; P] \ [Y[:, j]; zeros(m)]
        end
    end
    return C
end

acfv(x, L) = RF.autocorr(collect(x), L)          # lags 0:L
function colour(R)                               # R: rows = steps, cols = QoIs
    r1 = zeros(NQ); lrv = zeros(NQ)
    for i in 1:NQ
        a = acfv(view(R, :, i), KMAX)
        r1[i] = a[2]
        lrv[i] = 1 + 2 * sum(a[2:(KMAX + 1)])
    end
    return r1, lrv
end

"Residuals on the training rows and on 10-100 TU, for a mean with coefficients C at history h."
function residuals(C, h)
    Xt, Yt = train_design(h)
    Xf, Yf, st = full_design(h)
    held = findall(n -> 10 <= n * DT <= 100, st)
    return Yt .- Xt * C, Yf[held, :] .- Xf[held, :] * C
end

# paper 3's per-QoI rule: λ_i = λ (σ_i/σ_1)^2, σ_i the sd of the scaled correction (target minus the
# predictor row q*^n, the first NQ columns) on the training rows
function paper3_multipliers(h = 5)
    X, Y = train_design(h)
    s = vec(std(Y .- X[:, 1:NQ]; dims = 1))
    return (s ./ s[1]) .^ 2
end

rows = Pair{String,Any}[]
fmtp(v) = join((@sprintf("%5.2f / %5.2f", v[2i - 1], v[2i]) for i in 1:3), "  |  ")

# the correction itself: the prediction is q*^n, residual = scaled (q^{n+1} - q*^n)
let (Xt, Yt) = train_design(5), (Xf, Yf, st) = full_design(5)
    held = findall(n -> 10 <= n * DT <= 100, st)
    push!(rows, "dQ (prediction q*^n)" => (Yt .- Xt[:, 1:NQ], Yf[held, :] .- Xf[held, :][:, 1:NQ]))
end
fits = Dict{String,Any}()
for (lab, h, lam) in (("h = 1, λ = 0", 1, 0.0), ("h = 2, λ = 0", 2, 0.0), ("h = 5, λ = 0 (LinReg1)", 5, 0.0),
                      ("h = 7, λ = 0", 7, 0.0), ("h = 5, λ = 1e-4", 5, 1e-4), ("h = 5, λ = 1e-2", 5, 1e-2),
                      ("h = 5, λ = 0.1", 5, 0.1), ("h = 5, λ = 1 (LinReg7)", 5, 1.0), ("h = 5, λ = 10", 5, 10.0))
    X, Y = train_design(h)
    C = ridge(X, Y, lam)
    fits[lab] = C
    push!(rows, lab => residuals(C, h))
end
mult = paper3_multipliers()
for base in (0.03, 0.3)
    X, Y = train_design(5)
    C = ridge(X, Y, base .* mult)
    fits["paper 3, $base"] = C
    push!(rows, "paper 3's rule, λ = $base" => residuals(C, 5))
end
for lam in (0.1, 1.0, 10.0)
    X, Y = train_design(5)
    lams = zeros(NQ); lams[2] = lam                # E[0,6] only (LinReg^E)
    C = ridge(X, Y, lams)
    fits["LinReg^E, $lam"] = C
    push!(rows, "LinReg^E, λ = $lam on E[0,6]" => residuals(C, 5))
end

# 🔑 check the refits against the deployed models present on this machine
println("Refit vs deployed (max relative coefficient difference; training SSE relative difference):")
for (lab, n) in (("h = 5, λ = 0 (LinReg1)", "LinReg1"), ("h = 5, λ = 1e-4", "LinReg6"),
                 ("h = 5, λ = 1e-2", "LinReg2"), ("h = 5, λ = 1 (LinReg7)", "LinReg7"), ("h = 5, λ = 10", "LinReg8"))
    f = joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LRS", n, "LinReg.jld2")
    isfile(f) || (println("  $n: not here"); continue)
    m = load(f)
    Cd = permutedims(Matrix{Float64}(m["c"]))
    sd = m["scaling"].in_scaling
    @assert vec(sd.mu) ≈ vec(insc.mu) && vec(sd.sigma) ≈ vec(insc.sigma) "$n: scaling differs"
    X, Y = train_design(5)
    rel = maximum(abs, fits[lab] .- Cd) / maximum(abs, Cd)
    sse = (sum(abs2, Y .- X * Cd) - sum(abs2, Y .- X * fits[lab])) / sum(abs2, Y .- X * fits[lab])
    @printf("  %-8s %-26s coef %.2e   SSE %+.2e\n", n, lab, rel, sse)
end
@printf("paper 3 multipliers (σ_i/σ_1)^2: %s   (desktop's stored: 1 / 2.060240 / 0.233755 / 0.358555 / 8.992533 / 4.557659)\n",
        join((@sprintf("%.6f", m) for m in mult), " / "))

println("\nColour of the residual. Per band 'Z / E'.  rho1: training (1-10 TU) || 10-100 TU.  LRV = 1 + 2 sum_{k<=200} rho_k")
for (lab, (Rt, Rh)) in rows
    r1t, lt = colour(Rt)
    r1h, lh = colour(Rh)
    @printf("%-30s rho1  %s  ||  %s\n", lab, fmtp(r1t), fmtp(r1h))
    @printf("%-30s LRV   %s  ||  %s\n", "", fmtp(lt), fmtp(lh))
end

# LinReg7's residual size against LinReg1's, training rows
let R1 = residuals(fits["h = 5, λ = 0 (LinReg1)"], 5)[1], R7 = residuals(fits["h = 5, λ = 1 (LinReg7)"], 5)[1]
    @printf("\nresidual sd, LinReg7 / LinReg1 (training): %s\n",
            join((@sprintf("%s %.2f", LAB[i], std(R7[:, i]) / std(R1[:, i])) for i in 1:NQ), "  "))
end

# h = 7 against h = 5: fits on the first 80 % of the training rows, one-step error on the rest
# (8.2-10 TU), RMSE of the correction in units of sd(dQ) on the training window
println("\nOne-step error on 8.2-10 TU (fits on 1-8.2 TU), RMSE / sd(dQ):")
sdq = vec(std(q[:, (A + 1):B] .- qs[:, A:(B - 1)]; dims = 2)) ./ vec(insc.sigma)   # scaled units
for h in (5, 7)
    X, Y = train_design(h)
    n = floor(Int, 0.8 * size(X, 1))
    C = ridge(X[1:n, :], Y[1:n, :], 0.0)
    E = Y[(n + 1):end, :] .- X[(n + 1):end, :] * C
    @printf("  h = %d: %s   (rows %d-%d of %d)\n", h,
            join((@sprintf("%s %.4f", LAB[i], sqrt(mean(abs2, E[:, i])) / sdq[i]) for i in 1:NQ), "  "),
            n + 1, size(X, 1), size(X, 1))
end
