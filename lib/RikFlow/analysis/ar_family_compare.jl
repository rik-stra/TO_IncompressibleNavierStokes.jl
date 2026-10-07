# Offline: which residual model for step 1's coloured residual? (2026-10-07, Rik asked; before round 1)
#
#     julia --startup-file=no --project=lib/RikFlow/analysis lib/RikFlow/analysis/ar_family_compare.jl
#
# The constrained AR(2) (φ1 ≥ 0, φ2 ≤ 0; results_LSTMS.md §16) collapses to an AR(1) in Z[0,6] and
# Z[16,32] of LinReg7 and is one-step under-dispersed there, because an AR(2) with two non-negative
# poles has a ROUNDED ACF at the origin (fast component of negative weight) and cannot make the
# residual's fast drop + slow tail (`meta_files/lit_ar2_restriction_2026-10-07.md`, fact ii). Candidates,
# all fitted the same way (Eq. ACF fit: least squares on the ACF at lags 1-20, 1-10 TU, marginal
# variance = the residual's):
#
#   AR2u   AR(2), stationary (the pilot fit, `ar2_ls`)
#   AR2c   AR(2), φ1 ≥ 0, φ2 ≤ 0 (the round-1 rule, `ar2_ls(...; nonneg = true)`)
#   A1W    AR(1) + independent white noise = ARMA(1,1):        ρ_k = a r^k (k ≥ 1)
#   A1A1   two independent AR(1)s, poles in [0, 1) = ARMA(2,1): ρ_k = a r1^k + (1 - a) r2^k
#
# A1W and A1A1 are sampled continuous-time processes (OU + white, OU + OU): poles e^{-Δt/τ} ≥ 0.
# Per model and QoI: ACF misfit (lags 1-20) on 1-10 TU and 10-100 TU; long-run variance ratio
# (LRV, lags ≤ 200); the power the measured solver kernel passes on (diagonal G, K = 200, relative to
# white); variance share above f = 1/4 and above f = 0.45 (near Nyquist); and one-step: the model's
# exact best linear predictor from its ACF (Durbin-Levinson, 50 lags) applied to the residual, giving
# the realized one-step error variance over the model's predictive variance (1 = calibrated) and the
# one-step Gaussian NLL gain over the white constant-variance draw, nats/step.
#
# Residuals: LinReg7 (deployed coefficients); paper 3's rule at base 0.03 and 0.3 and LinReg^E at
# λ = 1 and 10 (E[0,6] only), refitted as colour_tables.jl does (exact per-column ridge, LinReg1's
# scaling; equal to the deployed fits to ≤ 5e-11 where those are on the laptop).
include(joinpath(@__DIR__, "..", "src", "ts_history.jl"))
include(joinpath(@__DIR__, "..", "src", "ts_scaling.jl"))
include(joinpath(@__DIR__, "..", "src", "ts_score.jl"))     # autocorr
using JLD2, LinearAlgebra, Statistics, Printf

const DT = 2.5e-3
const NQ = 6
const LAB = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const OUTD = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const REC = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const L = 20          # fitted lags (Eq. ACF fit)
const K = 200         # LRV and kernel window
const MPRED = 50      # lags of the one-step predictor

# `ar2_acf`, `ar2_stationary`, `ar2_ls` from m0c_checks.jl's own source (it loads RikFlow at top level)
function _defname(e)
    e isa Expr || return nothing
    e.head === :macrocall && return _defname(e.args[end])
    e.head in (:function, :(=)) && e.args[1] isa Expr && e.args[1].head === :call && return e.args[1].args[1]
    return nothing
end
let src = joinpath(@__DIR__, "m0c_checks.jl"), want = Set([:ar2_acf, :ar2_stationary, :ar2_ls]), got = Set{Symbol}()
    for e in Meta.parseall(read(src, String); filename = src).args
        n = _defname(e)
        n in want || continue
        Core.eval(@__MODULE__, e)
        push!(got, n)
    end
    got == want || error("m0c_checks.jl no longer defines $(setdiff(want, got))")
end

# --- the two continuous-time families, fitted to the ACF at lags 1..L ------------------------------
acf_a1w(a, r, n) = [k == 0 ? 1.0 : a * r^k for k in 0:n]
acf_a1a1(a, r1, r2, n) = [a * r1^k + (1 - a) * r2^k for k in 0:n]
sse(m, racf) = sum(abs2, m[2:(L + 1)] .- racf[2:(L + 1)])

"Grid + coordinate refinement of `f(θ)` over a box, θ clamped to [lo, hi]."
function boxmin(f, grids, lo, hi)
    best = (Inf, first.(grids))
    for θ in Iterators.product(grids...)
        e = f(collect(θ))
        e < best[1] && (best = (e, collect(θ)))
    end
    θ = best[2]
    for s in (0.005, 0.001, 0.0002, 4e-5), _ in 1:6, j in eachindex(θ), d in (-2s, -s, s, 2s)
        t = copy(θ); t[j] = clamp(t[j] + d, lo[j], hi[j])
        e = f(t)
        e < best[1] && (best = (e, t); θ = t)
    end
    return best[2]
end
fit_a1w(racf) = boxmin(θ -> sse(acf_a1w(θ..., L), racf), (0:0.02:1, 0:0.01:0.99), [0, 0], [1, 0.9999])
fit_a1a1(racf) = boxmin(θ -> sse(acf_a1a1(θ..., L), racf), (0:0.025:1, 0:0.01:0.99, 0:0.02:0.98),
                        [0, 0, 0], [1, 0.9999, 0.9999])

# --- diagnostics -------------------------------------------------------------------------------------
"Durbin-Levinson: best linear one-step predictor coefficients from ρ[0..m] and its variance / σ²."
function levinson(rho, m)
    phi = zeros(m); v = 1.0
    for k in 1:m
        kk = (rho[k + 1] - sum(phi[j] * rho[k - j + 1] for j in 1:(k - 1); init = 0.0)) / v
        new = copy(phi)
        new[k] = kk
        for j in 1:(k - 1)
            new[j] = phi[j] - kk * phi[k - j]
        end
        phi = new
        v *= (1 - kk^2)
    end
    return phi, v
end

"One-step on residual `x` with model ACF ρ and marginal variance s2: (realized/predicted var, NLL gain vs white)."
function onestep(x, rho, s2)
    phi, v = levinson(rho, MPRED)
    m = length(phi)
    e = [x[t] - sum(phi[j] * x[t - j] for j in 1:m) for t in (m + 1):length(x)]
    xs = x[(m + 1):end]
    vp = s2 * v
    nll(r, var) = mean(0.5 .* (log(2pi) .+ log(var) .+ r .^ 2 ./ var))
    return mean(abs2, e) / vp, nll(xs, s2) - nll(e, vp)
end
lrv(rho) = 1 + 2 * sum(rho[2:(K + 1)])
kp(g, rho) = sum(g[a] * g[b] * rho[abs(a - b) + 1] for a in eachindex(g), b in eachindex(g))
const FR = range(0, 0.5; length = 2001)
function share(S, f0)
    return sum(S[FR .>= f0]) / sum(S)
end
spec_model(rho) = [1 + 2 * sum(rho[k + 1] * cos(2pi * f * k) for k in 1:(length(rho) - 1)) for f in FR]
function spec_data(x)
    y = x .- mean(x); n = length(y)
    return abs2.([sum(y .* cis.(-2pi * f .* (0:(n - 1)))) for f in FR])
end

# --- residuals ---------------------------------------------------------------------------------------
m1 = load(joinpath(OUTD, "TO_LRS", "LinReg1", "LinReg.jld2"))
const SC = m1["scaling"].in_scaling
const HIST = HistorySpec(; h = 5, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
const XF, YF, ST = build_history(HIST, scale_input(REC["q_star"], SC), scale_input(REC["q"], SC))
const TR = findall(n -> 400 <= n <= 4000, ST)
const HO = findall(n -> 10 < n * DT <= 100, ST)
function ridge(lams)
    X, Y = XF[TR, :], YF[TR, :]
    m = size(X, 2)
    C = zeros(m, NQ)
    for j in 1:NQ
        if lams[j] == 0
            C[:, j] = X \ Y[:, j]
        else
            P = Matrix{Float64}(I, m, m) * sqrt(lams[j]); P[m, m] = 0
            C[:, j] = [X; P] \ [Y[:, j]; zeros(m)]
        end
    end
    return C
end
sdc = vec(std(YF[TR, :] .- XF[TR, 1:NQ]; dims = 1)); mult = (sdc ./ sdc[1]) .^ 2
lamE(l) = (v = zeros(NQ); v[2] = l; v)
const CASES = [("LinReg7", fill(1.0, NQ), 1:NQ), ("paper 3, 0.03", 0.03 .* mult, 1:NQ),
               ("paper 3, 0.3", 0.3 .* mult, 1:NQ), ("LinReg^E, 1", lamE(1.0), 2:2), ("LinReg^E, 10", lamE(10.0), 2:2)]
const G = load(joinpath(OUTD, "TO_LSTM", "response", "response_kernel.jld2"), "G")

# --- the comparison ------------------------------------------------------------------------------------
const MODELS = ("AR2u", "AR2c", "A1W", "A1A1")
tot = Dict(m => zeros(2) for m in MODELS)           # summed one-step NLL gain [train, held] over binding QoIs
for (name, lams, qois) in CASES
    R = YF .- XF * ridge(lams)
    println("\n==== $name   (columns: SSE 1-20 train | held, LRV train/held data -> model, kernel power data -> model,",
            " share f>=1/4 | f>=0.45 data -> model, one-step var ratio train | held, NLL gain train | held)")
    for i in qois
        xt, xh = R[TR, i] .- mean(R[TR, i]), R[HO, i] .- mean(R[TR, i])
        s2 = var(xt)
        at, ah = autocorr(xt, K), autocorr(xh, K)
        g = G[i, i, :]
        pw(rho) = kp(g, rho) / kp(g, [1.0; zeros(K)])
        Sd = spec_data(xt)
        u = ar2_ls(at; L); c = ar2_ls(at; L, nonneg = true)
        fits = Dict("AR2u" => ar2_acf(u..., K), "AR2c" => ar2_acf(c..., K),
                    "A1W" => acf_a1w(fit_a1w(at)..., K), "A1A1" => acf_a1a1(fit_a1a1(at)..., K))
        bind = !(u[1] >= 0 && u[2] <= 0)
        @printf("  %-8s%s data: LRV %.1f/%.1f, kernel %.2f, share %.3f|%.4f\n", LAB[i], bind ? " (constraint binds)" : "",
                lrv(at), lrv(ah), pw(at), share(Sd, 0.25), share(Sd, 0.45))
        for mname in MODELS
            rho = fits[mname]
            rt, gt = onestep(xt, rho, s2); rh, gh = onestep(xh, rho, s2)
            Sm = spec_model(rho)
            bind && (tot[mname] .+= [gt, gh])
            @printf("    %-5s SSE %.4f | %.4f  LRV %5.1f  kernel %5.2f  share %.3f|%.4f  var ratio %.3f | %.3f  NLL gain %.4f | %.4f\n",
                    mname, sse(rho, at), sse(rho, ah), lrv(rho), pw(rho), share(Sm, 0.25), share(Sm, 0.45), rt, rh, gt, gh)
        end
    end
end
println("\nSummed one-step NLL gain over the QoIs where the AR(2) constraint binds, nats/step [train | held]:")
for m in MODELS
    @printf("  %-5s %.4f | %.4f\n", m, tot[m]...)
end
