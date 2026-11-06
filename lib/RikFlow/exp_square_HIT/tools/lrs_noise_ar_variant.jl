# q1 / q2 builder (card `analysis/cards/q1q2.md`; Rik, 2026-11-06: "yes add q1 and q2 as todo", "start working on the
# coloured noise training"): the unregularized baseline's mean trained on histories that carry its OWN AR(2) draws and
# deployed with that AR(2) (q1), and the same with step 2's power-law scale on the draws, in training and in
# deployment (q2).
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/lrs_noise_ar_variant.jl [src] [--pl] [--report]
#     julia --startup-file=no --project=training exp_square_HIT/tools/lrs_noise_ar_variant.jl [src] --check
#
# default `LinReg1`: writes `output/TO_LRS/LinReg1_ntar2` (q1); with `--pl` `LinReg1_ntar2_pl` (q2). `--report` refits,
# checks the file holds exactly that fit, re-prints, and writes nothing. `--check` writes nothing: the white limit
# (φ ≡ 0) must reproduce `<src>_nt` (`lrs_noise_variant.jl`) and the AR fitted after it `<src>_nt_ar2nw`
# (`lrs_ar_variant.jl ... 2nw`), and q2's machinery with β held at 0 must reproduce q1. An existing target is refused.
#
# The model (scaled units: in/out_scaling of the source; card "Model"):
#
#   deployed (the LinReg path as it is, no run-path change): q^n = scale_out(c [x^n; 1] + η^n),
#     q1: η^n = μ + z^n,            q2: η^n = μ + f(q*^n) ⊙ z^n,   f_i = (clamp(q*_i) / q_ref,i)^β_i,
#     z^n = Φ1 z^{n-1} + Φ2 z^{n-2} + ξ^n, ξ ~ N(0, Σ_ξ), Φ diagonal, φ1 ≥ 0, φ2 ≤ 0 (the rule of 2026-10-07).
#
#   training: the window noise of Eq. `eq: window noise` (anchored at the prediction step, `tools/step3_innoise.jl`),
#     dx^t = M w^t, w^t = [η^{t-1}; …; η^{t-h}] (raw), its draws a path of the deployed noise. Least squares that
#     expects it is a ridge whose penalty is the noise:
#       c' = (X'X + Σ_t M diag(g_t) W_0 diag(g_t) M') \ X'Y = (X'X + M (W_0 ⊙ Σ_t g_t g_t') M') \ X'Y,
#     W_k = E[ẇ^t ẇ^{t-k}'] the block-Toeplitz covariance of the UNIT-scale window draws, block (j, j') = Γ^z_{k+j'-j},
#     Γ^z the deployed AR's lagged covariances (`ar_lagcov`: exact, discrete Lyapunov + recursion), and g_t the raw
#     scale of each draw: osg (q1), osg ⊙ f(q*^{t-j}) at the recorded predictor of the draw's own step (q2).
#     With Φ = 0 and g constant this is `lrs_noise_variant.jl`'s c' = (X'X + N M (I_h ⊗ DΣD) M') \ X'Y.
#
#   the residual given the noisy history, e^t = r^t − C'M diag(g_t) ẇ^t (r the clean residual): exact lagged
#     covariances (`resid_lagcov`), the noise independent of the record. With white draws, k < h only: `fit_ar_noisy`.
#
#   the noise model (as m2, `fit_ar_noisy`): Σ = Γ^e_0 (stored in `stoch_distr`, used for μ only on the AR path),
#     μ = mean(r); φ per QoI by `ar2_ls` (lags 1-20, restricted) on the ACF of e (q1) or of e/f (q2); Σ_ξ = D R D,
#     R the correlation of the exact innovation covariance, D matching the AR's marginal variance to diag Γ_0.
#     q2: β and the variance at q_ref by maximum likelihood (`RF.fit_powerlaw_scale`) on the EXPECTED squared residual
#     (r^t_i)² + (A_t W_0 A_t')_ii, A_t = C'M diag(g_t) (the Gaussian NLL is linear in e², so this is the exact
#     expected NLL); q_ref and the clip range from q* on the fit rows (the clip never binds there); Σ_ε = Γ^{e/f}_0
#     (diag = exp(logvar), asserted; not drawn on the AR path).
#
#   the fixed point over (c, Σ, φ, Σ_ξ[, β]), from m2's AR (`<src>_nt_ar2nw`) and, for q2, q1's AR and `<src>_pl`'s β:
#     stop when φ repeats exactly (`ar2_ls` is a grid + refinement, ~2e-4 lattice) and Σ_ξ (and β) change < 1e-12.
#
# Fit rows: the source's own, found by reproducing its c with no noise (asserted). 🔒 Nothing past 10 TU.
# Not modelled (card): the mean's own errors feeding back; online, the AR draw η^n is correlated with the draws in its
# window, which the training sees only through the residual's ACF.

include(joinpath(@__DIR__, "lrs_scale_variant.jl"))   # lrs_ar_variant.jl (+ m0c_ridge.jl): design, guard, ar2_ls, ...
using Dates

# ---------------------------------------------------------------------------------------------------------------
# The pure parts (test/test_noise_ar.jl, V97, evaluates them from this file's text)
# ---------------------------------------------------------------------------------------------------------------

"""
    ar_lagcov(phi, S, L) -> [Γ_0, …, Γ_L]

Lagged covariances Γ_k = E[z_t z_{t-k}'] of the stationary diagonal AR(p ≤ 2) z_t = Φ1 z_{t-1} + Φ2 z_{t-2} + ξ_t,
ξ ~ N(0, S), `phi` p × n (one column per component). Exact: the stationary covariance P of the companion form
s_t = [z_t; z_{t-1}] solves P = A P A' + Q (a discrete Lyapunov equation, solved as a linear system), Γ_0 and Γ_1 are
its blocks, and Γ_k = Φ1 Γ_{k-1} + Φ2 Γ_{k-2} for k ≥ 2.
"""
function ar_lagcov(phi::AbstractMatrix, S::AbstractMatrix, L::Integer)
    p, n = size(phi)
    1 <= p <= 2 || error("ar_lagcov: order $p")
    F1 = Matrix(Diagonal(Vector{Float64}(phi[1, :])))
    F2 = Matrix(Diagonal(p == 2 ? Vector{Float64}(phi[2, :]) : zeros(n)))
    A = [F1 F2; Matrix(1.0I, n, n) zeros(n, n)]
    Q = zeros(2n, 2n)
    Q[1:n, 1:n] .= S
    P = reshape((I - kron(A, A)) \ vec(Q), 2n, 2n)          # vec(A P A') = (A ⊗ A) vec(P)
    P = (P .+ P') ./ 2
    G = Vector{Matrix{Float64}}(undef, L + 1)
    G[1] = P[1:n, 1:n]
    L >= 1 && (G[2] = P[1:n, (n + 1):(2n)])                 # E[z_t z_{t-1}']
    for k in 2:L
        G[k + 1] = F1 * G[k] .+ F2 * G[k - 1]
    end
    return G
end

"E[z_s z_u'] at the lag k = s − u, from `G` = [Γ_0, …] (k < 0: the transpose of Γ_{-k})."
lagcov(G, k) = k >= 0 ? G[k + 1] : permutedims(G[1 - k])

"""
    window_lagcov(G, h, k) -> W_k

E[w_t w_{t-k}'] for the window w_t = [z^{t-1}; …; z^{t-h}] (block j = the draw j steps back) of a process with lagged
covariances `G` (lags up to k + h − 1 needed): block (j, j') = Γ_{k+j'-j}. For white draws (Γ_m = 0 for m ≠ 0) it is
`fit_ar_noisy`'s S_k ⊗ Γ_0, S_k[j, j'] = 1 iff j − j' = k.
"""
function window_lagcov(G, h, k)
    n = size(G[1], 1)
    W = zeros(n * h, n * h)
    for j in 1:h, jp in 1:h
        W[((j - 1) * n + 1):(j * n), ((jp - 1) * n + 1):(jp * n)] .= lagcov(G, k + jp - j)
    end
    return W
end

"""
    ar_from_lagcov(G; L = 20, nonneg = true, phi = nothing) -> (; phi, p1, p2, S, R, s2, sxi2, racf)

The restricted AR(2) of a residual with lagged covariances `G` = [Γ_0, …, Γ_L], fitted as `fit_ar_noisy` does after it
has its Γ_k: per QoI φ by `ar2_ls` on ρ_k = Γ_k[i,i] / Γ_0[i,i], k = 1..L; Σ_ξ = D R D, R the correlation of the exact
innovation covariance E[ξ ξ'] (ξ_t = e_t − Φ1 e_{t-1} − Φ2 e_{t-2}), D so that the AR's marginal variance is diag Γ_0.
A given `phi` (2 × n) is used instead of `ar2_ls`'s (`fit_noise_ar`'s tie-break of a φ cycle).
"""
function ar_from_lagcov(G; L = 20, nonneg = true, phi = nothing)
    n = size(G[1], 1)
    racf = [G[k + 1][i, i] / G[1][i, i] for k in 0:L, i in 1:n]
    if phi === nothing
        phi = zeros(2, n)
        for i in 1:n
            phi[:, i] .= ar2_ls(racf[:, i]; L, nonneg)
        end
    else
        phi = Matrix{Float64}(phi)
    end
    p1, p2 = phi[1, :], phi[2, :]
    F1, F2 = Diagonal(p1), Diagonal(p2)
    G0, G1, G2 = G[1], G[2], G[3]
    Cxi = G0 .+ F1 * G0 * F1 .+ F2 * G0 * F2 .- G1 * F1 .- F1 * G1' .- G2 * F2 .- F2 * G2' .+ F1 * G1 * F2 .+
          F2 * G1' * F1
    sd = sqrt.(diag(Cxi))
    R = Matrix(Symmetric(Cxi ./ (sd * sd')))
    s2 = diag(G0)
    sxi2 = [ar_innovation_var(s2[i], p1[i], p2[i]) for i in 1:n]
    D = Diagonal(sqrt.(sxi2))
    S = Matrix(Symmetric(D * R * D))
    return (; phi, p1, p2, S, R, s2, sxi2, racf)
end

"""
    noise_penalty(M, W0, g, N)

The training penalty Σ_t M diag(g_t) W_0 diag(g_t) M' = M (W_0 ⊙ Σ_t g_t g_t') M': `g` a vector (the same raw scale on
every row, N rows) or an N × nh matrix (one row per training row).
"""
noise_penalty(M, W0, g::AbstractVector, N) = N .* (M * (W0 .* (g * g')) * M')
noise_penalty(M, W0, g::AbstractMatrix, N) = (size(g, 1) == N || error("g has $(size(g, 1)) rows, not $N");
                                               M * (W0 .* (g' * g)) * M')

"diag(A_t W_0 A_t') per row (N × n), A_t = AM diag(g_t): the variance of the mean's response to the window's draws."
function noise_var_rows(AM, W0, g::AbstractMatrix)
    N, n = size(g, 1), size(AM, 1)
    v = zeros(N, n)
    for t in 1:N
        A = AM .* view(g, t, :)'
        v[t, :] .= vec(sum((A * W0) .* A; dims = 2))
    end
    return v
end
noise_var_rows(AM, W0, g::AbstractVector) = (A = AM .* g'; vec(sum((A * W0) .* A; dims = 2))')

"""
    resid_lagcov(Z, AM, Wz, g, f; L = 20) -> (Ge0, Gef)

Exact lagged covariances of the residual given the noisy history, e_t = z_t − AM diag(g_t) ẇ_t, rows t = 1..N
consecutive steps: `Z` (N × n) the clean residual minus μ, `AM` = C'M, `Wz[k+1]` = E[ẇ_t ẇ_{t-k}'] of the unit-scale
window draws. `g` a vector and `f === nothing` (q1: one scale) or `g` N × nh and `f` N × n (q2: each row's window
scales and its own scale). Returns Γ^e_0 and [Γ^{e/f}_0, …, Γ^{e/f}_L] (each entry of e_t divided by f_t; = Γ^e_k for
q1). The data part is (1/N) Σ_{t>k} (the biased estimate of `acf` and `fit_ar_noisy`), the noise part its expectation
per pair, i.e. the mean over the N − k pairs (`fit_ar_noisy`'s C'M (S_k ⊗ B) M'C for a constant scale).
"""
function resid_lagcov(Z, AM, Wz, g::AbstractVector, ::Nothing; L = 20)
    N = size(Z, 1)
    A = AM .* g'
    G = [Z[(k + 1):end, :]' * Z[1:(end - k), :] ./ N .+ A * Wz[k + 1] * A' for k in 0:L]
    return G[1], G
end
function resid_lagcov(Z, AM, Wz, g::AbstractMatrix, f::AbstractMatrix; L = 20)
    N, n = size(Z)
    size(g, 1) == N && size(f) == (N, n) || error("resid_lagcov: g, f do not match Z's rows")
    A = [AM .* view(g, t, :)' for t in 1:N]
    Zf = Z ./ f
    B, T = zeros(n, size(AM, 2)), zeros(n, n)
    G = map(0:L) do k
        acc = zeros(n, n)
        for t in (k + 1):N
            mul!(B, A[t], Wz[k + 1])
            mul!(T, B, A[t - k]')
            acc .+= T ./ (view(f, t, :) * view(f, t - k, :)')
        end
        Zf[(k + 1):end, :]' * Zf[1:(end - k), :] ./ N .+ acc ./ (N - k)
    end
    Ge0 = Z' * Z ./ N .+ sum(A[t] * Wz[1] * A[t]' for t in 1:N) ./ N
    return Ge0, G
end

"""
    scale_rows(qs, qwin, beta, qref, qclip, osg, h) -> (f, g)

Each row's own scale f_t = f(q*^t) (N × n, `qs` the rows' predictors) and the raw scales of its window's draws,
g_t = [osg ⊙ f(q*^{t-1}); …; osg ⊙ f(q*^{t-h})] (N × n h, `qwin` = [q*^{t-1} … q*^{t-h}] per row), with the deployed
factor `RF.powerlaw_factor`.
"""
function scale_rows(qs, qwin, beta, qref, qclip, osg, h)
    N, n = size(qs)
    f = permutedims(reduce(hcat, [RF.powerlaw_factor(view(qs, t, :), beta, qref, qclip) for t in 1:N]))
    g = zeros(N, n * h)
    for t in 1:N, j in 1:h
        c = ((j - 1) * n + 1):(j * n)
        g[t, c] .= osg .* RF.powerlaw_factor(view(qwin, t, c), beta, qref, qclip)
    end
    return f, g
end

"The long-run variance ratio 1 + 2 Σ_{k=1}^{K} ρ(k) of an AR(2) (App. `app:ar-fit`'s measure; K = 200 = 0.5 TU)."
lrv_ratio(p1, p2; K = 200) = 1 + 2 * sum(ar2_acf(p1, p2, K)[2:end])

"""
    fit_noise_ar(d, M, h; phi0, S0, scale = nothing, white = false, L = 20, tol = 1e-8, maxit = 500)

The fixed point of the header's model on the rows of `d` (`Xs`, `Ys`, `osg`: the scaled design of the source on its
fit rows). `phi0` (2 × n), `S0`: the AR to start from. `scale = nothing` is q1; for q2 a NamedTuple
`(; qs, qwin, beta0, fixbeta = false)` (`scale_rows`' predictors; `fixbeta` holds β at `beta0`, the check that q2 at
β = 0 is q1). `white = true` holds φ at 0: then Σ_ξ = Σ and this is `lrs_noise_variant.jl`'s `fit_noise` (which stops
at 1e-12; pass `tol = 1e-12` to reproduce it). Stops when φ repeats and Σ_ξ (and β) change by < `tol`.
🔑 `tol` is 1e-8, not 1e-12: with the AR the map has a rounding floor of ~2e-9 in Σ_ξ (measured 2026-11-06 on
LinReg1, deterministic, contraction ~0.7 above it). The normal equations solve for c at cond(X'X + pen) ~1e12 (the q*
block, gotcha #13): rounding moves c by ~3e-7 along directions that leave the fit-row residual unchanged (Γ_0 moves by
~1e-15) but reach the lag-k response C'M W_k M'C, hence Σ_ξ. The solve is kept as `fit_noise`'s so the white limit
reproduces `<src>_nt` bit for bit. Returns the converged state and every derived quantity.
🔑 **A φ cycle** (2026-11-06, q2 on the desktop: the continuous part converged to 3e-8 while φ alternated between two
neighbouring `ar2_ls` lattice points; no exact fixed point exists). Tie-break, deterministic and with `ar2_ls` unchanged:
each φ of the cycle is held fixed and (Σ_ξ, β, c, Σ) iterated to convergence; the one kept is the candidate closest to
self-consistent, i.e. with the smallest excess of the ACF-fit objective (Σ_k (ρ_AR(k) − ρ_e(k))², k = 1..L, summed over
QoIs) over that of `ar2_ls`'s own choice for the candidate's residual. The result carries `cycle` (the candidates) and
`cycle_gaps`; `fixed[2]` is then false (`ar2_ls` would pick the other lattice point).
"""
function fit_noise_ar(d, M, h; phi0, S0, scale = nothing, white = false, L = 20, tol = 1e-8, maxit = 500)
    N, n = size(d.Ys)
    osg = d.osg
    XtX, XtY = d.Xs' * d.Xs, d.Xs' * d.Ys
    sc = scale
    qref = sc === nothing ? nothing : [exp(sum(log, sc.qs[:, i]) / N) for i in 1:n]
    qclip = sc === nothing ? nothing : hcat(vec(minimum(sc.qs; dims = 1)), vec(maximum(sc.qs; dims = 1)))
    fixbeta = sc !== nothing && get(sc, :fixbeta, false)
    # one application of the map (φ, Σ_ξ, β) -> (φ', Σ_ξ', β'), with everything it computes; `phifix` holds φ
    function fpmap(phi, S, beta; phifix = nothing)
        Gz = ar_lagcov(phi, S, L + h)
        Wz = [window_lagcov(Gz, h, k) for k in 0:L]
        f, g = sc === nothing ? (nothing, repeat(osg, h)) : scale_rows(sc.qs, sc.qwin, beta, qref, qclip, osg, h)
        pen = noise_penalty(M, Wz[1], g, N)
        Ct = (XtX .+ pen) \ XtY
        r = d.Ys .- d.Xs * Ct
        mu = vec(mean(r; dims = 1))
        Z = r .- mu'
        AM = Ct' * M
        v = noise_var_rows(AM, Wz[1], g)
        if sc === nothing
            sfit, beta2, logvar, f2 = nothing, nothing, nothing, nothing
            Ge0, Gef = resid_lagcov(Z, AM, Wz, g, nothing; L)
        else
            sfit = [RF.fit_powerlaw_scale(sqrt.(Z[:, i] .^ 2 .+ v[:, i]), sc.qs[:, i]) for i in 1:n]
            maximum(abs, [s.qref for s in sfit] ./ qref .- 1) < 1e-12 || error("q_ref differs from the fit's")
            beta2 = fixbeta ? copy(beta) : [s.beta for s in sfit]
            logvar = [s.logvar for s in sfit]
            f2, _ = scale_rows(sc.qs, sc.qwin, beta2, qref, qclip, osg, h)
            Ge0, Gef = resid_lagcov(Z, AM, Wz, g, f2; L)
        end
        afree = ar_from_lagcov(Gef; L)                                    # ar2_ls's choice
        a = phifix === nothing ? afree : ar_from_lagcov(Gef; L, phi = phifix)   # the AR deployed
        phi2 = white ? zeros(2, n) : a.phi
        S2 = white ? Matrix(Symmetric(Gef[1])) : a.S
        return (; phi2, S2, beta2, Ct, mu, Z, AM, v, Gz, Wz, f, g, pen, Ge0, Gef, a, afree, sfit, logvar, f2)
    end
    function solve(phi, S, beta; phifix = nothing)
        phi = phifix === nothing ? Matrix{Float64}(phi) : Matrix{Float64}(phifix)
        seen = Matrix{Float64}[]
        steps = Float64[]
        for it in 1:maxit
            s = fpmap(phi, S, beta; phifix)
            dS = maximum(abs, s.S2 .- S) / maximum(diag(S))
            dB = sc === nothing ? 0.0 : maximum(abs, s.beta2 .- beta)
            push!(steps, max(dS, dB))
            same_phi = s.phi2 == phi
            k = findlast(==(s.phi2), seen)
            if !same_phi && k !== nothing && length(steps) > 20
                return (; cycle = [seen[k:end]; [copy(phi)]], S = s.S2, beta = s.beta2, iterations = it, dS)
            end
            push!(seen, copy(phi))
            phi, S, beta = s.phi2, s.S2, s.beta2
            if same_phi && last(steps) < tol
                fin = fpmap(phi, S, beta; phifix)        # the state's own quantities (c, Σ, Σ_ξ of this φ, S, β)
                return (; fin..., phi, S, beta, qref, qclip, iterations = it, steps,
                        Sig = Matrix(Symmetric(fin.Ge0)), Sdx = fin.pen ./ N,
                        Seps = fin.f2 === nothing ? nothing : Matrix(Symmetric(fin.Gef[1])),
                        fixed = (maximum(abs, fin.S2 .- S) / maximum(diag(S)), fin.afree.phi == phi || white),
                        cycle = nothing, cycle_gaps = nothing)
            end
        end
        error("the fixed point did not converge in $maxit iterations (last change $(last(steps)))")
    end
    beta = sc === nothing ? nothing : Vector{Float64}(sc.beta0)
    # (not `r`: the closure fpmap assigns its own `r`, which would capture and overwrite an outer one)
    res0 = solve(white ? zeros(2, n) : phi0, Matrix{Float64}(S0), beta)
    res0.cycle === nothing && return res0
    # the tie-break (docstring): each candidate held fixed, the one closest to self-consistent kept
    obj(p, racf) = sum(sum(abs2, ar2_acf(p[1, i], p[2, i], L)[2:end] .- racf[2:(L + 1), i]) for i in 1:n)
    cands = [solve(c, res0.S, res0.beta; phifix = c) for c in res0.cycle]
    any(x -> x.cycle !== nothing, cands) && error("a φ held fixed still cycles: impossible, check fpmap")
    gaps = [obj(x.phi, x.a.racf) - obj(x.afree.phi, x.a.racf) for x in cands]
    best = cands[argmin(gaps)]
    return merge(best, (; cycle = res0.cycle, cycle_gaps = gaps, iterations = res0.iterations + best.iterations))
end

# ---------------------------------------------------------------------------------------------------------------
# Data and reports
# ---------------------------------------------------------------------------------------------------------------

"The scaled design of a deployed LinReg `m` on steps `r` (rows = steps, bias last), its scales, and the predictors."
function scaled_design(m, r)
    X, Y = design(m["hist_len"], guard(r))
    si, so = m["scaling"].in_scaling, m["scaling"].out_scaling
    mu, sg = vec(si.mu), vec(si.sigma)
    Xs = copy(X)
    for j in 1:(size(X, 2) - 1)
        i = mod1(j, NQ)
        Xs[:, j] = (X[:, j] .- mu[i]) ./ sg[i]
    end
    Ys = (Y .- vec(so.mu)') ./ vec(so.sigma)'
    h = m["hist_len"]
    qs = X[:, 1:NQ]                                                   # q*^t
    qwin = reduce(hcat, [X[:, (2j * NQ + 1):((2j + 1) * NQ)] for j in 1:h])   # q*^{t-1} … q*^{t-h}
    return (; X, Xs, Ys, isg = sg, osg = vec(so.sigma), qs, qwin)
end

"Per QoI over `ref` (raw rms): clean rms, rms with the window's draws (exact), and the gain sqrt(response / drawn)."
function own_draw_line(io, name, r, v, drawn, osg, ref)
    clean = vec(sqrt.(mean(abs2, r; dims = 1))) .* osg
    noisy = sqrt.(vec(mean(abs2, r; dims = 1)) .+ vec(mean(v; dims = 1))) .* osg
    gain = sqrt.(vec(mean(v; dims = 1)) ./ drawn)
    f3(x) = join((@sprintf("%.3f", y) for y in x), " ")
    @printf(io, "  %-18s clean / ref %s | noisy / ref %s | gain %s\n", name, f3(clean ./ ref), f3(noisy ./ ref), f3(gain))
    return (; clean, noisy, gain)
end

"""
The response per unit draw and the AR's long-run variance ratio of a deployed LinReg + AR `name` under its OWN draws
(card D1 (a), (b)), exact, on the rows of `d`: the mean `c`, the drawn AR (`ar_phi`, `ar_sigma_xi`), the drawn
variance diag Σ (white: `stoch_distr`). A white LinReg draws N(0, Σ).
"""
function d1_numbers(name, d, h, M; L = 20)
    m = load(joinpath(LRS, name, "LinReg.jld2"))
    Ct = Matrix(Matrix(m["c"])')
    Sw = Matrix(cov(m["stoch_distr"]))
    haskey(m, "scale_beta") && error("d1_numbers: $name carries a scale")
    phi = haskey(m, "ar_phi") ? Matrix{Float64}(m["ar_phi"]) : zeros(2, NQ)
    S = haskey(m, "ar_phi") ? Matrix{Float64}(m["ar_sigma_xi"]) : Sw
    size(phi, 1) == 1 && (phi = [phi; zeros(1, NQ)])
    Gz = ar_lagcov(phi, S, L + h)
    W0 = window_lagcov(Gz, h, 0)
    v = noise_var_rows(Ct' * M, W0, repeat(d.osg, h))
    r = d.Ys .- d.Xs * Ct
    return (; name, Ct, r, v, drawn = diag(Gz[1]), gain = sqrt.(vec(v) ./ diag(Gz[1])),
            lrv = [lrv_ratio(phi[1, i], phi[2, i]) for i in 1:NQ], phi)
end

function source_rows(src, m, c0; io)
    for r in (405:3999, steps_of(1, 10), 400:3999, 405:4000)
        d = scaled_design(m, r)
        e = maximum(abs, (d.Xs \ d.Ys)' .- c0) / maximum(abs, c0)
        @printf(io, "  rows %d-%d: least squares vs %s's c, max rel. diff %.1e\n", first(r), last(r), src, e)
        if e < 1e-8
            n = first(r)
            d.X[1, 1:NQ] == QSR[:, n] && d.X[1, (NQ + 1):(2NQ)] == QR[:, n] && d.X[1, (2NQ + 1):(3NQ)] == QSR[:, n - 1] ||
                error("the design is not [q*^n; q^{n-1}, q^{n-1*}; ...]: the window map would be wrong")
            # the window's predictors are the earlier rows' own (consecutive steps)
            h = m["hist_len"]
            for j in 1:h
                d.qwin[(j + 1):end, ((j - 1) * NQ + 1):(j * NQ)] == d.qs[1:(end - j), :] ||
                    error("q*^{t-$j} of row t is not row t-$j's q*")
            end
            return r, d
        end
    end
    error("no candidate window reproduces $(src)'s c; its fit rows are unknown")
end

"The white limit and q2-at-β = 0 (`--check`; writes nothing)."
function check_limits(src; io = stdout)
    m = load(joinpath(LRS, src, "LinReg.jld2"))
    h = m["hist_len"]
    fitr, d = source_rows(src, m, Matrix(m["c"]); io)
    M = window_map(h, d.isg)
    S0 = Matrix(cov(m["stoch_distr"]))
    println(io, "\n==== check 1: the white limit (φ ≡ 0) against $(src)_nt and $(src)_nt_ar2nw")
    w = fit_noise_ar(d, M, h; phi0 = zeros(2, NQ), S0, white = true, tol = 1e-12)   # fit_noise's stop
    nt = load(joinpath(LRS, "$(src)_nt", "LinReg.jld2"))
    # c where it acts (the fit rows' prediction): its raw entries are ill-determined at ~1e-6 across machines (BLAS; the
    # desktop against the laptop-built <src>_nt, 2026-11-06: c rel. 5e-6 with Σ 5e-13), fit_noise_ar's note
    dc = maximum(abs, Matrix(nt["c"]) .- w.Ct') / maximum(abs, w.Ct)
    dp = maximum(abs, d.Xs * (Matrix(Matrix(nt["c"])') .- w.Ct)) / maximum(std(d.Ys; dims = 1))
    dS = maximum(abs, Matrix(cov(nt["stoch_distr"])) .- w.Sig) / maximum(diag(w.Sig))
    dm = maximum(abs, mean(nt["stoch_distr"]) .- w.mu)
    @printf(io, "  %d iterations; prediction rel. %.1e (c rel. %.1e), Σ rel. %.1e, |Δμ| %.1e against %s_nt\n", w.iterations,
            dp, dc, dS, dm, src)
    ar = load(joinpath(LRS, "$(src)_nt_ar2nw", "LinReg.jld2"))
    dphi = maximum(abs, Matrix(ar["ar_phi"]) .- w.a.phi)
    dxi = maximum(abs, Matrix(ar["ar_sigma_xi"]) .- w.a.S) / maximum(diag(w.a.S))
    @printf(io, "  the AR fitted after it: |Δφ| %.1e, Σ_ξ rel. %.1e against %s_nt_ar2nw\n", dphi, dxi, src)
    ok1 = dp < 1e-6 && dS < 1e-9 && dm < 1e-12 && dphi == 0 && dxi < 1e-7   # dp: 0 on the laptop, 2.9e-8 on the desktop
    println(io, ok1 ? "  PASS" : "  FAIL")
    println(io, "\n==== check 2: q2's machinery with β held at 0 against q1 (from the same start)")
    phi0, Sx0 = Matrix{Float64}(ar["ar_phi"]), Matrix{Float64}(ar["ar_sigma_xi"])
    q1 = fit_noise_ar(d, M, h; phi0, S0 = Sx0)
    q2 = fit_noise_ar(d, M, h; phi0, S0 = Sx0, scale = (; d.qs, d.qwin, beta0 = zeros(NQ), fixbeta = true))
    # c itself is ill-determined at ~1e-6 (fit_noise_ar's note): compare it where it acts, the fit rows' prediction and
    # the response to the window's draws
    dpred = maximum(abs, d.Xs * (q2.Ct .- q1.Ct)) / maximum(std(d.Ys; dims = 1))
    e = (dpred, maximum(abs, q2.AM .- q1.AM) / maximum(abs, q1.AM), q2.phi == q1.phi,
         maximum(abs, q2.S .- q1.S) / maximum(diag(q1.S)), maximum(abs, q2.Sig .- q1.Sig) / maximum(diag(q1.Sig)))
    @printf(io, "  prediction rel. %.1e, C'M rel. %.1e, φ equal %s, Σ_ξ rel. %.1e, Σ rel. %.1e (iterations %d / %d)\n",
            e..., q1.iterations, q2.iterations)
    ok2 = e[1] < 1e-7 && e[2] < 1e-5 && e[3] && e[4] < 1e-7 && e[5] < 1e-8   # both stop within tol = 1e-8
    println(io, ok2 ? "  PASS" : "  FAIL")
    return ok1 && ok2
end

function build_noise_ar(src; pl = false, io = stdout, report_only = false)
    dst = pl ? "$(src)_ntar2_pl" : "$(src)_ntar2"
    ddir = joinpath(LRS, dst)
    report_only || !isdir(ddir) || error("$ddir exists; refusing to overwrite (--report re-prints it)")
    m = load(joinpath(LRS, src, "LinReg.jld2"))
    @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
    isnothing(get(m, "ar_phi", nothing)) && isnothing(get(m, "scale_beta", nothing)) && !haskey(m, "noise_provenance") ||
        error("$src carries an AR, a scale or a noise training; give the white source")
    p = load(joinpath(LRS, src, "parameters.jld2"))["parameters"]
    lam = hasproperty(p, :lambda) ? p.lambda : 0.0
    lam == 0 || error("$src: λ = $lam; q1/q2 are on the unregularized baseline (the LinReg7 side is cut, Rik 2026-11-06)")
    h = m["hist_len"]
    c0 = Matrix(m["c"])
    fitr, d = source_rows(src, m, c0; io)
    M = window_map(h, d.isg)
    # the start: m2's AR (q1); q1's AR and <src>_pl's β (q2); white otherwise
    m2 = joinpath(LRS, "$(src)_nt_ar2nw", "LinReg.jld2")
    q1f = joinpath(LRS, "$(src)_ntar2", "LinReg.jld2")
    startf = pl && isfile(q1f) ? q1f : isfile(m2) ? m2 : nothing
    phi0, S0 = startf === nothing ? (zeros(2, NQ), Matrix(cov(m["stoch_distr"]))) :
               (Matrix{Float64}(load(startf, "ar_phi")), Matrix{Float64}(load(startf, "ar_sigma_xi")))
    plf = joinpath(LRS, "$(src)_pl", "LinReg.jld2")
    beta0 = isfile(plf) ? Vector{Float64}(load(plf, "scale_beta")) : zeros(NQ)
    scale = pl ? (; d.qs, d.qwin, beta0) : nothing
    println(io, "\n==== $dst: $src trained with its own AR(2)", pl ? " + power-law scaled" : "", " window noise, rows ",
            "$(first(fitr))-$(last(fitr)) (1-10 TU); start: ", startf === nothing ? "white" : relpath(startf, LRS),
            pl ? ", β from $(isfile(plf) ? "$(src)_pl" : "0")" : "")
    f = fit_noise_ar(d, M, h; phi0, S0, scale)
    @printf(io, "  fixed point in %d iterations (changes %s); one more step: Σ_ξ rel. %.1e, ar2_ls keeps φ %s\n", f.iterations,
            join((@sprintf("%.1e", s) for s in f.steps[unique([1, 2, 3, 5, 10, length(f.steps)] ∩ eachindex(f.steps))]), ", "),
            f.fixed...)
    if f.cycle !== nothing
        println(io, "  ⚠️ φ cycled between $(length(f.cycle)) lattice points of ar2_ls; tie-break (fit_noise_ar): each held fixed,",
                " the one closest to self-consistent kept")
        for (k, c) in enumerate(f.cycle)
            dif = [i for i in 1:NQ if c[:, i] != f.phi[:, i]]
            @printf(io, "    candidate %d: excess ACF-fit objective %.3e%s; differs from the kept φ in %s\n", k, f.cycle_gaps[k],
                    c == f.phi ? " (kept)" : "", isempty(dif) ? "nothing" :
                    join((@sprintf("%s (%.6f, %.6f) vs (%.6f, %.6f)", LABELS[i], c[1, i], c[2, i], f.phi[1, i], f.phi[2, i]) for i in dif), ", "))
        end
        @printf(io, "    ar2_ls's choice for the kept state: %s\n",
                join((@sprintf("%s (%.4f, %.4f)", LABELS[i], f.afree.phi[1, i], f.afree.phi[2, i]) for i in 1:NQ
                      if f.afree.phi[:, i] != f.phi[:, i]), ", "))
    end
    sd0 = Matrix(cov(m["stoch_distr"]))
    @printf(io, "  max |c - c_%s| / max |c_%s| = %.3f\n", src, src, maximum(abs, f.Ct' .- c0) / maximum(abs, c0))
    @printf(io, "  sd(η) (Γ^e_0) / %s's: %s\n", src, fmt(sqrt.(diag(f.Sig) ./ diag(sd0)); d = 3))
    # self-consistency (card: Γ_0 = Σ): the deployed AR's marginal covariance against the residual's
    Gz0 = f.Gz[1]
    target = pl ? f.Seps : f.Sig
    @printf(io, "  deployed AR marginal variance / diag Γ_0 of the residual%s: %s (D matches them)\n", pl ? " / f" : "",
            join((@sprintf("%.6f", x) for x in diag(Gz0) ./ diag(target)), " "))
    offd = [(Gz0[i, j] / sqrt(Gz0[i, i] * Gz0[j, j]), target[i, j] / sqrt(target[i, i] * target[j, j])) for i in 1:NQ for j in (i + 1):NQ]
    @printf(io, "  lag-0 correlations, deployed AR vs the residual's: max |diff| %.3f (Σ_ξ = D R D, R from the innovations)\n",
            maximum(abs(a - b) for (a, b) in offd))
    if pl
        @printf(io, "  %-9s %7s %7s | %11s %9s | %s\n", "QoI", "beta", "beta_pl", "q_ref", "sd at ref", "diag Σ_ε / exp(logvar)")
        for i in 1:NQ
            @printf(io, "  %-9s %7.3f %7.3f | %11.4e %9.3e | %.2e\n", LABELS[i], f.beta[i], beta0[i], f.qref[i],
                    sqrt(f.Seps[i, i]), f.Seps[i, i] / exp(f.logvar[i]) - 1)
        end
        maximum(abs, diag(f.Seps) ./ exp.(f.logvar) .- 1) < 1e-6 || error("diag Σ_ε is not the ML variance at q_ref")
    end
    # the AR: q1/q2 against m2
    mm2 = isfile(m2) ? load(m2) : nothing
    @printf(io, "  %-9s %7s %7s %7s | %13s | %15s | %9s\n", "QoI", "rho1", "phi1", "phi2", "|poles|", "m2 phi1 phi2", "σ_ξ/sd")
    for i in 1:NQ
        mp = mm2 === nothing ? (NaN, NaN) : (mm2["ar_phi"][1, i], mm2["ar_phi"][2, i])
        @printf(io, "  %-9s %7.3f %7.3f %7.3f | %6.3f %6.3f | %7.3f %7.3f | %9.3f\n", LABELS[i], f.a.racf[2, i], f.phi[1, i],
                f.phi[2, i], ar_poles(f.phi[1, i], f.phi[2, i])..., mp..., sqrt(f.a.sxi2[i] / f.a.s2[i]))
    end
    lags = (1, 2, 5, 10, 20)
    println(io, "  ACF at lags ", join(lags, " "), ": the residual given the noisy history", pl ? " / f" : "",
            " | the fitted AR | the clean residual")
    for i in 1:NQ
        zc = f.Z[:, i]
        @printf(io, "  %-9s %s | %s | %s\n", LABELS[i], fmt(f.a.racf[[l + 1 for l in lags], i]),
                fmt(ar_acf(f.phi[1, i], f.phi[2, i], 20)[[l + 1 for l in lags]]), fmt(acf(zc, lags)))
    end
    # card D1 (a), (b), and the own-draw errors (the fit window; 10-100 TU: analysis/noise_ar_offline.jl)
    ref = vec(sqrt.(mean(abs2, d.Ys .- d.Xs * c0'; dims = 1))) .* d.osg
    println(io, "  per QoI, raw units over $(src)'s clean rms on the fit rows, under each closure's OWN draws (exact):")
    for nm in (src, "$(src)_nt", "$(src)_nt_ar2nw")
        isfile(joinpath(LRS, nm, "LinReg.jld2")) || continue
        q = d1_numbers(nm, d, h, M)
        own_draw_line(io, nm, q.r, q.v, q.drawn, d.osg, ref)
    end
    drawn = pl ? vec(mean(f.f .^ 2; dims = 1)) .* diag(f.Gz[1]) : diag(f.Gz[1])
    own_draw_line(io, dst, f.Z, f.v, drawn, d.osg, ref)
    lrv = [lrv_ratio(f.phi[1, i], f.phi[2, i]) for i in 1:NQ]
    @printf(io, "  long-run variance ratio of the AR (200 lags): %s\n", fmt(lrv; d = 2))
    if mm2 !== nothing
        @printf(io, "                                      m2's: %s\n",
                fmt([lrv_ratio(mm2["ar_phi"][1, i], mm2["ar_phi"][2, i]) for i in 1:NQ]; d = 2))
    end
    if report_only
        b = load(joinpath(ddir, "LinReg.jld2"))
        # tolerant (fit_noise_ar's note: c is ill-determined at ~1e-6, Σ_ξ has a ~1e-9 floor): c where it acts
        dc = maximum(abs, d.Xs * (Matrix(Matrix(b["c"])') .- f.Ct)) / maximum(std(d.Ys; dims = 1))
        dS = maximum(abs, Matrix(cov(b["stoch_distr"])) .- f.Sig) / maximum(diag(f.Sig))
        dx = maximum(abs, Matrix(b["ar_sigma_xi"]) .- f.S) / maximum(diag(f.S))
        db = pl ? maximum(abs, b["scale_beta"] .- f.beta) : 0.0
        dc < 1e-6 && dS < 1e-8 && dx < 1e-7 && b["ar_phi"] == f.phi && db < 1e-6 ||   # dc: a refit on another machine
            error(@sprintf("the file differs from a refit: prediction %.1e, Σ %.1e, Σ_ξ %.1e, φ equal %s, |Δβ| %.1e", dc,
                           dS, dx, b["ar_phi"] == f.phi, db))
        @printf(io, "  (report only: %s holds this fit: prediction %.1e, Σ %.1e, Σ_ξ %.1e, |Δβ| %.1e)\n", ddir, dc, dS,
                dx, db)
        return (; dst, f)
    end
    # write: the source's keys, c and stoch_distr replaced, noise_*, ar_* (and scale_*) added
    a = load(joinpath(LRS, src, "LinReg.jld2"))
    b = copy(a)
    b["c"] = Matrix(f.Ct)'
    b["stoch_distr"] = RF.Distributions.MvNormal(f.mu, f.Sig)
    b["noise_sigma_dx"] = f.Sdx
    b["noise_fit_rows"] = (first(fitr), last(fitr))
    b["noise_provenance"] = (; source = src, lambda = 0.0,
        form = "c' = (X'X + M (W_0 .* sum_t g_t g_t') M') \\ X'Y, W_0 the block-Toeplitz covariance of the window's unit-scale AR(2) draws, g_t their raw scale; Sigma = Gamma^e_0, the AR fitted to the residual given the noisy history (fixed point)",
        noise = "the closure's own AR(2)" * (pl ? " x power-law scale" : "") * " draws as window noise, anchored at the prediction step (Eq. window noise; tools/step3_innoise.jl)",
        start = startf === nothing ? "white" : relpath(startf, LRS), iterations = f.iterations,
        units = "scaled (in/out_scaling of the source)", card = "analysis/cards/q1q2.md",
        record = "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2",
        built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_noise_ar_variant.jl" * (pl ? " --pl" : ""))
    b["ar_phi"] = f.phi
    b["ar_sigma_xi"] = f.S
    b["ar_provenance"] = (; source = src, order = 2,
        method = "ar2_ls on the ACF (lags 1-20) of the residual given the noisy history" * (pl ? ", divided by f(q*)" : "") *
                 ", restricted to phi1 >= 0, phi2 <= 0; inside the fixed point of the noise training",
        sigma = "exact innovation covariance's correlation, marginal variance matched to diag Gamma_0" * (pl ? " of e/f" : ""),
        fit_steps = (first(fitr), last(fitr)), units = pl ? "unit scale: residual (scaled, minus mu) divided by f(q*)" :
                                                           "scaled (out_scaling), minus mean(stoch_distr)",
        built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_noise_ar_variant.jl" * (pl ? " --pl" : ""))
    if pl
        b["scale_beta"] = f.beta
        b["scale_qref"] = f.qref
        b["scale_qclip"] = f.qclip
        b["scale_sigma_eps"] = f.Seps
        b["scale_logvar"] = f.logvar
        b["scale_provenance"] = (; source = src, form = "eta = mu_eta + f(q*) .* z, z the AR(2), f_i = (clamp(q*_i, clip)/q_ref_i)^beta_i",
            fit = "RF.fit_powerlaw_scale per QoI on the expected squared residual given the noisy history (exact expected NLL), inside the fixed point; Sigma_eps = Gamma^{e/f}_0 (not drawn with AR)",
            state = "q*^n (the predictor), physical units", fit_steps = (first(fitr), last(fitr)),
            built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_noise_ar_variant.jl --pl")
    end
    mkpath(ddir)
    jldopen(joinpath(ddir, "LinReg.jld2"), "w") do fh
        for (k, v) in b
            fh[k] = v
        end
    end
    cp(joinpath(LRS, src, "parameters.jld2"), joinpath(ddir, "parameters.jld2"))
    b2 = load(joinpath(ddir, "LinReg.jld2"))
    @assert Matrix(b2["c"]) == f.Ct' && mean(b2["stoch_distr"]) == f.mu && Matrix(cov(b2["stoch_distr"])) == f.Sig
    @assert b2["ar_phi"] == f.phi && b2["ar_sigma_xi"] == f.S
    pl && @assert b2["scale_beta"] == f.beta && b2["scale_sigma_eps"] == f.Seps
    for k in keys(a)
        k in ("c", "stoch_distr") || @assert isequal(a[k], b2[k]) "key $k differs from the source"
    end
    file = joinpath(ddir, "LinReg.jld2")
    lr = RF.LinReg(file, Xoshiro(1), Array; q_hist = zeros(2NQ, h), spinnup_data = zeros(NQ, h))
    @assert !isnothing(lr.ar) && isnothing(lr.scale) == !pl && Matrix(lr.c) == f.Ct'
    println(io, "  wrote $ddir (source keys identical except c, stoch_distr; + noise_*, ar_*", pl ? ", scale_*" : "",
            "); the deployed RikFlow.LinReg loads it")
    println(io, "  offline sanity (deployed RikFlow.LinReg on the new file, teacher-forced 1-10 TU):")
    if pl
        ar_scale_sanity(file, dst, (; f.beta, f.qref, f.qclip), f.a; io)
    else
        offline_sanity(file, dst, f.a; io)
    end
    return (; dst, f)
end

if abspath(PROGRAM_FILE) == @__FILE__
    flags = ("--report", "--pl", "--check")
    args = filter(a -> !(a in flags), ARGS)
    src = isempty(args) ? "LinReg1" : args[1]
    if "--check" in ARGS
        check_limits(src) || error("check failed")
    else
        build_noise_ar(src; pl = "--pl" in ARGS, report_only = "--report" in ARGS)
    end
end
