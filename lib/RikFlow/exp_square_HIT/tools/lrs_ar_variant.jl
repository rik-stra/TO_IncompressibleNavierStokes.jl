# M0ᶜ-ridge variant builder: a deployed LinReg + a stationary AR(p) residual (results_LSTMS §12).
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/lrs_ar_variant.jl [src] [p ...]
#
# default `LinReg7 2 1`: writes `output/TO_LRS/LinReg7_ar2` and `.../LinReg7_ar1`. `2c` writes
# `LinReg7_ar2c`, the AR(2) with the residual's lag 1 exact (`ar2_ls_lag1`); `2n` writes `LinReg7_ar2n`,
# the AR(2) restricted to φ1 ≥ 0, φ2 ≤ 0 (`ar2_ls(...; nonneg = true)`: no step-to-step alternating
# component; 2026-10-07, paper Sec. 6.1). `2e` / `2en` write `LinReg7_ar2e` / `LinReg7_ar2en`, the AR(2) by
# overdetermined Yule–Walker on lags 1-20 (`ar2_eyw`, free / restricted; 2026-10-13). `1ls` writes
# `LinReg7_ar1ls`, the AR(1) fitted as the AR(2) is, by LS on the residual ACF at lags 1-20 (`ar1_ls`;
# 2026-10-15, the AR(1) of paper Fig. colour-resacf-ar1); plain `1` stays φ = ρ̂(1). `--report`
# re-prints the tables for existing variants (checking the file holds exactly the refit) and writes nothing.
# `2nw` (a noise-trained
# source, `<src>_nt` of `lrs_noise_variant.jl`) writes `<src>_ar2nw`: the restricted AR(2) fitted to the residual
# GIVEN THE NOISY HISTORY, exactly, its marginal variance the source's self-consistent Σ (`fit_ar_noisy`;
# Rik, 2026-11-04: "run the both noise trained linear models with AR(2) coupled", fitted "the same way" as their MVG).
#
# The new directory holds a byte copy of `<src>/LinReg.jld2` with three keys ADDED (`ar_phi`,
# `ar_sigma_xi`, `ar_provenance`) -- the mean, Σ, scaling and every other key are untouched -- and a
# copy of `parameters.jld2`. Nothing else (no online runs) is copied. An existing target is refused.
#
# The AR is fitted to the source model's OWN residual on its train window (1-10 TU, steps
# 400-4000 of the R1 tracked cache), evaluated exactly as deployed (per-QoI input scaling,
# `c [x; 1]`, level target) and expressed in SCALED units (out_scaling), minus μ_η = mean(stoch_distr):
#
#   z_t = (q^t − mu_out) / sigma_out − c [x_t; 1] − μ_η
#
#  * p = 2: per-QoI AR(2) by least squares on the residual ACF at lags 1-20 (`ar2_ls`, stationary
#    by construction). p = 1: φ = the residual's lag-1 autocorrelation, or with `ls` (`1ls`) by
#    least squares on the ACF at lags 1-20 (`ar1_ls`, φ in [0, 1)).
#  * Σ_ξ = D R D: R the lag-0 correlation of the implied innovations ξ_t = z_t − Σ φ_k z_{t−k}, D the
#    per-QoI scale that makes the AR's stationary MARGINAL variance equal var(z):
#    σ_ξ² = var(z) (1 − φ1 ρ1 − φ2 ρ2), ρ the AR's own ACF.
#
# Also prints: roots' moduli, the kernel-weighted power ratio (§10h), and the offline sanity check
# (the deployed `RikFlow.LinReg` on the new file, teacher-forced on 1-10 TU: noise ACF vs the AR ACF
# and the residual's, marginal and one-step spread, warm-start state vs the data residual).
# 🔒 Nothing past 10 TU is read here (guard()).

include(normpath(joinpath(@__DIR__, "..", "..", "analysis", "m0c_ridge.jl")))   # LRS, design, acf, ar2_*, kpower, ...

"The deployed model's residual in scaled units on steps `r` (rows = steps), and the pieces."
function scaled_resid(name, r)
    m = load(joinpath(LRS, name, "LinReg.jld2"))
    @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
    X, Y = design(m["hist_len"], r)
    si, so = m["scaling"].in_scaling, m["scaling"].out_scaling
    mu, sg = vec(si.mu), vec(si.sigma)
    Xs = copy(X)
    for j in 1:(size(X, 2) - 1)
        i = mod1(j, NQ)
        Xs[:, j] = (X[:, j] .- mu[i]) ./ sg[i]
    end
    Ys = (Y .- vec(so.mu)') ./ vec(so.sigma)'
    mueta = Vector{Float64}(mean(m["stoch_distr"]))
    Z = Ys .- Xs * Matrix(m["c"])' .- mueta'
    return (; X, Y, Xs, Ys, Z, m, mueta)
end

"AR(1)/AR(2) ACF at lags 0:L for coefficients (φ1, φ2)."
ar_acf(p1, p2, L) = ar2_acf(p1, p2, L)

"Moduli of the roots of z² − φ1 z − φ2 (the AR's poles)."
function ar_poles(p1, p2)
    d = complex(p1^2 + 4p2)
    return sort(abs.([(p1 + sqrt(d)) / 2, (p1 - sqrt(d)) / 2]); rev = true)
end

"""
AR(2) whose lag-1 autocorrelation equals the data's exactly (φ1 = r1 (1 − φ2)); φ2 by least squares on
the ACF at lags 2..L. Removes `ar2_ls`'s one-step over-dispersion in the middle bands (§12b).
"""
function ar2_ls_lag1(racf; L = 20)
    r1 = racf[2]
    err(p2) = (p1 = r1 * (1 - p2); ar2_stationary(p1, p2) ? sum(abs2, ar2_acf(p1, p2, L)[3:end] .- racf[3:(L + 1)]) : Inf)
    grid = range(-0.99, 0.99; length = 1981)
    p2 = grid[argmin(err.(grid))]
    for s in (1e-4, 1e-5), _ in 1:3
        p2 = argmin(err, p2 .+ s .* (-5:5))
    end
    return r1 * (1 - p2), p2
end

"""
    ar_innovation_var(s2, p1, p2)

σ_ξ² of the AR(2) `η_t = φ1 η_{t−1} + φ2 η_{t−2} + ξ_t` whose stationary MARGINAL variance is `s2`:
`s2 (1 − φ1 ρ1 − φ2 ρ2)`, ρ the AR's own ACF (`p2 = 0` is the AR(1), `s2 (1 − φ1²)`). The `D` of
`Σ_ξ = D R D` in `fit_ar`; pure, so `test/test_round1.jl` checks it against the closed-form AR(2)
variance and a long simulation.
"""
ar_innovation_var(s2, p1, p2) = (rho = ar_acf(p1, p2, 2); s2 * (1 - p1 * rho[2] - p2 * rho[3]))

function fit_ar(Z, p; lag1 = false, nonneg = false, eyw = false, ls = false)
    @assert !(lag1 && nonneg) "lag1 and nonneg are separate variants"
    @assert !(lag1 && eyw) "lag1 and eyw are separate variants"
    @assert !ls || p == 1 "ls is the AR(1) variant"
    phi = zeros(p, NQ)
    for i in 1:NQ
        r = acf(Z[:, i], 0:20)
        if p == 2
            phi[:, i] .= lag1 ? ar2_ls_lag1(r; L = 20) :
                         eyw ? ar2_eyw(r; L = 20, nonneg) : ar2_ls(r; L = 20, nonneg)
        else
            phi[1, i] = ls ? ar1_ls(r; L = 20) : r[2]
        end
    end
    p1 = phi[1, :]
    p2 = p == 2 ? phi[2, :] : zeros(NQ)
    Xi = Z[3:end, :] .- p1' .* Z[2:(end - 1), :] .- p2' .* Z[1:(end - 2), :]
    R = cor(Xi)
    s2 = vec(var(Z; dims = 1))
    sxi2 = [ar_innovation_var(s2[i], p1[i], p2[i]) for i in 1:NQ]
    D = Diagonal(sqrt.(sxi2))
    S = Matrix(Symmetric(D * R * D))
    return (; phi, p1, p2, S, R, Xi, s2, sxi2)
end

"""
    window_map(h, isg) -> M

As `lrs_noise_variant.jl`'s: dx in the regressor's scaled units (`[q*^n; q^{n-1}, q^{n-1*}; ...; 1]`, input
sds `isg`) per unit RAW η^{n-j} of QoI i, column (j - 1) NQ + i.
"""
function window_map(h, isg)
    M = zeros(NQ * (2h + 1) + 1, NQ * h)
    for j in 1:h, i in 1:NQ, k in 1:h
        q = NQ + 2(k - 1) * NQ + i
        k >= j + 1 && (M[q, (j - 1) * NQ + i] = -1 / isg[i])
        k >= j && (M[q + NQ, (j - 1) * NQ + i] = -1 / isg[i])
    end
    return M
end

"""
    fit_ar_noisy(d, p; nonneg, L = 20)

The AR of a NOISE-TRAINED source (`lrs_noise_variant.jl`, `<src>_nt`) fitted to its residual GIVEN THE NOISY
HISTORY, as its MVG's Σ is (Rik, 2026-11-04: "Yes, same way"). Exact, no draws: with z_t the clean scaled
residual (minus μ_η) and the closure's white window noise of covariance Σ (the source's self-consistent
`stoch_distr`), the residual is z_t - C' M w_t, w_t = [η^{t-1}; ...; η^{t-h}] (raw), and its lagged covariances
are

    Γ_k = (1/N) Σ_t z_t z_{t-k}'  +  C' M (S_k ⊗ D Σ D) M' C,   S_k[j, j'] = 1 if j - j' = k (k < h),

the noise of consecutive windows overlapping in h - k draws; Γ_0 is the self-consistent Σ (asserted). Per QoI
the AR's φ by `ar2_ls` on ρ_k = Γ_k[i,i] / Γ_0[i,i], k = 1..L (`nonneg` the AR(2) rule); Σ_ξ = D R D with R the
correlation of the exact innovation covariance E[ξ ξ'] (ξ_t = z_t - Φ1 z_{t-1} - Φ2 z_{t-2}, Φ diagonal) and D
matching the AR's marginal variance to diag Σ. The mean model was trained with WHITE draws; the closure draws
the AR, so the fit is the residual under white draws (an approximation, not iterated).
"""
function fit_ar_noisy(d, p; nonneg = false, L = 20)
    @assert p == 2 "the noisy-residual fit is the AR(2)"
    m = d.m
    haskey(m, "noise_provenance") || error("not a noise-trained LinReg (no noise_provenance key)")
    h = m["hist_len"]
    isg, osg = vec(m["scaling"].in_scaling.sigma), vec(m["scaling"].out_scaling.sigma)
    Ct = Matrix(Matrix(m["c"])')                       # scaled regressor -> scaled output
    Sig = Matrix(cov(m["stoch_distr"]))
    M = window_map(h, isg)
    B = Diagonal(osg) * Sig * Diagonal(osg)            # raw η covariance
    Z = d.Z
    N = size(Z, 1)
    G = map(0:L) do k
        Gd = Z[(k + 1):end, :]' * Z[1:(end - k), :] ./ N
        k >= h && return Gd
        Sk = zeros(h, h)
        for jp in 1:(h - k)
            Sk[jp + k, jp] = 1.0
        end
        Gd .+ Ct' * M * kron(Sk, B) * M' * Ct
    end
    e0 = maximum(abs, Symmetric(G[1]) .- Sig) / maximum(diag(Sig))
    e0 < 1e-8 || error(@sprintf("Γ_0 is not the source's self-consistent Σ (rel. %.1e): wrong rows or units", e0))
    racf = [G[k + 1][i, i] / G[1][i, i] for k in 0:L, i in 1:NQ]
    racf_clean = [Z[(k + 1):end, i]' * Z[1:(end - k), i] / (Z[:, i]' * Z[:, i]) for k in 0:L, i in 1:NQ]
    phi = zeros(2, NQ)
    for i in 1:NQ
        phi[:, i] .= ar2_ls(racf[:, i]; L, nonneg)
    end
    p1, p2 = phi[1, :], phi[2, :]
    F1, F2 = Diagonal(p1), Diagonal(p2)
    G0, G1, G2 = G[1], G[2], G[3]
    Cxi = G0 .+ F1 * G0 * F1 .+ F2 * G0 * F2 .- G1 * F1 .- F1 * G1' .- G2 * F2 .- F2 * G2' .+ F1 * G1 * F2 .+
          F2 * G1' * F1
    sd = sqrt.(diag(Cxi))
    R = Matrix(Symmetric(Cxi ./ (sd * sd')))
    s2 = diag(G0)
    sxi2 = [ar_innovation_var(s2[i], p1[i], p2[i]) for i in 1:NQ]
    D = Diagonal(sqrt.(sxi2))
    S = Matrix(Symmetric(D * R * D))
    return (; phi, p1, p2, S, R, s2, sxi2, racf, racf_clean, e0)
end

"Simulate the diagonal AR with innovation covariance S (NQ x n, after burn-in)."
function sim_ar_S(p1, p2, S, n, rng)
    L = cholesky(Symmetric(S)).L
    E = zeros(NQ, n + 2000)
    for t in 3:size(E, 2)
        xi = L * randn(rng, NQ)
        @inbounds for i in 1:NQ
            E[i, t] = p1[i] * E[i, t - 1] + p2[i] * E[i, t - 2] + xi[i]
        end
    end
    return E[:, 2001:end]
end

"""
Kernel-weighted power. The diagonal ratios are scale-free; the full 6x6 kernel mixes QoIs, so it is
applied in RAW units (the kernel's own, as §10h did): `Z` and the covariances are rescaled by
`sigma_out` first.
"""
function power_report(Zs, f, Sws, sig; io)
    Z = Zs .* sig'
    Dg = Diagonal(sig)
    Sw = Dg * Sws * Dg
    fS = Dg * f.S * Dg
    G = load(joinpath(TO, "response", "response_kernel.jld2"), "G")
    K = size(G, 3)
    println(io, "  kernel-weighted power P / P_white at the residual's variance (measured kernel, K = $K)")
    @printf(io, "    %-9s %8s %8s\n", "QoI", "diag AR", "diag data")
    for i in 1:NQ
        e = Z[:, i]; s2 = var(e); g = G[i, i, :]
        Pw = kpower(g, [1.0; zeros(K - 1)], s2)
        Pa = kpower(g, ar_acf(f.p1[i], f.p2[i], K - 1), s2)
        @printf(io, "    %-9s %8.2f %8.2f\n", LABELS[i], Pa / Pw, kfilter_var(g, e) / Pw)
    end
    rng = Xoshiro(20260928)
    n = 200_000
    s2v = vec(var(Z; dims = 1)); Cor = cor(Z)
    pw = kfilter_var_full(G, sim_ar(zeros(NQ), zeros(NQ), Cor, s2v, n, rng))
    pa = kfilter_var_full(G, sim_ar(f.p1, f.p2, Cor, s2v, n, rng))           # §10h's construction
    pd = kfilter_var_full(G, permutedims(Z))
    # as built: the deployed Σ_ξ against the deployed white Σ (stoch_distr)
    pwd = kfilter_var_full(G, sim_ar_S(zeros(NQ), zeros(NQ), Sw, n, rng))
    pab = kfilter_var_full(G, sim_ar_S(f.p1, f.p2, fS, n, rng))
    @printf(io, "    full 6x6 kernel, §10h construction: AR/white %s | data/white %s\n", fmt(pa ./ pw), fmt(pd ./ pw))
    @printf(io, "    full 6x6 kernel, as built: AR(Σ_ξ) / white(stoch_distr Σ) %s\n", fmt(pab ./ pwd))
    return (; full_ar = pa ./ pw, full_data = pd ./ pw, as_built = pab ./ pwd)
end

"""
Offline sanity: the deployed `RikFlow.LinReg` on the new file, teacher-forced on the train window.
Warm-up = the 100 rows before 400 (replayed record), then one prediction per row with the level
history overwritten by the record's level after each step.
"""
function offline_sanity(file, name, f; io, seed = 1)
    nw = 100
    r = (first(steps_of(1, 10)) - nw):last(steps_of(1, 10))
    d = scaled_resid(name, r)
    m = d.m
    qs = d.X[:, 1:NQ]                 # q*_t
    dQrec = d.Y .- qs                 # the record's dQ (level target)
    h = m["hist_len"]
    lr = RF.LinReg(file, Xoshiro(seed), Array; q_hist = zeros(2NQ, h), spinnup_data = permutedims(dQrec[1:nw, :]))
    p = size(lr.ar.phi, 1)
    so = m["scaling"].out_scaling
    N = size(qs, 1)
    zmod = zeros(N - nw, NQ)
    warm = zeros(NQ, 0)
    for t in 1:N
        dQ = RF.get_next_item_timeseries(lr, qs[t, :])
        if t == nw
            warm = copy(lr.ar.z)
        end
        if t > nw
            lev = (qs[t, :] .+ dQ .- vec(so.mu)) ./ vec(so.sigma)
            zmod[t - nw, :] = lev .- vec(d.Xs[t, :]' * Matrix(m["c"])') .- d.mueta
            lr.q_hist[1:NQ, 1] .= d.Y[t, :]            # teacher forcing: the record's level
        end
    end
    Zd = d.Z[(nw + 1):end, :]
    werr = maximum(abs.(warm .- permutedims(d.Z[nw:-1:(nw - p + 1), :])))
    @printf(io, "  warm-start state vs the data residual at the last %d warm-up steps: max |diff| %.2e (scaled; sd(z) %.2e-%.2e)\n",
            p, werr, extrema(sqrt.(f.s2))...)
    lags = (1, 2, 5, 10, 20)
    @printf(io, "    %-9s %-31s %-31s %-31s %8s %8s\n", "QoI", "model noise ACF 1 2 5 10 20", "fitted AR ACF",
            "residual ACF", "sd m/res", "xi_d/σξ")
    Xid = Zd[3:end, :] .- f.p1' .* Zd[2:(end - 1), :] .- f.p2' .* Zd[1:(end - 2), :]
    rows = []
    for i in 1:NQ
        am = acf(zmod[:, i], lags)
        aa = ar_acf(f.p1[i], f.p2[i], 20)[[l + 1 for l in lags]]
        ad = acf(Zd[:, i], lags)
        sr = std(zmod[:, i]) / std(Zd[:, i])
        xr = std(Xid[:, i]) / sqrt(f.sxi2[i])
        @printf(io, "    %-9s %s %s %s %8.3f %8.3f\n", LABELS[i], fmt(am), fmt(aa), fmt(ad), sr, xr)
        push!(rows, (; qoi = LABELS[i], model = am, ar = aa, data = ad, sd_ratio = sr, xi_ratio = xr))
    end
    println(io, "    (model noise: its own draws minus the deployed mean, one 9 TU realisation, so its ACF carries",
            " sampling error of order sqrt(τ/9 TU); sd m/res = marginal spread; xi_d/σξ = the DATA's one-step",
            " innovation sd under the fitted φ against the model's σ_ξ, i.e. one-step spread calibration)")
    return (; werr, rows)
end

function build(src, p; io = stdout, report_only = false, lag1 = false, nonneg = false, eyw = false, ls = false,
               noisy = false)
    @assert (!lag1 && !nonneg && !eyw) || p == 2
    @assert !ls || p == 1
    @assert !noisy || (p == 2 && nonneg && !lag1 && !eyw) "the noisy-residual fit is the restricted AR(2), 2nw"
    dst = "$(src)_ar$(p)" * (lag1 ? "c" : "") * (eyw ? "e" : "") * (nonneg ? "n" : "") * (ls ? "ls" : "") *
          (noisy ? "w" : "")
    ddir = joinpath(LRS, dst)
    report_only || !isdir(ddir) || error("$ddir exists; refusing to overwrite (--report re-prints it)")
    # a noise-trained source: its own fit rows, those of its self-consistent Σ
    fitr = noisy ? guard(range(load(joinpath(LRS, src, "LinReg.jld2"), "noise_fit_rows")...)) : guard(steps_of(1, 10))
    d = scaled_resid(src, fitr)
    f = noisy ? fit_ar_noisy(d, p; nonneg) : fit_ar(d.Z, p; lag1, nonneg, eyw, ls)
    Sw = Matrix(cov(d.m["stoch_distr"]))
    println(io, "\n==== $dst: AR($p) on $(src)'s scaled residual", noisy ? " GIVEN THE NOISY HISTORY (exact)" : "",
            ", steps $(first(fitr))-$(last(fitr)) (1-10 TU)")
    noisy && @printf(io, "  Γ_0 against the source's self-consistent Σ: rel. %.1e\n", f.e0)
    @printf(io, "  %-9s %7s %7s %7s %7s | %13s | %9s %9s %11s\n", "QoI", "rho1", noisy ? "clean" : "", "phi1", "phi2",
            "|poles|", "var(z)", "Σ_white", "σ_ξ/sd(z)")
    for i in 1:NQ
        r1 = noisy ? f.racf[2, i] : acf(d.Z[:, i], 1:2)[1]
        rc = noisy ? @sprintf("%7.3f", f.racf_clean[2, i]) : " "^7
        pl = ar_poles(f.p1[i], f.p2[i])
        @printf(io, "  %-9s %7.3f %s %7.3f %7.3f | %6.3f %6.3f | %9.3e %9.3e %11.3f\n", LABELS[i], r1, rc, f.p1[i],
                f.p2[i], pl..., f.s2[i], Sw[i, i], sqrt(f.sxi2[i] / f.s2[i]))
    end
    if noisy
        lags = (1, 2, 5, 10, 20)
        println(io, "  ACF at lags ", join(lags, " "), ": the residual given the noisy history | the fitted AR | the clean residual")
        for i in 1:NQ
            @printf(io, "  %-9s %s | %s | %s\n", LABELS[i], fmt(f.racf[[l + 1 for l in lags], i]),
                    fmt(ar_acf(f.p1[i], f.p2[i], 20)[[l + 1 for l in lags]]), fmt(f.racf_clean[[l + 1 for l in lags], i]))
        end
    end
    @printf(io, "  innovation lag-0 correlation (off-diagonal range) %.2f .. %.2f\n",
            extrema(f.R[i, j] for i in 1:NQ, j in 1:NQ if i != j)...)
    pw = noisy ? nothing : power_report(d.Z, f, Sw, vec(d.m["scaling"].out_scaling.sigma); io)
    if report_only
        phi2, S2 = load(joinpath(ddir, "LinReg.jld2"), "ar_phi", "ar_sigma_xi")
        @assert phi2 == f.phi && S2 == f.S "the file's AR differs from a refit"
        println(io, "  (report only: $ddir holds exactly this fit)")
        sane = offline_sanity(joinpath(ddir, "LinReg.jld2"), src, f; io)
        return (; dst, phi = f.phi, S = f.S, power = pw, sane)
    end
    # write: byte copy + added keys
    mkpath(ddir)
    cp(joinpath(LRS, src, "LinReg.jld2"), joinpath(ddir, "LinReg.jld2"))
    cp(joinpath(LRS, src, "parameters.jld2"), joinpath(ddir, "parameters.jld2"))
    prov = (; source = src, order = p,
            method = lag1 ? "ar2_ls_lag1: lag 1 exact, LS on residual ACF lags 2-20" :
                     eyw ? "ar2_eyw: overdetermined Yule-Walker on residual ACF lags 1-20, stationary" *
                           (nonneg ? ", restricted to phi1 >= 0, phi2 <= 0" : "") :
                     nonneg ? "ar2_ls on residual ACF lags 1-20, restricted to phi1 >= 0, phi2 <= 0" :
                     p == 2 ? "ar2_ls on residual ACF lags 1-20" :
                     ls ? "ar1_ls: AR(1) by LS on residual ACF lags 1-20, phi in [0, 1)" : "AR(1) at residual lag 1",
            residual = noisy ? "GIVEN THE NOISY HISTORY: exact lagged covariances, the clean residual's plus the " *
                               "response to the white window noise of the self-consistent Sigma (fit_ar_noisy)" :
                               "the recorded history",
            sigma = noisy ? "exact innovation covariance's correlation, marginal variance matched to diag Sigma (self-consistent)" :
                            "innovation lag-0 correlation, marginal variance matched to var(z)",
            fit_steps = (first(fitr), last(fitr)), units = "scaled (out_scaling), minus mean(stoch_distr)",
            record = "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2",
            built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_ar_variant.jl")
    jldopen(joinpath(ddir, "LinReg.jld2"), "a+") do fh
        fh["ar_phi"] = f.phi
        fh["ar_sigma_xi"] = f.S
        fh["ar_provenance"] = prov
    end
    # round trip
    phi2, S2 = load(joinpath(ddir, "LinReg.jld2"), "ar_phi", "ar_sigma_xi")
    @assert phi2 == f.phi && S2 == f.S
    a = load(joinpath(LRS, src, "LinReg.jld2")); b = load(joinpath(ddir, "LinReg.jld2"))
    for k in keys(a)
        @assert isequal(a[k], b[k]) || (k == "stoch_distr" && a[k].μ == b[k].μ && Matrix(a[k].Σ) == Matrix(b[k].Σ)) "key $k differs"
    end
    println(io, "  wrote $ddir (source keys identical, + ar_phi, ar_sigma_xi, ar_provenance)")
    println(io, "  offline sanity (deployed RikFlow.LinReg on the new file, teacher-forced 1-10 TU):")
    sane = offline_sanity(joinpath(ddir, "LinReg.jld2"), src, f; io)
    return (; dst, phi = f.phi, S = f.S, power = pw, sane)
end

using Dates
if abspath(PROGRAM_FILE) == @__FILE__
    report_only = "--report" in ARGS
    args = filter(!=("--report"), ARGS)
    src = isempty(args) ? "LinReg7" : args[1]
    ps = length(args) > 1 ? args[2:end] : ["2", "1"]
    # "2c" = AR(2) with lag 1 exact; "2n" = AR(2) restricted to phi1 >= 0, phi2 <= 0; "2e" = overdetermined
    # Yule-Walker (`ar2_eyw`), "2en" the same restricted; "1ls" = AR(1) by `ar1_ls`
    for p in ps
        flags = lstrip(p, collect('0':'9'))
        @assert flags in ("", "c", "n", "e", "en", "ls", "nw") "unknown variant $p"
        build(src, parse(Int, p[1:(end - length(flags))]); report_only, lag1 = 'c' in flags,
              nonneg = 'n' in flags, eyw = 'e' in flags, ls = flags == "ls", noisy = 'w' in flags)
    end
end
