# M0ᶜ-ridge variant builder: a deployed LinReg + a stationary AR(p) residual (results_LSTMS §12).
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/lrs_ar_variant.jl [src] [p ...]
#
# default `LinReg7 2 1`: writes `output/TO_LRS/LinReg7_ar2` and `.../LinReg7_ar1`. `--report`
# re-prints the tables for existing variants (checking the file holds exactly the refit) and writes nothing.
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
#    by construction). p = 1: φ = the residual's lag-1 autocorrelation.
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

function fit_ar(Z, p)
    phi = zeros(p, NQ)
    for i in 1:NQ
        r = acf(Z[:, i], 0:20)
        if p == 2
            phi[:, i] .= ar2_ls(r; L = 20)
        else
            phi[1, i] = r[2]
        end
    end
    p1 = phi[1, :]
    p2 = p == 2 ? phi[2, :] : zeros(NQ)
    Xi = Z[3:end, :] .- p1' .* Z[2:(end - 1), :] .- p2' .* Z[1:(end - 2), :]
    R = cor(Xi)
    s2 = vec(var(Z; dims = 1))
    sxi2 = [begin
                rho = ar_acf(p1[i], p2[i], 2)
                s2[i] * (1 - p1[i] * rho[2] - p2[i] * rho[3])
            end for i in 1:NQ]
    D = Diagonal(sqrt.(sxi2))
    S = Matrix(Symmetric(D * R * D))
    return (; phi, p1, p2, S, R, Xi, s2, sxi2)
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

function build(src, p; io = stdout, report_only = false)
    dst = "$(src)_ar$(p)"
    ddir = joinpath(LRS, dst)
    report_only || !isdir(ddir) || error("$ddir exists; refusing to overwrite (--report re-prints it)")
    fitr = guard(steps_of(1, 10))
    d = scaled_resid(src, fitr)
    f = fit_ar(d.Z, p)
    Sw = Matrix(cov(d.m["stoch_distr"]))
    println(io, "\n==== $dst: AR($p) on $(src)'s scaled residual, steps $(first(fitr))-$(last(fitr)) (1-10 TU)")
    @printf(io, "  %-9s %7s %7s %7s | %13s | %9s %9s %11s\n", "QoI", "rho1", "phi1", "phi2", "|poles|",
            "var(z)", "Σ_white", "σ_ξ/sd(z)")
    for i in 1:NQ
        r = acf(d.Z[:, i], 1:2)
        pl = ar_poles(f.p1[i], f.p2[i])
        @printf(io, "  %-9s %7.3f %7.3f %7.3f | %6.3f %6.3f | %9.3e %9.3e %11.3f\n", LABELS[i], r[1], f.p1[i],
                f.p2[i], pl..., f.s2[i], Sw[i, i], sqrt(f.sxi2[i] / f.s2[i]))
    end
    @printf(io, "  innovation lag-0 correlation (off-diagonal range) %.2f .. %.2f\n",
            extrema(f.R[i, j] for i in 1:NQ, j in 1:NQ if i != j)...)
    pw = power_report(d.Z, f, Sw, vec(d.m["scaling"].out_scaling.sigma); io)
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
    prov = (; source = src, order = p, method = p == 2 ? "ar2_ls on residual ACF lags 1-20" : "AR(1) at residual lag 1",
            sigma = "innovation lag-0 correlation, marginal variance matched to var(z)",
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
    ps = length(args) > 1 ? parse.(Int, args[2:end]) : [2, 1]
    for p in ps
        build(src, p; report_only)
    end
end
