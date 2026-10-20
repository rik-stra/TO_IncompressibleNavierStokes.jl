# Step 2 builder (step 4p before 2026-10-13): a deployed LinReg + the power-law scale of its residual (paper Sec. 6.3, Eq. power law;
# 2026-10-07, results_LSTMS.md §17).
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/lrs_scale_variant.jl [src] [--report]
#
# default `LinReg1`: writes `output/TO_LRS/LinReg1_pl`. `--report` re-fits, checks the existing file
# holds exactly that fit, and writes nothing. The new directory holds a byte copy of
# `<src>/LinReg.jld2` with keys ADDED (`scale_beta`, `scale_qref`, `scale_qclip`, `scale_sigma_eps`,
# `scale_logvar`, `scale_provenance`) -- the mean, Σ, scaling and every other key untouched -- and a
# copy of `parameters.jld2`.
#
# The fit, per QoI i, on the source model's OWN residual on its training window (1-10 TU, steps
# 400-4000 of the R1 tracked cache), evaluated exactly as deployed (`scaled_resid`: out_scaling units,
# minus μ_η), with the state q*_i^n in physical units:
#   * β_i and the variance at q_ref,i by maximum likelihood, `RF.fit_powerlaw_scale` (src/ts_scale.jl);
#     q_ref,i = geometric mean of q*_i, clip range = its training range;
#   * Σ_ε = the (zero-mean) covariance of the residual divided by f(q*) on the same rows, so
#     diag Σ_ε = exp(logvar) (asserted).
# 🔒 Nothing past 10 TU is read here (`guard`).
#
#     julia ... lrs_scale_variant.jl LinReg7 --ar2n [--report]
#
# 🔑 `--ar2n` (2026-10-16, paper Eq. power law AR, step 2 on LinReg7 + AR(2)): writes `<src>_ar2n_pl`
# from a WHITE source: the scale keys exactly as `<src>_pl` (same fit), plus an AR(2) fitted as
# `lrs_ar_variant.jl <src> 2n` (`fit_ar(...; nonneg = true)`: ACF lags 1-20, φ1 ≥ 0, φ2 ≤ 0) but to the
# residual divided by f(q*), with Σ_ξ = D R D and the stationary variance of z matched to var(Z ./ f).
# Keys added: the six `scale_*` and `ar_phi`, `ar_sigma_xi`, `ar_provenance`.

include(joinpath(@__DIR__, "lrs_ar_variant.jl"))   # scaled_resid, guard, steps_of, LRS, NQ, LABELS, RF; main guarded
using Dates

"The power-law scale of residual `Z` (rows = steps) with state `qs` (physical q*, same rows)."
function fit_scale(Z, qs)
    fits = [RF.fit_powerlaw_scale(Z[:, i], qs[:, i]) for i in 1:NQ]
    beta = [f.beta for f in fits]
    qref = [f.qref for f in fits]
    qclip = reduce(vcat, [[f.qclip[1] f.qclip[2]] for f in fits])
    F = permutedims(reduce(hcat, [RF.powerlaw_factor(view(qs, t, :), beta, qref, qclip) for t in axes(qs, 1)]))
    Zf = Z ./ F
    S = Matrix(Symmetric(Zf' * Zf ./ size(Zf, 1)))
    logvar = [f.logvar for f in fits]
    @assert diag(S) ≈ exp.(logvar) rtol = 1e-6 "diag Σ_ε is not the ML variance at q_ref"
    return (; beta, qref, qclip, S, logvar, F)
end

"Gaussian NLL per step and QoI of `Z` under a constant variance and under the power law."
function nll_gain(Z, F, logvar)
    nll(e, v) = mean(0.5 .* (log(2pi) .+ log.(v) .+ e .^ 2 ./ v))
    return [nll(Z[:, i], fill(mean(abs2, Z[:, i]), size(Z, 1))) - nll(Z[:, i], exp(logvar[i]) .* F[:, i] .^ 2)
            for i in 1:NQ]
end

"""
Offline sanity: the deployed `RikFlow.LinReg` on the new file draws `η` at every training row's q*;
its draws divided by f(q*) must have covariance Σ_ε, and their sd per quintile of the level must follow
the data's.
"""
function scale_sanity(file, d, f; io, seed = 1)
    lr = RF.LinReg(file, Xoshiro(seed), Array; q_hist = zeros(2NQ, d.m["hist_len"]),
                   spinnup_data = zeros(NQ, d.m["hist_len"]))
    qs = d.X[:, 1:NQ]
    D = permutedims(reduce(hcat, [RF.draw_eta(lr, qs[t, :]) for t in axes(qs, 1)]))
    Dm = D .- lr.scale.mu_eta'
    Sd = cov(Dm ./ f.F; corrected = false)
    @printf(io, "  draws / f: max |cov - Σ_ε| / max diag Σ_ε = %.3f  (sampling, %d draws)\n",
            maximum(abs, Sd .- f.S) / maximum(diag(f.S)), size(D, 1))
    println(io, "  sd by quintile of q*, lowest..highest, relative to the QoI's overall sd: model draws | data residual")
    for i in 1:NQ
        edges = quantile(qs[:, i], 0.2:0.2:0.8)
        b = searchsortedfirst.(Ref(edges), qs[:, i])
        rm = [std(Dm[b .== k, i]) for k in 1:5] ./ std(Dm[:, i])
        rd = [std(d.Z[b .== k, i]) for k in 1:5] ./ std(d.Z[:, i])
        @printf(io, "    %-9s %s | %s\n", LABELS[i], fmt(rm), fmt(rd))
    end
end

function build_scale(src; io = stdout, report_only = false)
    dst = "$(src)_pl"
    ddir = joinpath(LRS, dst)
    report_only || !isdir(ddir) || error("$ddir exists; refusing to overwrite (--report re-prints it)")
    fitr = guard(steps_of(1, 10))
    d = scaled_resid(src, fitr)
    isnothing(get(d.m, "ar_phi", nothing)) || error("$src carries an AR residual; the power-law scale is for white LinRegs")
    qs = d.X[:, 1:NQ]                                    # q*^n, physical units
    f = fit_scale(d.Z, qs)
    g = nll_gain(d.Z, f.F, f.logvar)
    println(io, "\n==== $dst: power-law scale of $(src)'s residual, steps $(first(fitr))-$(last(fitr)) (1-10 TU)")
    @printf(io, "  %-9s %7s %11s %17s %10s %12s\n", "QoI", "beta", "q_ref", "clip / q_ref", "sd at ref", "NLL gain")
    for i in 1:NQ
        @printf(io, "  %-9s %7.3f %11.4e %8.3f-%-8.3f %10.3e %12.4f\n", LABELS[i], f.beta[i], f.qref[i],
                f.qclip[i, 1] / f.qref[i], f.qclip[i, 2] / f.qref[i], sqrt(f.S[i, i]), g[i])
    end
    @printf(io, "  NLL gain over the constant variance on the fit window: %.4f nats/step (sum over QoIs)\n", sum(g))
    if report_only
        b2, q2, c2, S2 = load(joinpath(ddir, "LinReg.jld2"), "scale_beta", "scale_qref", "scale_qclip", "scale_sigma_eps")
        # Tolerant, not `==`: a refit on another machine or BLAS differs in the last bits, and the Newton
        # stop of `fit_powerlaw_scale` can fire a few iterations early (|Δβ| ~ 3e-9; review 2026-10-15).
        db = maximum(abs, b2 .- f.beta)
        dq = max(maximum(abs, q2 ./ f.qref .- 1), maximum(abs, c2 ./ f.qclip .- 1))
        dS = maximum(abs, S2 .- f.S) / maximum(diag(f.S))
        db < 1e-6 && dq < 1e-10 && dS < 1e-6 ||
            error(@sprintf("the file's scale differs from a refit: |Δβ| %.1e, q_ref/clip rel. %.1e, Σ_ε rel. %.1e", db, dq, dS))
        @printf(io, "  (report only: %s holds this fit: |Δβ| %.1e, q_ref/clip rel. %.1e, Σ_ε rel. %.1e)\n", ddir, db, dq, dS)
        scale_sanity(joinpath(ddir, "LinReg.jld2"), d, f; io)
        return (; dst, f, gain = g)
    end
    mkpath(ddir)
    cp(joinpath(LRS, src, "LinReg.jld2"), joinpath(ddir, "LinReg.jld2"))
    cp(joinpath(LRS, src, "parameters.jld2"), joinpath(ddir, "parameters.jld2"))
    prov = (; source = src, form = "eta = mu_eta + f(q*) .* eps, f_i = (clamp(q*_i, clip)/q_ref_i)^beta_i",
            fit = "RF.fit_powerlaw_scale per QoI (ML, mean fixed); Sigma_eps = zero-mean cov of Z ./ f",
            state = "q*^n (the predictor), physical units", fit_steps = (first(fitr), last(fitr)),
            units = "scaled (out_scaling), minus mean(stoch_distr)",
            record = "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2",
            built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_scale_variant.jl")
    jldopen(joinpath(ddir, "LinReg.jld2"), "a+") do fh
        fh["scale_beta"] = f.beta
        fh["scale_qref"] = f.qref
        fh["scale_qclip"] = f.qclip
        fh["scale_sigma_eps"] = f.S
        fh["scale_logvar"] = f.logvar
        fh["scale_provenance"] = prov
    end
    b2, q2, c2, S2 = load(joinpath(ddir, "LinReg.jld2"), "scale_beta", "scale_qref", "scale_qclip", "scale_sigma_eps")
    @assert b2 == f.beta && q2 == f.qref && c2 == f.qclip && S2 == f.S
    a = load(joinpath(LRS, src, "LinReg.jld2")); b = load(joinpath(ddir, "LinReg.jld2"))
    for k in keys(a)
        @assert isequal(a[k], b[k]) || (k == "stoch_distr" && a[k].μ == b[k].μ && Matrix(a[k].Σ) == Matrix(b[k].Σ)) "key $k differs"
    end
    println(io, "  wrote $ddir (source keys identical, + scale_*)")
    println(io, "  offline sanity (deployed RikFlow.LinReg on the new file, draws at the training rows' q*):")
    scale_sanity(joinpath(ddir, "LinReg.jld2"), d, f; io)
    return (; dst, f, gain = g)
end

"Tolerant comparison of a stored scale with a refit (a refit on another machine differs in the last bits)."
function scale_check(file, f; io)
    b2, q2, c2, S2 = load(file, "scale_beta", "scale_qref", "scale_qclip", "scale_sigma_eps")
    db = maximum(abs, b2 .- f.beta)
    dq = max(maximum(abs, q2 ./ f.qref .- 1), maximum(abs, c2 ./ f.qclip .- 1))
    dS = maximum(abs, S2 .- f.S) / maximum(diag(f.S))
    db < 1e-6 && dq < 1e-10 && dS < 1e-6 ||
        error(@sprintf("the file's scale differs from a refit: |Δβ| %.1e, q_ref/clip rel. %.1e, Σ_ε rel. %.1e", db, dq, dS))
    @printf(io, "  scale holds this fit: |Δβ| %.1e, q_ref/clip rel. %.1e, Σ_ε rel. %.1e\n", db, dq, dS)
end

"""
Offline sanity of `<src>_ar2n_pl`: the deployed `RikFlow.LinReg` on the file, teacher-forced on 1-10 TU
as `offline_sanity` (warm-up = the 100 replayed rows before 400), with the AR process recovered as the
model's residual divided by f(q*): its ACF against the AR's and the data's (Z ./ f), its spread, the
data's one-step innovation sd against σ_ξ, and the warm-start state against the data's Z ./ f.
"""
function ar_scale_sanity(file, src, f, a; io, seed = 1)
    nw = 100
    r = (first(steps_of(1, 10)) - nw):last(steps_of(1, 10))
    d = scaled_resid(src, r)
    m = d.m
    qs = d.X[:, 1:NQ]
    dQrec = d.Y .- qs
    lr = RF.LinReg(file, Xoshiro(seed), Array; q_hist = zeros(2NQ, m["hist_len"]), spinnup_data = permutedims(dQrec[1:nw, :]))
    @assert !isnothing(lr.scale) && !isnothing(lr.ar) "the deployed LinReg should carry both the scale and the AR"
    F = permutedims(reduce(hcat, [RF.powerlaw_factor(view(qs, t, :), f.beta, f.qref, f.qclip) for t in axes(qs, 1)]))
    Zf = d.Z ./ F
    so = m["scaling"].out_scaling
    N = size(qs, 1)
    zmod = zeros(N - nw, NQ)
    warm = zeros(NQ, 0)
    for t in 1:N
        dQ = RF.get_next_item_timeseries(lr, qs[t, :])
        t == nw && (warm = copy(lr.ar.z))
        if t > nw
            lev = (qs[t, :] .+ dQ .- vec(so.mu)) ./ vec(so.sigma)
            zmod[t - nw, :] = (lev .- vec(d.Xs[t, :]' * Matrix(m["c"])') .- d.mueta) ./ F[t, :]
            lr.q_hist[1:NQ, 1] .= d.Y[t, :]            # teacher forcing: the record's level
        end
    end
    werr = maximum(abs.(warm .- permutedims(Zf[nw:-1:(nw - 1), :])))
    @printf(io, "  warm-start state vs the data's Z ./ f at the last 2 warm-up steps: max |diff| %.2e\n", werr)
    Zd = Zf[(nw + 1):end, :]
    lags = (1, 2, 5, 10, 20)
    @printf(io, "    %-9s %-31s %-31s %-31s %8s %8s\n", "QoI", "model z ACF 1 2 5 10 20", "fitted AR ACF",
            "data Z/f ACF", "sd m/dat", "xi_d/σξ")
    Xid = Zd[3:end, :] .- a.p1' .* Zd[2:(end - 1), :] .- a.p2' .* Zd[1:(end - 2), :]
    for i in 1:NQ
        @printf(io, "    %-9s %s %s %s %8.3f %8.3f\n", LABELS[i], fmt(acf(zmod[:, i], lags)),
                fmt(ar_acf(a.p1[i], a.p2[i], 20)[[l + 1 for l in lags]]), fmt(acf(Zd[:, i], lags)),
                std(zmod[:, i]) / std(Zd[:, i]), std(Xid[:, i]) / sqrt(a.sxi2[i]))
    end
    return werr
end

"Step 2 on the coloured residual: `<src>_ar2n_pl` (see the header)."
function build_scale_ar(src; io = stdout, report_only = false)
    dst = "$(src)_ar2n_pl"
    ddir = joinpath(LRS, dst)
    report_only || !isdir(ddir) || error("$ddir exists; refusing to overwrite (--report re-prints it)")
    fitr = guard(steps_of(1, 10))
    d = scaled_resid(src, fitr)
    isnothing(get(d.m, "ar_phi", nothing)) || error("$src carries an AR residual; give the white source (the AR is refitted to the residual / f)")
    qs = d.X[:, 1:NQ]
    f = fit_scale(d.Z, qs)
    a = fit_ar(d.Z ./ f.F, 2; nonneg = true)
    a0 = fit_ar(d.Z, 2; nonneg = true)                   # the fit of `<src>_ar2n` (step 1), for comparison
    println(io, "\n==== $dst: power-law scale of $(src)'s residual + AR(2) on residual / f, steps $(first(fitr))-$(last(fitr)) (1-10 TU)")
    @printf(io, "  %-9s %7s | %7s %7s | %13s | %15s | %9s %11s\n", "QoI", "beta", "phi1", "phi2", "|poles|",
            "phi1 phi2 on Z", "var(Z/f)", "σ_ξ/sd(Z/f)")
    for i in 1:NQ
        @printf(io, "  %-9s %7.3f | %7.3f %7.3f | %6.3f %6.3f | %7.3f %7.3f | %9.3e %11.3f\n", LABELS[i], f.beta[i],
                a.p1[i], a.p2[i], ar_poles(a.p1[i], a.p2[i])..., a0.p1[i], a0.p2[i], a.s2[i], sqrt(a.sxi2[i] / a.s2[i]))
    end
    @printf(io, "  innovation lag-0 correlation (off-diagonal range) %.2f .. %.2f\n",
            extrema(a.R[i, j] for i in 1:NQ, j in 1:NQ if i != j)...)
    file = joinpath(ddir, "LinReg.jld2")
    if report_only
        scale_check(file, f; io)
        phi2, S2 = load(file, "ar_phi", "ar_sigma_xi")
        dp, dS = maximum(abs, phi2 .- a.phi), maximum(abs, S2 .- a.S) / maximum(diag(a.S))
        dp < 1e-6 && dS < 1e-6 || error(@sprintf("the file's AR differs from a refit: |Δφ| %.1e, Σ_ξ rel. %.1e", dp, dS))
        @printf(io, "  AR holds this fit: |Δφ| %.1e, Σ_ξ rel. %.1e\n", dp, dS)
        ar_scale_sanity(file, src, f, a; io)
        return (; dst, f, a)
    end
    mkpath(ddir)
    cp(joinpath(LRS, src, "LinReg.jld2"), file)
    cp(joinpath(LRS, src, "parameters.jld2"), joinpath(ddir, "parameters.jld2"))
    sprov = (; source = src, form = "eta = mu_eta + f(q*) .* z, z the AR(2) below, f_i = (clamp(q*_i, clip)/q_ref_i)^beta_i",
             fit = "RF.fit_powerlaw_scale per QoI (ML, mean fixed), as $(src)_pl; Sigma_eps = zero-mean cov of Z ./ f (not drawn with AR)",
             state = "q*^n (the predictor), physical units", fit_steps = (first(fitr), last(fitr)),
             units = "scaled (out_scaling), minus mean(stoch_distr)",
             record = "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2",
             built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_scale_variant.jl --ar2n")
    aprov = (; source = src, order = 2,
             method = "ar2_ls on the ACF (lags 1-20) of the residual divided by f(q*), restricted to phi1 >= 0, phi2 <= 0",
             sigma = "innovation lag-0 correlation, marginal variance matched to var(Z ./ f)",
             fit_steps = (first(fitr), last(fitr)), units = "unit scale: the residual (scaled, minus mean(stoch_distr)) divided by f(q*)",
             record = "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2",
             built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_scale_variant.jl --ar2n")
    jldopen(file, "a+") do fh
        fh["scale_beta"] = f.beta
        fh["scale_qref"] = f.qref
        fh["scale_qclip"] = f.qclip
        fh["scale_sigma_eps"] = f.S
        fh["scale_logvar"] = f.logvar
        fh["scale_provenance"] = sprov
        fh["ar_phi"] = a.phi
        fh["ar_sigma_xi"] = a.S
        fh["ar_provenance"] = aprov
    end
    b2, S2, phi2, Sx2 = load(file, "scale_beta", "scale_sigma_eps", "ar_phi", "ar_sigma_xi")
    @assert b2 == f.beta && S2 == f.S && phi2 == a.phi && Sx2 == a.S
    A = load(joinpath(LRS, src, "LinReg.jld2")); B = load(file)
    for k in keys(A)
        @assert isequal(A[k], B[k]) || (k == "stoch_distr" && A[k].μ == B[k].μ && Matrix(A[k].Σ) == Matrix(B[k].Σ)) "key $k differs"
    end
    println(io, "  wrote $ddir (source keys identical, + scale_*, ar_phi, ar_sigma_xi, ar_provenance)")
    println(io, "  offline sanity (deployed RikFlow.LinReg on the new file, teacher-forced 1-10 TU):")
    ar_scale_sanity(file, src, f, a; io)
    return (; dst, f, a)
end

if abspath(PROGRAM_FILE) == @__FILE__
    report_only = "--report" in ARGS
    ar2n = "--ar2n" in ARGS
    args = filter(a -> !(a in ("--report", "--ar2n")), ARGS)
    src = isempty(args) ? "LinReg1" : args[1]
    ar2n ? build_scale_ar(src; report_only) : build_scale(src; report_only)
end
