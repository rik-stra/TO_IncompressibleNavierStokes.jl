# Step 4p builder: a deployed LinReg + the power-law scale of its residual (paper Sec. 6.4, Eq. power law;
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
        @assert b2 == f.beta && q2 == f.qref && c2 == f.qclip && S2 == f.S "the file's scale differs from a refit"
        println(io, "  (report only: $ddir holds exactly this fit)")
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

if abspath(PROGRAM_FILE) == @__FILE__
    report_only = "--report" in ARGS
    args = filter(!=("--report"), ARGS)
    build_scale(isempty(args) ? "LinReg1" : args[1]; report_only)
end
