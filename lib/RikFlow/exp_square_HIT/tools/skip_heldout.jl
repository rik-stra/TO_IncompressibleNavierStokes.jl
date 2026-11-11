# Step 5 on other skips (card analysis/cards/step5_skips.md, "the held-out stage 1 on another skip"; Rik, 2026-11-11, via
# the paper session: "start step 5 on both linreg1_nt and LinReg7"). The generator's stage 1 (settings, early stop,
# offline check) must not see the early-stopping rows (8.2-10 TU) through its skip either, so its skip is the skip's own
# RECIPE refitted on the stage-1 fit rows:
#   - a white least-squares / ridge LinReg (LinReg1: λ = 0; LinReg7: λ = 1, bias unpenalised): the same ridge solve as
#     `5_train_LinReg.jl` on the fit rows, its MLE Gaussian (μ, Σ / N) as the source's `fit(MvNormal, ...)`;
#   - a noise-trained LinReg (`*_nt`, `lrs_noise_variant.jl`): its closed form c' = (X'X + λP + N S_dx(Σ)) \ X'Y with
#     the self-consistent Σ on the fit rows, from the source's Σ, exactly as `build_noise` does on 1-10 TU.
# For LinReg1 this is least squares on the fit rows, step 5's stage 1 (`cnn_mean_fit.jl` asserts it).
#
#     julia --startup-file=no --project=. exp_square_HIT/tools/skip_heldout.jl <skip> <first step> <last step> [--report]
#
# (from lib/RikFlow; record steps n, q^{n*} = column n of q_star, as `step3_data.jl`'s `cols`). Writes
# `output/TO_LRS/<skip>_s1` (not deployed): the skip's LinReg.jld2 with `c` and `stoch_distr` replaced (and, for a
# noise-trained skip, `noise_sigma_dx` / `noise_fit_rows`), plus `s1_fit_rows` and `s1_provenance`; a copy of
# `parameters.jld2`. First the recipe is checked: it must reproduce the skip's own `c` on the skip's own fit rows. An
# existing target is refused; `--report` refits, checks the file holds that fit, and writes nothing.

include(joinpath(@__DIR__, "lrs_noise_variant.jl"))   # scaled_design, window_map, fit_noise, ridge_ls, LRS, NQ, ...

"The fit `(c', μ, Σ)` of the skip's recipe on rows `r`, and the recipe's name."
function recipe_fit(m, r; lam, noise, Sig0 = nothing)
    d = scaled_design(m, r)
    if noise
        f = fit_noise(d, window_map(m["hist_len"], d.isg), Sig0, m["hist_len"]; lam)
        return (; Ct = f.Ct, mu = f.mu, Sig = f.Sig, Sdx = f.Sdx, d)
    end
    Ct = ridge_ls(d, lam)
    res = d.Ys .- d.Xs * Ct
    return (; Ct, mu = vec(mean(res; dims = 1)), Sig = Matrix(Symmetric(cov(res; corrected = false))), Sdx = nothing, d)
end

function build_s1(skip, fr; io = stdout, report_only = false)
    dst = "$(skip)_s1"
    ddir = joinpath(LRS, dst)
    report_only || !isdir(ddir) || error("$ddir exists; refusing to overwrite (--report re-prints it)")
    m = load(joinpath(LRS, skip, "LinReg.jld2"))
    @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
    isnothing(get(m, "ar_phi", nothing)) && isnothing(get(m, "scale_beta", nothing)) ||
        error("$skip carries an AR or a scale; the held-out refit is for white LinRegs")
    h = m["hist_len"]
    noise = haskey(m, "noise_provenance")
    if noise
        prov = m["noise_provenance"]
        src = prov.source
        lam = float(get(prov, :lambda, 0.0))       # LinReg1_nt's provenance predates the ridge source: λ = 0
        Sig0 = Matrix(cov(load(joinpath(LRS, src, "LinReg.jld2"))["stoch_distr"]))
        own = UnitRange(m["noise_fit_rows"]...)
        recipe = "noise-trained closed form of $src (lambda = $lam), self-consistent Sigma from $(src)'s"
    else
        p = load(joinpath(LRS, skip, "parameters.jld2"))["parameters"]
        lam = hasproperty(p, :lambda) ? float(p.lambda) : 0.0
        Sig0 = nothing
        own = nothing
        for r in (405:3999, steps_of(1, 10), 400:3999, 405:4000)   # the skip's own fit rows, as build_noise finds them
            e = maximum(abs, ridge_ls(scaled_design(m, r), lam)' .- Matrix(m["c"])) / maximum(abs, m["c"])
            e < 1e-8 && (own = r; break)
        end
        own === nothing && error("no candidate window reproduces $(skip)'s c")
        recipe = lam == 0 ? "least squares" : "ridge lambda = $lam (bias unpenalised)"
    end
    # the recipe reproduces the skip on the skip's own rows (c, and the Gaussian)
    f0 = recipe_fit(m, own; lam, noise, Sig0)
    c0, sd0 = Matrix(m["c"]), m["stoch_distr"]
    ec = maximum(abs, f0.Ct' .- c0) / maximum(abs, c0)
    eS = maximum(abs, f0.Sig .- Matrix(cov(sd0))) / maximum(diag(Matrix(cov(sd0))))
    em = maximum(abs, f0.mu .- mean(sd0))
    @printf(io, "  %s: %s; own rows %d-%d reproduce c (rel. %.1e), Σ (rel. %.1e), μ (abs. %.1e)\n", skip, recipe,
            first(own), last(own), ec, eS, em)
    ec < 1e-8 && eS < 1e-6 && em < 1e-6 || error("the recipe does not reproduce $skip on its own rows")
    # the refit on the stage-1 rows
    f = recipe_fit(m, fr; lam, noise, Sig0)
    println(io, "\n==== $dst: $skip's recipe refitted on rows $(first(fr))-$(last(fr)) ",
            @sprintf("(%.3f-%.3f TU)", first(fr) * DT, last(fr) * DT))
    @printf(io, "  max |c - c_%s| / max |c_%s| = %.3f; sd(η) / %s's: %s\n", skip, skip,
            maximum(abs, f.Ct' .- c0) / maximum(abs, c0), skip, fmt(sqrt.(diag(f.Sig) ./ diag(Matrix(cov(sd0)))); d = 3))
    # clean rms (raw, over the skip's own clean rms on its rows) on the fit rows and on the held-out rows up to 10 TU
    ho = (last(fr) + 1):last(own)
    dho, dfr = scaled_design(m, ho), f.d
    rms(d, Ct) = vec(sqrt.(mean(abs2, d.Ys .- d.Xs * Ct; dims = 1))) .* d.osg
    ref = rms(f0.d, Matrix(c0'))
    println(io, "  clean rms / $(skip)'s on its rows (", join(LABELS, " "), "):")
    @printf(io, "    fit rows: %s %s | %s_s1 %s\n", skip, fmt(rms(dfr, Matrix(c0')) ./ ref; d = 3), skip, fmt(rms(dfr, f.Ct) ./ ref; d = 3))
    @printf(io, "    held out %d-%d: %s %s (in sample) | %s_s1 %s\n", first(ho), last(ho), skip,
            fmt(rms(dho, Matrix(c0')) ./ ref; d = 3), skip, fmt(rms(dho, f.Ct) ./ ref; d = 3))
    if report_only
        b = load(joinpath(ddir, "LinReg.jld2"))
        dc = maximum(abs, Matrix(b["c"]) .- f.Ct') / maximum(abs, f.Ct)
        dS = maximum(abs, Matrix(cov(b["stoch_distr"])) .- f.Sig) / maximum(diag(f.Sig))
        dc < 1e-9 && dS < 1e-9 && b["s1_fit_rows"] == (first(fr), last(fr)) ||
            error(@sprintf("the file differs from a refit: c rel. %.1e, Σ rel. %.1e, rows %s", dc, dS, b["s1_fit_rows"]))
        @printf(io, "  (report only: %s holds this fit: c rel. %.1e, Σ rel. %.1e)\n", ddir, dc, dS)
        return (; dst, f)
    end
    b = copy(m)
    b["c"] = Matrix(f.Ct)'                                      # the source's type: Adjoint of a Matrix
    b["stoch_distr"] = RF.Distributions.MvNormal(f.mu, f.Sig)
    if noise
        b["noise_sigma_dx"] = f.Sdx
        b["noise_fit_rows"] = (first(fr), last(fr))
    end
    b["s1_fit_rows"] = (first(fr), last(fr))
    b["s1_provenance"] = (; source = skip, recipe, lambda = lam, rows = (first(fr), last(fr)),
        purpose = "step 5's held-out stage 1 (card analysis/cards/step5_skips.md); not deployed",
        units = "scaled (in/out_scaling of the source)", built = string(Dates.now()),
        script = "exp_square_HIT/tools/skip_heldout.jl")
    mkpath(ddir)
    jldopen(joinpath(ddir, "LinReg.jld2"), "w") do fh
        for (k, v) in b
            fh[k] = v
        end
    end
    cp(joinpath(LRS, skip, "parameters.jld2"), joinpath(ddir, "parameters.jld2"))
    b2 = load(joinpath(ddir, "LinReg.jld2"))
    @assert Matrix(b2["c"]) == f.Ct' && mean(b2["stoch_distr"]) == f.mu && Matrix(cov(b2["stoch_distr"])) == f.Sig
    for k in keys(m)
        k in ("c", "stoch_distr", "noise_sigma_dx", "noise_fit_rows") || @assert isequal(m[k], b2[k]) "key $k differs from the source"
    end
    println(io, "  wrote $ddir (source keys identical except c, stoch_distr", noise ? ", noise_*" : "", "; + s1_*)")
    return (; dst, f)
end

if abspath(PROGRAM_FILE) == @__FILE__
    report_only = "--report" in ARGS
    args = filter(!=("--report"), ARGS)
    length(args) == 3 || error("usage: skip_heldout.jl <skip> <first step> <last step> [--report]")
    build_s1(args[1], parse(Int, args[2]):parse(Int, args[3]); report_only)
end
