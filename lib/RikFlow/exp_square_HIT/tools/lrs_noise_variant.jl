# The linear analogue of input-noise training (Rik, 2026-10-26: "Build both: first LinReg1 + noise (the
# closed-form linear analogue, self-consistent Σ, exported as a LinReg closure)"; claude_memory.md, standing
# decision "input-noise training of a network mean"; the offline probe behind it: `cnn_mean_fit.jl` with
# RIKFLOW_CNN_PROBE).
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/lrs_noise_variant.jl [src] [--report]
#
# default `LinReg1`: writes `output/TO_LRS/LinReg1_nt` ("noise-trained"). `--report` refits, checks the file
# holds exactly that fit, and writes nothing. An existing target is refused.
#
# Online the closure's own draws enter its history: every correction in the window carries one η and the
# levels carry the running sum, anchored at the prediction step (`tools/step3_innoise.jl`). Least squares
# that expects this noise minimises E|y - c (x + dx)|² = |y - c x|² + tr(c S_dx c') per row, so
#
#     c' = (X'X + N S_dx(Σ)) \ X'Y,      S_dx = E[dx dx'] = M (I_h ⊗ D Σ D) M',
#
# M the exact linear map from the window's raw η to dx in the regressor's scaled units, D the output
# scaling (raw = D x scaled): a ridge whose penalty is the closure's own noise; the bias and q^{n*} are not
# penalised. Σ is SELF-CONSISTENT: the covariance of the residual given noisy history,
# Σ = cov(y - c x) + c S_dx(Σ) c' (the noise is independent of the data and zero-mean), iterated from the
# source's Σ to a fixed point; μ_η = the mean of the clean residual and Σ the MLE (/N), as the source's
# `fit(MvNormal, ...)`. Only `c` and `stoch_distr` change; every other key of the source's LinReg.jld2 is
# copied and `noise_*` keys are added, with a copy of `parameters.jld2`.
# Fit rows: the source's own, found by reproducing its `c` with no noise (asserted). 🔒 Nothing past 10 TU.

include(normpath(joinpath(@__DIR__, "..", "..", "analysis", "m0c_ridge.jl")))   # design, guard, steps_of, LRS, NQ, ...
using Dates

"The scaled design of a deployed LinReg `m` on steps `r` (rows = steps, bias last), and its scales."
function scaled_design(m, r)
    X, Y = design(m["hist_len"], guard(r))
    si, so = m["scaling"].in_scaling, m["scaling"].out_scaling
    mu, sg = vec(si.mu), vec(si.sigma)
    Xs = copy(X)
    for j in 1:(size(X, 2) - 1)                     # column j carries QoI mod1(j, NQ)
        i = mod1(j, NQ)
        Xs[:, j] = (X[:, j] .- mu[i]) ./ sg[i]
    end
    Ys = (Y .- vec(so.mu)') ./ vec(so.sigma)'
    return (; X, Xs, Ys, isg = sg, osg = vec(so.sigma))
end

"""
    window_map(h, isg) -> M

dx in the regressor's scaled units (`[q*^n; q^{n-1}, q^{n-1*}; ...; 1]`, input sds `isg`) per unit RAW
η^{n-j} of QoI i, column (j - 1) NQ + i: noise(q^{n-k}) = -(η^{n-k+1} + ... + η^{n-1}),
noise(q^{n-k*}) = -(η^{n-k} + ... + η^{n-1}); q^{n*} and the bias get none.
"""
function window_map(h, isg)
    M = zeros(NQ * (2h + 1) + 1, NQ * h)
    for j in 1:h, i in 1:NQ, k in 1:h
        q = NQ + 2(k - 1) * NQ + i                  # the q^{n-k} row of QoI i; its q^{n-k*} row is q + NQ
        k >= j + 1 && (M[q, (j - 1) * NQ + i] = -1 / isg[i])
        k >= j && (M[q + NQ, (j - 1) * NQ + i] = -1 / isg[i])
    end
    return M
end

"S_dx = E[dx dx'] for corrections of covariance Σ (scaled output units)."
window_cov(M, Sig, osg, h) = M * kron(Matrix(1.0I, h, h), Diagonal(osg) * Sig * Diagonal(osg)) * M'

"The fixed point (c', μ_η, Σ) of the noise-trained least squares, from Σ0 (see the header)."
function fit_noise(d, M, Sig0, h; tol = 1e-12, maxit = 1000)
    N = size(d.Xs, 1)
    XtX, XtY = d.Xs' * d.Xs, d.Xs' * d.Ys
    Sig, steps = Matrix{Float64}(Sig0), Float64[]
    for it in 1:maxit
        Sdx = window_cov(M, Sig, d.osg, h)
        Ct = (XtX .+ N .* Sdx) \ XtY
        r = d.Ys .- d.Xs * Ct
        Snew = Matrix(Symmetric(cov(r; corrected = false) .+ Ct' * Sdx * Ct))
        push!(steps, maximum(abs, Snew .- Sig) / maximum(diag(Sig)))
        Sig = Snew
        if last(steps) < tol
            Sdx = window_cov(M, Sig, d.osg, h)
            Ct = (XtX .+ N .* Sdx) \ XtY
            r = d.Ys .- d.Xs * Ct
            return (; Ct, mu = vec(mean(r; dims = 1)), Sig, Sdx, r, iterations = it, steps)
        end
    end
    error("the self-consistent Σ did not converge in $maxit iterations (last change $(last(steps)))")
end

"""
Per QoI, in RAW units over `ref` (a raw rms): the clean rms one-step residual, the rms with the window noise
of the model's own Σ, and the gain sqrt(diag(c S_dx c') / diag Σ), the share of one step's η that reaches the
mean. Exact (no draws): the noise is independent of the data.
"""
function noise_report(name, c, Sig, d, M, h; ref, io)
    r = d.Ys .- d.Xs * Matrix(c)'
    resp = Matrix(c) * window_cov(M, Sig, d.osg, h) * Matrix(c)'
    clean = vec(sqrt.(mean(abs2, r; dims = 1))) .* d.osg
    noisy = sqrt.(vec(mean(abs2, r; dims = 1)) .+ diag(resp)) .* d.osg
    f3(v) = join((@sprintf("%.3f", x) for x in v), " ")
    @printf(io, "  %-14s clean / ref %s | noisy / ref %s | gain %s\n", name, f3(clean ./ ref), f3(noisy ./ ref),
            f3(sqrt.(diag(resp) ./ diag(Sig))))
    return clean
end

function build_noise(src; io = stdout, report_only = false)
    dst = "$(src)_nt"
    ddir = joinpath(LRS, dst)
    report_only || !isdir(ddir) || error("$ddir exists; refusing to overwrite (--report re-prints it)")
    m = load(joinpath(LRS, src, "LinReg.jld2"))
    @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
    isnothing(get(m, "ar_phi", nothing)) && isnothing(get(m, "scale_beta", nothing)) ||
        error("$src carries an AR or a scale; the noise-trained fit is for white LinRegs")
    h = m["hist_len"]
    # the source's own fit rows: the window whose least squares IS its c
    c0 = Matrix(m["c"])
    found = nothing
    for r in (405:3999, steps_of(1, 10), 400:3999, 405:4000)
        d = scaled_design(m, r)
        e = maximum(abs, (d.Xs \ d.Ys)' .- c0) / maximum(abs, c0)
        @printf(io, "  rows %d-%d: least squares vs %s's c, max rel. diff %.1e\n", first(r), last(r), src, e)
        e < 1e-8 && (found = (r, d); break)
    end
    found === nothing && error("no candidate window reproduces $(src)'s c; its fit rows are unknown")
    fitr, d = found
    # the regressor layout the window map assumes, on the first fit row (n = first(fitr))
    n = first(fitr)
    d.X[1, 1:NQ] == QSR[:, n] && d.X[1, (NQ + 1):(2NQ)] == QR[:, n] && d.X[1, (2NQ + 1):(3NQ)] == QSR[:, n - 1] ||
        error("the design is not [q*^n; q^{n-1}, q^{n-1*}; ...]: the window map would be wrong")
    sd0 = m["stoch_distr"]
    r0 = d.Ys .- d.Xs * c0'
    @printf(io, "  %s's stoch_distr vs the MLE of its residual on these rows: |Δμ| %.1e, |ΔΣ| / max diag %.1e\n", src,
            maximum(abs, mean(sd0) .- vec(mean(r0; dims = 1))),
            maximum(abs, Matrix(cov(sd0)) .- cov(r0; corrected = false)) / maximum(diag(Matrix(cov(sd0)))))
    M = window_map(h, d.isg)
    f = fit_noise(d, M, Matrix(cov(sd0)), h)
    println(io, "\n==== $dst: $src trained with its own window noise, rows $(first(fitr))-$(last(fitr)) (1-10 TU)")
    @printf(io, "  self-consistent Σ in %d iterations (changes %s)\n", f.iterations,
            join((@sprintf("%.1e", s) for s in f.steps[unique([1, 2, 3, 5, 10, length(f.steps)] ∩ eachindex(f.steps))]), ", "))
    @printf(io, "  sd(η) / %s's: %s\n", src, fmt(sqrt.(diag(f.Sig) ./ diag(Matrix(cov(sd0)))); d = 3))
    @printf(io, "  max |c - c_%s| / max |c_%s| = %.3f\n", src, src, maximum(abs, f.Ct' .- c0) / maximum(abs, c0))
    println(io, "  per QoI, raw units over $(src)'s clean rms (", join(LABELS, " "), "):")
    ref = vec(sqrt.(mean(abs2, r0; dims = 1))) .* d.osg
    noise_report(src, c0, Matrix(cov(sd0)), d, M, h; ref, io)
    for o in ("LinReg7",)
        isfile(joinpath(LRS, o, "LinReg.jld2")) || continue
        mo = load(joinpath(LRS, o, "LinReg.jld2"))
        mo["hist_len"] == h || continue
        doo = scaled_design(mo, fitr)
        noise_report(o, Matrix(mo["c"]), Matrix(cov(mo["stoch_distr"])), doo, window_map(h, doo.isg), h;
                     ref = ref, io)
    end
    noise_report(dst, f.Ct', f.Sig, d, M, h; ref, io)
    if report_only
        b = load(joinpath(ddir, "LinReg.jld2"))
        dc = maximum(abs, Matrix(b["c"]) .- f.Ct') / maximum(abs, f.Ct)
        dS = maximum(abs, Matrix(cov(b["stoch_distr"])) .- f.Sig) / maximum(diag(f.Sig))
        dc < 1e-9 && dS < 1e-9 || error(@sprintf("the file differs from a refit: c rel. %.1e, Σ rel. %.1e", dc, dS))
        @printf(io, "  (report only: %s holds this fit: c rel. %.1e, Σ rel. %.1e)\n", ddir, dc, dS)
        return (; dst, f)
    end
    a = load(joinpath(LRS, src, "LinReg.jld2"))
    b = copy(a)
    b["c"] = Matrix(f.Ct)'                                      # the source's type: Adjoint of a Matrix
    b["stoch_distr"] = RF.Distributions.MvNormal(f.mu, f.Sig)
    b["noise_sigma_dx"] = f.Sdx
    b["noise_fit_rows"] = (first(fitr), last(fitr))
    b["noise_provenance"] = (; source = src,
        form = "c' = (X'X + N S_dx(Sigma)) \\ X'Y, S_dx = M (I_h kron D Sigma D) M'; Sigma = cov(y - c x) + c S_dx c' (fixed point)",
        noise = "the closure's window noise, anchored at the prediction step (tools/step3_innoise.jl)",
        iterations = f.iterations, units = "scaled (in/out_scaling of the source)",
        record = "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2",
        built = string(Dates.now()), script = "exp_square_HIT/tools/lrs_noise_variant.jl")
    mkpath(ddir)
    jldopen(joinpath(ddir, "LinReg.jld2"), "w") do fh
        for (k, v) in b
            fh[k] = v
        end
    end
    cp(joinpath(LRS, src, "parameters.jld2"), joinpath(ddir, "parameters.jld2"))
    b2 = load(joinpath(ddir, "LinReg.jld2"))
    @assert Matrix(b2["c"]) == f.Ct' && mean(b2["stoch_distr"]) == f.mu && Matrix(cov(b2["stoch_distr"])) == f.Sig
    for k in keys(a)
        k in ("c", "stoch_distr") || @assert isequal(a[k], b2[k]) "key $k differs from the source"
    end
    lr = RF.LinReg(joinpath(ddir, "LinReg.jld2"), Xoshiro(1), Array; q_hist = zeros(2NQ, h), spinnup_data = zeros(NQ, h))
    @assert isnothing(lr.ar) && isnothing(lr.scale) && Matrix(lr.c) == f.Ct'
    println(io, "  wrote $ddir (source keys identical except c, stoch_distr; + noise_*); the deployed RikFlow.LinReg loads it")
    return (; dst, f)
end

if abspath(PROGRAM_FILE) == @__FILE__
    report_only = "--report" in ARGS
    args = filter(!=("--report"), ARGS)
    build_noise(isempty(args) ? "LinReg1" : args[1]; report_only)
end
