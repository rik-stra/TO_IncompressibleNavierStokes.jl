# M0ᶜ check (ii) under ridge: residual colour of the DEPLOYED λ-ladder fits (Rik, 2026-09-28).
#
#     julia --startup-file=no --project=training analysis/m0c_ridge.jl
#
# m0c_checks.jl (ii) used λ = 0 least squares only. Here every archived `TO_LRS/LinReg<n>` fit is
# applied exactly as deployed (`time_series_methods.jl` `get_next_item_timeseries`, target :q:
# per-QoI input scaling, `c * [x_sc; 1]`, output scaling) teacher-forced on R1's tracked record,
# and the residual q^n - mu(x) is scored per QoI: ACF at lags 1, 2, 5, 10 on the fit window
# (1-10 TU, the ladder's train_range 400-4000) and on held-out 52-74 TU, plus its sd relative to
# LinReg1's, and the kernel-weighted power ratio of an AR(1)/AR(2) fitted to it. LinReg1 (λ = 0)
# must reproduce m0c_checks (ii)'s h = 5 least-squares residual: that is the check on the wiring.
# 🔒 Nothing past 74 TU is read.

include(joinpath(@__DIR__, "m0c_checks.jl"))   # design, lsfit, acf, fmt, kernel helpers, R1 cache

const LRS = normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "TO_LRS"))

function deployed_resid(name, r)
    d = joinpath(LRS, name)
    m = load(joinpath(d, "LinReg.jld2"))
    p = load(joinpath(d, "parameters.jld2"))["parameters"]
    @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
    h = m["hist_len"]
    X, Y = design(h, r)                             # raw [q*_n; q_{n-1}, q*_{n-1}; ...; 1], Y = q^n
    si, so = m["scaling"].in_scaling, m["scaling"].out_scaling
    mu, sg = vec(si.mu), vec(si.sigma)
    Xs = copy(X)
    for j in 1:(size(X, 2) - 1)                     # column j carries QoI mod1(j, NQ)
        i = mod1(j, NQ)
        Xs[:, j] = (X[:, j] .- mu[i]) ./ sg[i]
    end
    pred = (Matrix(m["c"]) * Xs')' .* vec(so.sigma)' .+ vec(so.mu)'
    lam = hasproperty(p, :lambda) ? p.lambda : 0.0
    return (; name, h, lam, R = Y .- pred)
end

function main(; io = stdout)
    names = ["LinReg1", "LinReg5", "LinReg6", "LinReg2", "LinReg7", "LinReg8", "LinReg9", "LinReg10"]
    names = filter(n -> isfile(joinpath(LRS, n, "LinReg.jld2")), names)
    fitr = steps_of(1, 10)
    ref = lsfit(5, fitr)
    K = load(joinpath(TO, "response", "response_kernel.jld2"))
    println(io, "==== residual colour of the deployed λ ladder (h, λ as fitted), R1 tracked record")
    base = nothing
    for n in names
        f = deployed_resid(n, fitr)
        fh = deployed_resid(n, WIN_HELD)
        base === nothing && (base = f)
        if n == "LinReg1"
            @printf(io, "  wiring check: LinReg1 vs Float64 LS h=5 residual, max|diff|/sd = %.2e\n",
                    maximum(abs.(f.R .- ref.R) ./ std(ref.R; dims = 1)))
        end
        @printf(io, "\n%s  h = %d  λ = %g\n", n, f.h, f.lam)
        @printf(io, "  %-9s %-26s %-26s %s\n", "band", "fit 1-10 ACF 1 2 5 10", "held 52-74 ACF 1 2 5 10",
                "sd/sd(LinReg1) fit, held")
        for i in 1:NQ
            @printf(io, "  %-9s %s   %s   %.3f %.3f\n", LABELS[i], fmt(acf(f.R[:, i], LAGS2)),
                    fmt(acf(fh.R[:, i], LAGS2)), std(f.R[:, i]) / std(base.R[:, i]),
                    std(fh.R[:, i]) / std(base.R[:, i]))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
