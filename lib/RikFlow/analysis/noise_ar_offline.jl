# q1 / q2 offline check on 10-100 TU (card `analysis/cards/q1q2.md`, Stage 2; Rik, 2026-11-06): the quantities of paper
# Table `tab: own-draws` for closures that draw AR(2), or AR(2) x power-law, noise, EXACT (expectations, no draws).
#
#     julia --startup-file=no --project=training analysis/noise_ar_offline.jl LinReg1 LinReg1_nt LinReg1_nt_ar2nw \
#         [LinReg1_ntar2 LinReg1_ntar2_pl]
#
# Each closure (a deployed `TO_LRS/<name>`) is scored under its OWN deployed noise entering its history as Eq.
# `eq: window noise` (white N(0, Σ); AR(2) (φ, Σ_ξ); AR(2) x f(q*) with the recorded q* of each draw's step), on the
# held-out 10-100 TU rows. The first name must be LinReg1: its clean rms there is the unit (as the table's).
#   error: the rms one-step error on the recorded history | on noisy inputs (expected: + the response to the draws) |
#          the drawn sd ("fitted noise") | noisy / drawn (target 1) | the gain sqrt(response / drawn variance)
#   acf:   the residual given the noisy history (/ f with a scale) at lags 1 2 5 10 20 | the drawn noise's
#   lrv:   the long-run variance ratio 1 + 2 Σ_{k=1}^{200} ρ(k) (App. app:ar-fit): residual given the noisy history |
#          the drawn noise (card D1 (b))
# The white rows must reproduce the paper table's LinReg1 and LinReg1_nt (one Monte Carlo path there) within its
# sampling error. Functions: `exp_square_HIT/tools/lrs_noise_ar_variant.jl` (tests V97). Fits are not touched here:
# nothing is fitted on 10-100 TU.

include(normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "tools", "lrs_noise_ar_variant.jl")))

const HO_LAGS = 200

"The scaled design of the deployed LinReg `m` on rows `r`, as `scaled_design` but held out (no 74 TU guard: evaluation)."
function heldout_design(m, r)
    X, Y = design(m["hist_len"], r)
    si, so = m["scaling"].in_scaling, m["scaling"].out_scaling
    mu, sg = vec(si.mu), vec(si.sigma)
    Xs = copy(X)
    for j in 1:(size(X, 2) - 1)
        Xs[:, j] = (X[:, j] .- mu[mod1(j, NQ)]) ./ sg[mod1(j, NQ)]
    end
    Ys = (Y .- vec(so.mu)') ./ vec(so.sigma)'
    h = m["hist_len"]
    qs = X[:, 1:NQ]
    qwin = reduce(hcat, [X[:, (2j * NQ + 1):((2j + 1) * NQ)] for j in 1:h])
    return (; X, Xs, Ys, osg = vec(so.sigma), isg = sg, qs, qwin)
end

"One closure under its own noise on rows `r`: everything the report prints."
function own_noise(name, r)
    m = load(joinpath(LRS, name, "LinReg.jld2"))
    @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
    h = m["hist_len"]
    d = heldout_design(m, r)
    M = window_map(h, d.isg)
    Ct = Matrix(Matrix(m["c"])')
    mu = Vector{Float64}(mean(m["stoch_distr"]))
    hasar, hassc = haskey(m, "ar_phi"), haskey(m, "scale_beta")
    hassc && !hasar && error("$name: a white power-law closure is not covered here")
    phi = hasar ? Matrix{Float64}(m["ar_phi"]) : zeros(2, NQ)
    size(phi, 1) == 1 && (phi = [phi; zeros(1, NQ)])
    S = hasar ? Matrix{Float64}(m["ar_sigma_xi"]) : Matrix(cov(m["stoch_distr"]))
    Gz = ar_lagcov(phi, S, HO_LAGS + h)
    Wz = [window_lagcov(Gz, h, k) for k in 0:HO_LAGS]
    if hassc
        beta, qref, qclip = m["scale_beta"], m["scale_qref"], m["scale_qclip"]
        f, g = scale_rows(d.qs, d.qwin, beta, qref, qclip, d.osg, h)
    else
        f, g = nothing, repeat(d.osg, h)
    end
    r_ = d.Ys .- d.Xs * Ct
    Z = r_ .- mu'
    AM = Ct' * M
    v = noise_var_rows(AM, Wz[1], g)
    _, Ge = resid_lagcov(Z, AM, Wz, g, f; L = HO_LAGS)
    drawn = hassc ? vec(mean(f .^ 2; dims = 1)) .* diag(Gz[1]) : diag(Gz[1])   # scaled units, mean over rows
    racf = [Ge[k + 1][i, i] / Ge[1][i, i] for k in 0:HO_LAGS, i in 1:NQ]
    zacf = [Gz[k + 1][i, i] / Gz[1][i, i] for k in 0:HO_LAGS, i in 1:NQ]
    return (; name, kind = hassc ? "AR(2) x f" : hasar ? "AR(2)" : "white", osg = d.osg,
            clean = vec(sqrt.(mean(abs2, r_; dims = 1))) .* d.osg,
            noisy = sqrt.(vec(mean(abs2, r_; dims = 1)) .+ vec(mean(v; dims = 1))) .* d.osg,
            sdraw = sqrt.(drawn) .* d.osg, gain = sqrt.(vec(mean(v; dims = 1)) ./ drawn),
            racf, zacf, lrv_e = 1 .+ 2 .* vec(sum(racf[2:end, :]; dims = 1)),
            lrv_z = 1 .+ 2 .* vec(sum(zacf[2:end, :]; dims = 1)))
end

function main(names; io = stdout)
    first(names) == "LinReg1" || error("the first closure must be LinReg1 (the unit)")
    last_ = min(last(steps_of(10, 100)), size(QR, 2) - 1)
    r = (first(steps_of(10, 100)) + 1):last_
    println(io, "==== own-draw offline check, 10-100 TU (steps $(first(r))-$(last(r))), exact; unit: LinReg1's clean rms")
    println(io, "     QoIs: ", join(LABELS, " "))
    res = [own_noise(n, r) for n in names]
    ref = res[1].clean
    p3(v) = join((@sprintf("%.3f", x) for x in v), " ")
    for q in res
        @printf(io, "error: %-20s %-9s clean %s | noisy %s | drawn sd %s\n", q.name, q.kind, p3(q.clean ./ ref),
                p3(q.noisy ./ ref), p3(q.sdraw ./ ref))
        @printf(io, "ratio: %-20s %-9s noisy / drawn %s | gain %s\n", q.name, q.kind, p3(q.noisy ./ q.sdraw), p3(q.gain))
    end
    lags = (1, 2, 5, 10, 20)
    for q in res
        println(io, "acf:   $(q.name) ($(q.kind)), lags ", join(lags, " "), ": residual given the noisy history | drawn noise")
        for i in 1:NQ
            @printf(io, "       %-9s %s | %s\n", LABELS[i], fmt(q.racf[[l + 1 for l in lags], i]), fmt(q.zacf[[l + 1 for l in lags], i]))
        end
    end
    for q in res
        @printf(io, "lrv:   %-20s residual given the noisy history %s | drawn noise %s\n", q.name, fmt(q.lrv_e; d = 2),
                fmt(q.lrv_z; d = 2))
    end
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? ["LinReg1", "LinReg1_nt", "LinReg1_nt_ar2nw"] : ARGS)
end
