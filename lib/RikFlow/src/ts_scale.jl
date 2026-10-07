# The state-dependent scale of the residual, step 4p (paper Sec. 6.4, Eq. power law; added 2026-10-07).
#
#     η = μ_η + f(q*) ⊙ ε,   ε ~ N(0, Σ_ε),   f_i(q) = (clamp(q_i, lo_i, hi_i) / q_ref,i)^β_i
#
# with q* the predicted level (physical units), q_ref,i the geometric mean of q*_i on the training
# window and [lo_i, hi_i] its range there, so the law is never extrapolated. Evidence and the choice
# of the state: `analysis/results_LSTMS.md` §17 (`analysis/state_dependence.jl`).
#
# Stdlib-only, like the rest of the `ts_*` layer: the deployed sampler (`time_series_methods.jl`),
# the builder (`exp_square_HIT/tools/lrs_scale_variant.jl`), the offline check and the tests all use
# THIS fit and THIS factor, so the paper's number, the artefact and the deployment cannot drift apart.

"""
    powerlaw_factor(q, beta, qref, qclip)

`f_i = (clamp(q_i, qclip[i, 1], qclip[i, 2]) / qref[i])^beta[i]`, one factor per QoI. `β = 0` gives
exactly 1.0.
"""
powerlaw_factor(q, beta, qref, qclip) =
    [(clamp(q[i], qclip[i, 1], qclip[i, 2]) / qref[i])^beta[i] for i in eachindex(beta)]

"""
    fit_powerlaw_scale(e, q; iters = 100) -> (; beta, logvar, qref, qclip)

Maximum likelihood for `e ~ N(0, exp(logvar) (q / qref)^(2β))` with the mean held at zero: Newton on
`(logvar, β)`. `qref` is the geometric mean of `q`, `qclip = (minimum(q), maximum(q))`, so on the fit
data the clamp never binds. Refuses a non-positive state.
"""
function fit_powerlaw_scale(e::AbstractVector, q::AbstractVector; iters::Integer = 100)
    length(e) == length(q) || throw(DimensionMismatch("e has $(length(e)) entries, q $(length(q))"))
    all(>(0), q) || error("fit_powerlaw_scale: the state must be positive")
    qref = exp(sum(log, q) / length(q))
    F = hcat(ones(length(e)), 2 .* log.(q ./ qref))           # log variance = logvar + 2β log(q/qref)
    e2 = Float64.(e) .^ 2
    w = [log(sum(e2) / length(e2)), 0.0]
    L(w) = 0.5 * sum(F * w .+ e2 .* exp.(-(F * w)))
    for _ in 1:iters
        u = e2 .* exp.(-(F * w))
        g = 0.5 .* (F' * (1 .- u))
        H = 0.5 .* (F' * (F .* u))
        d = H \ g
        t = 1.0
        while L(w - t * d) > L(w) && t > 1e-8
            t /= 2
        end
        w -= t * d
        maximum(abs, t * d) < 1e-12 && break
    end
    return (; beta = w[2], logvar = w[1], qref, qclip = (minimum(q), maximum(q)))
end
