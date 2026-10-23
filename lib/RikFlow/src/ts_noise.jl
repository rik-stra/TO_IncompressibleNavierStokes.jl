# The residual noise models shared by the two closure implementations (2026-10-23, Rik: "the noise
# models could be shared between the paths"): the diagonal AR(p <= 2) recursion of `LinReg`'s
# coloured residual (step 1) and, through `ts_scale.jl`'s `powerlaw_factor`, step 2's power-law
# scale, used by `LinReg` (`time_series_methods.jl`) and by `StochLSTM` (`ts_lstm_online.jl`).
#
# Only the recursion is shared; the innovation is drawn by each closure in its own way (`LinReg`:
# one `rand(rng, MvNormal(0, Σ_ξ))` per step; `StochLSTM`: its emission draw, `D LR ε`), so each
# keeps its RNG stream. `LinReg`'s AR code was moved here unchanged, so its draws are bit-identical
# to before the move.
#
# Stdlib only, like the rest of the `ts_*` layer.

"Steps of the AR recursion from zero used as the stationary-draw fallback (see `LinReg`, `StochLSTM`)."
const AR_BURNIN = 2000

"Stationarity of a diagonal AR(p ≤ 2) with coefficients `phi` (`p × n`), per column."
function ar_stationary(phi::AbstractMatrix)
    p = size(phi, 1)
    return all(eachcol(phi)) do f
        p == 1 ? abs(f[1]) < 1 : (abs(f[2]) < 1 && f[2] + f[1] < 1 && f[2] - f[1] < 1)
    end
end

"Push `znew` into the AR state (column 1 newest)."
function ar_push!(ar, znew)
    p = size(ar.z, 2)
    p > 1 && (ar.z[:, 2:p] .= ar.z[:, 1:(p - 1)])
    ar.z[:, 1] .= znew
    return ar
end

"""
    ar_advance!(ar, xi)

One step of the diagonal AR recursion from a given innovation, `z_n = ξ_n + Σ_k φ_k ⊙ z_{n-k}`:
advances the state `ar` (`phi`, `p × n`; `z`, `n × p`, column 1 newest) and returns `z_n`. The caller
draws `ξ_n`.
"""
function ar_advance!(ar, xi)
    znew = xi .+ vec(sum(ar.phi' .* ar.z; dims = 2))
    ar_push!(ar, znew)
    return znew
end

"""
    ar_state(phi)

A fresh AR state for coefficients `phi` (`p × n`, one column per QoI): `(; phi, z, nz)` with `z` the
`n × p` state (column 1 newest) and `nz[]` how many lags were filled from data. Refuses an order
other than 1 or 2 and a non-stationary AR.
"""
function ar_state(phi::AbstractMatrix)
    phi = Matrix{Float64}(phi)
    p, n = size(phi)
    1 <= p <= 2 || error("AR order $p: only p = 1, 2 are implemented")
    ar_stationary(phi) || error("AR coefficients $phi are not stationary")
    return (; phi, z = zeros(n, p), nz = zeros(Int))
end
