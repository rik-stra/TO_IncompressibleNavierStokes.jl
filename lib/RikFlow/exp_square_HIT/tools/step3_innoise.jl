# Input-noise training of the step-3 network means (Rik, 2026-10-26: "lock this way of adding noise";
# claude_memory.md, standing decision). Online the closure adds η^m ~ N(0, Σ) to its correction at every
# step; the solver carries the corrected level into the next prediction, noise(q^{m+1*}) = noise(q^m), and
# the closure adds a fresh sample, noise(q^m) = noise(q^{m*}) + η^m. So every correction in the history
# carries one η, and the levels carry the running sum.
#
# 🔒 The sum is ANCHORED AT THE PREDICTION STEP: the current predictor q^{n*} is the record's, so the
# target (the record's dQ^n) belongs to the state the network predicts from, and the noise shapes an
# alternative past that arrives there:
#
#     noise(q^{n-k})  = -(η^{n-k+1} + ... + η^{n-1})      (0 at k = 1)
#     noise(q^{n-k*}) = -(η^{n-k}   + ... + η^{n-1})
#
# hence noise(dQ^{n-k}) = noise(Dq*^{n-k}) = η^{n-k}, every lag once. Anchoring at the window's start
# instead moves the current state off the record while the oldest entry stays on it, which rewards
# reliance on the oldest entry -- rejected. Not modelled: the solver's damping over a step (< 1 %) and the
# network's own errors feeding back.
#
# `include` after step3_data.jl (it uses `lagrows`, `P`, `dat`, `so`, `dq_s`, `in_s`).

FEAT === :raw && error("step3_innoise.jl: the window noise is defined on the dQ history (FEAT = dq or dqinc)")

"""
    window_noise(eta) -> dx

The perturbation of LinReg1's regressor (nreg x B, in its scaled level units, `in_scaling`) for the window
noise `eta` (nq x h x B, raw dQ units; `eta[:, j, b]` = η^{n-j}, j = 1 the newest). The rows of q^{n*} and
of the bias stay 0. The network reads `P * (x + dx)`.
"""
function window_noise(eta::AbstractArray{<:Real,3})
    nq_, h_, B = size(eta)
    (nq_, h_) == (nq, cfg.h) || error("window_noise: eta is $(size(eta)), expected ($nq, $(cfg.h), B)")
    C = cumsum(eta; dims = 2) ./ vec(dat.scaling.in_scaling.sigma)   # C[:, k, b] = η^{n-1} + ... + η^{n-k}
    dx = zeros(nreg, B)
    for k in 1:h_
        r = lagrows(k)
        k > 1 && (dx[r.q, :] .= .-C[:, k - 1, :])        # q^{n-k}
        dx[r.s, :] .= .-C[:, k, :]                       # q^{n-k*}
    end
    return dx
end

"`B` windows of η = L E, E standard normal (nq x h B), Σ = L L' in the STANDARDISED target units, in raw dQ units."
eta_from(L, E) = reshape(so .* (L * E), nq, cfg.h, size(E, 2) ÷ cfg.h)
"`B` windows of i.i.d. η ~ N(0, L L') (standardised target units), in raw dQ units (nq x h x B)."
draw_window_eta(rng, L, B) = eta_from(L, randn(rng, nq, cfg.h * B))
"Lower factor D A^{-1} of a head's Σ = D R D, R^{-1} = A'A (`(; bd, Araw)`); diagonal for the CRPS head (A = I)."
function head_factor(hd)
    Ar = Float64.(Array(hd.Araw))
    A = LowerTriangular(tril(Ar, -1) + Diagonal(exp.(diag(Ar))))
    return Diagonal(exp.(Float64.(Array(hd.bd)))) * Matrix(inv(A))
end
"Lower Cholesky factor of a covariance."
cov_factor(S) = Matrix(cholesky(Symmetric(Matrix{Float64}(S))).L)
"Second moment about zero of residuals `r` (nq x N): the MVG the closures draw from (Rik: \"about zero\")."
mom0(r) = r * r' ./ size(r, 2)

# 🔴 The pairing, through the input map, against the formulas above: the dQ and Dq* entry of every lag carry
# their own η once, q^{n-k*} the running sum back from the prediction, q^{n*} and the bias nothing.
let rng = Xoshiro(20261026), eta = randn(rng, nq, cfg.h, 3)
    sg = vec(dat.scaling.in_scaling.sigma)
    dz = P * window_noise(eta)
    worst = max(maximum(abs, dz[1:nq, :]), maximum(abs, dz[nreg, :]))
    for k in 1:cfg.h, b in 1:3
        r = lagrows(k)
        worst = max(worst, maximum(abs, dz[r.q, b] .- eta[:, k, b] ./ sg ./ dq_s[:, k]))              # dQ^{n-k}
        worst = max(worst, maximum(abs, dz[r.s, b] .+ vec(sum(eta[:, 1:k, b]; dims = 2)) ./ sg))     # q^{n-k*}
        FEAT === :dqinc && (worst = max(worst, maximum(abs, dz[nreg .+ (((k - 1) * nq + 1):(k * nq)), b] .-
                                                          eta[:, k, b] ./ sg ./ in_s[:, k])))        # Dq*^{n-k}
    end
    rel = worst / maximum(abs, dz)
    rel < 1e-12 || error("the window noise does not pair as the closure does (max error $worst, relative $rel)")
    @info "window noise checked through the input map" max_rel_err = rel
end
