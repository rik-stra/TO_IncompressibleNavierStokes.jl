# Offline scoring and residual diagnostics.
#
# Nothing in the repository scored a predictive density on held-out data before this file: the
# LinReg pipeline fits and evaluates on the same range, and there is no residual diagnostic of any
# kind. The battery below is G0-d of `methodology.md` §6.

"""
    autocorr(x, maxlag)

Sample autocorrelation of `x` at lags `0:maxlag`. A series with no variance returns all zeros.

🔴 **The degeneracy guard is relative, and it has to be.** An earlier version tested `den == 0`,
which almost never fires: on `fill(3.7, 1000)` the centred sum of squares is `2.8e-26` rather than
exactly zero, because `sum(x)/n` does not reproduce `x` bit-for-bit. The autocorrelation was then
computed from pure rounding noise and came back as `[1.0, 0.999]` -- indistinguishable from a
strongly autocorrelated series. That mattered immediately: the data-driven noise model's predictive
mean **is** a constant, so its reported lag-1 autocorrelation came out as 1.0000 when the truthful
answer is that it has none. The threshold below is scaled by the magnitude of the data, so a
constant series of any size is caught.

The criterion is **relative to the series' own variation**, not to its magnitude: a series whose
centred RMS is below `CONSTANT_RTOL` times its largest absolute value is treated as constant. An
absolute or magnitude-scaled threshold does not work -- scaling by `maximum(abs, x)` declares an
AR(1) sitting on a large offset to be constant, because the offset dominates the scale while the
variation it is being compared against is real.
"""
function autocorr(x::AbstractVector, maxlag::Int)
    n = length(x)
    mu = sum(x) / n
    xc = x .- mu
    den = sum(abs2, xc)
    amax = maximum(abs, x)
    # No variation at all, or variation only at the level of the centring's own rounding error.
    (den == 0 || (amax > 0 && sqrt(den / n) <= CONSTANT_RTOL * amax)) &&
        return zeros(Float64, maxlag + 1)
    return [sum(xc[i] * xc[i - k] for i in (k + 1):n) / den for k in 0:maxlag]
end

"""
    CONSTANT_RTOL

A series varying by less than this fraction of its own magnitude counts as constant in
[`autocorr`](@ref). Far above the `1e-16` that centring rounding produces and far below any real
signal, so it separates the two without a judgement call at the call site.
"""
const CONSTANT_RTOL = 1e-12

"""
    ljung_box(x, nlags)

Ljung-Box portmanteau statistic `Q = n(n+2) sum_k rho_k^2/(n-k)` on lags `1:nlags`, with its
degrees of freedom. Compare against a chi-squared quantile; `Q` far above `nlags` rejects
serial independence.
"""
function ljung_box(x::AbstractVector, nlags::Int)
    n = length(x)
    r = autocorr(x, nlags)
    Q = n * (n + 2) * sum(r[k + 1]^2 / (n - k) for k in 1:nlags)
    return (; Q, dof = nlags)
end

"""
    residual_battery(Res, Z; dt, nlags = 50)

The G0-d diagnostics, run on the raw residual `Res` and the standardised residual `Z = D^{-1}Res`.

Returns, per QoI, `phi_ee` (raw autocorrelation, outside its band implies colour), `phi_e2e2`
(squared raw, implies heteroscedasticity), `phi_zz` and `phi_z2z2` (the same on the standardised
residual: both inside the band is the acceptance test for the variance head), plus skewness and
excess kurtosis of `Z`, and the Ljung-Box statistic on the standardised cross-products
`Z_i Z_j`, whose serial correlation rejects constant `R`.

The 95% band for a white series is `+/- 1.96/sqrt(n)`.
"""
function residual_battery(Res::AbstractMatrix, Z::AbstractMatrix; dt = 1.0, nlags::Int = 50)
    n, nq = size(Res)
    band = 1.96 / sqrt(n)
    phi_ee = [autocorr(view(Res, :, i), nlags) for i in 1:nq]
    phi_e2e2 = [autocorr(view(Res, :, i) .^ 2, nlags) for i in 1:nq]
    phi_zz = [autocorr(view(Z, :, i), nlags) for i in 1:nq]
    phi_z2z2 = [autocorr(view(Z, :, i) .^ 2, nlags) for i in 1:nq]

    skew = [moment_std(view(Z, :, i), 3) for i in 1:nq]
    kurt = [moment_std(view(Z, :, i), 4) - 3.0 for i in 1:nq]

    cross = NamedTuple[]
    for i in 1:nq, j in (i + 1):nq
        c = vec(view(Z, :, i) .* view(Z, :, j))
        push!(cross, (; i, j, lb = ljung_box(c, 20)))
    end

    tcorr = [correlation_time(view(Res, :, i), dt) for i in 1:nq]
    return (; band, phi_ee, phi_e2e2, phi_zz, phi_z2z2, skew, kurt, cross, tcorr)
end

function moment_std(x, k)
    n = length(x)
    mu = sum(x) / n
    s = sqrt(sum(abs2, x .- mu) / n)
    return s == 0 ? 0.0 : sum((xi - mu)^k for xi in x) / (n * s^k)
end

"""
    correlation_time(x, dt)

Correlation time of a residual series, in physical units. Two estimates are returned: `T_exp` from
the lag-one autocorrelation, `-dt/log(rho_1)`, which is what a diagonal AR(1) predicts, and
`T_int`, the integral of the autocorrelation up to its first zero crossing.

This is the quantity Rességuier et al. estimate for their own residual before down-sampling the
time step towards it. That escape is closed here: `dt` is the LES time step.
"""
function correlation_time(x::AbstractVector, dt; maxlag = min(500, length(x) ÷ 4))
    r = autocorr(x, maxlag)
    T_exp = (r[2] > 0 && r[2] < 1) ? -dt / log(r[2]) : NaN
    zc = findfirst(<=(0.0), r[2:end])
    # 🔴 `truncated` is the difference between a measurement and a lower bound. The sum runs to the
    # first non-positive rho (Sokal's window); if that never happens inside `maxlag`, the window is
    # the cap and `T_int` is whatever fitted in it. On the QoI **level** that is a live risk --
    # rho_1(q) is about 1 and the level's integral time is 0.5-1.1 TU against a default window of
    # 500 lags = 1.25 TU -- whereas on `dQ` the ACF crosses within tens of lags. A truncated value
    # must be reported as a lower bound, never as the timescale.
    truncated = zc === nothing
    k = truncated ? length(r) - 1 : zc
    T_int = dt * (0.5 + sum(r[2:k]))
    return (; T_exp, T_int, rho1 = r[2], truncated, maxlag, window = k)
end

"""
    gaussian_nll_per_sample(model_nll, nrows, nq)

Convert a summed negative log-likelihood into a per-sample, per-QoI figure including the
`log(2pi)/2` constant, which is the number to compare across models.
"""
gaussian_nll_per_sample(model_nll, nrows, nq) =
    (model_nll + 0.5 * nrows * nq * log(2pi)) / (nrows * nq)

"""
    ks_distance(a, b)

Two-sample Kolmogorov-Smirnov distance between the empirical distributions of `a` and `b`. The
statistic is a valid distance on serially correlated samples; what is not valid is a p-value or an
i.i.d. bootstrap over time samples.
"""
function ks_distance(a::AbstractVector, b::AbstractVector)
    x = sort(vcat(a, b))
    sa, sb = sort(a), sort(b)
    d = 0.0
    for v in x
        fa = searchsortedlast(sa, v) / length(sa)
        fb = searchsortedlast(sb, v) / length(sb)
        d = max(d, abs(fa - fb))
    end
    return d
end

# =============================================================================================
# Density, calibration and accuracy metrics.
#
# `metrics.md` numbers each of these and names the regime it is computed in. The regime matters
# more than the formula, so it is stated in every docstring:
#
#   0  fit-time      -- from the parameters and the design matrix, no trajectory
#   A  one-step      -- teacher-forced on the true history; errors never accumulate
#   B  offline unrolled, `q*` replayed -- ARTIFICIAL: the feedback loop through the solver is cut
#   C  online        -- the LF solver is in the loop; the only regime that measures deployment
#
# Stdlib-only, like the rest of the `ts_*` layer. That constraint costs one thing -- there is no
# `erf` in the standard library -- so `norm_cdf` is implemented here and checked against a BigFloat
# series in `test/test_score.jl` rather than against remembered constants.
# =============================================================================================

"""
    erf_series(x)

`erf(x)` by its Maclaurin series, `(2/sqrt(pi)) * sum (-1)^n x^(2n+1)/(n!(2n+1))`.

Used for `abs(x) <= 2`, where the alternating series reaches full double precision well inside the
term limit; beyond that the terms grow before they decay and [`erfc_cf`](@ref) takes over.
"""
function erf_series(x::Real)
    z = float(x)
    z == 0 && return zero(z)
    term = z          # the n = 0 term of the sum, before the prefactor
    s = z
    z2 = z * z
    for n in 1:500
        term *= -z2 / n
        add = term / (2n + 1)
        s += add
        abs(add) <= eps(typeof(s)) * abs(s) && break
    end
    return 2 / sqrt(oftype(s, pi)) * s
end

"""
    erfc_cf(x)

`erfc(x)` for `x > 0` from the continued fraction

    erfc(x) = exp(-x^2)/(x*sqrt(pi)) * 1/(1 + (1/2)/x^2/(1 + (2/2)/x^2/(1 + ...)))

evaluated by the modified Lentz algorithm. Accurate for `x >= 2`; below that the fraction
converges slowly and the series form is used instead.
"""
function erfc_cf(x::Real)
    z = float(x)
    z <= 0 && throw(DomainError(x, "erfc_cf is for x > 0"))
    tiny = eps(typeof(z))^2
    f = tiny
    C = f
    D = zero(z)
    for i in 1:10_000
        a = i == 1 ? one(z) : (i - 1) / 2 / (z * z)
        b = one(z)
        D = b + a * D
        iszero(D) && (D = tiny)
        C = b + a / C
        iszero(C) && (C = tiny)
        D = inv(D)
        delta = C * D
        f *= delta
        abs(delta - 1) <= eps(typeof(z)) && break
    end
    return exp(-z * z) / (z * sqrt(oftype(z, pi))) * f
end

"""
    erf_(x)
    erfc_(x)

`erf` and `erfc` in double precision, without `SpecialFunctions`.

The switch is at `abs(x) = 2`: the series is exact there to machine precision and the continued
fraction has begun to converge, so neither method is used outside its comfortable range.
"""
function erf_(x::Real)
    z = float(x)
    abs(z) <= 2 && return erf_series(z)
    return z > 0 ? 1 - erfc_cf(z) : erfc_cf(-z) - 1
end

function erfc_(x::Real)
    z = float(x)
    abs(z) <= 2 && return 1 - erf_series(z)
    return z > 0 ? erfc_cf(z) : 2 - erfc_cf(-z)
end

"""
    norm_cdf(z)
    norm_pdf(z)

Standard normal CDF and PDF. `norm_cdf` uses `erfc` on the far side of zero so that a small tail
probability keeps its relative accuracy instead of cancelling against 1.
"""
norm_cdf(z::Real) = z < 0 ? erfc_(-z / sqrt(oftype(float(z), 2))) / 2 :
                    1 - erfc_(z / sqrt(oftype(float(z), 2))) / 2
norm_pdf(z::Real) = exp(-float(z) * float(z) / 2) / sqrt(2 * oftype(float(z), pi))

# ---------------------------------------------------------------------------------------------
# 1 -- held-out negative log-likelihood.  Regime A.
# ---------------------------------------------------------------------------------------------

_row(Y::AbstractMatrix, n) = view(Y, n, :)
_mu_row(mu::AbstractMatrix, n, nq) = view(mu, n, :)
_mu_row(mu::AbstractVector, n, nq) = mu

"""
    nll_gaussian(Y, mu, Sigma)

Metric #1, regime **A**. Held-out NLL of a multivariate Gaussian predictive density,

    NLL = (1/2N) sum_n [ logdet(S_n) + (y_n-m_n)' inv(S_n) (y_n-m_n) + N_Q log(2pi) ]

`Y` is `N x N_Q`; `mu` is either an `N x N_Q` matrix of per-step means or one vector broadcast to
every step; `Sigma` is either one `N_Q x N_Q` matrix (a state-independent residual -- M0 and the
data-driven noise model) or a vector of `N` such matrices.

Returns `(; total, per_sample, per_sample_per_qoi, n, nq)`. Report **per sample**, so records of
different length compare directly.

Two terms pull opposite ways: `logdet(S)` punishes width, the quadratic form punishes a narrow
density that misses; the minimum sits where the width matches the actual error.

!!! warning
    It is unbounded, so one truth value far in the tail of a confident Gaussian can reverse a
    ranking. That is why #2 and #4 exist alongside it, and why M4's variational bound cannot be
    compared against an exact NLL at all.
"""
function nll_gaussian(Y::AbstractMatrix, mu, Sigma)
    N, nq = size(Y)
    # A vector `mu` means "one mean, broadcast to every step", so its length must be N_Q. Without
    # this check a single-QoI problem handed an N-length vector of per-step means would be read as
    # one mean vector of length N and silently scored against the wrong thing.
    mu isa AbstractVector && length(mu) != nq &&
        throw(DimensionMismatch("a vector mu is one mean broadcast to every step and must have " *
                                "length N_Q = $nq, got $(length(mu)); pass an N x N_Q matrix " *
                                "for per-step means"))
    mu isa AbstractMatrix && size(mu) != (N, nq) &&
        throw(DimensionMismatch("matrix mu must be $((N, nq)), got $(size(mu))"))
    Sigma isa AbstractVector && length(Sigma) != N &&
        throw(DimensionMismatch("a vector Sigma is per-step and must have length N = $N, " *
                                "got $(length(Sigma))"))
    total = 0.0
    if Sigma isa AbstractMatrix
        C = cholesky(Symmetric(Matrix{Float64}(Sigma)))
        ld = logdet(C)
        for n in 1:N
            r = Float64.(_row(Y, n)) .- Float64.(_mu_row(mu, n, nq))
            total += ld + dot(r, C \ r)
        end
    else
        for n in 1:N
            C = cholesky(Symmetric(Matrix{Float64}(Sigma[n])))
            r = Float64.(_row(Y, n)) .- Float64.(_mu_row(mu, n, nq))
            total += logdet(C) + dot(r, C \ r)
        end
    end
    total = (total + N * nq * log(2pi)) / 2
    return (; total, per_sample = total / N, per_sample_per_qoi = total / (N * nq), n = N, nq)
end

# ---------------------------------------------------------------------------------------------
# 2 -- CRPS.  Regime A, and the score every ensemble tier shares.
# ---------------------------------------------------------------------------------------------

"""
    crps_gaussian(mu, sigma, y)

Metric #2, closed form for one Gaussian margin:

    CRPS = sigma * ( z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ),   z = (y - mu)/sigma

In the units of the variable and read like an error: as `sigma -> 0` it reduces to `abs(mu - y)`.
"""
function crps_gaussian(mu::Real, sigma::Real, y::Real)
    s = float(sigma)
    s <= 0 && return abs(float(y) - float(mu))
    z = (float(y) - float(mu)) / s
    return s * (z * (2 * norm_cdf(z) - 1) + 2 * norm_pdf(z) - 1 / sqrt(oftype(z, pi)))
end

"""
    crps_ensemble(x, y; fair = true)

Metric #2, ensemble form, for `M` members `x` and one truth `y`:

    CRPS = (1/M) sum_i abs(x_i - y) - (1/(2M(M-1))) sum_i sum_j abs(x_i - x_j)

!!! warning
    The `fair = true` denominator `2M(M-1)` is not optional. The common `1/(2M^2)` form is biased
    at small `M` and **rewards under-dispersion** -- the exact defect this project is trying to
    measure, so the biased form would flatter the failure mode under investigation. At `M = 5`, the
    archive's ensemble size, the two denominators differ by 25% on the spread term.

    `fair = false` exists only so that a test can demonstrate the bias rather than describe it.

The double sum is evaluated in `O(M log M)` from the sorted sample: term `i` of the ordered sample
enters with weight `2i - M - 1`.
"""
function crps_ensemble(x::AbstractVector, y::Real; fair::Bool = true)
    M = length(x)
    M == 0 && return NaN
    M == 1 && return abs(float(x[1]) - float(y))
    s = sort(float.(collect(x)))
    a = sum(abs(xi - float(y)) for xi in s) / M
    g = 0.0
    for i in 1:M
        g += (2i - M - 1) * s[i]
    end
    g *= 2                                # the full unordered double sum
    den = fair ? 2 * M * (M - 1) : 2 * M * M
    return a - g / den
end

"""
    crps_ensemble_mean(ens, Y; fair = true)

Per-QoI mean CRPS over verification instances. `ens` is `N x N_Q x M`, `Y` is `N x N_Q`.
"""
function crps_ensemble_mean(ens::AbstractArray{<:Real,3}, Y::AbstractMatrix; fair::Bool = true)
    N, nq, M = size(ens)
    @assert size(Y) == (N, nq)
    return [sum(crps_ensemble(view(ens, n, i, :), Y[n, i]; fair) for n in 1:N) / N for i in 1:nq]
end

# ---------------------------------------------------------------------------------------------
# 4 -- rank histogram, and reading it as a number.
# ---------------------------------------------------------------------------------------------

"""
    ranks(ens, y; ties = :random, rng)

Rank of each truth value among its `M` ensemble members, `r = 1 + count(x_i < y)`, in `1:M+1`.
`ens` is `N x M` for one QoI and `y` has length `N`.

!!! warning
    Ties are broken at random and the RNG is an argument, because the inherited stabiliser
    `any(abs.(q_star) .< 1e-2) && (dQ .= 0)` produces **exact** duplicates by construction:
    whenever it fires every member returns the identical `dQ = 0` and the truth ties with all of
    them. Breaking ties toward either end would manufacture the U or the cap that this metric
    exists to detect. Seed and record the RNG.
"""
function ranks(ens::AbstractMatrix, y::AbstractVector; ties::Symbol = :random,
               rng = Random.default_rng())
    N, M = size(ens)
    @assert length(y) == N
    r = Vector{Int}(undef, N)
    for n in 1:N
        yn = y[n]
        below = 0
        equal = 0
        for m in 1:M
            x = ens[n, m]
            if x < yn
                below += 1
            elseif x == yn
                equal += 1
            end
        end
        r[n] = if equal == 0 || ties === :low
            below + 1
        elseif ties === :high
            below + equal + 1
        else
            below + 1 + rand(rng, 0:equal)
        end
    end
    return r
end

"""
    n_eff(x)

Effective sample size of a serially correlated series, `N / (1 + 2*sum_k rho_k)`, with the sum
truncated by Geyer's initial-positive-sequence rule.

!!! warning
    Every chi-squared statistic in this file runs on this, not on `N`. QoI series are strongly
    serially correlated -- `rho_1` is about 0.94 on HIT -- and an omnibus chi-squared on raw `N`
    rejects uniformity for **every** model ever fitted, which makes it useless as a discriminator
    rather than merely conservative.
"""
function n_eff(x::AbstractVector)
    N = length(x)
    N < 8 && return float(N)
    r = autocorr(collect(float.(x)), min(N ÷ 4, 1000))
    s = 0.0
    for k in 2:length(r)
        r[k] <= 0 && break
        s += r[k]
    end
    return N / max(1 + 2s, 1.0)
end

"""
    jolliffe_primo(counts)

Jolliffe-Primo orthogonal contrasts of a rank histogram, plus the omnibus chi-squared.

Two weighted sums of the bin counts, `c = sum_k a_k n_k` with `sum a_k = 0`, one degree of freedom
each, standardised to `Z = c / sqrt(e * sum a_k^2)` so each is asymptotically standard normal under
uniformity:

  * **linear weights -> the slope contrast -> mean bias.**
  * **quadratic weights -> the convexity contrast -> dispersion.** Positive means the end bins are
    over-populated, i.e. a **U**, i.e. **under-dispersion**; negative means over-dispersion.

The omnibus `chi2 = sum (n_k - e)^2 / e` is returned too, but it discards the shape, which is the
informative part. "Under-dispersed, convexity contrast +0.34 [0.21, 0.47]" is a quantified problem;
"chi2 = 812, p < 1e-9" is not.

Returns `(; slope, convexity, chi2, dof, expected, K, n)`.
"""
function jolliffe_primo(counts::AbstractVector{<:Integer})
    K = length(counts)
    n = sum(counts)
    e = n / K
    k = collect(1.0:K)
    mid = (K + 1) / 2
    a1 = k .- mid                                   # linear
    a2 = (k .- mid) .^ 2 .- (K^2 - 1) / 12          # quadratic, orthogonal to a1 and to the mean
    z(a) = sum(a .* counts) / sqrt(e * sum(abs2, a))
    chi2 = sum((c - e)^2 / e for c in counts)
    return (; slope = z(a1), convexity = z(a2), chi2, dof = K - 1, expected = e, K, n)
end

"""
    quantile_sorted(s, p)

Linear-interpolation quantile of an already sorted vector, so the layer keeps no dependency on
`Statistics.quantile`.
"""
function quantile_sorted(s::AbstractVector, p::Real)
    n = length(s)
    n == 0 && return NaN
    n == 1 && return float(s[1])
    h = (n - 1) * float(p) + 1
    lo = clamp(floor(Int, h), 1, n)
    hi = clamp(lo + 1, 1, n)
    return s[lo] + (h - lo) * (s[hi] - s[lo])
end

"""
    block_bootstrap_indices(N, blocklen, rng)

Index vector for one moving-block bootstrap replicate covering `N` samples. Blocks are contiguous,
so within-block serial dependence survives the resample.

!!! warning
    This is the only bootstrap these metrics may use over time samples. The samples are consecutive
    time steps, and an i.i.d. resample would treat `rho_1` of about 0.94 as independent and produce
    confidence intervals several times too narrow.
"""
function block_bootstrap_indices(N::Int, blocklen::Int, rng)
    b = clamp(blocklen, 1, N)
    idx = Int[]
    while length(idx) < N
        s = rand(rng, 1:(N - b + 1))
        append!(idx, s:(s + b - 1))
    end
    return idx[1:N]
end

"""
    rank_histogram(ens, y; ties = :random, rng, nboot = 1000, blocklen = nothing)

Metric #4 for one QoI. `ens` is `N x M`, `y` has length `N`. Bins are `M+1`, flat at `N/(M+1)`
under the null that the truth is exchangeable with a member.

Returns the counts, the ranks, the Jolliffe-Primo contrasts with block-bootstrap 95% intervals, the
omnibus chi-squared **and its rescaling to `N_eff`**, and the block length used.

| shape | reading |
|---|---|
| **U**, ends over-populated | truth keeps falling outside the members, so **under-dispersed** |
| **cap**, centre over-populated | over-dispersed |
| **sloped** | mean bias |
| **flat** | calibrated *at this lead, for this QoI, and nothing more* |

!!! warning
    Flat is reliability, not skill. A climatological ensemble is perfectly flat and has zero
    resolution, and it is flat only when pooled over attractor-distributed verification instances.
    So a histogram from this function is reported **beside** CRPS (#2) and spread-skill (#17),
    never alone. The state-independent data-driven noise model is the live demonstration: it
    reproduces the marginal `dQ` distribution by construction while having a lag-1 autocorrelation
    of 0.0006 against a reference 0.9434.

`blocklen` defaults to twice the integral correlation time of the rank series, which is the
dependence the bootstrap has to respect.
"""
function rank_histogram(ens::AbstractMatrix, y::AbstractVector; ties::Symbol = :random,
                        rng = Random.default_rng(), nboot::Int = 1000, blocklen = nothing)
    N, M = size(ens)
    r = ranks(ens, y; ties, rng)
    K = M + 1
    counts = zeros(Int, K)
    for ri in r
        counts[ri] += 1
    end
    jp = jolliffe_primo(counts)

    ne = n_eff(r)
    # 🔴 One correction, applied consistently to both statistics.
    #
    # `jolliffe_primo` standardises each contrast as if the bin counts were an independent
    # multinomial. They are not: the rank series inherits the serial correlation of the QoIs, so
    # the true standard deviation of a contrast is inflated by `sqrt(N/N_eff)`. Measured on a null
    # of exchangeable AR(1) paths with `rho_1 = 0.94`: the raw contrast is unbiased but has a
    # standard deviation of about 4.6 instead of 1, and `N_eff/N` is about 0.03.
    #
    # `chi2` scales linearly in the sample size and `Z^2` is one of its one-degree-of-freedom
    # components, so the same factor appears as `N_eff/N` on the chi-squared and as its square root
    # on each contrast. Correcting only the chi-squared -- which an earlier version of this
    # function did -- leaves the signed statistics reading several sigma on calibrated data.
    #
    # Both forms are returned. The `_eff` pair is the one to quote; the raw pair is kept so a test
    # can assert that the uncorrected version fails, which is what stops the correction being
    # dropped again later.
    f_eff = sqrt(ne / N)
    chi2_eff = jp.chi2 * ne / N
    slope_eff = jp.slope * f_eff
    convexity_eff = jp.convexity * f_eff

    b = blocklen === nothing ?
        max(1, ceil(Int, 2 * correlation_time(collect(float.(r)), 1.0).T_int)) : blocklen
    slopes = Vector{Float64}(undef, nboot)
    convs = Vector{Float64}(undef, nboot)
    cb = zeros(Int, K)
    for t in 1:nboot
        idx = block_bootstrap_indices(N, b, rng)
        fill!(cb, 0)
        for i in idx
            cb[r[i]] += 1
        end
        j = jolliffe_primo(cb)
        slopes[t] = j.slope * f_eff
        convs[t] = j.convexity * f_eff
    end
    ci(v) = isempty(v) ? (NaN, NaN) :
            (quantile_sorted(sort(v), 0.025), quantile_sorted(sort(v), 0.975))
    return (; counts, ranks = r, K, n = N, M, expected = jp.expected,
            slope = slope_eff, convexity = convexity_eff,
            slope_raw = jp.slope, convexity_raw = jp.convexity,
            slope_ci = ci(slopes), convexity_ci = ci(convs),
            chi2 = jp.chi2, chi2_eff, dof = jp.dof, n_eff = ne, blocklen = b,
            eff_factor = f_eff)
end

# ---------------------------------------------------------------------------------------------
# 10, 11 -- summed and ensemble KS.  Regime C.
# ---------------------------------------------------------------------------------------------

"""
    summed_ks(traj, ref)

Metrics #10. For **one replica**,

    KS_r = sum over QoIs of sup_x abs(F_i_replica(x) - F_i_reference(x))

`traj` and `ref` are `N_Q x nstep` (the two need not share a length). Returns
`(; total, per_qoi)`.

!!! warning
    "Summed" means summed over the six QoIs and **never** over replicas. Replicas are separate
    samples of this statistic, reported as a spread and bootstrapped over at the replica level.
    Pooling them into one empirical CDF is [`ensemble_ks`](@ref), a different object: a model whose
    replicas are individually wrong but collectively cover the reference scores well on the second
    and badly on the first. Both are reported; they are never averaged together.

!!! warning
    KS sees the histogram of values, not their order -- shuffle a series and it is unchanged. This
    is not hypothetical here: on HIT, summed KS ranked the **memoryless** data-driven noise model
    best of nine at 0.183 while its correction had a lag-1 autocorrelation of 0.0006 against a
    reference 0.9434. KS is a guard, never a selector.
"""
function summed_ks(traj::AbstractMatrix, ref::AbstractMatrix)
    nq = size(traj, 1)
    @assert size(ref, 1) == nq
    per_qoi = [ks_distance(collect(float.(view(traj, i, :))), collect(float.(view(ref, i, :))))
               for i in 1:nq]
    return (; total = sum(per_qoi), per_qoi)
end

"""
    ensemble_ks(trajs, ref)

Metric #11. Pool every replica's time series into **one** empirical CDF per QoI first, then take
the supremum against the reference and sum over QoIs. The counterpart to [`summed_ks`](@ref); see
its warning for why the two are never mixed.
"""
function ensemble_ks(trajs::AbstractVector{<:AbstractMatrix}, ref::AbstractMatrix)
    nq = size(ref, 1)
    per_qoi = Float64[]
    for i in 1:nq
        pooled = reduce(vcat, [collect(float.(view(t, i, :))) for t in trajs])
        push!(per_qoi, ks_distance(pooled, collect(float.(view(ref, i, :)))))
    end
    return (; total = sum(per_qoi), per_qoi)
end

"""
    ks_noise_floor(ref; nsplit = 2, rng = nothing)

Metric #10/#11's reference-vs-reference noise floor (data object **D8**): split the HF record into
`nsplit` disjoint contiguous halves and score them against each other.

Any summed-KS difference smaller than this is not a model difference. Contiguous, not resampled,
because the record is serially correlated.
"""
function ks_noise_floor(ref::AbstractMatrix; nsplit::Int = 2)
    nq, T = size(ref)
    @assert nsplit == 2 "only a two-way split is defined; a k-way floor needs a stated pairing"
    h = T ÷ 2
    a = view(ref, :, 1:h)
    b = view(ref, :, (h + 1):T)
    return summed_ks(a, b)
end

# ---------------------------------------------------------------------------------------------
# 12-15 -- autocorrelation error.  Regime C.
# ---------------------------------------------------------------------------------------------

"""
    delta_rho(traj, ref; lag, maxlag = nothing, dt = 1.0)

Metrics #12-#15. Where KS asks whether the values are right, this asks whether the **order** is
right; together they are the S2' accuracy coordinate.

    delta_rho(tau) = sum over QoIs of abs(rho_model_i(tau) - rho_ref_i(tau))

Returns `(; at_lag, per_qoi, integral, integral_per_qoi, rho_model, rho_ref)`.

The named lags have distinct jobs:

  * `lag = 1` -- **#12**, full dynamic range at the degenerate end of a sweep, where the model has
    no dynamics at all. The adequacy statistic.
  * `lag = ceil(t_int/dt)` -- **#13**, the S2' coordinate. On HIT `rho_1` is about 0.94, so a model
    whose correlation time is wrong by a factor two moves `delta_rho(1)` by only about 6% of its
    scale; at the integral-timescale lag the same two models separate about four times better.
  * `lag = 2*ceil(t_int/dt)` -- **#14**, for coloured cells only. An AR(1) has `rho_k = a^k`
    exactly, so matching lag 1 forces every other lag if the process really is exponential and
    mismatches them otherwise. The statistic a degenerate coloured cell cannot fake.

`integral` is **#15**, `sum_i int abs(rho_model - rho_ref) d tau` over `0:maxlag`: the whole curve
as one number with no lag choice. Recommended as the reported headline, because all three named
lags ride on `t_int`, which is currently disputed by a factor 4-8.
"""
function delta_rho(traj::AbstractMatrix, ref::AbstractMatrix; lag::Int = 1, maxlag = nothing,
                   dt = 1.0)
    nq = size(traj, 1)
    @assert size(ref, 1) == nq
    ml = maxlag === nothing ? max(lag * 4, 200) : maxlag
    ml = min(ml, size(traj, 2) ÷ 4, size(ref, 2) ÷ 4)
    # The clamp above can fall below `lag` on a short record, which would index past the end of the
    # autocorrelation vector. Refuse instead: a lag the record cannot support is a study-design
    # error, and silently returning rho at some other lag is the worst possible response to it.
    lag >= 1 || throw(ArgumentError("lag must be >= 1, got $lag"))
    ml >= lag || throw(ArgumentError(
        "lag $lag exceeds the largest supportable lag $ml for series of length " *
        "$(size(traj, 2)) and $(size(ref, 2)); a quarter of the shorter series is the cap"))
    rm = [autocorr(collect(float.(view(traj, i, :))), ml) for i in 1:nq]
    rr = [autocorr(collect(float.(view(ref, i, :))), ml) for i in 1:nq]
    per_qoi = [abs(rm[i][lag + 1] - rr[i][lag + 1]) for i in 1:nq]
    ipq = [dt * sum(abs, rm[i][2:end] .- rr[i][2:end]) for i in 1:nq]
    return (; at_lag = sum(per_qoi), per_qoi, integral = sum(ipq), integral_per_qoi = ipq,
            rho_model = rm, rho_ref = rr, lag, maxlag = ml)
end

# ---------------------------------------------------------------------------------------------
# 16 -- stability fraction.  Regime C.
# ---------------------------------------------------------------------------------------------

"""
    stability_fraction(trajs; horizon = nothing)

Metric #16: the fraction of replicas that complete the horizon without a NaN.

Returns `(; fraction, nstable, n, first_nan)`, where `first_nan[r]` is the first non-finite column
of replica `r` or `nothing`.

!!! warning
    The binomial bound this feeds is **conditional on the initial-condition and forcing pool**.
    20/20 gives a one-sided 95% lower bound of 0.86 on `P(stable)` only if the IC and the forcing
    seed vary between replicas; the archived HIT ensembles give every replica the same `ustart` and
    vary only the model seed, so a bound from them is conditional on one initial condition.
"""
function stability_fraction(trajs::AbstractVector{<:AbstractMatrix}; horizon = nothing)
    first_nan = Union{Int,Nothing}[]
    for t in trajs
        T = horizon === nothing ? size(t, 2) : min(horizon, size(t, 2))
        k = nothing
        for n in 1:T
            if !all(isfinite, view(t, :, n))
                k = n
                break
            end
        end
        push!(first_nan, k)
    end
    nstable = count(isnothing, first_nan)
    return (; fraction = nstable / length(trajs), nstable, n = length(trajs), first_nan)
end

# ---------------------------------------------------------------------------------------------
# 17, 18 -- spread and skill.  Regime C.
# ---------------------------------------------------------------------------------------------

"""
    spread_skill(ens, truth; correct = true)

Metrics #17 and #18. `ens` is `K x N_Q x M` -- `K` verification instances, `M` members -- and
`truth` is `K x N_Q`.

    spread = sqrt(mean_k mean over members of (x - member mean)^2, with the M-1 denominator)
    skill  = sqrt(mean_k (ensemble mean - truth)^2)
    ratio  = sqrt((M+1)/M) * spread / skill

Below 1 is over-confident, above 1 over-dispersed.

!!! warning
    Apply the finite-`M` factor. A reliable ensemble satisfies
    `E[RMSE^2] = ((M+1)/M) E[spread^2]`, so an **uncorrected** ratio reads `sqrt(M/(M+1))` for a
    perfect ensemble: 0.953 at `M = 10`, and **0.913 at `M = 5`**, which is the archive's ensemble
    size. Against a [0.8, 1.25] band that consumes a third of the lower margin at `M = 5`. Either
    correct the ratio, as `correct = true` does, or restate the band for a named `M`.

!!! warning
    Both spread and skill are expectations over initial conditions, so this needs `K` much greater
    than 1. Every archived online run is a single trajectory from one initial condition, which is
    why the lead-resolved form (#17) is not computable from the archive and the version that is
    computable from it is the **climatological** one (#18): evaluated where the members have fully
    decorrelated from their common start, it tests whether the long-run variance matches, which is
    close to what KS already measures and is explicitly not error growth. Report it labelled as
    climatological, never as a lead.
"""
function spread_skill(ens::AbstractArray{<:Real,3}, truth::AbstractMatrix; correct::Bool = true)
    K, nq, M = size(ens)
    @assert size(truth) == (K, nq)
    @assert M > 1 "spread needs at least two members"
    sp2 = 0.0
    sk2 = 0.0
    per_qoi_sp = zeros(nq)
    per_qoi_sk = zeros(nq)
    for k in 1:K, i in 1:nq
        m = view(ens, k, i, :)
        mbar = sum(m) / M
        v = sum(abs2, m .- mbar) / (M - 1)
        e = (mbar - truth[k, i])^2
        per_qoi_sp[i] += v
        per_qoi_sk[i] += e
        sp2 += v
        sk2 += e
    end
    f = correct ? sqrt((M + 1) / M) : 1.0
    spread = sqrt(sp2 / (K * nq))
    skill = sqrt(sk2 / (K * nq))
    # A zero skill would make the ratio infinite rather than wrong; report NaN so a degenerate
    # QoI is visible as missing instead of dominating a plot axis.
    ratio_or_nan(sp, sk) = sk > 0 ? f * sqrt(sp) / sqrt(sk) : NaN
    return (; ratio = ratio_or_nan(sp2 / (K * nq), sk2 / (K * nq)), spread, skill, M, K,
            correction = f,
            per_qoi = [ratio_or_nan(per_qoi_sp[i] / K, per_qoi_sk[i] / K) for i in 1:nq],
            # Per-QoI spread and skill in their own units, not only their ratio. The ratio cannot
            # answer "has this lead saturated?", which is what the lead-resolved form has to report
            # rather than extrapolate -- see `saturation_lead`.
            per_qoi_spread = [sqrt(per_qoi_sp[i] / K) for i in 1:nq],
            per_qoi_skill = [sqrt(per_qoi_sk[i] / K) for i in 1:nq])
end

# ---------------------------------------------------------------------------------------------
# 17 lead-resolved, and RH-3.  Regime C, and computable only from the multi-IC ensemble D6.
# ---------------------------------------------------------------------------------------------
#
# Both metrics below are expectations over **initial conditions** at a fixed lead, which is the
# whole reason D6 exists: every archived online run is one trajectory from one initial condition
# (`6_online_TO_LRS.jl:56-59`), so each lead has exactly one verification instance and an RMSE from
# one sample is not an RMSE.
#
# The layout they share is `fc :: K x N_Q x M x L` and `truth :: K x N_Q x L`, where the fourth axis
# runs over a **sorted grid of step leads** rather than over every step of the forecast. The grid is
# a few dozen entries against 1208 steps, which is what keeps the whole object a few MB rather than
# a hundred.

"""
    lead_grid(T_int; dt, multipliers = (0.25, 0.5, 1, 2, 5, 10), nlead)

Per-QoI lead grids, in **steps**, from a per-QoI integral timescale in physical time.

🔴 One grid cannot serve six QoIs. `T_int` spans 0.0082-0.3017 TU on the reference `dQ`, a factor
**36.8** (`meta_files/claude_memory.md` gotcha #30), so a grid expressed in units of `t_int` is six
different grids and a single grid in physical time is right for at most one QoI. Hence: physical
time, one grid per QoI, and runs long enough that the slowest QoI's largest multiple fits inside
them -- 10 x 0.3017 TU = 1207 steps against a 1208-step forecast.

A lead the run cannot support raises. It is never silently clipped, because a clipped lead reads as
a saturated one and saturation is the thing being measured.

!!! warning
    The default multipliers are anchored on `T_int` of the **correction**, which is what set the
    3.02 TU forecast length. The level decorrelates far more slowly (`rho_1(q)` is about 1, gotcha
    #28), so the level's saturation lead may lie beyond the largest multiple. That is a reportable
    outcome, not a licence to extrapolate -- see [`saturation_lead`](@ref).
"""
function lead_grid(T_int::AbstractVector; dt, multipliers = (0.25, 0.5, 1, 2, 5, 10),
                   nlead::Integer)
    grids = Vector{Vector{Int}}(undef, length(T_int))
    for (i, T) in pairs(T_int)
        T > 0 || throw(ArgumentError("lead_grid: T_int[$i] = $T is not positive"))
        g = sort(unique(max.(1, round.(Int, collect(multipliers) .* (T / dt)))))
        maximum(g) <= nlead || throw(ArgumentError(
            "lead_grid: QoI $i asks for a lead of $(maximum(g)) steps but the forecast is only " *
            "$nlead steps. Lengthen the run or drop the multiplier; never extrapolate past the grid."))
        grids[i] = g
    end
    return grids
end

"""
    union_grid(leads)

The sorted union of a set of per-QoI lead grids -- the fourth axis `fc` and `truth` are built on.
"""
union_grid(leads::AbstractVector{<:AbstractVector{<:Integer}}) =
    sort(unique(reduce(vcat, leads)))

"""
    lead_positions(grid, leads)

Map each per-QoI lead to its column in `grid`, refusing any lead the grid does not carry.
"""
function lead_positions(grid::AbstractVector{<:Integer},
                        leads::AbstractVector{<:AbstractVector{<:Integer}})
    pos = Dict{Int,Int}(g => j for (j, g) in pairs(grid))
    return [[haskey(pos, g) ? pos[g] :
             throw(ArgumentError("lead $g is not on the grid $(collect(grid)); build the " *
                                 "forecast array from `union_grid` of the same leads"))
             for g in gl] for gl in leads]
end

"""
    spread_skill_by_lead(fc, truth; grid, leads = nothing, correct = true)

Metric **#17**, lead-resolved. `fc` is `K x N_Q x M x L` and `truth` is `K x N_Q x L`, both built on
the same sorted `grid` of step leads; `leads` gives the per-QoI subset of that grid to report and
defaults to the whole grid for every QoI.

Returns `(; grid, leads, ratio, spread, skill, pooled_ratio, K, M, correction)`, where `ratio[i]`,
`spread[i]` and `skill[i]` are vectors over `leads[i]`.

Everything is per QoI: `sd(dQ)` spans four orders of magnitude across the six bands, so a pooled
spread or skill in raw units is an enstrophy statistic with the energy bands contributing nothing.
`pooled_ratio` is kept because it is dimensionless and therefore does pool honestly.

!!! warning
    The finite-`M` correction is on by default and must stay on. An uncorrected ratio reads
    `sqrt(M/(M+1)) = 0.953` at `M = 10` for a **perfect** ensemble, which against a [0.8, 1.25] band
    spends a fifth of the lower margin on an artefact.
"""
function spread_skill_by_lead(fc::AbstractArray{<:Real,4}, truth::AbstractArray{<:Real,3};
                              grid::AbstractVector{<:Integer}, leads = nothing,
                              correct::Bool = true)
    K, nq, M, L = size(fc)
    size(truth) == (K, nq, L) ||
        throw(DimensionMismatch("truth is $(size(truth)), expected $((K, nq, L))"))
    length(grid) == L ||
        throw(DimensionMismatch("grid has $(length(grid)) entries, forecast has $L lead columns"))
    issorted(grid) && allunique(grid) || throw(ArgumentError("grid must be sorted and unique"))
    ll = leads === nothing ? [collect(grid) for _ in 1:nq] : leads
    length(ll) == nq || throw(DimensionMismatch("leads has $(length(ll)) grids, expected $nq"))
    idx = lead_positions(grid, ll)

    # One `spread_skill` call per grid column, then pick out the per-QoI entries each grid asks for.
    per = [spread_skill(view(fc, :, :, :, j), view(truth, :, :, j); correct) for j in 1:L]
    ratio = [[per[j].per_qoi[i] for j in idx[i]] for i in 1:nq]
    spread = [[per[j].per_qoi_spread[i] for j in idx[i]] for i in 1:nq]
    skill = [[per[j].per_qoi_skill[i] for j in idx[i]] for i in 1:nq]
    return (; grid = collect(grid), leads = ll, ratio, spread, skill,
            pooled_ratio = [p.ratio for p in per], K, M, correction = per[1].correction)
end

"""
    climatological_skill(ref_row, M)

The skill a *calibrated* ensemble converges to once its members have decorrelated from their common
initial condition: members and truth become independent draws from the same marginal, so
`E[(mean - y)^2] = sigma^2 (1 + 1/M)`.

`ref_row` is the reference series for one QoI over the verification window; returns
`sigma * sqrt(1 + 1/M)`. At that level a corrected spread-skill ratio reads 1, which is what makes
this the right yardstick for saturation and not merely a plausible one.
"""
function climatological_skill(ref_row::AbstractVector, M::Integer)
    n = length(ref_row)
    n > 1 || return NaN
    mu = sum(ref_row) / n
    sigma = sqrt(sum(abs2, ref_row .- mu) / (n - 1))
    return sigma * sqrt(1 + 1 / M)
end

"""
    saturation_lead(leads, skill; sat_level, frac = 0.95)

The first lead in `leads` whose `skill` has reached `frac` of the climatological level, or
`nothing` when the grid never gets there.

🔴 `nothing` is an answer, and it is the one to report. If the longest lead has not saturated, say
so; do not fit a curve through the grid and read a crossing off it. The grid is the evidence, and a
saturation lead outside it has not been measured.
"""
function saturation_lead(leads::AbstractVector{<:Integer}, skill::AbstractVector;
                         sat_level, frac = 0.95)
    length(leads) == length(skill) ||
        throw(DimensionMismatch("leads and skill differ in length"))
    (isfinite(sat_level) && sat_level > 0) || return nothing
    for (j, s) in pairs(skill)
        isfinite(s) && s >= frac * sat_level && return leads[j]
    end
    return nothing
end

"""
    rank_histogram_by_lead(fc, truth; grid, leads = nothing, rng, nboot = 1000,
                           blocklen = nothing, ties = :random)

**RH-3**: a rank histogram per (QoI, lead), each over the `K` initial conditions.

Same layout as [`spread_skill_by_lead`](@ref). Returns `(; grid, leads, hist)` with `hist[i][t]` the
full [`rank_histogram`](@ref) result for QoI `i` at `leads[i][t]`.

!!! warning
    The verification axis here is **initialisation time**, not time within a trajectory, and the
    instances on it are not independent: D6's ICs are 0.4818 TU apart on average and 0.25 TU apart
    at their narrowest, against a slowest `T_int` of 0.3017 TU (gotcha #34). The inherited block
    bootstrap is therefore not optional, and its default block length -- twice the integral
    correlation time of the rank series itself, measured along the IC axis -- is the right
    estimator. Never claim `K` independent instances.

!!! warning
    `M = 10` gives 11 bins, and `K = 180` about 16 instances per bin. Comfortable for the
    Jolliffe-Primo contrasts, marginal for the omnibus chi-squared -- the second reason the
    contrasts are primary.
"""
function rank_histogram_by_lead(fc::AbstractArray{<:Real,4}, truth::AbstractArray{<:Real,3};
                                grid::AbstractVector{<:Integer}, leads = nothing,
                                rng = Random.default_rng(), nboot::Int = 1000,
                                blocklen = nothing, ties::Symbol = :random)
    K, nq, M, L = size(fc)
    size(truth) == (K, nq, L) ||
        throw(DimensionMismatch("truth is $(size(truth)), expected $((K, nq, L))"))
    length(grid) == L ||
        throw(DimensionMismatch("grid has $(length(grid)) entries, forecast has $L lead columns"))
    ll = leads === nothing ? [collect(grid) for _ in 1:nq] : leads
    idx = lead_positions(grid, ll)
    hist = [[rank_histogram(Array(view(fc, :, i, :, j)), Array(view(truth, :, i, j));
                            ties, rng, nboot, blocklen) for j in idx[i]] for i in 1:nq]
    return (; grid = collect(grid), leads = ll, hist)
end

# ---------------------------------------------------------------------------------------------
# 26 -- clamp census.  Regimes 0, B and C.
# ---------------------------------------------------------------------------------------------

"""
    clamp_census(q_star; threshold = 1e-2)

Metric #26: how often the inherited stabiliser would fire on a record.

The stabiliser is `any(abs.(q_star) .< 1e-2) && (dQ .= 0)` in the shared `LinReg` path
(`time_series_methods.jl:162,165,190,193`). It tests the **unscaled** predictor and zeroes the
**entire** `dQ` vector whenever **any** QoI is small.

Returns `(; rate, nfired, nsteps, per_qoi_rate, threshold, fired)`, where `per_qoi_rate[i]` is how
often QoI `i` alone was below the threshold -- which is what says whether one band drives the whole
census.

!!! warning
    Two different numbers hide under this name and only one of them is model-specific.

    The rate computed here is a property of the **record**: it counts steps whose predictor is
    small, and it is computable for any run, including one produced by a model whose code path has
    no clamp. What is model-specific is whether the clamp actually *fired*: it lives in the
    `LinReg` path only, while the data-driven noise model's `get_next_item_timeseries`
    (`time_series_methods.jl:37-39`) is a bare `rand` with no clamp at all. So report this rate for
    every record touched, and attribute firing only to the LinReg lineage. Any stability or
    accuracy comparison between the two carries that asymmetry and has to state it.

!!! warning
    The training and deployment thresholds differ. Taylor-Green's training path drops rows at
    `0.5e-2` (`taylor-green/7_train_LinReg.jl:35-36`) while deployment clamps at `1e-2`, a factor
    two apart, so rows with a predictor in `[0.5e-2, 1e-2)` are trained on and then clamped at run
    time. HIT and the channel filter nothing during training.
"""
function clamp_census(q_star::AbstractMatrix; threshold = 1e-2)
    nq, T = size(q_star)
    fired = falses(T)
    per = zeros(Int, nq)
    for n in 1:T
        hit = false
        for i in 1:nq
            if abs(q_star[i, n]) < threshold
                per[i] += 1
                hit = true
            end
        end
        fired[n] = hit
    end
    nfired = count(fired)
    return (; rate = nfired / T, nfired, nsteps = T, per_qoi_rate = per ./ T, threshold, fired)
end
