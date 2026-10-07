# This file implements various time series methods for generating time series data:
# - `Reference_reader`: Reads time series data from a reference dataset.
# - `MVG_sampler`: Samples from a multivariate Gaussian distribution fitted to the data.
# - `Resampler`: Resamples from a given dataset.
# - `ANN`: Uses a trained artificial neural network to predict the next item in the time series.
# - `LinReg`: Uses a linear regression model to predict the next item in the time series.


struct Reference_reader
    vals
    index::Array{Int64, 0}
    stds
    means
    function Reference_reader(vals)
        index = ones(Int)
        index[] = 2
        stds = std(vals, dims = 2)
        means = mean(vals, dims = 2)
        new(vals, index, stds, means)
    end
end

"""
    TURBULENCE_GATE

Below this QoI magnitude the TO SGS term is switched off entirely for that step.

🔑 **This is a laminar-start gate, not a numerical guard** (Rik, 2026-09-15, from recollection --
no written source found; treat as the design intent rather than as a citation). It was introduced
for the **Taylor-Green vortex**, which begins laminar: until the cascade has filled the smallest
resolved scales there is no sub-grid content for the TO correction to represent, and applying one
would be forcing a flow that has no turbulence yet. `any(...)` and a whole-vector `dQ .= 0` are
therefore correct by intent -- the question is "is the flow turbulent yet?", answered over all
bands at once, not "is band i singular?".

⚠️ The nearby comment `# set dQ_i to 0 if q*_i is 0` describes a per-band guard that this is not
and should not become. Making it per-band would let the SGS term act on a laminar flow through
whichever bands happen to have content.

🔴 **On a statistically stationary turbulent testbed the correct firing rate is ZERO, so a nonzero
rate is an alarm about the RUN, not a property of the model.** HIT launches from a spun-up field
and paper 2's archived runs never fired (min |q*| 2.19e-2). The 2026-09-15 rebaselined runs fire on
0.4-1.8% of steps, all through `E[16,32]`, because that band parks at ~0.010 -- below the
regenerated reference's own minimum of 0.0243 and a factor 6.6 under its median. The gate is
reporting that the run has left the attractor; it is not what put it there.

⚠️ **The value is testbed-specific and 1e-2 was chosen for Taylor-Green.** On HIT it is 0.15x the
reference's median `E[16,32]` and 0.41x its minimum -- close enough to the physical range that a
mildly degraded run trips it. A per-testbed threshold, or one set as a fraction of the reference
band's own distribution, would be better; it is left alone here because every run to date used this
value and changing it would split the comparison.

🔴 **The gate lives only in the `LinReg`/`ANN` path.** `MVG_sampler` never receives `q_star`, so the
DDN has no laminar-start gate at all -- a real defect on Taylor-Green, where it would inject an SGS
term into a laminar flow. On HIT it means the two closures are not treated alike; see
`analysis/results.md`.
"""
const TURBULENCE_GATE = 1e-2

function get_next_item_timeseries(time_series_method::Reference_reader)
    val = time_series_method.vals[:,time_series_method.index[]]
    time_series_method.index[] += 1
    return val
end

"""
    MVG_sampler(dQ_data, rng; spinnup_data = nothing)

Paper 1's data-driven noise model (the **DDN**): one multivariate Gaussian fitted to `dQ`, sampled
i.i.d. at every step. State-independent by construction -- that is the point of it as a control.

# Warm-up ("pseudo spin-up")

`spinnup_data`, when given, is replayed column by column for its full width before any sampling
starts, exactly as `LinReg` replays it.

🔑 **It is "pseudo" because the DDN has no internal state to spin up.** `LinReg`'s warm-up does two
jobs: fill `q_hist` so the first prediction has a valid history, *and* drive the solver along the
recorded trajectory. The DDN has no history, so only the second job exists here -- and that job is
the whole reason D6 needs this. D6 forecasts from `K` initial conditions and compares closures at
matched leads; if the LRS is advanced through `nwarm` recorded steps and the DDN is not, the two
enter the forecast from **different physical states**, and every lead-resolved difference between
them carries that offset (memory #55).

🔴 **The warm-up does not consume `rng`.** The replay returns before the `rand` call, so member `m`'s
first *sampled* value is identical with and without a warm-up. This matches `LinReg`, and it is what
keeps a member seed meaning the same thing across the two closures and across warm-up lengths.

⚠️ The replayed columns are returned **unconverted**, again as `LinReg` does. D6's validation gate is
`dQ` bit-identity over the replayed window (memory #48); converting a Float64 record to Float32 on
the way out would break it.
"""
struct MVG_sampler
    dQ_distribution
    rng
    spinnup_data
    counter
    function MVG_sampler(dQ_data, rng; spinnup_data = nothing)
        dQ_distribution = fit(MvNormal, dQ_data .|> Float64)
        if !isnothing(spinnup_data)
            size(spinnup_data, 1) == length(dQ_distribution) || error(
                "MVG_sampler: spinnup_data has $(size(spinnup_data, 1)) rows but the fitted " *
                "distribution has $(length(dQ_distribution)) components")
            size(spinnup_data, 2) >= 1 ||
                error("MVG_sampler: spinnup_data has no columns; pass `nothing` for no warm-up")
        end
        new(dQ_distribution, rng, spinnup_data, zeros(Int))
    end
end

function get_next_item_timeseries(time_series_method::MVG_sampler)
    sd = time_series_method.spinnup_data
    # Replay first, and return BEFORE touching the rng -- see the note in the docstring.
    if !isnothing(sd) && time_series_method.counter[] < size(sd, 2)
        time_series_method.counter[] += 1
        return sd[:, time_series_method.counter[]]
    end
    return rand(time_series_method.rng, time_series_method.dQ_distribution) .|> Float32
end

struct Resampler
    vals
    rng
end

function get_next_item_timeseries(time_series_method::Resampler)
    # sample a random integer from 1 to the length of the data
    index = rand(time_series_method.rng, 1:size(time_series_method.vals, 2))
    return time_series_method.vals[:,index]
end

struct ANN
    model
    ps
    st
    scaling
    q_hist  # history of q values, newest first
    counter
    hist_var
    function ANN(file_name; q_hist = nothing)
        model, ps, st, scaling, hist_var = load_ANN(file_name)
        counter = zeros(Int)
        new(model, ps, st, scaling, q_hist, counter, hist_var)
    end
end

function get_next_item_timeseries(time_series_method::ANN, q_star)
    if !isnothing(time_series_method.q_hist)  # if the NN uses history
        if time_series_method.counter[] < size(time_series_method.q_hist, 2) # for the first few steps, directly read dQ
            time_series_method.counter[] += 1
            dQ = time_series_method.q_hist[:,end]
        else    # after that, predict dQ  (we now have enough history)
            input = vcat(q_star, time_series_method.q_hist[:]) 
            data = scale_input(input, time_series_method.scaling.in_scaling)
            pred = Lux.apply(time_series_method.model, data, time_series_method.ps, time_series_method.st)[1]
            dQ = scale_output(pred, time_series_method.scaling.out_scaling)
        end
        time_series_method.q_hist[:,2:end] = time_series_method.q_hist[:,1:end-1] # shift history
        if time_series_method.hist_var == :q
            time_series_method.q_hist[:,1] .= q_star + dQ                             # add new q to history
        elseif time_series_method.hist_var == :q_star
            time_series_method.q_hist[:,1] .= q_star
        end
    else    # if the NN does not use history, predict dQ directly from q_star
        input = q_star
        data = scale_input(input, time_series_method.scaling.in_scaling)
        pred = Lux.apply(time_series_method.model, data, time_series_method.ps, time_series_method.st)[1]
        dQ = scale_output(pred, time_series_method.scaling.out_scaling)
    end
    return dQ
end

"""
    LinReg(file_name, rng, ArrayType; q_hist = nothing, spinnup_data = nothing)

The linear-regression closure (M0): a level prediction `q^n = scale_output(c [x; 1] + η)`, with
`x` the scaled predictor + history and `η` the residual draw in **scaled units**.

# The residual `η`

  * **White (every file written before 2026-09-28):** `η ~ stoch_distr`, i.i.d. per step.
  * **AR(p) (optional, `p ≤ 2`, "M0ᶜ-ridge", `analysis/results_LSTMS.md` §12):** when the file
    carries the keys `"ar_phi"` (`p × n_qoi`, diagonal AR coefficients, one column per QoI) and
    `"ar_sigma_xi"` (`n_qoi × n_qoi` innovation covariance),

        η_n = μ_η + z_n,   z_n = Σ_{k=1..p} φ_k ⊙ z_{n-k} + ξ_n,   ξ_n ~ N(0, Σ_ξ),

    with `μ_η = mean(stoch_distr)`. The innovation `ξ`, not `η`, is what is drawn (plan §5 D2): the
    closure knows its own past draws. `η` enters exactly where the white draw does.

  * **Power-law scale (optional, step 4p, 2026-10-07; `ts_scale.jl`):** when the file carries
    `"scale_beta"`, `"scale_qref"`, `"scale_qclip"` (`n_qoi × 2`) and `"scale_sigma_eps"`,

        η_n = μ_η + f(q*_n) ⊙ ε_n,   ε_n ~ N(0, Σ_ε),   f_i = (clamp(q*_i, lo_i, hi_i) / q_ref,i)^β_i,

    with `q*` the predictor in physical units. One `rand(rng, MvNormal(0, Σ_ε))` replaces the white
    call, so the stream is LinReg1's; with `β = 0` and `Σ_ε = Σ` the draws equal the white ones bit
    for bit. Not combined with AR (refused).

🔴 **Without the AR keys the behaviour is bit-identical to the pre-AR code, RNG stream included**:
the white branch is the same `rand(rng, stoch_distr)` call at the same point, and nothing else
touches `rng` (`test/test_linreg_ar.jl` pins this against a verbatim copy of the old code).
With AR, one `rand(rng, MvNormal(0, Σ_ξ))` replaces that call, so a member seed consumes the same
number of normals per step in both variants -- which is what makes a CRN pairing of LinReg7 and
LinReg7_ar2 meaningful.

# Warm start of the AR state

During the replayed warm-up (`spinnup_data`) `dQ` is emitted verbatim, exactly as before. In
addition, on each of the **last `p` warm-up steps** the residual the record actually realised is
computed with the history the model holds at that point,

    z = scale(q* + dQ_replayed) − c [x; 1] − μ_η      (scaled units, out_scaling),

and pushed into the AR state, so the forecast's first `z` continues the data's residual rather
than starting from zero. ⚠️ A warm-up step whose history is not yet full (step index ≤ `hist_len`)
is skipped. **Fallback:** if fewer than `p` lags got filled that way (warm-up shorter than
`hist_len + p`, or no history at all), the whole state is replaced at the first prediction by an
approximately stationary draw: `AR_BURNIN` steps of the recursion from zero with fresh `ξ` draws.
This consumes `rng`, only in the AR path, and never in the D6 configuration (`nwarm = 100 ≫ h + p`).

# Turbulence gate

When `TURBULENCE_GATE` zeroes `dQ`, the AR state is **still advanced** (the `ξ` draw happens before
the gate is applied, as the white draw always did). The gate censors the closure's output, not its
noise process, so a gated step does not shift every later draw of the member's stream.
"""
struct LinReg
    c
    stoch_distr
    scaling
    q_hist
    spinnup_data
    counter
    hist_var
    include_predictor
    fitted_qois
    target
    rng
    ArrayType
    ar          # nothing (white η), or the AR(p) residual state -- see `load_ar_residual`
    scale       # nothing, or step 4p's power-law scale -- see `load_powerlaw_scale`

    function LinReg(file_name, rng, ArrayType; q_hist = nothing, spinnup_data = nothing)

        c, stoch_distr, scaling, hist_var, include_predictor, fitted_qois = load(file_name, "c", "stoch_distr", "scaling", "hist_var", "include_predictor", "fitted_qois")
        target = :q
        scaling = adapt(ArrayType, scaling)
        c= adapt(ArrayType, c)
        ar = load_ar_residual(file_name, stoch_distr)
        scale = load_powerlaw_scale(file_name, stoch_distr)
        !isnothing(ar) && !isnothing(scale) && error("$file_name: AR residual and power-law scale together are not implemented")

        counter = zeros(Int)
        if !isnothing(q_hist)
            @assert size(spinnup_data, 2) >= size(q_hist, 2) "Need spinnup data to fill history"
        end
        if !isnothing(spinnup_data) && isnothing(q_hist)
            @error "Spinnup not implemented without history"
        end
        new(c, stoch_distr, scaling, q_hist, spinnup_data, counter, hist_var, include_predictor, fitted_qois, target, rng, ArrayType, ar, scale)
    end
end

"""
    load_powerlaw_scale(file_name, stoch_distr)

Step 4p's optional scale of a `LinReg` file: `nothing` without the `"scale_*"` keys (every file
before 2026-10-07), else `(; beta, qref, qclip, eps_distr, mu_eta)`. Refuses a partial key set, a
size mismatch, a non-positive reference or clip range, or a missing `stoch_distr`.
"""
function load_powerlaw_scale(file_name, stoch_distr)
    keys_ = ("scale_beta", "scale_qref", "scale_qclip", "scale_sigma_eps")
    has = jldopen(f -> [haskey(f, k) for k in keys_], file_name, "r")
    any(has) || return nothing
    all(has) || error("$file_name has only some of the power-law scale keys $(keys_[has])")
    isnothing(stoch_distr) && error("$file_name: a power-law scale needs stoch_distr (its mean is μ_η)")
    beta, qref, qclip, S = load(file_name, keys_...)
    n = length(stoch_distr)
    length(beta) == n && length(qref) == n && size(qclip) == (n, 2) && size(S) == (n, n) ||
        error("$file_name: power-law scale sizes do not match $n QoIs")
    all(>(0), qref) && all(>(0), qclip) && all(qclip[:, 1] .<= qclip[:, 2]) ||
        error("$file_name: q_ref and the clip range must be positive and ordered")
    eps_distr = MvNormal(zeros(n), Matrix{Float64}(Symmetric(Matrix{Float64}(S))))
    return (; beta = Vector{Float64}(beta), qref = Vector{Float64}(qref), qclip = Matrix{Float64}(qclip),
            eps_distr, mu_eta = Vector{Float64}(mean(stoch_distr)))
end

"Steps of the AR recursion from zero used as the stationary-draw fallback (see `LinReg`)."
const AR_BURNIN = 2000

"Stationarity of a diagonal AR(p ≤ 2) with coefficients `phi` (`p × n`), per column."
function ar_stationary(phi::AbstractMatrix)
    p = size(phi, 1)
    return all(eachcol(phi)) do f
        p == 1 ? abs(f[1]) < 1 : (abs(f[2]) < 1 && f[2] + f[1] < 1 && f[2] - f[1] < 1)
    end
end

"""
    load_ar_residual(file_name, stoch_distr)

The optional AR(p) residual of a `LinReg` file: `nothing` when the file has no `"ar_phi"` key
(every pre-2026-09-28 file), else `(; phi, xi_distr, mu_eta, z, nz)` with `z` the `n × p` state
(column 1 newest) and `nz[]` how many lags were filled from data. Refuses a non-stationary or
malformed AR rather than deploying it.
"""
function load_ar_residual(file_name, stoch_distr)
    has_phi, has_sig = jldopen(file_name, "r") do f
        haskey(f, "ar_phi"), haskey(f, "ar_sigma_xi")
    end
    if !has_phi
        has_sig && error("$file_name has ar_sigma_xi but no ar_phi")
        return nothing
    end
    has_sig || error("$file_name has ar_phi but no ar_sigma_xi")
    isnothing(stoch_distr) && error("$file_name: an AR residual needs stoch_distr (its mean is μ_η)")
    phi, S = load(file_name, "ar_phi", "ar_sigma_xi")
    phi = Matrix{Float64}(phi)
    p, n = size(phi)
    n == length(stoch_distr) || error("ar_phi has $n columns, stoch_distr $(length(stoch_distr)) components")
    1 <= p <= 2 || error("AR order $p: only p = 1, 2 are implemented")
    ar_stationary(phi) || error("$file_name: AR coefficients $phi are not stationary")
    size(S) == (n, n) || error("ar_sigma_xi is $(size(S)), expected ($n, $n)")
    xi_distr = MvNormal(zeros(n), Matrix{Float64}(Symmetric(Matrix{Float64}(S))))
    return (; phi, xi_distr, mu_eta = Vector{Float64}(mean(stoch_distr)), z = zeros(n, p),
            nz = zeros(Int))
end

"AR order of a `LinReg`'s residual; 0 = white."
ar_order(m::LinReg) = isnothing(m.ar) ? 0 : size(m.ar.phi, 1)

"Push `znew` into the AR state (column 1 newest)."
function ar_push!(ar, znew)
    p = size(ar.z, 2)
    p > 1 && (ar.z[:, 2:p] .= ar.z[:, 1:(p - 1)])
    ar.z[:, 1] .= znew
    return ar
end

"One step of the AR recursion: draw ξ, advance the state, return the new `z`."
function ar_step!(ar, rng)
    xi = rand(rng, ar.xi_distr)
    znew = xi .+ vec(sum(ar.phi' .* ar.z; dims = 2))
    ar_push!(ar, znew)
    return znew
end

"""
    draw_eta(m::LinReg, q_star = nothing)

The residual draw in scaled units: `rand(rng, stoch_distr)` for a white `LinReg` (the pre-AR call,
unchanged), `μ_η + z_n` for an AR one, `μ_η + f(q*) ⊙ ε` with step 4p's scale (`q_star` in physical
units, required then).
"""
function draw_eta(m::LinReg, q_star = nothing)
    sc = m.scale
    if !isnothing(sc)
        isnothing(q_star) && error("a power-law scale needs q_star")
        eps = rand(m.rng, sc.eps_distr)
        return sc.mu_eta .+ powerlaw_factor(vec(Array(q_star)), sc.beta, sc.qref, sc.qclip) .* eps
    end
    ar = m.ar
    isnothing(ar) && return rand(m.rng, m.stoch_distr)
    p = size(ar.z, 2)
    if ar.nz[] < p            # fallback: approximately stationary state (see the docstring)
        ar.z .= 0
        for _ in 1:AR_BURNIN
            ar_step!(ar, m.rng)
        end
        ar.nz[] = p
    end
    return ar.mu_eta .+ ar_step!(ar, m.rng)
end

"The regression input `[x; 1]` (scaled) of the history branch, exactly as the prediction uses it."
function linreg_data(time_series_method::LinReg, q_star)
    n_qoi = size(q_star,1)
    q_star_sc = scale_input(q_star, time_series_method.scaling.in_scaling)
    if time_series_method.hist_var == :q_star_q
        q_hist_sc1 = scale_input(time_series_method.q_hist[1:n_qoi,:], time_series_method.scaling.in_scaling)
        q_hist_sc2 = scale_input(time_series_method.q_hist[n_qoi+1:end,:], time_series_method.scaling.in_scaling)
        q_hist_sc = cat(q_hist_sc1, q_hist_sc2, dims = 1)
    else
        q_hist_sc = scale_input(time_series_method.q_hist, time_series_method.scaling.in_scaling)
    end

    if time_series_method.include_predictor
        input = vcat(q_star_sc, q_hist_sc[:])
    else
        input = q_hist_sc
    end

    return vcat(input,ones(eltype(input), (1,1)))
end

"""
    ar_warm_residual!(m::LinReg, q_star, dQ)

Warm start: the residual the replayed step realised, `scale(q* + dQ) − c [x; 1] − μ_η`, from the
history the model holds now (call BEFORE the history shift), pushed into the AR state.
"""
function ar_warm_residual!(m::LinReg, q_star, dQ)
    ar = m.ar
    n = length(ar.mu_eta)
    data = linreg_data(m, q_star)
    mu = zeros(n)
    mu[m.fitted_qois] .= vec(Array(m.c * data))
    lev = vec(Array(scale_input(q_star + dQ, m.scaling.out_scaling)))
    ar_push!(ar, lev .- mu .- ar.mu_eta)
    ar.nz[] = min(ar.nz[] + 1, size(ar.z, 2))
    return ar
end

function get_next_item_timeseries(time_series_method::LinReg, q_star)
    if !isnothing(time_series_method.q_hist)  # if the model uses history
        n_qoi = size(q_star,1)
        nspin = size(time_series_method.spinnup_data,2)
        if time_series_method.counter[] < nspin # for the first few steps, directly read dQ
            time_series_method.counter[] += 1
            dQ = time_series_method.spinnup_data[1:n_qoi, time_series_method.counter[]]
            # AR warm start: the last p replayed steps whose history is full (see the docstring)
            if !isnothing(time_series_method.ar)
                j = time_series_method.counter[]
                if j > nspin - ar_order(time_series_method) && j > size(time_series_method.q_hist, 2)
                    ar_warm_residual!(time_series_method, q_star, dQ)
                end
            end
        else    # after that, predict dQ  (we now have enough history)
            data = linreg_data(time_series_method, q_star)
            if !isnothing(time_series_method.stoch_distr)
                pred = draw_eta(time_series_method, q_star).|> Float32 |> adapt(time_series_method.ArrayType)
            else
                pred = zeros(eltype(data), (n_qoi,1)) |> adapt(time_series_method.ArrayType)
            end

            pred[time_series_method.fitted_qois,:] += time_series_method.c * data

            pred = scale_output(pred, time_series_method.scaling.out_scaling)[:]

            if time_series_method.target == :dq
                dQ = pred
                # set dQ_i to 0 if q*_i is 0
                any(abs.(q_star) .< TURBULENCE_GATE) && (dQ .= 0)
            elseif time_series_method.target == :q
                dQ = pred - q_star
                any(abs.(q_star) .< TURBULENCE_GATE) && (dQ .= 0)
            end
        end
        time_series_method.q_hist[:,2:end] = time_series_method.q_hist[:,1:end-1] # shift history
        if time_series_method.hist_var == :q
            time_series_method.q_hist[:,1] .= q_star + dQ                             # add new q to history
        elseif time_series_method.hist_var == :q_star
            time_series_method.q_hist[:,1] .= q_star
        elseif time_series_method.hist_var == :q_star_q
            time_series_method.q_hist[1:n_qoi,1] .= q_star + dQ
            time_series_method.q_hist[n_qoi+1:end,1] .= q_star
        end
    else    # if the model does not use history, predict dQ directly from q_star
        q_star_sc = scale_input(q_star, time_series_method.scaling.in_scaling)
        data = vcat(q_star_sc, ones(eltype(q_star_sc), (1,1)))
        if !isnothing(time_series_method.stoch_distr)
            pred = draw_eta(time_series_method, q_star) |> adapt(time_series_method.ArrayType)
        else
            pred = zeros(eltype(q_star_sc), n_qoi)
        end
        pred[time_series_method.fitted_qois,:] += time_series_method.c * data
        pred = scale_output(pred, time_series_method.scaling.out_scaling)[:]

        if time_series_method.target == :dq
            dQ = pred
            any(abs.(q_star) .< TURBULENCE_GATE) && (dQ .= 0)
        elseif time_series_method.target == :q
            dQ = pred - q_star
            any(abs.(q_star) .< TURBULENCE_GATE) && (dQ .= 0)
        end
    end
    return dQ
end


# ---------------------------------------------------------------------------------------------
# Which closures are handed the current predictor
# ---------------------------------------------------------------------------------------------

"""
    needs_qstar(time_series_method)

Whether `to_sgs_term` must compute the current predictor `q*` and pass it on to
[`get_next_item_timeseries`](@ref).

🔑 **This trait replaces a literal type list that used to sit inside `to_sgs_term`**
(`typeof(...) in [MVG_sampler, Resampler]` / `[ANN, LinReg]`). Adding a closure meant editing a
branch buried in the middle of the SGS assembly, which is the same shape of hazard as the live
`model_index = 2` that once let training and deployment disagree silently (`claude_memory.md` #55).
Adding a closure is now one line **here**, next to the docstring that says so.

🔴 **There is deliberately no fallback method.** An unregistered closure raises
`MethodError: no method matching needs_qstar(::Foo)`, which names exactly what is missing. A
`needs_qstar(::Any) = false` default would instead silently pick the no-predictor branch and fail
later, deep inside `get_next_item_timeseries`, with a message about the wrong thing.
"""
function needs_qstar end

needs_qstar(::Reference_reader) = false
needs_qstar(::MVG_sampler) = false
needs_qstar(::Resampler) = false
needs_qstar(::ANN) = true
needs_qstar(::LinReg) = true

export get_next_item_timeseries, Reference_reader, MVG_sampler, Resampler, ANN
export needs_qstar