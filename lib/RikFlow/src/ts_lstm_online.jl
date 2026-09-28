# M4 -- the deployed closure. `StochLSTM`, as `to_sgs_term` sees it.
#
# 🔑 Stdlib-only, like `ts_lstm.jl`, and for the same reason: this is the code that runs inside
# the solver, so it is the code the verification matrix has to be able to reach. It needs
# `HistoryBuffer`/`inputvec` (ts_history.jl), `scale_input`/`scale_output` (ts_scaling.jl) and
# `lstm_step!` (ts_lstm.jl), all of which are stdlib, and it deliberately needs nothing else --
# no Adapt, no CUDA, no Lux.
#
# ⚠️ CPU by design. `to_sgs_term` already brings `dQ` back to the host on the next line
# (`dQ = Array(dQ)`), and at H = 60 / N_Q = 6 the cell is far too small to amortise a kernel
# launch. This is the same reasoning S4 uses: the cost here is launch-bound, not FLOP-bound.

# `needs_qstar` and `get_next_item_timeseries` are declared in time_series_methods.jl, which
# RikFlow includes before this file. The test suite includes this file bare, without that one, so
# the generics have to be brought into existence when they are absent. Adding methods is the same
# either way.
if !isdefined(@__MODULE__, :needs_qstar)
    function needs_qstar end
end
if !isdefined(@__MODULE__, :get_next_item_timeseries)
    function get_next_item_timeseries end
end

"""
    StochLSTM(spec, weights, scaling; spinnup_data, rng, gate, stochastic = true, T = Float32)

M4 deployed as a TO closure: one `dQ` per solver step, from a stochastic LSTM.

# The warm-up, which is the part that is new

`LinReg` and `MVG_sampler` both replay `spinnup_data` before they start predicting, and their
replays do at most two jobs: return the recorded `dQ` so the solver follows the recorded
trajectory, and (for `LinReg`) fill the lag window. 🔑 **M4's replay does a third job — it drives
the recurrence, so `(h, c)` are charged before the first prediction.** The cell is stepped on the
recorded inputs, teacher-forced, and its output is thrown away.

🔴 **The replay still draws nothing from `rng`.** That is V38's invariant — a member seed has to
mean the same thing across closures and across warm-up lengths — and M4 cannot satisfy it by
returning early the way the other two do, because it has to run the forward pass. It satisfies it
instead by using the posterior mean, `sample_latent = false`. Member `m`'s first *sampled* value
is therefore identical with and without a warm-up, exactly as for `LinReg` and `MVG_sampler`.

⚠️ **The replayed columns are returned unconverted**, again as both other closures do. D6's
validation gate is `dQ` bit-identity over the replayed window (`claude_memory.md` #48); converting
a Float64 record to the model's Float32 on the way out would break it.

# Precision: the model is Float32, the solver is Float64

🔴 **The conversion happens at the two boundaries of `get_next_item_timeseries` and nowhere else.**
`q*` arrives at the solver's precision and is scaled and cast **down** to `T` for the cell; the
predicted level is cast **up** to `eltype(q*)` before `dQ = qhat - q*` is formed, so the closure
returns the solver's precision whatever `T` and the `Scaling` happen to be. The lag window is
stored in `T` (`_push_scaled!`), which is the model's own state, not an output.

⚠️ Do not let promotion do this implicitly. It gives the right answer while the `Scaling` is
Float64 and quietly stops as soon as one fitted on a Float32 record is loaded -- the same shape of
trap as the Float32 `Re` that used to win a `params_track` splat (`claude_memory.md` #57).

⚠️ **`nwarm` must now cover the recurrence's memory, not just the lag window.** `LinReg`'s 100
steps were sized from the measured ACF; M4's requirement is its own and is larger in general.
Measure `‖h_t − h_t^∞‖` against warm-up length on tracked data and report the number chosen.

# RNG consumption, per predicted step

`n_latent` draws for `z`, then `N_Q` draws for the emission — in that order, and only after the
warm-up. Nothing else consumes `rng`.

# Window mode (`spec.window = W > 0`)

The recurrent state is **reset for every prediction** and the last `W` inputs -- this step's and
the `W - 1` before it -- are replayed from `h = c = 0`. The deployed prediction is then exactly the
last output of a training segment of length `W`, which is what lets training use every row as the
end of a short window.

🔴 **The latent draw belongs to the PHYSICAL STEP, not to the window it is replayed in.** Step `t`
is fed through the cell in `W` consecutive windows. Drawing `z_t` afresh each time would give the
model a different noise history at every prediction, so consecutive corrections would not come
from one noise realisation at all. So each step's standard-normal `eps_t` is drawn ONCE, when the
step is first predicted, and stored beside its input; every replay forms
`z_t = mu_z(x_t) + sigma_z(x_t) .* eps_t` from the stored pair. Because the encoder reads only
`x_t`, which is also stored, the replayed `z_t` is bit-identical to the one first used.

- **Warm-up steps enter the window with `eps = 0`**, i.e. the posterior mean -- the replay draws
  nothing (V38), as in the persistent mode. That touches only the first `W - 1` predictions.
- The warm-up must be at least `W - 1` steps, so the first prediction already sees a full window.
- Per predicted step the `rng` gives `n_latent` draws for `eps_t`, then `N_Q` for the emission --
  the same order and count as the persistent mode.

# Fields
- `spec`, `weights`: the model. `check_shapes` is run at construction.
- `scaling`: the `(in_scaling, out_scaling)` pair, applied exactly as `LinReg` applies it.
- `gate`: the turbulence gate. M4 sits in the `q*`-consuming path, so it inherits `LinReg`'s gate
  rather than introducing a third convention -- see `TURBULENCE_GATE`'s docstring, and note that
  `MVG_sampler` has none, which is a known asymmetry.
- `stochastic`: `false` returns the emission mean instead of a draw. A diagnostic, not a mode to
  run the paper on.
- `tie_noise`: window mode only. `true` (default) draws each step's latent once and replays it;
  `false` re-draws the whole window at every prediction -- an ablation, not a model.
"""
struct StochLSTM{T,S}
    spec::LSTMSpec
    weights::LSTMWeights{T}
    scaling::S
    state::LSTMState{T}
    buf::HistoryBuffer{T}
    spinnup_data::Union{Nothing,AbstractMatrix}
    counter::Vector{Int}
    rng::Any
    gate::Float64
    stochastic::Bool
    scratch::Vector{T}
    # window mode only (`spec.window = W > 0`; zero columns otherwise): the last `W` inputs and
    # their latent draws, oldest in column 1, and how many columns are filled
    xwin::Matrix{T}
    ewin::Matrix{T}
    nwin::Vector{Int}
    tie_noise::Bool
    eprev::Vector{T}        # the previous step's emission noise, for `scaling.eta_ar`

    function StochLSTM(spec::LSTMSpec, weights::LSTMWeights{T}, scaling;
                       spinnup_data = nothing, rng, gate::Real = 1e-2,
                       stochastic::Bool = true, tie_noise::Bool = true) where {T}
        check_shapes(weights, spec)
        nq = spec.hist.n_qoi
        W = spec.window
        if W > 0
            nw = spinnup_data === nothing ? 0 : size(spinnup_data, 2)
            nw >= W - 1 || error(
                "StochLSTM: window = $W needs a warm-up of at least $(W - 1) steps so the first " *
                "prediction sees a full window; got $nw.")
        end
        if spinnup_data !== nothing
            size(spinnup_data, 1) >= nq || error(
                "StochLSTM: spinnup_data has $(size(spinnup_data, 1)) rows but N_Q = $nq")
            size(spinnup_data, 2) >= spec.hist.h || error(
                "StochLSTM: the warm-up is $(size(spinnup_data, 2)) steps but the lag window " *
                "needs $(spec.hist.h). The history would still be part zero at the first " *
                "prediction.")
        end
        new{T,typeof(scaling)}(spec, weights, scaling, LSTMState(spec, T),
                               HistoryBuffer(spec.hist, T), spinnup_data, zeros(Int, 1), rng,
                               Float64(gate), stochastic, zeros(T, spec.hist.n_qoi),
                               zeros(T, n_input(spec), W), zeros(T, spec.n_latent, W), zeros(Int, 1),
                               tie_noise, zeros(T, spec.hist.n_qoi))
    end
end

# Append one step's input and latent draw to the window, dropping the oldest once it is full.
# A column loop, not `xwin[:, 1:W-1] .= xwin[:, 2:W]`: an aliased broadcast copies (allocates).
function _push_window!(m::StochLSTM{T}, x, eps) where {T}
    W = m.spec.window
    if m.nwin[] == W
        @inbounds for j in 1:(W - 1), i in axes(m.xwin, 1)
            m.xwin[i, j] = m.xwin[i, j + 1]
        end
        @inbounds for j in 1:(W - 1), i in axes(m.ewin, 1)
            m.ewin[i, j] = m.ewin[i, j + 1]
        end
    else
        m.nwin[] += 1
    end
    j = m.nwin[]
    @inbounds for i in axes(m.xwin, 1)
        m.xwin[i, j] = T(x[i])
    end
    @inbounds for i in axes(m.ewin, 1)
        m.ewin[i, j] = eps === nothing ? zero(T) : T(eps[i])
    end
    return m
end

# Reset the recurrence and replay the filled window, oldest first, each step with its OWN stored
# latent draw. Leaves the last step's output in `m.state`.
function _replay_window!(m::StochLSTM)
    reset!(m.state)
    if is_dense(m.spec)
        dense_window!(m.state, m.weights, m.spec, view(m.xwin, :, 1:m.nwin[]), view(m.ewin, :, 1:m.nwin[]))
        return m
    end
    if m.spec.prior === :learned
        # 🔑 ewin holds the stored LATENTS z of the earlier steps, and the fresh standard-normal draw
        # of the newest one: replay the earlier steps, draw the newest z from the learned prior at the
        # resulting h_{n-1}, store it (later windows reuse it), then take the last step with it
        k = m.nwin[]
        for j in 1:(k - 1)
            lstm_step!(m.state, m.weights, m.spec, view(m.xwin, :, j); eps = view(m.ewin, :, j))
        end
        learned_prior_z!(view(m.ewin, :, k), m.state, m.weights, copy(view(m.ewin, :, k)))
        lstm_step!(m.state, m.weights, m.spec, view(m.xwin, :, k); eps = view(m.ewin, :, k))
        return m
    end
    for j in 1:m.nwin[]
        lstm_step!(m.state, m.weights, m.spec, view(m.xwin, :, j); eps = view(m.ewin, :, j))
    end
    return m
end

needs_qstar(::StochLSTM) = true

"""
    nwarm(m::StochLSTM)

Number of steps the warm-up replays, zero when there is none.
"""
nwarm(m::StochLSTM) = m.spinnup_data === nothing ? 0 : size(m.spinnup_data, 2)

"""
    in_warmup(m::StochLSTM)

Whether the next call will replay rather than predict.
"""
in_warmup(m::StochLSTM) = m.counter[] < nwarm(m)

# Push one completed step onto the lag window, in the scaled units the regressor is built in.
"What the fit predicts: `:q` (the level; every fit before 2026-09-23) or `:dQ` (the correction)."
_lstm_target(scaling) = hasproperty(scaling, :target) ? scaling.target : :q

"""
    _apply_input_map(x, scaling)

Apply the fit's fixed input map, `x -> P x`, when its `scaling` carries one (`input_map`), else
return `x` unchanged. 🔑 Preprocessing, like the standardisation, and stored with it: a square,
invertible `P` changes no information, only the conditioning -- e.g. replacing `q*^{n-1}` by the
standardised increment `(q*^n - q*^{n-1} - m) / s`, which a least-squares map otherwise reaches only
through coefficients of ~130 that a network does not learn (`results_LSTMS.md` §7f). The training
driver applies the same `P` to its design, so training and deployment see one input.
"""
function _apply_input_map(x::AbstractVector{T}, scaling) where {T}
    hasproperty(scaling, :input_map) || return x
    P = scaling.input_map
    size(P, 2) == length(x) ||
        error("input_map is $(size(P)) but the regressor has $(length(x)) entries")
    return T.(P * x)
end

"The fit's calibrated noise scale (`scaling.noise_scale`), 1 when absent."
_noise_scale(scaling) = hasproperty(scaling, :noise_scale) ? scaling.noise_scale : 1.0

function _push_scaled!(m::StochLSTM{T}, q, q_star) where {T}
    qs = scale_input(collect(q), m.scaling.in_scaling)
    qss = scale_input(collect(q_star), m.scaling.in_scaling)
    push!(m.buf, T.(vec(qs)), T.(vec(qss)))
    return m
end

"""
    get_next_item_timeseries(m::StochLSTM, q_star)

One solver step. Returns `dQ`.

The order of operations matters and is the same on both branches: form the regressor from the
*current* `q*` and the lag window, step the cell, then push the completed step. Pushing first
would put `q^n` into the window the row for step `n` is built from, which is the off-by-one V1 and
V2 exist to catch on the linear cells.
"""
function get_next_item_timeseries(m::StochLSTM{T}, q_star) where {T}
    nq = m.spec.hist.n_qoi
    # 🔴 `Tsolve` is the SOLVER's precision and it is Float64 in production; `T` is the model's and
    # is Float32. The two boundaries are here and nowhere else: the regressor is converted DOWN on
    # the way in, the prediction UP on the way out, and `dQ` leaves this function at the solver's
    # precision whatever the weights are. Relying on Julia's promotion would give the same answer
    # today and stop doing so the moment a Float32 `Scaling` is loaded, which is exactly the class
    # of silent precision loss `rf_setup`'s grid/Re check exists for (claude_memory.md #57).
    qs_host = collect(vec(Array(q_star)))
    Tsolve = eltype(qs_host)

    # the regressor for the step about to be taken
    qs_scaled = T.(vec(scale_input(qs_host, m.scaling.in_scaling)))
    x = _apply_input_map(inputvec(m.buf, qs_scaled), m.scaling)

    windowed = m.spec.window > 0
    if in_warmup(m)
        # 🔴 Charge the recurrence, draw nothing, and return the recorded dQ UNCONVERTED. In window
        # mode there is nothing to charge -- the state is reset at every prediction -- so the step
        # only enters the window, at the posterior mean (`eps = 0`).
        if windowed
            _push_window!(m, x, nothing)
        else
            lstm_step!(m.state, m.weights, m.spec, x; sample_latent = false)
        end
        m.counter[] += 1
        dQ = m.spinnup_data[1:nq, m.counter[]]
        _push_scaled!(m, qs_host .+ dQ, qs_host)
        return dQ
    end

    if windowed
        # 🔴 This step's latent draw is taken ONCE, here, and stored with its input; the W - 1
        # later windows that replay this step reuse it (see "Window mode" above). Scalar draws in
        # the order `lstm_step!` makes them, so the rng is consumed identically in both modes.
        _push_window!(m, x, nothing)
        if latent_sampled(m.spec)
            # ⚠️ `tie_noise = false` is a DIAGNOSTIC ablation only: every step of the window is
            # re-drawn at every prediction, so consecutive corrections share no latent draw. It
            # measures how much of the correction's persistence the tying produces.
            jr = m.tie_noise ? (m.nwin[]:m.nwin[]) : (1:m.nwin[])
            # 🔑 `scaling.noise_scale` (window mode): every latent draw, and the emission noise
            # below, is multiplied by it -- the spread parameter a closed-loop calibration adjusts
            ns = T(_noise_scale(m.scaling))
            @inbounds for j in jr, i in axes(m.ewin, 1)
                m.ewin[i, j] = ns * T(randn(m.rng))
            end
        end
        _replay_window!(m)
    else
        lstm_step!(m.state, m.weights, m.spec, x; rng = m.rng, sample_latent = true)
    end
    if m.stochastic
        sample_emission!(m.scratch, m.state, m.weights, m.spec, m.rng)
        # `scaling.emission_scale` (per QoI, window mode) multiplies the EMISSION noise on top of
        # `noise_scale` -- whiter corrections in chosen bands without touching the shared latent
        es = hasproperty(m.scaling, :emission_scale) ? m.scaling.emission_scale : nothing
        if windowed && (_noise_scale(m.scaling) != 1 || es !== nothing)
            ns = T(_noise_scale(m.scaling))
            @inbounds for k in eachindex(m.scratch)
                f = es === nothing ? ns : ns * T(es[k])
                m.scratch[k] = m.state.y[k] + f * (m.scratch[k] - m.state.y[k])
            end
        end
        # 🔑 COLOURED emission noise (2026-09-27): `scaling.eta_ar = a` (per QoI) turns the white
        # emission draw e_n into a stationary AR(1), e'_n = a e'_{n-1} + sqrt(1 - a^2) e_n -- same
        # marginal variance, lag-k autocorrelation a^k. The linear model with it is M0^c (plan L7).
        if hasproperty(m.scaling, :eta_ar)
            a = m.scaling.eta_ar
            @inbounds for k in eachindex(m.scratch)
                e = m.scratch[k] - m.state.y[k]
                ec = T(a[k]) * m.eprev[k] + sqrt(one(T) - T(a[k])^2) * e
                m.eprev[k] = ec
                m.scratch[k] = m.state.y[k] + ec
            end
        end
    else
        copyto!(m.scratch, m.state.y)
    end

    # the model predicts the LEVEL q^n in scaled units; the closure owes the solver the correction
    # 🔑 `scaling.target` (absent on every fit before 2026-09-23, hence `:q`) says what the output
    # IS: the level, from which the correction is `qhat - q*`, or the correction itself.
    out = Tsolve.(vec(scale_output(m.scratch, m.scaling.out_scaling)))
    tgt = _lstm_target(m.scaling)
    # :q -- the level, :dQ -- the additive correction, :logr -- the multiplicative one, r = log1p(dQ/q*)
    dQ = tgt === :dQ ? out : tgt === :logr ? qs_host .* expm1.(out) : out .- qs_host
    # 🔑 `scaling.dq_offset` (physical units, per QoI): a constant added to every deployed correction
    # -- the parameter a closed-loop CALIBRATION of the level adjusts (`tools/m4_calibrate.jl`)
    # `scaling.offset_ref` (per QoI) makes the offset STATE-PROPORTIONAL, `c .* q* ./ q*_ref`: the
    # same correction at the typical level, shrinking as a band drains (2026-09-27: a constant
    # offset pushed low excursions into collapse, the clamp firing)
    if hasproperty(m.scaling, :dq_offset)
        if hasproperty(m.scaling, :offset_ref)
            dQ .+= m.scaling.dq_offset .* (qs_host ./ m.scaling.offset_ref)
        else
            dQ .+= m.scaling.dq_offset
        end
    end
    any(abs.(qs_host) .< m.gate) && (dQ .= 0)

    _push_scaled!(m, qs_host .+ dQ, qs_host)
    return dQ
end
