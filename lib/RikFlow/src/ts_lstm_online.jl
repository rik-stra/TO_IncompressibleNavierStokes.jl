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

# Fields
- `spec`, `weights`: the model. `check_shapes` is run at construction.
- `scaling`: the `(in_scaling, out_scaling)` pair, applied exactly as `LinReg` applies it.
- `gate`: the turbulence gate. M4 sits in the `q*`-consuming path, so it inherits `LinReg`'s gate
  rather than introducing a third convention -- see `TURBULENCE_GATE`'s docstring, and note that
  `MVG_sampler` has none, which is a known asymmetry.
- `stochastic`: `false` returns the emission mean instead of a draw. A diagnostic, not a mode to
  run the paper on.
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

    function StochLSTM(spec::LSTMSpec, weights::LSTMWeights{T}, scaling;
                       spinnup_data = nothing, rng, gate::Real = 1e-2,
                       stochastic::Bool = true) where {T}
        check_shapes(weights, spec)
        nq = spec.hist.n_qoi
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
                               Float64(gate), stochastic, zeros(T, spec.hist.n_qoi))
    end
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
    x = inputvec(m.buf, qs_scaled)

    if in_warmup(m)
        # 🔴 Charge the recurrence, draw nothing, and return the recorded dQ UNCONVERTED.
        lstm_step!(m.state, m.weights, m.spec, x; sample_latent = false)
        m.counter[] += 1
        dQ = m.spinnup_data[1:nq, m.counter[]]
        _push_scaled!(m, qs_host .+ dQ, qs_host)
        return dQ
    end

    lstm_step!(m.state, m.weights, m.spec, x; rng = m.rng, sample_latent = true)
    if m.stochastic
        sample_emission!(m.scratch, m.state, m.weights, m.spec, m.rng)
    else
        copyto!(m.scratch, m.state.y)
    end

    # the model predicts the LEVEL q^n in scaled units; the closure owes the solver the correction
    qhat = Tsolve.(vec(scale_output(m.scratch, m.scaling.out_scaling)))
    dQ = qhat .- qs_host
    any(abs.(qs_host) .< m.gate) && (dQ .= 0)

    _push_scaled!(m, qs_host .+ dQ, qs_host)
    return dQ
end
