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

    function LinReg(file_name, rng, ArrayType; q_hist = nothing, spinnup_data = nothing)

        c, stoch_distr, scaling, hist_var, include_predictor, fitted_qois = load(file_name, "c", "stoch_distr", "scaling", "hist_var", "include_predictor", "fitted_qois")
        target = :q
        scaling = adapt(ArrayType, scaling)
        c= adapt(ArrayType, c)

        counter = zeros(Int)
        if !isnothing(q_hist)
            @assert size(spinnup_data, 2) >= size(q_hist, 2) "Need spinnup data to fill history"
        end
        if !isnothing(spinnup_data) && isnothing(q_hist)
            @error "Spinnup not implemented without history"
        end
        new(c, stoch_distr, scaling, q_hist, spinnup_data, counter, hist_var, include_predictor, fitted_qois, target, rng, ArrayType)
    end
end

function get_next_item_timeseries(time_series_method::LinReg, q_star)
    if !isnothing(time_series_method.q_hist)  # if the model uses history
        n_qoi = size(q_star,1)
        if time_series_method.counter[] < size(time_series_method.spinnup_data,2) # for the first few steps, directly read dQ
            time_series_method.counter[] += 1
            dQ = time_series_method.spinnup_data[1:n_qoi, time_series_method.counter[]]
        else    # after that, predict dQ  (we now have enough history)
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
            
            data = vcat(input,ones(eltype(input), (1,1)))
            if !isnothing(time_series_method.stoch_distr)
                pred = rand(time_series_method.rng, time_series_method.stoch_distr).|> Float32 |> adapt(time_series_method.ArrayType)
            else
                pred = zeros(eltype(input), (n_qoi,1)) |> adapt(time_series_method.ArrayType)
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
            pred = rand(time_series_method.rng, time_series_method.stoch_distr) |> adapt(time_series_method.ArrayType)
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


export get_next_item_timeseries, Reference_reader, MVG_sampler, Resampler, ANN