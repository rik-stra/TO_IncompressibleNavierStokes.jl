# Verbatim copy of the pre-AR `LinReg` (src/time_series_methods.jl:173-277 at the 2026-09-28 HEAD,
# 82c1e63a), renamed `LegacyLinReg`. The oracle of test_linreg_ar.jl's bit-identity test: DO NOT EDIT.
# Included by that file's test module, never by the package.

struct LegacyLinReg
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

    function LegacyLinReg(file_name, rng, ArrayType; q_hist = nothing, spinnup_data = nothing)

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

function get_next_item_timeseries(time_series_method::LegacyLinReg, q_star)
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

