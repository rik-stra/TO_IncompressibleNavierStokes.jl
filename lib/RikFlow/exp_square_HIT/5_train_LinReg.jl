if false
    include("../src/RikFlow.jl")
    using .RikFlow
end

using RikFlow
using JLD2
using Random
# 🔴 `using CairoMakie` was here and would have killed this job at load: CairoMakie is not in
# `lib/RikFlow`'s `[deps]` (only `Makie`, and only as a weak dependency behind `RikFlowMakieExt`).
# Nothing in this file plots. Exactly the defect #53 found in `2_HF_ref.jl`, in the second step of
# the fit pipeline.
using Distributions
using LinearAlgebra
using RegularizedLeastSquares

# parse input ARGS
model_index = parse(Int, ARGS[1])
# or set model_index manually
#model_index = 1
inputs_file_name = "/inputs_example.jld2"
TO_folder = @__DIR__()*"/output/TO_LRS"
# The tracking record produced by `3_track_ref.jl` on the REGENERATED HF reference.
#
# 🔴 The `_f64_lmwray3` suffix is deliberate and must not be dropped: a Float64/LMWray3 record must
# never be confusable with the archived Float32/RK44 one, which is otherwise identically named. The
# archive sits on the pre-`09954be1` Nyquist convention, which changed `∂` and therefore `tau` and
# `dQ`, so it is a *different dynamical system* rather than a less accurate measurement of this one
# (claude_memory.md #45, #46).
#
# 🔑 100 TU, not 10 (Rik, 2026-09-14). One tracking run carries the 1–10 TU fit window *and* the 401
# fields at 0.25 TU that D6 draws its initial conditions from. `train_range = (400, 4000)` selects
# t ∈ [1, 10] out of whatever record it is given, so fitting "to 10 TU" needs no change here.
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
    @__DIR__()*"/output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2")

function create_history(hist_len, q_star, q, dQ; include_predictor = true)
    if hist_len == 0
        return q_star, dQ
    end
    qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
    if include_predictor
        return vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
    else
        return vcat(qs...), dQ[:,hist_len:end]
    end
end

function create_history(hist_len, q_star, q, dQ, hist_var; include_predictor = true)
    if hist_var == :q
        inputs,outputs = create_history(hist_len, q_star[:,:], q[:,:], dQ[:,:]; include_predictor)
    elseif hist_var == :q_star
        inputs,outputs = create_history(hist_len, q_star[:,2:end], q_star[:,1:end-1], dQ[:,2:end]; include_predictor)
    elseif hist_var == :q_star_q
        inputs,outputs = create_history(hist_len, q_star[:,2:end], cat(q[:,2:end],q_star[:,1:end-1],dims = 1), dQ[:,2:end]; include_predictor)
    end
    return inputs,outputs
end


## Load parameters
inputs = load(TO_folder*inputs_file_name, "inputs")
(; name, hist_len, hist_var, n_replicas, normalization, include_predictor, tracking_noise, train_range, indep_normals, lambda, fitted_qois, model_noise) = inputs[model_index]


out_dir = TO_folder*"/$(name)/"
save(out_dir*"parameters.jld2", "parameters", (; name, hist_len, hist_var, n_replicas, normalization, include_predictor))


data = load(track_file, "data_track");

# normalize the data
q_scaled, in_scaling = RikFlow._normalise(data.q[:,train_range[1]:train_range[2]-1], normalization = normalization)
q_star_scaled = RikFlow.scale_input(data.q_star[:,train_range[1]:train_range[2]-1], in_scaling)
dQ_scaled     = RikFlow.scale_input(data.q[:,train_range[1]+1:train_range[2]], in_scaling)
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)

inputs, outputs = create_history(hist_len, q_star_scaled, q_scaled, dQ_scaled, hist_var; include_predictor)


function fit_model(inputs, outputs, fitted_qois; indep_normals = false, lambda = 0.0, regularizer = :l2)
    n_targets = length(fitted_qois)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term

    # solve linear regression
    if lambda > 0.0
        inp_r = kron(Matrix(I, n_targets,n_targets),inp)
        if regularizer == :l2
            reg = L2Regularization(lambda)
        elseif regularizer == :nuclear
            reg = NuclearRegularization(lambda, (size(inp,2), size(outputs,1)))
        end
        solver = createLinearSolver(ADMM, inp_r; reg=reg)
        b = reshape(outputs[fitted_qois,:]', length(fitted_qois)*size(outputs,2),1)
        c = solve!(solver, b)
        c = reshape(c, :, length(fitted_qois))
    else
        c = inp \ outputs[fitted_qois,:]'
    end 
    
    # fit the stochastic part to the residuals
    preds = inp * c
    stoch_part = copy(outputs)
    stoch_part[fitted_qois,:] -= preds'
    # fit MVG
    if indep_normals
        stoch_distr = fit(DiagNormal, stoch_part .|> Float64)
    else
        stoch_distr = fit(MvNormal, stoch_part .|> Float64)
    end
    
    return c, stoch_distr
end

function run_model(inputs, c, stoch_distr, fitted_qois)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    preds = inp * c
    rand_part = rand(stoch_distr, size(inputs,2))'
    return rand_part[fitted_qois,:]+=preds
end

# fit model
c, stoch_distr = fit_model(inputs, outputs, fitted_qois; indep_normals, lambda, regularizer = :l2)

# overwrite the nose distribution
if model_noise == :tracking_noise
    stds_ref_data = load(@__DIR__()*"/output/tracking/stds_refdata.jld2", "stds")
    stds = stds_ref_data.*tracking_noise./scaling.out_scaling.sigma
    stoch_distr = MvNormal(diagm(reshape(stds,6).^2))
elseif model_noise == :no_noise
    stoch_distr = nothing
end

## save model
save(out_dir*"/LinReg.jld2", "c", c', "stoch_distr", stoch_distr, 
    "scaling", scaling, "hist_var", hist_var, "hist_len", hist_len, "include_predictor", include_predictor, "fitted_qois", fitted_qois)

