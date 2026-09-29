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
using Statistics
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
# Added 2026-09-15; read with a fallback so an `inputs_example.jld2` written before they existed
# still loads and keeps the behaviour it had (ADMM, intercept penalized -- what paper 2 ran).
ridge_solver      = get(inputs[model_index], :ridge_solver, :admm)
penalize_intercept = get(inputs[model_index], :penalize_intercept, true)
# Added 2026-09-29: `:paper3` scales the penalty per QoI as paper 3 does -- column i of C uses
# λ_i = λ (σ_i/σ_1)^2, σ_i the sd of the SCALED correction (target minus the predictor row of the
# design) on the training rows. `lambda` stays the base λ. `:none` (default) is one λ for all.
lambda_scaling = get(inputs[model_index], :lambda_scaling, :none)


out_dir = TO_folder*"/$(name)/"
# 🔴 `lambda`, `ridge_solver` and `penalize_intercept` are recorded here. They were NOT, and
# `test_g1.jl:52` says so outright -- "parameters.jld2 does not store lambda" -- so a fitted model
# on disk could not be told apart from another at a different penalty. With λ > 0 cells in the
# table that is the difference between two experiments.
save(out_dir*"parameters.jld2", "parameters",
     (; name, hist_len, hist_var, n_replicas, normalization, include_predictor,
        lambda, lambda_scaling, ridge_solver, penalize_intercept, train_range, track_file))


data = load(track_file, "data_track");

# normalize the data
q_scaled, in_scaling = RikFlow._normalise(data.q[:,train_range[1]:train_range[2]-1], normalization = normalization)
q_star_scaled = RikFlow.scale_input(data.q_star[:,train_range[1]:train_range[2]-1], in_scaling)
dQ_scaled     = RikFlow.scale_input(data.q[:,train_range[1]+1:train_range[2]], in_scaling)
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)

inputs, outputs = create_history(hist_len, q_star_scaled, q_scaled, dQ_scaled, hist_var; include_predictor)


"""
Per-QoI penalties λ_i = λ (σ_i/σ_1)^2 (paper 3), σ_i the sd of the scaled correction: the target row
minus the predictor row `q*` (the first rows of the design when `include_predictor`). Returns the
λ vector and σ over `fitted_qois`.
"""
function paper3_lambdas(inputs, outputs, fitted_qois, lambda)
    corr = outputs[fitted_qois, :] .- inputs[fitted_qois, :]   # predictor rows come first
    sig = vec(std(corr; dims = 2))
    return lambda .* (sig ./ sig[1]) .^ 2, sig
end

function fit_model(inputs, outputs, fitted_qois; indep_normals = false, lambda = 0.0,
                  regularizer = :l2, ridge_solver = :exact, penalize_intercept = false,
                  lambda_per_qoi = nothing)
    n_targets = length(fitted_qois)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term

    # solve linear regression
    if lambda_per_qoi !== nothing
        # One exact ridge solve per output column, each with its own λ_i -- exact for the same
        # reason the single-λ solve is column-separable (same design; P differs only in scale).
        (regularizer == :l2 && ridge_solver == :exact) ||
            error("per-QoI λ is implemented for the exact :l2 ridge only")
        T = eltype(inp)
        m = size(inp, 2)
        c = zeros(T, m, n_targets)
        for (j, i) in enumerate(fitted_qois)
            P = Matrix{T}(I, m, m) * T(sqrt(lambda_per_qoi[j]))
            penalize_intercept || (P[m, m] = zero(T))   # the bias column is the last one
            c[:, j] = [inp; P] \ [outputs[i, :]; zeros(T, m)]
        end
    elseif lambda > 0.0 && (regularizer == :nuclear || ridge_solver == :admm)
        # 🔴 The historical path, and for `:l2` it does not solve the problem it claims to.
        # Measured on R1's record 2026-09-15 (the parity check `results.md` §7 lists as never
        # run): against the exact ridge minimiser the ADMM iterate differs by a relative 0.97 at
        # λ = 1e-5, 0.90 at 1e-4 and 0.45 at 1e-2, and its TRAINING RMSE is ~0.00714 at every one
        # of those λ -- it is iteration-limited, not λ-limited, so a sweep run through it is not a
        # sweep in λ. Kept because paper 2's archived λ > 0 models came out of it and reproducing
        # one needs it, and because `:nuclear` has no closed form and must come here.
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
    elseif lambda > 0.0
        # The squared-ridge minimiser in closed form, as the augmented system rather than the
        # normal equations: `cond(X)` is 1.9e6 here, so `X'X` would square that to 3.4e12 and lose
        # most of the coefficient vector. Identical construction to `RikFlow.fit_ridge`
        # (`src/ts_fit.jl`), which V1 covers; written out here because this script does not load
        # the `ts_*` layer.
        T = eltype(inp)
        m = size(inp, 2)
        P = Matrix{T}(I, m, m) * T(sqrt(lambda))
        penalize_intercept || (P[m, m] = zero(T))   # the bias column is the last one
        c = [inp; P] \ [outputs[fitted_qois,:]'; zeros(T, m, n_targets)]
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
lambda_per_qoi, corr_sigma = if lambda_scaling == :paper3
    include_predictor || error("lambda_scaling = :paper3 needs include_predictor (correction = target - q*)")
    paper3_lambdas(inputs, outputs, fitted_qois, lambda)
elseif lambda_scaling == :none
    nothing, nothing
else
    error("unknown lambda_scaling $(lambda_scaling)")
end
lambda_per_qoi === nothing || println("per-QoI λ (paper 3, base $(lambda)): ", lambda_per_qoi,
                                      "  from σ(scaled correction) = ", corr_sigma)
c, stoch_distr = fit_model(inputs, outputs, fitted_qois; indep_normals, lambda,
                          regularizer = :l2, ridge_solver, penalize_intercept, lambda_per_qoi)
# the per-QoI λ are only known now; record them next to the base λ
lambda_per_qoi === nothing ||
    save(out_dir*"parameters.jld2", "parameters",
         (; name, hist_len, hist_var, n_replicas, normalization, include_predictor,
            lambda, lambda_scaling, lambda_per_qoi = Tuple(lambda_per_qoi), corr_sigma = Tuple(corr_sigma),
            ridge_solver, penalize_intercept, train_range, track_file))

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

