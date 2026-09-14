# This script creates a data set with hyperparameters for the TO LRS model. This allows for systematic hyperparameter search.
using JLD2

# standard settings
fixed_parameters = (
                    # stuff you want to explore
                    hist_len = 5, # number of history point included in the linear regression
                    lambda = 0,   # regularization strength in linear regression
                    train_range = (400,4000), # range of training data to use when fitting linear regression (we used Δt = 0.0025, so this is timeunit 1 to 10)
                    n_replicas = 5, # number of replicas to run when evaluating the model online
                    
                    # stuff you might want to explore
                    model_noise = :MVG,   # noise model for residual of linear regression options :MVG multi variate gaussian, :no_noise no noise added to linreg, :model_noise use the same noise as during tracking (see "tracking_noise")
                    fitted_qois = [1,2,3,4,5,6],  # choose which qois to fit the linear regression to.
                    # 🔴 `:normal` — FULL standardisation, mean subtracted AND divided by the
                    # standard deviation (Rik, 2026-09-14). `:standardise` sets `mu = 0` and only
                    # divides, leaving the mean in the data; that is what paper 2 used and it is
                    # `plan.md`'s TODO-0. Options: :normal, :standardise, :minmax.
                    #
                    # 🔑 At λ = 0 this does not change the fitted model. The two conventions differ
                    # by a constant shift of every design column and of the target, and a
                    # least-squares fit carrying an intercept is invariant to that — measured at
                    # 4.7e-11 in Float64 (claude_memory.md #26). What it changes is conditioning
                    # and interpretability: the uncentred design multiplies σ²_max by 31.5 and κ by
                    # 5.6 (#23, #29). At λ > 0 it changes the fit itself, because the penalty is
                    # not shift-invariant.
                    #
                    # ⚠️ The other half of TODO-0 — keeping the intercept out of the penalty — is
                    # not set here and does not bite at λ = 0. It must be decided before any λ > 0
                    # cell is fitted.
                    normalization = :normal,

                    # stuff you probably don't want to explore 
                    hist_var = :q_star_q,  # include both q_star and q in the history, options: :q, :q_star_q
                    indep_normals = false, # if true: fit MVG with diagonal covariance matrix (so "independent normal distributions")
                    include_predictor = true, # include the predictor q_star in the model inputs
                    tracking_noise = 0,    # add noise to the reference trajectories data (results in a data-assimilation-like problem). If you want to explore this you also need to run multiple tracking simulations with different noise levels and possibly different randomseeds 
                    )


# test two history lengths and regularization strengths
hist_lens = [5, 20]
labs = [0, 0.01]

i = 0
inputs = []
for hist_len in hist_lens
    for lambda in labs
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, lambda))
    end
end

outdir = @__DIR__()*"/output/TO_LRS"
ispath(outdir) || mkpath(outdir)
save(outdir*"/inputs_example.jld2", "inputs", inputs)
# 🔴 `using DataFrames` and an exploratory `DataFrame(inputs)` query used to sit here. DataFrames is
# not in `lib/RikFlow`'s `[deps]`, so under `julia --project` this script died at load — after the
# queue wait, with nothing done. Gotcha #53's class, and this file is the FIRST step of the fit
# pipeline, so it would have taken R2 down before anything ran. Nothing here needs a DataFrame: the
# sweep is a vector of NamedTuples and is already saved above.
for e in inputs
    println("  ", e.name, "  h = ", e.hist_len, "  lambda = ", e.lambda,
            "  normalization = ", e.normalization, "  train_range = ", e.train_range)
end