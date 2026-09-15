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

                    # 🔴 How the λ > 0 solve is done, and it is NOT cosmetic.
                    #
                    # `:exact` solves the augmented system `[X; sqrt(λ)P] \ [Y; 0]`, which is the
                    # squared-ridge minimiser in closed form. `:admm` is the historical path through
                    # `RegularizedLeastSquares`, which paper 2's λ > 0 fits used.
                    #
                    # ⚠️ **They do not agree, and ADMM is the one that is wrong.** Measured on R1's
                    # record, 2026-09-15 -- the QR-vs-ADMM parity check `results.md` §7 lists as
                    # never run:
                    #
                    #     λ      rel(C_admm, C_ridge)   train RMSE admm / ridge
                    #     1e-5   0.970                  0.00715 / 0.00670
                    #     1e-4   0.903                  0.00716 / 0.00671
                    #     1e-2   0.452                  0.00714 / 0.00685
                    #
                    # ADMM's training RMSE is ~0.00714 at EVERY λ from 1e-5 to 1e-2, i.e. it is
                    # iteration-limited rather than λ-limited: it returns roughly the same
                    # under-converged iterate whatever penalty it is given. A λ sweep run through it
                    # would not be a sweep in λ at all. `:admm` is kept because paper 2's archived
                    # λ > 0 models came out of it and reproducing one needs it -- not because it
                    # should be used for new work. The `:nuclear` regularizer has no closed form and
                    # always goes through ADMM.
                    ridge_solver = :exact,

                    # ⚠️ TODO-0's second half, and it has to be set now that λ > 0 cells exist.
                    # 🔑 **Measured: it makes no difference here.** ρ(C̃), the starred gain and the
                    # training RMSE are identical to five figures with and without the intercept in
                    # the penalty, at every λ from 0 to 1e-2 on R1's record -- the design is
                    # standardised and the intercept is one column of 67. `false` is paper 4's
                    # harmonized convention and is chosen on that ground, not on a measured effect.
                    penalize_intercept = false,

                    # stuff you probably don't want to explore 
                    hist_var = :q_star_q,  # include both q_star and q in the history, options: :q, :q_star_q
                    indep_normals = false, # if true: fit MVG with diagonal covariance matrix (so "independent normal distributions")
                    include_predictor = true, # include the predictor q_star in the model inputs
                    tracking_noise = 0,    # add noise to the reference trajectories data (results in a data-assimilation-like problem). If you want to explore this you also need to run multiple tracking simulations with different noise levels and possibly different randomseeds 
                    )


# 🔴 THE INDEX IS THE IDENTITY. `LinReg<i>` is the i-th entry of this list; that name is the output
# directory, the batch-script argument and how every results table refers to a configuration.
# **APPEND ONLY.** This used to be a cartesian product of `hist_lens` and `labs`, which meant that
# adding one regularization strength silently renamed every configuration after the insertion
# point -- `LinReg2` would have stopped being (h = 5, λ = 0.01) the moment a λ was added, while
# `output/TO_LRS/LinReg2/` kept the old runs. The product is still available through `grid` below
# for an exploratory sweep into a fresh directory; the deployed table is written out by hand.
const CELLS = [
    (hist_len =  5, lambda = 0.0),     # LinReg1 -- paper 2's headline configuration; R2 ran this
    (hist_len =  5, lambda = 0.01),    # LinReg2
    (hist_len = 20, lambda = 0.0),     # LinReg3
    (hist_len = 20, lambda = 0.01),    # LinReg4
    # --- appended 2026-09-15: the precision-regularization probe -------------------------------
    #
    # Measured on R1's record in Float64 at h = 5: the λ = 0 fit has ρ(C̃) = 2.689 and a starred
    # gain of 270, while the SAME data fitted in Float32 gives 1.003 and 7.2, and paper 2's
    # archived Float32 fit gives 1.0002 and 10.3. Every LRS deployed before R2 was regularized by
    # its own round-off; R2's is the first Float64 fit and the first to run the actual λ = 0
    # solution. A ridge of 1e-5 restores the archive's operator norms at +0.1% training RMSE:
    #
    #     λ        ρ(C̃)     starred gain   train RMSE
    #     0        2.6888   269.80         0.00669
    #     1e-5     1.0031     8.01         0.00670
    #     1e-4     1.0033     4.10         0.00671
    #
    # These two cells test whether that is what produces the LRS's downward excursions and its
    # 0.836 summed KS against the archive's 0.198. λ = 1e-6 is deliberately absent: ρ falls to
    # 1.129 there but the starred gain spikes to 2362, a transitional regime worth not stepping on.
    (hist_len =  5, lambda = 1e-5),    # LinReg5
    (hist_len =  5, lambda = 1e-4),    # LinReg6
]

"""
    grid(hist_lens, lambdas)

The cartesian product, for an exploratory sweep. ⚠️ Not what `CELLS` is built from any more -- see
the warning above. Writing a grid over the deployed table renumbers it.
"""
grid(hist_lens, lambdas) = [(; hist_len, lambda) for hist_len in hist_lens, lambda in lambdas] |> vec

# 🔴 In a function, not a bare loop. `i += 1` inside a top-level `for` when a global `i` exists
# makes `i` a NEW LOCAL, and the read throws `UndefVarError` -- so this script has never run as a
# script. It worked only in the REPL, where Julia special-cases soft-scope assignment to an existing
# global. Gotcha #47's last bullet; do not reach for `global` to fix it.
function build_inputs(cells, fixed)
    inputs = NamedTuple[]
    for (i, cell) in enumerate(cells)
        push!(inputs, (name = "LinReg$i", fixed..., cell.hist_len, cell.lambda))
    end
    return inputs
end

inputs = build_inputs(CELLS, fixed_parameters)

outdir = @__DIR__()*"/output/TO_LRS"
ispath(outdir) || mkpath(outdir)
save(outdir*"/inputs_example.jld2", "inputs", inputs)
# 🔴 `using DataFrames` and an exploratory `DataFrame(inputs)` query used to sit here. DataFrames is
# not in `lib/RikFlow`'s `[deps]`, so under `julia --project` this script died at load — after the
# queue wait, with nothing done. Gotcha #53's class, and this file is the FIRST step of the fit
# pipeline, so it would have taken R2 down before anything ran. Nothing here needs a DataFrame: the
# sweep is a vector of NamedTuples and is already saved above.
for e in inputs
    println("  ", e.name, "  h = ", lpad(e.hist_len, 2), "  lambda = ", rpad(e.lambda, 8),
            "  normalization = ", e.normalization, "  solver = ", e.ridge_solver,
            "  penalize_intercept = ", e.penalize_intercept)
end