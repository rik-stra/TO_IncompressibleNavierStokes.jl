# Score TO+LRS (M0) and the data-driven noise model (DDN) on HIT.
#
# The two models paper 2 and paper 1 already contain, scored on one axis for the first time. No new
# model is fitted beyond refitting M0 from the tracked record, and no simulation is run: 0 SBU.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/score_m0_ddn.jl
#
# Writes `analysis/output/paper4_scores.jld2`, which `analysis/plot_paper4.jl` reads.
#
# ---------------------------------------------------------------------------------------------
# Two decisions that determine whether the numbers mean anything
# ---------------------------------------------------------------------------------------------
#
# **Everything is scored on `dQ`, in physical units.** The two models have different native
# targets: DDN samples `dQ` directly from a state-independent `fit(MvNormal, dQ)` on raw unscaled
# data, while M0 predicts the *level* `q` in scaled coordinates and deployment then forms
# `dQ = pred - q_star`. Given the history the map from `q` to `dQ` is a deterministic shift, so the
# predictive law transforms exactly:
#
#     q  ~ N(mu_scaled, Sigma_scaled)                  (what M0 fits)
#     mu_phys = mu_scaled .* sigma .+ mu               Sigma_phys = D(sigma) Sigma_scaled D(sigma)
#     dQ ~ N(mu_phys - q_star, Sigma_phys)             (same covariance, shifted mean)
#
# and both models are then Gaussian densities on `dQ` in physical units. Scoring the level instead
# would throw the signal away: with the predictor stream replayed, `q` cannot separate models at
# all -- measured spread across nine models was 0.027-0.289 on `q` against 0.18-2.82 on `dQ`.
#
# **DDN is the negative control, and that is the point of running it.** Its predictive density is
# the same multivariate Gaussian at every step, so it *is* the climatological ensemble for `dQ`. It
# should therefore produce a nearly flat rank histogram while having no conditional skill
# whatever. Flat histogram + good marginal KS + zero dynamics, in one pair of panels.

using LinearAlgebra
using Statistics
using Random
using Printf
using JLD2
using Dates

const HERE = @__DIR__
const SRC = normpath(joinpath(HERE, "..", "src"))
for f in ("ts_scaling", "ts_history", "ts_models", "ts_fit", "ts_score", "ts_rollout",
          "ts_spectrum")
    include(joinpath(SRC, "$f.jl"))
end
include(joinpath(HERE, "extract_qois.jl"))
include(joinpath(HERE, "extract_archive.jl"))
include(joinpath(HERE, "extract_rebaseline.jl"))

const OUT = joinpath(HERE, "output")
const DT = 2.5e-3                      # HIT LES time step
const SEED = 20260908

# QoI band labels. HIT shells are [0,6] [7,15] [16,32] (`RikFlow.jl:185-193`), and the rows
# alternate enstrophy / energy per band.
const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

# ---------------------------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------------------------

"""
    windows(rec; train = (400, 4000), tsim_steps = nothing)

The training and held-out ranges, in steps of the tracked record.

`plan.md` §7: **no time unit ever serves two roles.** HIT trains on `[t0, T]` and selects on
`[T, 10]`, so the held-out window is everything after the training range, and the 100 TU record is
the *online evaluation* reference rather than a training extension. Paper 2's archived fits used
`train_range = (400, 4000)`, i.e. 1 to 10 TU at `dt = 2.5e-3`, leaving no held-out set on the 10 TU
record -- which is why the held-out scores below are computed on the 100 TU record, where the same
training range leaves 36 000 unseen steps.
"""
function windows(rec; train = (400, 4000))
    T = size(rec.q_star, 2)
    a, b = train
    b < T || error("training range $(train) leaves no held-out steps in a record of $T")
    return (; train = (a, b), heldout = (b + 1, T), nsteps = T)
end

# ---------------------------------------------------------------------------------------------
# the two models
# ---------------------------------------------------------------------------------------------

"""
    fit_m0(rec, train; h, lambda, normalization, penalize_intercept)

Refit TO+LRS on a tracked record and return everything needed to score it.

The fit is paper 2's: regress the **level** `q^{n+1}` on `[q^{n*} | q^{n-1}, q^{n-1*} | ... | 1]`
in scaled coordinates, then fit one multivariate Gaussian to the residuals. Both conventions are
reachable -- `normalization = :standardise, penalize_intercept = true` is the paper-2-faithful
path that G1 reproduces, and the `:normal` default with a free intercept is paper 4's harmonized
one.
"""
function fit_m0(rec, train; h = 5, lambda = 0.0, normalization = :normal,
                penalize_intercept = false, T = Float64)
    a, b = train
    # 🔴 Promote to Float64 before the solve, and do not take the archive's precision as given.
    #
    # The tracked records are Float32 and this design has `cond(X)` of 1.7e6 to 9.5e6, so
    # `cond(X) * eps(Float32)` is 0.2: in single precision the coefficient vector is not resolved
    # at all. Measured, by fitting the identical system in both precisions:
    #
    #   * the slope blocks differ between Float32 and Float64 by a **relative 0.993**;
    #   * `rho(Ctilde)` reads 1.012 in Float32 and **2.174** in Float64 -- marginally unstable
    #     against violently unstable, i.e. opposite conclusions from the same data;
    #   * the H-infinity starred-block gain reads 25.3 against 108.6.
    #
    # The clean way to see that this is precision and not modelling: at `lambda = 0` the two
    # normalization conventions differ only by a constant shift of every design column and of the
    # target, and a least-squares fit with an intercept is invariant to that, so their slope blocks
    # must agree exactly. In Float64 they agree to 4.7e-11. In Float32 they differ by a relative
    # 3.32. (Test: `test_spectrum.jl`, "the conventions agree on slopes in Float64".)
    #
    # Consequence for what may be quoted: the aggregate diagnostics survive single precision
    # because they average -- the total block sum's deviation from the identity is 3.07e-2 against
    # 3.15e-2 -- but **anything reading individual coefficients (#21 rho, #22 gain) must be
    # computed in Float64**, and those numbers cannot be taken from an archived Float32 fit.
    # `fit_ridge` itself deliberately keeps its argument's precision, because G1 has to reproduce
    # the archive bit-for-bit under the archive's own arithmetic.
    q = T.(rec.q)
    q_star = T.(rec.q_star)
    _, scal = fit_scaling(q[:, a:(b - 1)]; normalization, penalize_intercept)
    spec = HistorySpec(; h, n_qoi = size(q, 1), hist_var = :q_star_q,
                       include_predictor = true)
    X, Y, steps = build_history(spec, scale_input(q_star[:, a:(b - 1)], scal),
                                scale_input(q[:, a:b], scal))
    C = fit_ridge(X, Y; lambda, penalize_intercept)
    Res = Y - X * C
    Sigma_scaled = cov(Float64.(Res); corrected = false)
    return (; spec, scaling = scal, C, X, Y, steps, Res, Sigma_scaled, h, lambda,
            normalization, penalize_intercept)
end

"""
    m0_predictive_dQ(m0, rec, range)

M0's one-step predictive density for `dQ`, in **physical** units, over a range of steps.

Returns `(; mu, Sigma, Y)`: an `N x N_Q` matrix of predictive means, one shared covariance, and the
realised `dQ` to score against. The covariance is the same at every step -- M0's residual is
state-independent by construction, which is the defect the ladder's L2 lever exists to address.

The transformation is the exact one in the header comment: undo the scaling on the predicted level,
then subtract the predictor.
"""
function m0_predictive_dQ(m0, rec, range)
    a, b = range
    s = m0.scaling
    X, Yq, steps = build_history(m0.spec, scale_input(rec.q_star[:, a:(b - 1)], s),
                                 scale_input(rec.q[:, a:b], s))
    sig = vec(collect(s.sigma))
    mu0 = s.mu isa Number ? fill(Float64(s.mu), length(sig)) : vec(Float64.(collect(s.mu)))

    mu_q_phys = Float64.(X * m0.C) .* transpose(sig) .+ transpose(mu0)     # levels, physical
    qstar = transpose(Float64.(rec.q_star[:, (a .+ steps .- 1)]))          # N x N_Q
    mu_dQ = mu_q_phys .- qstar
    D = Diagonal(sig)
    Sigma = D * m0.Sigma_scaled * D
    # The realised dQ at those steps: q^n - q^{n*}, with q's one-column offset applied.
    Y_dQ = transpose(Float64.(rec.q[:, (a .+ steps)])) .- qstar
    return (; mu = mu_dQ, Sigma, Y = Y_dQ, steps, X, qstar)
end

"""
    fit_ddn(rec, train)

Paper 1's data-driven noise model: one multivariate Gaussian fitted to the **raw, unscaled** `dQ`
of the training window, sampled independently at every step.

`MVG_sampler`'s `get_next_item_timeseries` (`time_series_methods.jl:37-39`) is a bare `rand` -- no
history, no state dependence, and **no clamp**, unlike the LinReg path. So this is a
state-independent predictive density, identical at every step, and therefore exactly the
climatological ensemble for `dQ`.
"""
function fit_ddn(rec, train)
    a, b = train
    dQ = Float64.(rec.q[:, (a + 1):b] .- rec.q_star[:, a:(b - 1)])
    mu = vec(mean(dQ; dims = 2))
    Sigma = cov(transpose(dQ); corrected = false)
    return (; mu, Sigma, ntrain = size(dQ, 2))
end

"""
    ddn_predictive_dQ(ddn, rec, range, nrows)

DDN's predictive density over a range, aligned to M0's rows so the two are scored on identical
verification instances. The mean is one vector broadcast to every step.
"""
function ddn_predictive_dQ(ddn, rec, range, steps)
    a, b = range
    qstar = transpose(Float64.(rec.q_star[:, (a .+ steps .- 1)]))
    Y_dQ = transpose(Float64.(rec.q[:, (a .+ steps)])) .- qstar
    return (; mu = ddn.mu, Sigma = ddn.Sigma, Y = Y_dQ, steps)
end

# ---------------------------------------------------------------------------------------------
# regime A -- the one-step scores, including RH-1
# ---------------------------------------------------------------------------------------------

"""
    draw_members(mu, Sigma, M, rng)

`M` draws from a Gaussian predictive density per verification instance, as an `N x N_Q x M` array.

RH-1's members come from the **fitted density**, not from a simulation, so `M` is a free parameter
and 50-200 is affordable. That is the whole reason RH-1 rather than RH-2 carries the quantitative
statement: the archived online ensembles have `n_replicas = 5`, giving six bins and a spread
estimate on four degrees of freedom.
"""
function draw_members(mu, Sigma, N, nq, M, rng)
    L = cholesky(Symmetric(Sigma)).L
    ens = Array{Float64}(undef, N, nq, M)
    z = Array{Float64}(undef, nq)
    for n in 1:N
        m = mu isa AbstractMatrix ? view(mu, n, :) : mu
        for j in 1:M
            randn!(rng, z)
            ens[n, :, j] .= m .+ L * z
        end
    end
    return ens
end

"""
    score_regime_a(name, pred; M, rng, nboot)

Metrics #1, #2 and #4 for one model on one held-out window.
"""
function score_regime_a(name, pred; M = 100, rng, nboot = 500, ref_scale = nothing)
    N, nq = size(pred.Y)
    nll = nll_gaussian(pred.Y, pred.mu, pred.Sigma)
    ens = draw_members(pred.mu, pred.Sigma, N, nq, M, rng)
    crps = crps_ensemble_mean(ens, pred.Y)
    sd = sqrt.(diag(pred.Sigma))
    crps_closed = [sum(crps_gaussian(pred.mu isa AbstractMatrix ? pred.mu[n, i] : pred.mu[i],
                                     sd[i], pred.Y[n, i]) for n in 1:N) / N for i in 1:nq]
    rh = [rank_histogram(view(ens, :, i, :), view(pred.Y, :, i); rng, nboot) for i in 1:nq]
    # The lag-1 autocorrelation of the realised and predicted correction, which is what exposes a
    # flat histogram with no dynamics.
    rho1_truth = [autocorr(collect(view(pred.Y, :, i)), 1)[2] for i in 1:nq]
    rho1_mean = [autocorr(pred.mu isa AbstractMatrix ? collect(view(pred.mu, :, i)) :
                          fill(pred.mu[i], N), 1)[2] for i in 1:nq]
    # CRPS is in the units of the variable, and the six QoIs differ by four orders of magnitude:
    # the energy bands have a dQ standard deviation of 5e-4 to 9e-3 while Z[16,32] has 30.
    # Averaging raw CRPS across them is an enstrophy average with the energy bands contributing
    # nothing -- the trap `metrics.md` names as "standardise before mixing rows". So the only
    # pooled number reported is the **CRPS skill ratio**: each QoI's CRPS divided by the standard
    # deviation of the reference correction for that QoI. Dimensionless, comparable across bands,
    # and it reads as "how large is the score relative to the spread of what is being predicted".
    scale = ref_scale === nothing ? ones(nq) : collect(ref_scale)
    crps_norm = crps ./ scale
    return (; name, n = N, nq, M, nll, crps, crps_closed, crps_norm,
            crps_norm_mean = sum(crps_norm) / nq, rh, rho1_truth, rho1_mean,
            sd, mu_is_state_dependent = pred.mu isa AbstractMatrix)
end

# ---------------------------------------------------------------------------------------------
# regime C -- the archived online ensembles
# ---------------------------------------------------------------------------------------------

"""
    score_regime_c(name, ens, q_ref, ref_dQ; dt, lag_int)

Metrics #10, #11, #12-15, #16 and #18 for one archived online ensemble.

🔑 **Scored on the QoI level `q`, against the high-fidelity reference level.** `dQ` is computed
alongside as a secondary diagnostic, and the primary is `q`.

This is the opposite of regime A's framing and the difference is not a matter of taste; it is what
the regime dictates:

  * In **regime B** the predictor stream is replayed from a tracked record, so `q = q* + dQ` with
    `q*` fixed and dominant. Every model's level trajectory is then pinned to the reference --
    measured spread across nine models was 0.027-0.289 on `q` against 0.18-2.82 on `dQ` -- and only
    the correction discriminates. That is what O7 and `metrics.md` §1 are about, and it applies
    **only** where `q*` is replayed.
  * In **regime C** nothing is replayed: `q*` is produced by the solver from a field the model has
    been perturbing since step one. The level is therefore free, and the level is precisely what
    the physical claims are about -- the long-term QoI distribution (#10, #11), the QoI
    decorrelation time (#12-#15), and the spread of the QoIs (#18). Whether the *correction's* own
    marginal is right is not a claim anyone makes.

It also restores comparability. Paper 2's `compute_ks.jl:48` computes
`ks_dist(q_ref[i,:], q_rep[r][i,:])` -- on the level -- so a `dQ`-based summed KS cannot be checked
against Fig. 6 or against the archived `ks_dists_*.jld2` tables, and cannot serve G1's online
acceptance. A `q`-based one can, and [`compare_archived_ks`](@ref) does it.

In regime **A** the question does not arise: given the history, `q` and `dQ` differ by a shift
common to the forecast and the truth, so the NLL, the CRPS and the ranks are identical either way
(verified to 1e-16 on CRPS and bit-identically on ranks; V26 covers the ranks).
"""
function score_regime_c(name, ens, q_ref, ref_dQ; dt = DT, lag_int = nothing)
    trajs = [Float64.(t) for t in ens.q]            # the level: the primary object
    dtrajs = [Float64.(t) for t in ens.dQ]          # the correction: secondary
    qref = Float64.(q_ref)
    # 🔴 Smagorinsky and no-model emit no correction at all -- no `dQ`, no `q*`. Every secondary
    # statistic below is then **undefined, not zero**; returning 0.0 would put a perfect-looking
    # `dQ` score beside a closure that has no `dQ` (the shape of memory #59's mistake). The level
    # statistics are unaffected and are the ones every claim rests on.
    has_dQ = !isempty(dtrajs)
    li = lag_int === nothing ? 1 : lag_int

    ks_per = [summed_ks(t, qref) for t in trajs]
    eks = ensemble_ks(trajs, qref)
    stab = stability_fraction(trajs)
    dr1 = [delta_rho(t, qref; lag = 1, maxlag = 400) for t in trajs]
    drt = [delta_rho(t, qref; lag = li) for t in trajs]

    # The same three on the correction, kept because it is what the model actually emits and
    # because the contrast between the two is itself the O7 finding.
    ks_per_dQ = has_dQ ? [summed_ks(t, ref_dQ) for t in dtrajs] : nothing
    eks_dQ    = has_dQ ? ensemble_ks(dtrajs, ref_dQ) : nothing
    dr1_dQ    = has_dQ ? [delta_rho(t, ref_dQ; lag = 1, maxlag = 400) for t in dtrajs] : nothing
    drt_dQ    = has_dQ ? [delta_rho(t, ref_dQ; lag = li) for t in dtrajs] : nothing
    # #18: the climatological spread-skill of the **level**, from replicas that have fully
    # decorrelated from their common start. Reported and labelled, never quoted as a lead.
    M = length(trajs)
    nq = size(trajs[1], 1)
    function ss_of(ts, ref)
        K = min(minimum(size(t, 2) for t in ts), size(ref, 2))
        arr = Array{Float64}(undef, K, nq, length(ts))
        for (j, t) in enumerate(ts)
            arr[:, :, j] .= transpose(view(t, :, 1:K))
        end
        return length(ts) > 1 ? spread_skill(arr, transpose(view(ref, :, 1:K))) : nothing
    end
    ss = ss_of(trajs, qref)
    ss_dQ = has_dQ ? ss_of(dtrajs, ref_dQ) : nothing
    return (; name, M, replicas = ens.replicas, family = ens.family, root = ens.root, has_dQ,
            # --- primary: the QoI level ---
            ks_summed = [k.total for k in ks_per], ks_per_qoi = [k.per_qoi for k in ks_per],
            ks_ensemble = eks.total, ks_ensemble_per_qoi = eks.per_qoi,
            stability = stab.fraction, first_nan = stab.first_nan,
            drho1 = [d.at_lag for d in dr1], drho_tau = [d.at_lag for d in drt],
            drho_int = [d.integral for d in dr1], lag_tau = li,
            rho_model = dr1[1].rho_model, rho_ref = dr1[1].rho_ref,
            spread_skill = ss,
            # --- secondary: the correction the model emits ---
            ks_summed_dQ = has_dQ ? [k.total for k in ks_per_dQ] : Float64[],
            ks_ensemble_dQ = has_dQ ? eks_dQ.total : NaN,
            drho1_dQ = has_dQ ? [d.at_lag for d in dr1_dQ] : Float64[],
            drho_tau_dQ = has_dQ ? [d.at_lag for d in drt_dQ] : Float64[],
            drho_int_dQ = has_dQ ? [d.integral for d in dr1_dQ] : Float64[],
            rho_model_dQ = has_dQ ? dr1_dQ[1].rho_model : Float64[],
            rho_ref_dQ = has_dQ ? dr1_dQ[1].rho_ref : Float64[],
            spread_skill_dQ = ss_dQ)
end

"""
    compare_archived_ks(name)

Paper 2's own summed KS for a named configuration, read from the archived `ks_dists_*.jld2`
tables, so the level-based numbers computed here can be checked against the published ones.

This is G1's online acceptance in `plan.md` §12: *"online, summed ensemble KS per config within
paper 2 Fig. 6's replica min-max range, pinned as a number in the test file"*. It is only
checkable because regime C is now scored on the level; a `dQ`-based score has nothing to compare
against.

✅ **The family question is closed and it was benign.** `compute_ks.jl` decides which replicas are
stable by testing `data_online_tsim100.0_replica<i>.jld2` with `isfile` (`:25`) and then loads
`..._replica<i>_rand_initial_dQ.jld2` (`:34`). An earlier version of this docstring called that a
table mixing *"a stability check on one set of runs with distances computed on another"*. Measured
2026-09-10: the two filenames are the **same runs**, bit-identical in `q` over all 40 001 columns
for LinReg1 (replicas 1-3) and LinReg63/64 (replicas 1-2), and `paper_runs/online_sgs.jl:62,84`
seeds its warm start from `data_track.dQ[:, 1:100]` exactly as the frozen driver does. So the
published table mixes nothing, and this comparison is against one consistent set of trajectories.

⚠️ The remaining caveat is real: the tables are split by configuration index across several files,
so a name absent from one is not absent from the archive.
"""
function compare_archived_ks(name::AbstractString)
    dir = joinpath(Archive_root(), "ks_data")
    isdir(dir) || return nothing
    for f in sort(readdir(dir))
        endswith(f, ".jld2") || continue
        t = try
            load(joinpath(dir, f), "ks_table")
        catch
            continue
        end
        names = getproperty(getproperty(t, :colindex), :names)
        cols = getproperty(t, :columns)
        i_nm = findfirst(==(:name), names)
        i_r = findfirst(==(:ks_dist_replicas), names)
        i_e = findfirst(==(:ks_dist_ensemble), names)
        i_u = findfirst(==(:n_unstable), names)
        (i_nm === nothing || i_r === nothing) && continue
        row = findfirst(==(name), cols[i_nm])
        row === nothing && continue
        reps = cols[i_r][row]
        reps === nothing && continue
        return (; file = f, replicas = [r[1] for r in reps],
                ensemble = cols[i_e][row][1],
                n_unstable = i_u === nothing ? -1 : cols[i_u][row])
    end
    return nothing
end

Archive_root() = get(ENV, "RIKFLOW_ARCHIVE", FROZEN)

# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# which data this run scores
# ---------------------------------------------------------------------------------------------

"R1's tracking record, cached QoIs. The source is 2.6 GB; `extract_qois.jl` makes this 7 MB."
const NEW_TRACK_QOIS = joinpath(HERE, "data",
    "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")

"""
    DATASET

`:archive` (the default) scores paper 2's frozen data -- the 100 TU `data_track2` record, the
archived HF reference and the five archived online ensembles. `:new` scores P2r's rebaselined
pipeline: R1's tracking record, the regenerated HF reference and R2's four closures.

Select with `RIKFLOW_DATASET=new`. The two write different score files, and `results.md` reports
the `:new` numbers as primary; `:archive` stays because G1's reproduction of paper 2 is defined
only against paper 2's own data and cannot be rebased onto a different record.

🔴 **Never pool them.** The rebaselined runs are post-`09954be1`, which changed the Nyquist
convention and therefore `tau` and `dQ`: a different dynamical system, not a better measurement of
the same one (memory #45, #46).
"""
const DATASET = Symbol(get(ENV, "RIKFLOW_DATASET", "archive"))

"""
    rebaselined_ensemble(key)

`load_rebaseline` reshaped into what [`score_regime_c`](@ref) expects.

The deterministic baselines carry no `dQ` and no `q_star`, so those come back empty and every
correction-based statistic is skipped rather than zeroed -- the `has_dQ` branch there.
"""
function rebaselined_ensemble(key::AbstractString)
    e = load_rebaseline(key)
    return (; e.q, e.q_star, e.dQ, e.tau, replicas = e.replica_index,
            family = e.label, root = :rebaselined, e.stochastic, e.clampable, e.key)
end

"The data sources for `which`, and what is and is not defined on them."
function dataset(which::Symbol)
    if which === :archive
        return (; tag = "archive",
                rec = load_qois(joinpath(HERE, "data",
                    "data_track2_dns512_les64_Re2000.0_tsim100.0_qois.jld2")),
                # Paper 2 fitted LinReg1 on a 10 TU record; kept for the clamp census, the one
                # place where the two records answer different questions.
                rec10 = load_qois(joinpath(HERE, "data",
                    "data_track_trackingnoise_std_0.0_Re2000.0_tsim10.0_replica1_qois.jld2")),
                q_ref = Float64.(load_reference()),
                configs = collect(ARCHIVED_CONFIGS), load_ens = load_ensemble,
                g1 = true, out = "paper4_scores.jld2")
    elseif which === :new
        return (; tag = "rebaselined",
                rec = load_qois(NEW_TRACK_QOIS),
                # 🔴 No 10 TU counterpart exists and none is coming: R1 tracked for 100 TU exactly
                # so one record carries both the fit window and D6's IC pool (#58).
                rec10 = nothing,
                q_ref = Float64.(load_new_reference().q_ref),
                # Only the cells whose ensembles have landed. A rung of the lambda ladder is in
                # `REBASE_MODELS` from the moment it is fitted, hours before its replicas finish.
                configs = [m.key for m in REBASE_MODELS if rebase_available(m.key)],
                load_ens = rebaselined_ensemble,
                g1 = false, out = "paper4_scores_new.jld2")
    else
        error("RIKFLOW_DATASET must be `archive` or `new`; got $which")
    end
end


function main()
    mkpath(OUT)
    rng = Random.MersenneTwister(SEED)
    println("=" ^ 92)
    println("Scoring M0 (TO+LRS) and DDN on HIT -- 0 SBU, no new model, no new simulation")
    println("=" ^ 92)
    flush(stdout)

    # ---- data ----------------------------------------------------------------------------
    ds = dataset(DATASET)
    rec10 = ds.rec10
    rec100 = ds.rec
    q_ref = ds.q_ref
    nq = size(rec100.q, 1)
    @printf("  dataset           : %s\n", ds.tag)
    rec10 === nothing ? println("  10 TU tracked     : none in this dataset") :
        @printf("  10 TU tracked     : q %s  q_star %s\n", size(rec10.q), size(rec10.q_star))
    @printf("  100 TU tracked    : q %s  q_star %s\n", size(rec100.q), size(rec100.q_star))
    @printf("  HF reference      : q_ref %s\n", size(q_ref))
    flush(stdout)

    # The reference correction, for the regime-C comparison. The HF record is a reference QoI
    # trajectory and carries no predictor stream, so the reference `dQ` is taken from the tracked
    # record -- which is what the tracked run's nudging produced to hold the LF QoIs on the
    # reference, and therefore the right target for a model of `dQ`.
    ref_dQ = Float64.(rec100.q[:, 2:end] .- rec100.q_star)
    @printf("  reference dQ      : %s (from the 100 TU tracked record)\n", size(ref_dQ))

    win = windows(rec100)
    @printf("  train  steps %d:%d  (%.2f-%.2f TU)\n", win.train[1], win.train[2],
            win.train[1] * DT, win.train[2] * DT)
    @printf("  heldout steps %d:%d  (%.2f-%.2f TU)\n", win.heldout[1], win.heldout[2],
            win.heldout[1] * DT, win.heldout[2] * DT)
    flush(stdout)

    # ---- t_int, both ways, per QoI -------------------------------------------------------
    # 🔴 Reported explicitly, because every document said 0.04 TU and the integral timescale is
    # several times that. They are two different quantities: an exponential fit to rho_1 gives a
    # lag-1-equivalent time, and the integral of the autocorrelation gives another. Delta_rho's
    # lag, the bootstrap block length, N_eff and P2c's whole cost hang off which is which.
    # 🔑 Both series, because they are different objects and this project has quoted one for the
    # other before. The CORRECTION decorrelates in tens of steps; the LEVEL inherits the QoI's own
    # smoothness, `rho_1(q)` is about 1, and its integral time is hundreds. D6's grid sizing runs
    # off the LEVEL's `T_int` (#58) while `Delta_rho`'s lag below runs off the CORRECTION's --
    # quoting the wrong one mis-sizes either the forecast grid or the temporal metric.
    #
    # The level gets a 4000-lag (10 TU) window: at the 500-lag default its ACF need not have
    # crossed zero yet, and `T_int` would come back as the window rather than the timescale.
    # `truncated` records which happened instead of leaving it to be assumed.
    tint = [correlation_time(collect(Float64.(view(ref_dQ, i, :))), DT) for i in 1:nq]
    tint_q = [correlation_time(collect(Float64.(view(q_ref, i, :))), DT; maxlag = 4000)
              for i in 1:nq]
    tint_qtrack = [correlation_time(collect(Float64.(view(rec100.q, i, :))), DT; maxlag = 4000)
                   for i in 1:nq]
    println("\n  timescales per QoI -- CORRECTION dQ and LEVEL q (metric O5):")
    @printf("    %-12s %-27s | %-27s\n", "", "dQ   rho1  T_exp   T_int", "q    rho1  T_exp   T_int")
    for i in 1:nq
        @printf("    %-12s %7.4f %7.4f %7.4f %-3s | %7.4f %7.4f %7.4f %s\n", LABELS[i],
                tint[i].rho1, tint[i].T_exp, tint[i].T_int, tint[i].truncated ? "(t)" : "",
                tint_q[i].rho1, tint_q[i].T_exp, tint_q[i].T_int,
                tint_q[i].truncated ? "(truncated -- lower bound)" : "")
    end
    @printf("    %-12s tracked-record q, T_int: %s\n", "",
            join((@sprintf("%.3f", t.T_int) for t in tint_qtrack), "  "))
    t_int_med = median([t.T_int for t in tint])
    lag_tau = max(1, round(Int, t_int_med / DT))
    @printf("    median T_int = %.4f TU  =>  Delta_rho lag = %d steps\n", t_int_med, lag_tau)
    flush(stdout)

    # ---- clamp census --------------------------------------------------------------------
    # #26. A property of the record, so computable for both models; the *firing* belongs to the
    # LinReg path alone, because DDN's sampler has no clamp.
    census = Dict{String,Any}()
    recs = rec10 === nothing ? (("tracked_100TU", rec100),) :
           (("tracked_10TU", rec10), ("tracked_100TU", rec100))
    for (nm, r) in recs
        c = clamp_census(Float64.(r.q_star))
        census[nm] = c
        @printf("\n  clamp census %-14s rate = %.3e over %d steps\n", nm, c.rate, c.nsteps)
        for i in 1:nq
            @printf("      %-12s %.3e\n", LABELS[i], c.per_qoi_rate[i])
        end
    end
    flush(stdout)

    # ---- fit both models -----------------------------------------------------------------
    println("\n  fitting M0 (harmonized and paper-2-faithful) and DDN ...")
    flush(stdout)
    m0 = fit_m0(rec100, win.train; h = 5, lambda = 0.0)
    m0f = fit_m0(rec100, win.train; h = 5, lambda = 0.0, normalization = :standardise,
                 penalize_intercept = true)
    ddn = fit_ddn(rec100, win.train)
    @printf("    M0  h=%d lambda=%.3g  %s  X = %s  intercept |b|max = %.3e\n", m0.h, m0.lambda,
            string(m0.scaling.mode), size(m0.X), maximum(abs, m0.C[bias_column(m0.spec), :]))
    @printf("    M0f h=%d lambda=%.3g  %s  intercept |b|max = %.3e\n", m0f.h, m0f.lambda,
            string(m0f.scaling.mode), maximum(abs, m0f.C[bias_column(m0f.spec), :]))
    @printf("    DDN fitted on %d steps of raw dQ; it has no history and no clamp\n", ddn.ntrain)
    flush(stdout)

    # ---- regime 0 diagnostics ------------------------------------------------------------
    println("\n  regime 0: Gram spectrum, rank, companion, block sum ...")
    flush(stdout)
    gram = Dict{String,Any}()
    for (nm, m) in (("harmonized", m0), ("faithful", m0f))
        for lam in (0.0, 0.01, 0.1, 1.0, 10.0)
            gram["$(nm)_lambda$(lam)"] = gram_diagnostics(m.X; lambda = lam)
        end
        gram["$(nm)_pinv"] = pinv_gap(m.X, m.Y)
        gram["$(nm)_rho"] = rho(m.C, m.spec)
        gram["$(nm)_gain"] = starred_gain(m.C, m.spec)
        gram["$(nm)_tbs"] = total_block_sum(m.C, m.spec)
    end
    g = gram["harmonized_lambda0.0"]
    @printf("    harmonized: sigma_min^2 = %.4e  kappa = %.4e  rank = %d/%d  branch(l=0.01) = %s\n",
            g.sigma_min2, g.cond, g.rank, size(m0.X, 2),
            string(gram["harmonized_lambda0.01"].branch))
    gf = gram["faithful_lambda0.0"]
    @printf("    faithful  : sigma_min^2 = %.4e  kappa = %.4e  sigma_max^2 ratio = %.3f\n",
            gf.sigma_min2, gf.cond, gf.sigma_max2 / g.sigma_max2)
    @printf("    rho(Ctilde) harmonized = %.6f   faithful = %.6f  (Float64 solve)\n",
            gram["harmonized_rho"], gram["faithful_rho"])
    @printf("    total block sum dev from I: harmonized %.4e  faithful %.4e\n",
            gram["harmonized_tbs"].dev_from_identity, gram["faithful_tbs"].dev_from_identity)

    # The precision comparison, reported rather than assumed. Under the archive's own Float32
    # arithmetic the coefficient-level diagnostics take different values; the invariance check is
    # what shows which of the two is the arithmetic and which is the model.
    m0_32 = fit_m0(rec100, win.train; h = 5, lambda = 0.0, T = Float32)
    m0f_32 = fit_m0(rec100, win.train; h = 5, lambda = 0.0, normalization = :standardise,
                    penalize_intercept = true, T = Float32)
    prec = (; rho32 = rho(m0_32.C, m0_32.spec), rho64 = gram["harmonized_rho"],
            gain32 = starred_gain(m0_32.C, m0_32.spec).hinf,
            gain64 = gram["harmonized_gain"].hinf,
            tbs32 = total_block_sum(m0_32.C, m0_32.spec).dev_from_identity,
            tbs64 = gram["harmonized_tbs"].dev_from_identity,
            slope_rel_f32_vs_f64 = norm(Float64.(m0_32.C[1:(end - 1), :]) -
                                        m0.C[1:(end - 1), :]) / norm(m0.C[1:(end - 1), :]),
            slope_rel_conventions_f32 = norm(m0_32.C[1:(end - 1), :] -
                                             m0f_32.C[1:(end - 1), :]) /
                                        norm(m0f_32.C[1:(end - 1), :]),
            slope_rel_conventions_f64 = norm(m0.C[1:(end - 1), :] - m0f.C[1:(end - 1), :]) /
                                        norm(m0f.C[1:(end - 1), :]),
            kappa_eps32 = gram["harmonized_lambda0.0"].cond * eps(Float32))
    @printf("    precision: rho f32 %.4f vs f64 %.4f | hinf f32 %.2f vs f64 %.2f | ",
            prec.rho32, prec.rho64, prec.gain32, prec.gain64)
    @printf("|S-I| f32 %.3e vs f64 %.3e\n", prec.tbs32, prec.tbs64)
    @printf("    slope-block rel diff: f32 vs f64 %.3e | conventions in f32 %.3e | in f64 %.3e",
            prec.slope_rel_f32_vs_f64, prec.slope_rel_conventions_f32,
            prec.slope_rel_conventions_f64)
    @printf("   (kappa*eps32 = %.3f)\n", prec.kappa_eps32)
    flush(stdout)

    # ---- regime A ------------------------------------------------------------------------
    println("\n  regime A: held-out NLL, CRPS and RH-1 on dQ in physical units ...")
    flush(stdout)
    pm0 = m0_predictive_dQ(m0, rec100, win.heldout)
    pddn = ddn_predictive_dQ(ddn, rec100, win.heldout, pm0.steps)
    @assert pm0.Y ≈ pddn.Y "the two models must be scored on identical verification instances"
    ref_scale = vec(std(view(ref_dQ, :, win.heldout[1]:(win.heldout[2] - 1)); dims = 2))
    a_m0 = score_regime_a("M0 (TO+LRS)", pm0; M = 100, rng, nboot = 500, ref_scale)
    a_ddn = score_regime_a("DDN", pddn; M = 100, rng, nboot = 500, ref_scale)

    # 🔑 The autocorrelation of the correction, for the figure — and the one place DDN can appear on
    # it at all.
    #
    # DDN has **no archived online runs** (`7_online_DDN.jl` writes to `output/TO_DDN/`, which does
    # not exist in either archive root), so it has no regime-C level trajectory and cannot join the
    # five configurations there. What it does have is a fitted predictive density, and drawing one
    # sample path from it is the honest way to show what its correction time series looks like.
    #
    # Three curves, all over the held-out window and all on `dQ`:
    #   * `truth`  -- the realised correction. This is what a model has to reproduce.
    #   * `m0_mu`  -- M0's predictive *mean*. It tracks, so this decays like the truth.
    #   * `ddn`    -- one sampled DDN path. State-independent draws, so it is white by
    #                 construction and sits at zero from lag 1. Sampled rather than asserted so the
    #                 line carries its own sampling noise and can be read as a measurement.
    #
    # ⚠️ This is regime A, mixed into a regime-C figure deliberately and labelled as such: the
    # question "does the model have memory at all" is answered by the fitted density, and waiting
    # for DDN online runs (~51 SBU) to answer it would be spending compute on a foregone conclusion.
    acf_a = let nl = 400
        Ld = cholesky(Symmetric(pddn.Sigma)).L
        Nh = size(pddn.Y, 1)
        z = Array{Float64}(undef, nq)
        ddn_path = Array{Float64}(undef, Nh, nq)
        for n in 1:Nh
            randn!(rng, z)
            ddn_path[n, :] .= pddn.mu .+ Ld * z
        end
        (; lags = collect(0:nl) .* DT,
         truth = [autocorr(collect(view(pm0.Y, :, i)), nl) for i in 1:nq],
         m0_mu = [autocorr(collect(view(pm0.mu, :, i)), nl) for i in 1:nq],
         ddn = [autocorr(collect(view(ddn_path, :, i)), nl) for i in 1:nq])
    end
    println("\n  ACF of the correction over the held-out window, lag 1:")
    @printf("      %-10s %10s %10s %10s\n", "QoI", "truth", "M0 mean", "DDN")
    for i in 1:nq
        @printf("      %-10s %10.4f %10.4f %10.4f\n", LABELS[i], acf_a.truth[i][2],
                acf_a.m0_mu[i][2], acf_a.ddn[i][2])
    end
    flush(stdout)

    # 🔑 The in-sample histogram is RH-1's null, and flatness is not.
    #
    # On the training window M0's residual mean is 1e-4 standard deviations and its residual
    # standard deviation matches the fitted Sigma to 1.0001 -- the fit is exact there by
    # construction. The rank histogram is nonetheless cap-shaped, convexity -3.7 to -9.2, because
    # the residual is **leptokurtic**: excess kurtosis +0.9 to +2.0, so it is more sharply peaked
    # than the Gaussian fitted to it and the truth lands in the middle bins more often than a
    # Gaussian truth would. A simulated control confirms it -- resample the truth from the
    # empirical residual, draw the members from the Gaussian fitted to that same residual, and the
    # convexity reproduces without any error in either the mean or the variance.
    #
    # So a held-out convexity must be read **against this baseline**, not against zero. Reading it
    # against zero would attribute a pure shape effect to dispersion, in the direction that
    # flatters the model in the high bands and condemns it in the low ones.
    pm0_in = m0_predictive_dQ(m0, rec100, win.train)
    a_m0_in = score_regime_a("M0 in-sample", pm0_in; M = 100, rng, nboot = 500, ref_scale)
    res_in = pm0_in.Y .- pm0_in.mu
    shape = [(; skew = moment_std(view(res_in, :, i), 3),
              exkurt = moment_std(view(res_in, :, i), 4) - 3.0) for i in 1:nq]
    sd_fit = sqrt.(diag(pm0.Sigma))
    sd_ratio_in = vec(std(res_in; dims = 1)) ./ sd_fit
    sd_ratio_out = vec(std(pm0.Y .- pm0.mu; dims = 1)) ./ sd_fit
    println("\n  RH-1's null: M0 scored on its own training window")
    @printf("      %-10s %9s %9s %11s %11s %10s %10s\n", "QoI", "skew", "ex.kurt",
            "sd_in/fit", "sd_out/fit", "cvx_in", "cvx_out")
    for i in 1:nq
        @printf("      %-10s %9.3f %9.3f %11.4f %11.4f %10.3f %10.3f\n", LABELS[i],
                shape[i].skew, shape[i].exkurt, sd_ratio_in[i], sd_ratio_out[i],
                a_m0_in.rh[i].convexity, a_m0.rh[i].convexity)
    end
    flush(stdout)

    # Phase 0.4: the O(|sgs|^2) gap. QoIs are quadratic in the velocity while the TO correction is
    # linear-response and is added to `u` with no projection, so the recomputed correction
    # `q - q_star` and the stored `dQ` are not the same object. `plan.md` §0 item 8 calls this
    # structural rather than a bug and Phase 0.4 exists to size it; 0.4's stated consequence is
    # that at O(1e-2) the later rungs must buffer recomputed QoIs rather than `q_star + dQ`.
    o4 = if rec100.dQ === nothing
        nothing
    else
        rc = Float64.(rec100.q[:, 2:end] .- rec100.q_star)
        st = Float64.(rec100.dQ)
        n = min(size(rc, 2), size(st, 2))
        per = [maximum(abs, view(rc, i, 1:n) .- view(st, i, 1:n)) / std(view(st, i, 1:n))
               for i in 1:nq]
        (; overall = maximum(abs, view(rc, :, 1:n) .- view(st, :, 1:n)) /
                     maximum(abs, view(st, :, 1:n)), per_qoi_rel_sd = per)
    end
    if o4 !== nothing
        @printf("\n  Phase 0.4  |q - q* - dQ|: overall rel %.4e, per QoI vs own sd %.3f to %.3f\n",
                o4.overall, minimum(o4.per_qoi_rel_sd), maximum(o4.per_qoi_rel_sd))
        flush(stdout)
    end
    for s in (a_m0, a_ddn)
        @printf("    %-12s N = %d  NLL/sample = %9.4f  mean CRPS/sd(dQ) = %7.4f\n", s.name,
                s.n, s.nll.per_sample, s.crps_norm_mean)
        @printf("      %-10s %-12s %-9s %-9s %-10s %-9s\n", "QoI", "CRPS", "CRPS/sd", "slope",
                "convexity", "rho1_mu")
        for i in 1:s.nq
            @printf("      %-10s %12.4e %9.4f %9.3f %10.3f %9.4f\n", LABELS[i], s.crps[i],
                    s.crps_norm[i], s.rh[i].slope, s.rh[i].convexity, s.rho1_mean[i])
        end
    end
    flush(stdout)

    # ---- regime C ------------------------------------------------------------------------
    @printf("\n  regime C: %s online ensembles ...\n", ds.tag)
    flush(stdout)
    online = Dict{String,Any}()
    for nm in ds.configs
        e = try
            ds.load_ens(nm)
        catch err
            @warn "no online ensemble for $nm" err
            nothing
        end
        e === nothing && continue
        s = score_regime_c(nm, e, q_ref, ref_dQ; lag_int = lag_tau)
        online[nm] = s
        @printf("    %-10s M=%d %-6s | q: sumKS %.3f-%.3f ensKS %.3f drho1 %.3f drho%d %.3f | ",
                nm, s.M, string(s.root), minimum(s.ks_summed), maximum(s.ks_summed),
                s.ks_ensemble, median(s.drho1), s.lag_tau, median(s.drho_tau))
        s.has_dQ ?
            @printf("dQ: sumKS %.3f-%.3f drho1 %.3f | stab %.2f\n", minimum(s.ks_summed_dQ),
                    maximum(s.ks_summed_dQ), median(s.drho1_dQ), s.stability) :
            @printf("dQ: none (deterministic closure) | stab %.2f\n", s.stability)
        flush(stdout)
    end
    floor_ = ks_noise_floor(Float64.(q_ref))
    floor_dQ = ks_noise_floor(ref_dQ)
    @printf("    KS noise floor (D8, reference vs reference): q %.4f   dQ %.4f\n", floor_.total,
            floor_dQ.total)

    # G1's online acceptance: the level-based summed KS against paper 2's own published table.
    # 🔴 `archived` is declared OUTSIDE the branch: it is saved unconditionally, so leaving it
    # inside `else` makes it undefined on the rebaselined run -- an UndefVarError at the jldsave,
    # after every score has been computed.
    archived = Dict{String,Any}()
    if !ds.g1
        println("\n  G1 online: SKIPPED -- G1 reproduces paper 2's published KS table, and " *
                "these runs are a different dynamical system (#45, #46); nothing to reproduce. " *
                "An empty `archived_ks` in the score file means this, not a failed comparison.")
    else
    println("\n  G1 online: this round's level-based summed KS against paper 2's archived table")
    for nm in sort(collect(keys(online)))
        a = compare_archived_ks(nm)
        s = online[nm]
        if a === nothing
            @printf("    %-10s no archived KS row found\n", nm)
            continue
        end
        archived[nm] = a
        inside = minimum(a.replicas) <= median(s.ks_summed) <= maximum(a.replicas)
        @printf("    %-10s here %.3f-%.3f (med %.3f) | paper 2 %.3f-%.3f (ens %.3f) | %s | %s\n",
                nm, minimum(s.ks_summed), maximum(s.ks_summed), median(s.ks_summed),
                minimum(a.replicas), maximum(a.replicas), a.ensemble,
                inside ? "median INSIDE paper 2 range" : "median OUTSIDE", a.file)
        flush(stdout)
    end
    end

    # #26 on the free-running records. The tracked records answer only half the question: nudging
    # holds the QoIs near the reference, so a small predictor is unlikely there by construction. An
    # online run is free to wander, and it is the run the clamp actually sits in.
    println("\n  clamp census on the free-running online records:")
    for nm in sort(collect(keys(online)))
        e = ds.load_ens(nm)
        isempty(e.q_star) &&
            (@printf("    %-10s deterministic -- no q*, stabilizer does not apply\n", nm); continue)
        rates = Float64[]
        mins = Float64[]
        for qs in e.q_star
            c = clamp_census(Float64.(qs))
            push!(rates, c.rate)
            push!(mins, minimum(abs, qs))
        end
        census["online_$nm"] = (; rate = maximum(rates), nfired = 0,
                                nsteps = size(e.q_star[1], 2), per_qoi_rate = rates,
                                threshold = 1e-2)
        @printf("    %-10s max rate over %d replicas = %.3e   global min |q*| = %.4e\n", nm,
                length(rates), maximum(rates), minimum(mins))
        flush(stdout)
    end

    # ---- save ----------------------------------------------------------------------------
    out = joinpath(OUT, ds.out)
    jldsave(out;
            labels = LABELS, dt = DT, seed = SEED, dataset = String(DATASET),
            windows = (; train = win.train, heldout = win.heldout, nsteps = win.nsteps),
            tint = [(; t.rho1, t.T_exp, t.T_int) for t in tint], lag_tau, t_int_med,
            tint_q = [(; t.rho1, t.T_exp, t.T_int, t.truncated) for t in tint_q],
            tint_q_tracked = [(; t.rho1, t.T_exp, t.T_int, t.truncated) for t in tint_qtrack],
            t_int_q_med = median([t.T_int for t in tint_q]),
            census = Dict(k => (; v.rate, v.nfired, v.nsteps, v.per_qoi_rate, v.threshold)
                          for (k, v) in census),
            m0 = (; m0.h, m0.lambda, m0.normalization, m0.penalize_intercept, m0.C,
                  m0.Sigma_scaled, sigma = collect(m0.scaling.sigma),
                  mu = m0.scaling.mu, nfeatures = size(m0.X, 2)),
            m0_faithful = (; m0f.h, m0f.lambda, m0f.normalization, m0f.penalize_intercept, m0f.C,
                           sigma = collect(m0f.scaling.sigma)),
            ddn = (; ddn.mu, ddn.Sigma, ddn.ntrain),
            regime_a = Dict("M0" => strip_a(a_m0), "DDN" => strip_a(a_ddn),
                            "M0_insample" => strip_a(a_m0_in)),
            residual_shape = (; skew = [x.skew for x in shape],
                              exkurt = [x.exkurt for x in shape],
                              sd_ratio_in = sd_ratio_in, sd_ratio_out = sd_ratio_out,
                              sd_fit = sd_fit),
            phase04 = o4 === nothing ? (; overall = NaN, per_qoi_rel_sd = fill(NaN, nq)) : o4,
            regime_a_series = (; Y = pm0.Y, mu_m0 = pm0.mu, steps = pm0.steps),
            acf_regime_a = acf_a,
            gram = Dict(k => strip_g(v) for (k, v) in gram),
            precision = prec,
            online = Dict(k => strip_c(v) for (k, v) in online),
            ks_floor = (; floor_.total, floor_.per_qoi),
            ks_floor_dQ = (; total = floor_dQ.total, per_qoi = floor_dQ.per_qoi),
            archived_ks = archived,
            ref_dQ_stats = (; std = vec(std(ref_dQ; dims = 2)),
                            rho1 = [t.rho1 for t in tint]),
            # The overview is subsampled 1:40 to keep the file small; a full-resolution slice
            # straddling the train/held-out boundary is kept alongside it, because at 100 TU the
            # tracked and reference curves are visually one line -- nudging pins them together, and
            # a figure that shows one line cannot demonstrate that.
            trajectories = (; q_ref = q_ref[:, 1:40:end],
                            q_track = rec100.q[:, 1:40:end],
                            q_star_track = rec100.q_star[:, 1:40:end],
                            dQ_track = ref_dQ[:, 1:40:end], stride = 40,
                            zoom_range = (3600, 4400),
                            zoom_q_ref = q_ref[:, 3600:4400],
                            zoom_q_track = rec100.q[:, 3600:4400],
                            zoom_q_star = rec100.q_star[:, 3600:4400],
                            zoom_dQ = ref_dQ[:, 3600:4400],
                            track_vs_ref_rel_rms =
                                vec(sqrt.(mean(abs2, Float64.(rec100.q[:, 1:40001]) .-
                                                     Float64.(q_ref[:, 1:40001]); dims = 2))) ./
                                vec(std(Float64.(q_ref[:, 1:40001]); dims = 2))),
            created = string(now()))
    @printf("\n  wrote %s (%.2f MB)\n", out, filesize(out) / 2^20)
    return out
end

# JLD2 cannot store the closures and views hiding in the score named tuples, so each is reduced to
# plain arrays before saving. Keeping this explicit means the saved file is readable by anything.
strip_a(s) = (; s.name, s.n, s.nq, s.M, nll_per_sample = s.nll.per_sample,
              nll_total = s.nll.total, s.crps, s.crps_closed, s.crps_norm, s.crps_norm_mean,
              s.rho1_truth, s.rho1_mean, s.sd, s.mu_is_state_dependent,
              rh = [(; r.counts, r.K, r.n, r.M, r.expected, r.slope, r.convexity, r.slope_raw,
                     r.convexity_raw, r.slope_ci, r.convexity_ci, r.chi2, r.chi2_eff, r.dof,
                     r.n_eff, r.blocklen, r.eff_factor) for r in s.rh])

function strip_g(v)
    v isa NamedTuple || return v
    d = Dict{String,Any}()
    for k in propertynames(v)
        x = getfield(v, k)
        d[string(k)] = x isa AbstractArray ? collect(x) : (x isa Symbol ? string(x) : x)
    end
    return d
end

# The autocorrelation curves are kept to a fixed 400 lags (1 TU at dt = 2.5e-3) so the saved file
# stays small and every configuration is plotted over the same axis.
const ACF_LAGS = 400

strip_c(s) = (; s.name, s.M, s.replicas, s.family, root = string(s.root), s.ks_summed,
              s.ks_per_qoi, s.ks_ensemble, s.ks_ensemble_per_qoi, s.stability,
              first_nan = [x === nothing ? 0 : x for x in s.first_nan],
              s.drho1, s.drho_tau, s.drho_int, s.lag_tau,
              s.ks_summed_dQ, s.ks_ensemble_dQ, s.drho1_dQ, s.drho_tau_dQ, s.drho_int_dQ,
              rho_model_dQ = [r[1:min(end, ACF_LAGS + 1)] for r in s.rho_model_dQ],
              rho_ref_dQ = [r[1:min(end, ACF_LAGS + 1)] for r in s.rho_ref_dQ],
              spread_skill_dQ = s.spread_skill_dQ === nothing ? nothing :
                                (; s.spread_skill_dQ.ratio, s.spread_skill_dQ.spread,
                                 s.spread_skill_dQ.skill, s.spread_skill_dQ.M,
                                 s.spread_skill_dQ.K, s.spread_skill_dQ.correction,
                                 s.spread_skill_dQ.per_qoi),
              rho_model = [r[1:min(end, ACF_LAGS + 1)] for r in s.rho_model],
              rho_ref = [r[1:min(end, ACF_LAGS + 1)] for r in s.rho_ref],
              spread_skill = s.spread_skill === nothing ? nothing :
                             (; s.spread_skill.ratio, s.spread_skill.spread,
                              s.spread_skill.skill, s.spread_skill.M, s.spread_skill.K,
                              s.spread_skill.correction, s.spread_skill.per_qoi))

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
