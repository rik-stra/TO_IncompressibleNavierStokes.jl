# M4 as a POST-RUN CORRECTOR -- the faithful Sørensen setting, and regime A for the ladder.
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/analysis/postrun_lstm.jl <model_index> [seed]
#
# ⚠️ **Runs under the training project, not `--project=analysis`.** It needs the package (for
# `load_stochlstm` and the scorers) and, for the IWAE bound only, the Lux extension. This is the
# second exception of its kind in `analysis/`; `ou_replay.jl` is the first, and both say so in
# their own headers rather than leaving a reader to discover it.
#
# ---------------------------------------------------------------------------------------------
# Why this driver exists, and what it is NOT
# ---------------------------------------------------------------------------------------------
#
# 🔑 **Barthel Sørensen et al. do not run their model as a closure.** They train a low-fidelity to
# high-fidelity trajectory map and apply it as post-processing, outside the solver. This driver is
# that setting: the model is teacher-forced on the recorded history and never sees its own output,
# so there is no feedback, no exposure bias, and nothing that can go unstable. It is the honest
# reproduction of the claim M4 was imported for -- that the architectures with upstream
# stochasticity (STORN, VRNN) overfit less than the deterministic and output-only ones -- and it
# costs no solver time at all.
#
# `exp_square_HIT/12_online_StochLSTM.jl` is the other half: M4 *as a closure*, which is what this
# project actually needs and which they never did.
#
# 🔴 **M4 and M0 must be compared in the same mode.** An M4 scored here against an M0 scored in the
# solver is not a comparison; it is exactly the confound `plan.md` §22 item 9 is written to name.
#
# ---------------------------------------------------------------------------------------------
# What it reports, and what it deliberately does not
# ---------------------------------------------------------------------------------------------
#
# 🔴 A stochastic latent path has **no closed-form one-step predictive density**, so exact NLL,
# closed-form CRPS and `companion`/rho(C~) are all undefined for M4 (`plan.md` §3). What survives:
#
#   - `crps_ensemble` -- reads the same on M4 as on every linear cell.
#   - the rank histogram with Jolliffe-Primo contrasts -- §9 makes this the ladder-wide
#     calibration metric precisely BECAUSE it is the one metric that survives to M4.
#   - the IWAE-K bound, reported as `nll_iwae`. ⚠️ It is a LOWER bound on log p, hence an UPPER
#     bound on the NLL, and it is **not comparable to M0's exact NLL and never shares a column
#     with one**.
#   - KL(q||p) per latent dimension. 🔑 Not optional: if it collapses to zero the latent path is
#     unused and :storn/:vrnn have silently become :lstm with a heteroscedastic head, which voids
#     the very comparison M4 exists for while every other number still looks fine.

using RikFlow
# ⚠️ All three, and all three are needed: `RikFlowLuxExt` is triggered by Lux + Optimisers +
# Zygote *together*, and the IWAE bound lives in it. Loading only `RikFlow` leaves the extension
# dormant and the likelihood silently unreported -- which is exactly what happened the first time
# this driver was run.
using Lux, Optimisers, Zygote
using JLD2
using Random
using Statistics
using Printf
using LinearAlgebra

const RF = RikFlow

length(ARGS) >= 1 || error("usage: julia postrun_lstm.jl <model_index> [seed]")
model_index = parse(Int, ARGS[1])
seed_arg = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing

const M_MEMBERS = parse(Int, get(ENV, "RIKFLOW_M4_MEMBERS", "50"))
const K_IWAE = parse(Int, get(ENV, "RIKFLOW_M4_IWAE_K", "64"))

TO_folder = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
DATA = normpath(joinpath(@__DIR__, "data"))

qoi_cache = get(ENV, "RIKFLOW_QOI_CACHE", "")
function load_qois()
    path = if !isempty(qoi_cache)
        qoi_cache
    else
        hits = isdir(DATA) ? filter(f -> occursin("data_track_dns512", f) &&
                                         endswith(f, "_qois.jld2"), readdir(DATA)) : String[]
        isempty(hits) && error("no tracked QoI cache found under $DATA; set RIKFLOW_QOI_CACHE")
        length(hits) == 1 || error("ambiguous cache: $(sort(hits)). Set RIKFLOW_QOI_CACHE.")
        joinpath(DATA, only(hits))
    end
    d = load(path)
    return (; q = d["q"], q_star = d["q_star"], name = basename(path))
end

inputs = load(TO_folder * "/inputs_lstm.jld2", "inputs")
cfg = inputs[model_index]
out_dir = TO_folder * "/$(cfg.name)/"
seed = seed_arg !== nothing ? seed_arg :
       (isfile(out_dir * "seed_summary.jld2") ? load(out_dir * "seed_summary.jld2", "median_seed") : 1)
fit = RF.load_stochlstm(out_dir * "StochLSTM_seed$(seed).jld2")
spec, w, scaling = fit.spec, fit.weights, fit.scaling

rec = load_qois()
@info "post-run correction" cell=cfg.name arch=cfg.arch seed record=rec.name

# --- the evaluation window ---------------------------------------------------------------------
# 🔴 Disjoint from the training window by construction: training used `cfg.train_range`, this
# starts where that ended. Selection on a window that overlaps training is the one mistake that
# makes every number here meaningless.
a_tr, b_tr = cfg.train_range
a, b = b_tr, min(size(rec.q, 2), b_tr + 3600)
b - a > spec.hist.h + 10 || error("the evaluation window $(a):$(b) is too short")
@info "evaluation window" train=(a_tr, b_tr) eval=(a, b)

qs = RF.scale_input(rec.q[:, a:b], scaling.in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b - 1)], scaling.in_scaling)
X, Y, steps = RF.build_history(spec.hist, qss, qs)
Xc, Yc = permutedims(Float32.(X)), permutedims(Float32.(Y))
N = size(Xc, 2)
nq = RF.n_output(spec)
# 🔴 The scorer charges the recurrence for AT LEAST as long as training did. It used to use a
# fixed `h + 50 = 51`, which is under the level's 1/e crossing on every band (116-142 steps), so
# the first scored steps carried exactly the cold-start contamination the training burn-in exists
# to remove -- and the number was smaller than the burn-in the fit had been trained with, which is
# the part that makes it a defect rather than a choice.
burn = min(max(spec.hist.h + 50, get(cfg, :burn, 0)), N - 10)
score = (burn + 1):N

# --- the ensemble, teacher-forced ---------------------------------------------------------------
# Every member sees the same recorded inputs; they differ only in the latent draw and the emission
# draw. That is what "post-processing" means here -- the model never consumes its own output.
#
# ⚠️ In a function, not at top level. A bare `for` in a script is *soft scope*: `nkl += 1` inside
# one creates a fresh local and leaves the outer `nkl` at zero, so the KL diagnostic below would
# have divided by zero and reported NaN. `test_sources.jl`'s V36 scan caught exactly this.
function run_ensemble(spec, w, Xc, N, burn, nq, M)
    ens = zeros(Float32, N - burn, nq, M)
    klsum = zeros(Float64, spec.n_latent)
    nkl = 0
    for m in 1:M
        st = RF.LSTMState(spec, Float32)
        rng = Xoshiro(20_000 + m)
        draw = zeros(Float32, nq)
        for t in 1:N
            RF.lstm_step!(st, w, spec, view(Xc, :, t); rng, sample_latent = true)
            RF.sample_emission!(draw, st, w, spec, rng)
            t > burn && (ens[t - burn, :, m] .= draw)
            # KL( q(z|x) || N(0,I) ) per dimension -- the posterior-collapse diagnostic.
            # Accumulated on member 1 only: mu and sigma depend on the recorded input alone, so
            # every member would contribute identical numbers.
            if m == 1 && t > burn && RF.latent_sampled(spec)
                for k in 1:spec.n_latent
                    s, mu = st.sigz[k], st.muz[k]
                    klsum[k] += 0.5 * (s^2 + mu^2 - 1) - log(s)
                end
                nkl += 1
            end
        end
    end
    return ens, (nkl > 0 ? klsum ./ nkl : klsum), nkl
end

ens, kl, nkl = run_ensemble(spec, w, Xc, N, burn, nq, M_MEMBERS)

truth = permutedims(Yc[:, score])                    # N_score x N_Q

# --- scores --------------------------------------------------------------------------------------
crps = RF.crps_ensemble_mean(ens, truth)
rmse = [sqrt(mean((mean(ens[:, i, :]; dims = 2) .- truth[:, i]) .^ 2)) for i in 1:nq]

println()
# ⚠️ `chi2_eff`, `slope` and `convexity` are the effective-sample-size-corrected forms, and they
# are the ones to quote. `rank_histogram` also returns the raw pair, deliberately, so a test can
# assert that the uncorrected version fails -- see its long comment. Quoting the raw numbers on a
# serially correlated rank series reads several sigma on calibrated data.
@printf("%-12s %-10s %-10s %-10s %-9s %-9s\n",
        "QoI", "CRPS", "RMSE", "RH chi2_eff", "RH slope", "RH conv")
println("-"^66)
rh = [RF.rank_histogram(ens[:, i, :], truth[:, i]; rng = Xoshiro(7)) for i in 1:nq]
for i in 1:nq
    @printf("%-12s %-10.5f %-10.5f %-10.2f %-9.3f %-9.3f\n",
            "q$i", crps[i], rmse[i], rh[i].chi2_eff, rh[i].slope, rh[i].convexity)
end

# --- the IWAE bound ------------------------------------------------------------------------------
# 🔴 Reported as `nll_iwae` and NEVER in a column with an exact NLL. It is a lower bound on log p,
# so an upper bound on the NLL, and M0's number is exact -- the two are not on the same scale.
nll_iwae = missing
if !RF.emission_noise(spec)
    @info "emission = :none has a deterministic decoder, so there is no likelihood to bound. " *
          "CRPS and the rank histogram above are the scores for this cell."
elseif Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing
    @info "no Lux extension loaded, so no IWAE bound (run under --project=lib/RikFlow/training)"
elseif !haskey(fit.extras, :ps)
    @info "this fit predates `ps` being stored, so no IWAE bound; refit to get one"
else
    nll_iwae = RF.iwae_nll(spec, fit.extras.ps, Xc, Yc, score; K = K_IWAE, rng = Xoshiro(5))
    @printf("\nnll_iwae (K = %d): %.4f  per step, per QoI: %.4f\n",
            K_IWAE, nll_iwae, nll_iwae / nq)
    println("🔴 an UPPER bound on the NLL, and not comparable to M0's exact NLL.")
end

println()
if RF.latent_sampled(spec) && nkl > 0
    @printf("KL(q||p) per latent dim: mean %.4g  max %.4g  frac < 1e-3: %.2f\n",
            mean(kl), maximum(kl), count(<(1e-3), kl) / length(kl))
    if mean(kl) < 1e-3
        println("🔴 POSTERIOR COLLAPSE: the latent path is unused. :$(cfg.arch) has degenerated")
        println("   into a deterministic LSTM with a heteroscedastic head, and the")
        println("   upstream-stochasticity comparison this cell exists for is void.")
    end
else
    println("no latent path (arch = :$(cfg.arch)), so no KL to report")
end

mkpath(joinpath(@__DIR__, "output"))
jldsave(joinpath(@__DIR__, "output", "postrun_$(cfg.name)_seed$(seed).jld2");
        cell = cfg.name, arch = cfg.arch, seed, crps, rmse,
        rh_chi2_eff = [r.chi2_eff for r in rh], rh_slope = [r.slope for r in rh],
        rh_convexity = [r.convexity for r in rh], rh_n_eff = [r.n_eff for r in rh],
        kl = (RF.latent_sampled(spec) && nkl > 0) ? kl : nothing,
        members = M_MEMBERS, eval_window = (a, b), train_range = cfg.train_range,
        nll_iwae)
println("\nwrote analysis/output/postrun_$(cfg.name)_seed$(seed).jld2")
