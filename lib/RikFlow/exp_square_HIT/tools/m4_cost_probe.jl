# S4 cost probe for M4 -- what does one `get_next_item_timeseries` call actually cost?
#
#     julia --startup-file=no --project=lib/RikFlow lib/RikFlow/exp_square_HIT/tools/m4_cost_probe.jl
#
# 🔑 **S4 is a kill criterion, not a target** (`plan.md` §2): TO-LRS is already 2.04x no-model on
# HIT, and the budget is <=1.15x TO-LRS per step with no headroom. The surrogate's share is
# **1.85 ms/step** on HIT, and the plan is explicit that this is **launch-bound, not FLOP-bound**.
# So the question this probe answers is not "how many FLOPs" but "does a per-step call to the
# deployed cell fit inside 15% of 1.85 ms".
#
# ⚠️ Measures the closure only. It does not measure `to_sgs_term`'s FFTs, the QoI computation or
# the host/device round trip, all of which are already in the 1.85 ms. The number here is the
# *addition* M4 makes relative to `LinReg`'s matrix-vector product.
#
# Needs no Lux: the deployed path is stdlib. That is the point of the split.

using RikFlow
using Random
using Statistics
using Printf
using LinearAlgebra

const RF = RikFlow

# The deployed geometry: the source's dimensions at this project's N_Q.
const N_QOI = 6
const BUDGET_MS = 1.85          # the surrogate's measured share of a HIT step
const S4_HEADROOM = 0.15        # <=1.15x TO-LRS

# `emission` follows the deployed configuration: `:none` everywhere except the `:lstm` control,
# which `LSTMSpec` refuses without an emission head because it would have no stochasticity at all.
# The cost table therefore measures the model that runs, not a variant of it.
function build(; arch, h, n_hidden, n_latent, n_encoder, T = Float32,
               emission = arch === :lstm ? :constant : :none)
    spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h, n_qoi = N_QOI), n_hidden, n_latent,
                       n_encoder, arch, emission)
    H = spec.n_hidden
    nin, nout, nz = RF.n_input(spec), RF.n_output(spec), spec.n_latent
    ncin, nenc = RF.n_cell_input(spec), RF.n_encoder_out(spec)
    rng = Xoshiro(1)
    w = RF.LSTMWeights{T}(
        randn(rng, T, 4H, ncin), randn(rng, T, 4H, H), randn(rng, T, 4H),
        (n_encoder > 0 && RF.latent_sampled(spec)) ? randn(rng, T, nenc, nin) : nothing,
        (n_encoder > 0 && RF.latent_sampled(spec)) ? randn(rng, T, nenc) : nothing,
        randn(rng, T, nz, nenc), randn(rng, T, nz, nenc),
        randn(rng, T, nout, H),
        RF.latent_to_decoder(spec) ? randn(rng, T, nout, nz) : nothing,
        randn(rng, T, nout),
        randn(rng, T, nout, H), randn(rng, T, nout),
        Matrix{T}(I(nout)))
    mu = reshape(fill(0.1, N_QOI), N_QOI, 1)
    sigma = reshape(fill(1.0, N_QOI), N_QOI, 1)
    scaling = (in_scaling = (; mu, sigma), out_scaling = (; mu, sigma))
    nwarm = max(h, 4)
    spinnup = randn(Xoshiro(2), Float64, N_QOI, nwarm) ./ 10
    m = RF.StochLSTM(spec, w, scaling; spinnup_data = spinnup, rng = Xoshiro(3))
    return spec, m, nwarm
end

"Median wall time of one predicted step, in microseconds."
function time_step(m, nwarm; reps = 2000)
    q_star = randn(Xoshiro(11), Float64, N_QOI) .+ 1.0
    for _ in 1:nwarm                     # get past the replay
        RF.get_next_item_timeseries(m, q_star)
    end
    RF.get_next_item_timeseries(m, q_star)          # compile
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        RF.get_next_item_timeseries(m, q_star)
        ts[i] = (time_ns() - t0) / 1e3
    end
    return ts
end

println("M4 per-step cost, deployed closure only (CPU, Float32 weights)")
println("S4 budget: surrogate share $(BUDGET_MS) ms/step, allowance $(100 * S4_HEADROOM)% => ",
        @sprintf("%.0f us", 1000 * BUDGET_MS * S4_HEADROOM))
println()
@printf("%-8s %-4s %-7s %-7s %-7s | %9s %9s %9s %7s\n",
        "arch", "h", "hidden", "latent", "enc", "median", "mean", "p99", "% S4")
println("-"^88)

configs = [
    (; arch = :vrnn, h = 1, n_hidden = 60, n_latent = 60, n_encoder = 60),   # the source's size
    (; arch = :storn, h = 1, n_hidden = 60, n_latent = 60, n_encoder = 60),
    (; arch = :lstm, h = 1, n_hidden = 60, n_latent = 60, n_encoder = 60),
    (; arch = :vrnn, h = 5, n_hidden = 60, n_latent = 60, n_encoder = 60),   # at M0's h
    (; arch = :vrnn, h = 1, n_hidden = 128, n_latent = 60, n_encoder = 60),  # a wider cell
    (; arch = :vrnn, h = 1, n_hidden = 60, n_latent = 6, n_encoder = 0),     # the tex's lean form
]

allowance_us = 1000 * BUDGET_MS * S4_HEADROOM
for cfg in configs
    _, m, nwarm = build(; cfg...)
    ts = time_step(m, nwarm)
    med, mn, p99 = median(ts), mean(ts), quantile(ts, 0.99)
    @printf("%-8s %-4d %-7d %-7d %-7d | %7.2f us %7.2f us %7.2f us %6.1f%%\n",
            cfg.arch, cfg.h, cfg.n_hidden, cfg.n_latent, cfg.n_encoder,
            med, mn, p99, 100 * med / allowance_us)
end

println()
println("Read the last column as: percent of the S4 allowance consumed by the closure alone.")
println("🔴 A number near or above 100 means M4 fails S4 at that size and the size must come")
println("   down -- S4 is a kill criterion, not a target.")
