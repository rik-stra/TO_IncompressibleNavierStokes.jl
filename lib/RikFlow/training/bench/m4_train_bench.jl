# M4 training-cost bench -- what does one batched gradient step cost?
#
#     julia --startup-file=no --project=lib/RikFlow/training/bench \
#           lib/RikFlow/training/bench/m4_train_bench.jl
#
# 🔑 Companion to `exp_square_HIT/tools/m4_cost_probe.jl`, which times the *deployed* cell. This
# one times the *training* side, which is the other half of M4's cost and the one that multiplies
# by the sweep.
#
# The geometry is deliberately small: `N_Q = 6` is what the project actually predicts, and at that
# width `n_hidden = 60` is far more capacity than the target needs, so the bench runs at
# `n_hidden = 16`. That is the regime where Zygote's per-operation overhead, not BLAS, sets the
# cost -- which is the regime the optimisation work is aimed at and the one the right-sized M4
# configs run in.
#
# 🔑 **The number to watch is the reverse/forward ratio.** Reverse mode should cost 2-4x a forward
# pass; anything far above that is a pathological pullback rather than real work, and is worth a
# `m4_profile.jl` run. It read 26x before `_timeslices` and 5.8x after.

using RikFlow
using Lux, Optimisers, Zygote
using Random, LinearAlgebra, Statistics, Printf

const RF = RikFlow

const T = Float32
const N_QOI = 6          # what the project predicts
const N_HIDDEN = 16      # deliberately small: 6 QoIs do not need 60 hidden units
const N_LATENT = 4
const N_ENCODER = 8
const HIST_H = 2
const L = 200
const BURN = 50
const BATCH = 8

"Median wall time in milliseconds of `f()`, plus allocation count and bytes of one call."
function timeit(f; reps = 20, warmup = 3)
    for _ in 1:warmup
        f()
    end
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e6)
    end
    st = @timed f()
    return (; ms = median(ts), bytes = st.bytes, allocs = Base.gc_alloc_count(st.gcstats))
end

function main()
    spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = HIST_H, n_qoi = N_QOI),
                       n_hidden = N_HIDDEN, n_latent = N_LATENT, n_encoder = N_ENCODER,
                       arch = :vrnn)
    rng = Xoshiro(1)
    nin, nout, nz = RF.n_input(spec), RF.n_output(spec), spec.n_latent

    # one minibatch, in the layout `lstm_forward` takes: features x time x segment
    Xb = randn(rng, T, nin, L, BATCH)
    Yb = randn(rng, T, nout, L, BATCH)
    eb = randn(rng, T, nz, L, BATCH)
    score = (BURN + 1):L
    ps = RF.init_lstm_params(Xoshiro(2), spec; T)

    @printf("geometry : n_input=%d n_hidden=%d n_latent=%d n_out=%d arch=%s emission=%s\n",
            nin, spec.n_hidden, spec.n_latent, nout, spec.arch, spec.emission)
    @printf("minibatch: L=%d burn=%d batch=%d\n\n", L, BURN, BATCH)

    # --- V50 in miniature: batching must be a pure speed change ------------------------------
    per_seg = mean(RF.elbo(spec, ps, Xb[:, :, b:b], Yb[:, :, b:b], score, eb[:, :, b:b];
                           beta = 1e-4) for b in 1:BATCH)
    batched = RF.elbo(spec, ps, Xb, Yb, score, eb; beta = 1e-4)
    @printf("elbo, mean of per-segment : %.10f\n", per_seg)
    @printf("elbo, batched             : %.10f   (rel. diff %.3e)\n\n",
            batched, abs(batched - per_seg) / abs(per_seg))

    fw = timeit(() -> RF.elbo(spec, ps, Xb, Yb, score, eb; beta = 1e-4))
    @printf("elbo (forward)  : %8.3f ms  %9d allocs  %8.2f MiB\n",
            fw.ms, fw.allocs, fw.bytes / 2^20)

    gr = timeit(() -> Zygote.withgradient(p -> RF.elbo(spec, p, Xb, Yb, score, eb; beta = 1e-4),
                                          ps); reps = 10)
    @printf("withgradient    : %8.3f ms  %9d allocs  %8.2f MiB\n",
            gr.ms, gr.allocs, gr.bytes / 2^20)
    @printf("reverse/forward : %8.2fx\n", gr.ms / fw.ms)

    return nothing
end

main()
