# Where does one batched M4 gradient step spend its time?
#
#     julia --startup-file=no --project=lib/RikFlow/training/bench \
#           lib/RikFlow/training/bench/m4_profile.jl
#
# Companion to `m4_train_bench.jl`, which says *how much* and not *where*. Kept because it is what
# found the `G[:, t, :]` pullback: the flat profile put `fill!` and `_setindex!` at the top of the
# self-time list, which is the signature of a slice whose adjoint zeroes a parent-sized array once
# per loop iteration, and `accum(::@NamedTuple{Wx, Wh, ...})` just below it, which is the
# signature of reading a field of `ps` inside the loop. Both are invisible in the forward pass and
# neither shows up as a wrong number -- only as time.
#
# Read the output by the `Overhead` column, which is self time. A healthy profile here is
# dominated by `gemm!` and the broadcast kernels, not by allocation.

using RikFlow
using Lux, Optimisers, Zygote
using Random, LinearAlgebra, Statistics, Printf, Profile

const RF = RikFlow

const T = Float32
const L, B, BURN = 200, 8, 50

spec = RF.LSTMSpec(; hist = RF.HistorySpec(; h = 2, n_qoi = 6),
                   n_hidden = 16, n_latent = 4, n_encoder = 8, arch = :vrnn)

rng = Xoshiro(1)
Xb = randn(rng, T, RF.n_input(spec), L, B)
Yb = randn(rng, T, RF.n_output(spec), L, B)
eb = randn(rng, T, spec.n_latent, L, B)
score = (BURN + 1):L
ps = RF.init_lstm_params(Xoshiro(2), spec; T)

step() = Zygote.withgradient(p -> RF.elbo(spec, p, Xb, Yb, score, eb; beta = 1e-4), ps)

step()   # warm up / compile

Profile.clear()
Profile.init(; n = 10^7, delay = 0.0005)
@profile for _ in 1:20
    step()
end

println("=== flat profile, sorted by self time ===")
Profile.print(; format = :flat, sortedby = :overhead, mincount = 40, maxdepth = 100)
