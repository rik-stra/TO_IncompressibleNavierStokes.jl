"""
    RikFlowLuxExt

M4's training side. Triggered by `Lux`, `Optimisers` and `Zygote` together.

🔑 **This is the only part of M4 that needs Lux, and that is deliberate.** What runs inside the
solver is `lstm_step!` in `src/ts_lstm.jl`, on plain arrays, with no Lux anywhere -- see that
file's header for the four reasons. This extension fits the weights; `LSTMWeights(ps, spec)` at
the bottom converts them into the form the solver uses.

⚠️ **Parameters are a plain NamedTuple of arrays rather than a `Lux.AbstractLuxLayer`.** Lux's
parameters *are* NamedTuples of arrays, so this is not un-Lux-like, but the layer API is skipped on
purpose: M4 is a single bespoke recurrence with a bespoke objective, so `Chain`/`Dense` would buy
nothing, while a 1:1 correspondence between the trained NamedTuple and `LSTMWeights` buys a great
deal -- the conversion is a copy, and V41 can compare the two forward passes field by field
instead of through a translation layer that could itself be wrong. Lux supplies the initialisers.

# The covariance is parametrised differently here than it is deployed

Training carries a free lower-triangular **precision** factor `A`:

    Sigma^{-1} = D^{-1} A' A D^{-1}      quad = || A (r ./ d) ||^2
    log det Sigma = 2 sum(log d) - 2 sum(log diag(A))

which is unconstrained, needs no triangular solve, and so differentiates cleanly. The deployed form
is the one `methods_overview.tex` specifies, `Sigma = D R D` with `R` a **correlation** matrix.
These are the same family: `A` leaves `R`'s scale confounded with `d`, by a per-coordinate constant
only, and `LSTMWeights` resolves it by normalising `R` to unit diagonal and folding the scale into
`bd`. The arithmetic is in `LSTMWeights(ps, spec)` and V41 is what says the conversion is right.
"""
module RikFlowLuxExt

using RikFlow
using Lux
using Optimisers
using Zygote
using Random
using LinearAlgebra
using Statistics
# ⚠️ Only for `m4_device`'s `CuArray` and its `functional()` check -- no other line in this file
# mentions a device, by design (see "Device placement"). `CUDA` is a hard dependency of `RikFlow`,
# so this costs nothing that was not already loaded, and it loads on a machine with no GPU.
using CUDA

const RF = RikFlow

# Elementwise helpers, written locally so nothing depends on which of Lux/NNlib exported what.
_sig(x) = one(x) / (one(x) + exp(-x))
_sp(x) = x > zero(x) ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    _precision_factor(Araw)

Lower-triangular `A` with a positive diagonal, from an unconstrained square matrix: the strict
lower triangle of `Araw`, plus `exp` of its diagonal.

⚠️ **Built from `LowerTriangular`/`Diagonal` rather than from a boolean mask, and that is not a
style choice.** The obvious `Araw .* [i > j for i in 1:n, j in 1:n]` puts an array comprehension
inside the objective, and a comprehension fills its array by `setindex!`, so Zygote refuses the
whole gradient with *"Mutating arrays is not supported"* — pointing at the mask rather than at
anything the model does. Wrapping the mask in `ignore_derivatives` does **not** help either: the
argument is evaluated, and therefore traced, before `ignore_derivatives` ever sees it. These three
operations all have ChainRules rules and sidestep the problem entirely.
"""
function _precision_factor(Araw::AbstractMatrix)
    d = diag(Araw)
    return LowerTriangular(Araw) - Diagonal(d) + Diagonal(exp.(d))
end

# ---------------------------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------------------------

"""
    init_lstm_params(rng, spec; T = Float32)

Initial parameters, as a NamedTuple whose fields are exactly [`RikFlow.LSTMWeights`](@ref)'s.

Two initialisations are not arbitrary and are worth keeping:

- 🔑 **The forget-gate bias is set to 1.** Without it the forget gate starts at `sigmoid(0) = 0.5`
  and the cell halves its memory every step, so gradients over a 200-step segment vanish before
  training has a chance to learn the timescale. Standard, and load-bearing at our segment lengths.
- 🔑 **`Bsig` starts at zero**, matching the source's `kernel_initializer='zeros'`, so every latent
  scale starts at `softplus(0) = log 2` rather than somewhere the KL immediately punishes.
"""
function RF.init_lstm_params(rng::AbstractRNG, spec::RF.LSTMSpec; T::Type = Float32)
    H = spec.n_hidden
    nin, nout, nz = RF.n_input(spec), RF.n_output(spec), spec.n_latent
    ncin, nenc = RF.n_cell_input(spec), RF.n_encoder_out(spec)
    has_latent = RF.latent_sampled(spec)

    b = zeros(T, 4H)
    b[(H + 1):(2H)] .= one(T)              # forget-gate bias

    return (;
        Wx = T.(Lux.glorot_uniform(rng, 4H, ncin)),
        Wh = T.(Lux.glorot_uniform(rng, 4H, H)),
        b,
        We = (has_latent && spec.n_encoder > 0) ? T.(Lux.glorot_uniform(rng, nenc, nin)) : nothing,
        be = (has_latent && spec.n_encoder > 0) ? zeros(T, nenc) : nothing,
        Bmu = has_latent ? T.(Lux.glorot_uniform(rng, nz, nenc)) : zeros(T, nz, nenc),
        Bsig = zeros(T, nz, nenc),         # softplus(0) = log 2
        V1 = T.(Lux.glorot_uniform(rng, nout, H)),
        V2 = RF.latent_to_decoder(spec) ? T.(Lux.glorot_uniform(rng, nout, nz)) : nothing,
        cdec = zeros(T, nout),
        Wd = zeros(T, nout, H),            # start homoscedastic: the scale head learns from flat
        bd = zeros(T, nout),
        Araw = Matrix{T}(I, nout, nout) .* zero(T),   # A = I at init, i.e. R = I
    )
end

# ---------------------------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------------------------
#
# 🔑 **There is no CUDA in this file, and that is the design.** `lstm_forward` allocates its
# recurrent state and its hidden-state buffer with `similar(X, ...)`, so every array it makes
# follows the array type it was *given*. Device choice therefore lives entirely at the call site:
# `train_stochlstm(...; device = CuArray)` moves the parameters and each batch across and the
# arithmetic follows, with no device-specific branch anywhere in the model.
#
# `device` is any `Array -> AbstractArray` function. `identity` is the CPU path and the default.
#
# ⚠️ **What this does NOT do is make a GPU worth using.** `results_LSTMS.md` §5 measures the fit at
# ~75 MFLOP and ~0.34 GFLOP/s per gradient step over a recurrence whose `L` steps are strictly
# sequential, i.e. overhead-bound on the host side, where a device cannot help. This plumbing
# exists so the question can be *measured* rather than argued.

_to_device(device, x::AbstractArray) = device(x)
_to_device(device, ::Nothing) = nothing
_to_device(device, nt::NamedTuple) = map(v -> _to_device(device, v), nt)

"Bring a parameter set back to the host. `Array` on a host array is a no-op."
_to_host(x) = _to_device(Array, x)

"""
    m4_device(name) -> f

Resolve `"cpu"` / `"cuda"` into the function `train_stochlstm`'s `device` keyword wants.

🔴 **`"cuda"` throws when no device is functional rather than falling back to the host.** A silent
fallback is the worst outcome available here: the job would report a GPU run, take CPU time, and
put a number in a table that says something it does not mean.

⚠️ `CUDA` is a hard dependency of `RikFlow`, so `using CUDA` costs nothing extra here and works on
a machine with no GPU — loading it without a device is supported; only *using* one is not.
"""
function RF.m4_device(name::AbstractString)
    d = lowercase(strip(name))
    d in ("cpu", "host") && return identity
    if d in ("cuda", "gpu")
        CUDA.functional() || error(
            "m4_device: M4_DEVICE=$name was requested but CUDA.functional() is false -- no " *
            "usable device. Refusing to fall back to the CPU silently; either run on a node " *
            "with a GPU or set M4_DEVICE=cpu.")
        return CUDA.CuArray
    end
    return error("m4_device: expected \"cpu\" or \"cuda\"; got \"$name\"")
end

"""
    _timeslices(G, L)

Split a `4H x L x B` gate-preactivation array into `L` matrices of `4H x B`, one per time step.

🔴 **This exists only for its adjoint, and the slice it replaces was the single largest cost in
the reverse pass.** Writing `G[:, t, :]` inside the recurrence looks free -- one small strided
copy -- but Zygote's pullback for a slice allocates a *parent-sized* zero array and scatters the
cotangent into it. Inside an `L`-step loop that is `L` allocations of `4H x L x B` and `L` full
zero-fills, so the reverse pass is O(L^2) in a model whose forward pass is O(L). Profiling one
`L = 200, B = 8, H = 16` gradient step put `fill!` and `setindex!` at the top of the self-time
list, and the step allocated 172 MiB for a forward pass that allocates 3.8 MiB.

Slicing once, through a rule that accumulates every step's cotangent into **one** buffer, makes
the reverse pass O(L) again. Measured on that geometry: 246 ms -> 49 ms, 172 MiB -> 18 MiB, and
the reverse/forward ratio from 26x down to 5.8x, which is what reverse mode should cost. Nothing
about the model changes -- this is the same arithmetic with a cheaper pullback, and V50 is what
says so.
"""
_timeslices(G::AbstractArray{<:Any,3}, L::Integer) = [G[:, t, :] for t in 1:L]

Zygote.@adjoint function _timeslices(G::AbstractArray{<:Any,3}, L::Integer)
    return _timeslices(G, L), function (Δ)
        dG = zero(G)
        for t in 1:L
            Δt = Δ[t]
            Δt === nothing && continue
            @views dG[:, t, :] .+= Δt
        end
        return (dG, nothing)
    end
end

# ---------------------------------------------------------------------------------------------
# Architecture branches, in the type domain
# ---------------------------------------------------------------------------------------------
#
# See the note inside `lstm_forward`. Each of these is a branch that Zygote must NOT compile into
# a single pullback, so each is a pair of methods on `Val` rather than a conditional.

# the encoder: the repository's dense tanh layer, or the tex's linear form
_encode(::Val{true}, ps, X) = tanh.(ps.We * X .+ ps.be)
_encode(::Val{false}, ps, X) = X

# the cell input: with or without z concatenated on (STORN/VRNN vs the rest)
_cell_input(::Val{true}, X, Z) = vcat(X, Z)
_cell_input(::Val{false}, X, Z) = X

# the V2 decoder skip
_decoder_skip(::Val{true}, Y, ps, Z) = Y .+ ps.V2 * Z
_decoder_skip(::Val{false}, Y, ps, Z) = Y

# the emission log-scale head: state-dependent, or `bd` alone with `Wd` cut out of the graph
_logd(::Val{false}, ps, Hm2) = ps.Wd * Hm2 .+ ps.bd
_logd(::Val{true}, ps, Hm2) = ps.bd .+ zero(eltype(Hm2)) .* sum(Hm2; dims = 1)

# the whole latent path, or a constant stand-in for the deterministic architecture
function _latent(::Val{true}, encv, ps, X, epsz, ::Type{T}, nz, L) where {T}
    E = _encode(encv, ps, X)
    MU = ps.Bmu * E
    SIG = _sp.(ps.Bsig * E)
    return MU, SIG, MU .+ SIG .* epsz
end
# 🔴 Allocated with `zero(similar(X, ...))`, NOT `zeros(T, ...)`, so the constants follow the
# INPUT'S array type instead of being pinned to the host. That is what lets the same code run on a
# device (see `train_stochlstm`'s `device` keyword); `zeros` here would silently mix a host array
# into a device graph. ⚠️ `zero(similar(...))` and not `fill!(similar(...), 0)`: the latter is a
# `setindex!` on an array Zygote can see and is refused with *"Mutating arrays is not supported"* --
# measured 2026-09-18, not assumed.
# 🔑 For `:lstm` these three are dead downstream -- `_cell_input`/`_decoder_skip` on `Val{false}`
# ignore `Z`, and `elbo` reads `MU`/`SIG` only when `latent_sampled` -- so this is about not
# leaving a host array inside a device NamedTuple, not about the arithmetic.
function _latent(::Val{false}, encv, ps, X, epsz, ::Type{T}, nz, L) where {T}
    mu = zero(similar(X, T, nz, L))
    z = zero(similar(X, T, nz, L))
    return (mu, zero(similar(X, T, nz, L)) .+ one(T), z)
end

# ---------------------------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------------------------

"""
    lstm_forward(spec, ps, X, epsz)

Forward pass over one segment. `X` is `n_input x L`, columns in increasing time; `epsz` is
`n_latent x L` of standard normals, supplied by the caller so the reparametrisation is explicit
and the function itself is deterministic.

Returns `(; Y, LOGD, MU, SIG, Hm)`.

🔑 Only the recurrence is a loop. The encoder, the latent draw, the decoder and the log-scale head
are all applied to the whole segment as matrix products, which is both faster and far less for
Zygote to get wrong.
"""
function RF.lstm_forward(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractArray{T,3},
                         epsz::AbstractArray) where {T}
    H = spec.n_hidden
    nin, L, B = size(X)
    nz, nout = spec.n_latent, RF.n_output(spec)

    # 🔴 The architecture branches are DISPATCHED, not written as `?:` or `if`, and this is not
    # cosmetic. Two of them change the shape of what they return -- the encoder gives `n_encoder x
    # L` or `n_input x L`, the cell input gives `n_cell_input x L` or `n_input x L` -- and Zygote
    # compiles ONE pullback for a conditional and then accumulates cotangents across both arms. The
    # symptom is a `DimensionMismatch` between two of the model's own dimensions ("axes OneTo(5)
    # and OneTo(16)") raised from inside Zygote, with nothing wrong at all in the forward pass.
    # Every piece of this function differentiates correctly on its own; only the conditionals do
    # not. `Val` puts the branch in the type domain, so only the taken arm is ever traced.
    encv = Val(spec.n_encoder > 0)
    hasz = Val(RF.latent_sampled(spec))
    cellv = Val(RF.latent_to_cell(spec))
    decv = Val(RF.latent_to_decoder(spec))

    # 🔑 **Everything that is pointwise in time is done on the flattened (L*B) axis**, so the
    # encoder, the latent draw, the decoder and the log-scale head are four matrix products for
    # the whole batch rather than 4*L*B small ones. Only the recurrence has to be a loop.
    X2 = reshape(X, nin, L * B)
    MU2, SIG2, Z2 = _latent(hasz, encv, ps, X2, reshape(epsz, nz, L * B), T, nz, L * B)

    # --- recurrence, batched over segments ----------------------------------------------------
    #
    # 🔑 `h` and `c` are `H x B`, not `H`. This is the single biggest cost decision in the file.
    # Running a chunk of `B` segments as `B` separate loops traces `B * L` recurrent steps through
    # Zygote; running them as one loop over matrices traces `L`, for identical arithmetic, and
    # turns every GEMV into a GEMM. Measured on R1's record at B = 8: 3.42 s/epoch -> see
    # `analysis/results_LSTMS.md` §4. Zygote's per-operation overhead dominates the actual flops
    # at this model size, so the win is close to the full factor of B.
    XZ2 = _cell_input(cellv, X2, Z2)
    # The bias is folded in here rather than inside the loop: it is the same vector at every step,
    # so adding it once to the whole batch is one broadcast instead of `L` of them.
    GX = _timeslices(reshape(ps.Wx * XZ2 .+ ps.b, 4H, L, B), L)

    # 🔑 `Wh` is pulled out of `ps` **before** the loop, and that is a performance fix, not tidying.
    # Reading `ps.Wh` inside the loop makes every iteration's pullback a `getfield` on the whole
    # parameter NamedTuple, so Zygote builds a full params-shaped tangent and `accum`s it `L`
    # times; profiling showed that one line as the second-largest cost in the reverse pass, after
    # the slice `_timeslices` replaces. Bound to a local, the per-step cotangent is just `H x H`.
    Wh = ps.Wh

    # 🔴 `similar(X, ...)` / `zero(similar(X, ...))`, not `zeros(T, ...)`: the recurrent state and
    # the hidden-state buffer follow the INPUT'S array type, which is the whole mechanism by which
    # this function runs on a device without a single device-specific line in it. The `Buffer`'s
    # backing array is left uninitialised on purpose -- every one of its `L` slices is written in
    # the loop below -- and `Zygote.Buffer` is what makes that mutation legal at all.
    Hb = Zygote.Buffer(similar(X, T, H, L, B))
    h = zero(similar(X, T, H, B))
    c = zero(similar(X, T, H, B))
    for t in 1:L
        g = GX[t] .+ Wh * h
        # Views, not slices: each of these used to copy an `H x B` block, and each copy is an
        # operation Zygote traces and a cotangent it has to allocate; a view's pullback writes
        # straight into the parent's.
        gi = @view g[1:H, :]
        gf = @view g[(H + 1):(2H), :]
        gc = @view g[(2H + 1):(3H), :]
        go = @view g[(3H + 1):(4H), :]
        # 🔴 **The four gate activations are FUSED INTO THE TWO STATE UPDATES, and that is a
        # kernel-count fix, not a style change.** Written out as `i = _sig.(gi); f = _sig.(gf);
        # gg = tanh.(gc); o = _sig.(go); c = f .* c .+ i .* gg; h = o .* tanh.(c)` this is SIX
        # broadcasts per timestep; as two `@.` expressions with the activations inside, it is TWO,
        # for identical arithmetic on identical values in the same order.
        # 🔑 Why it matters more than it looks: the recurrence is the one part of the model that
        # cannot be batched over time, so its cost is `L` x (ops per step) and nothing amortises
        # it. Each broadcast is a separate kernel launch on a device and a separate traced
        # operation with its own pullback under Zygote, so removing four removes four launches AND
        # four pullbacks per step -- 2000 of each over an `L = 500` segment. The GPU is
        # launch-bound here (`results_LSTMS.md` §5: ~17 000-20 000 dependent launches per update
        # at ~25 us each), which is the setting this was measured against.
        # ✅ **Bit-identical, verified rather than assumed**: the validation curves and summed
        # weights of all four architectures are unchanged to the last digit (2026-09-18), on a
        # deterministic synthetic fit run before and immediately after the change.
        # ⚠️ **It buys nothing measurable on the CPU, and the claim that it did was not
        # reproduced.** Measured here at the real scan geometry, pre against post:
        # 0.308/0.304, 0.498/0.458, 0.399/0.502, 0.842/0.829, 2.003/1.967 s/epoch -- scatter
        # around 1.0 with one point worse, and 58 vs 59 min over the whole scan. It is kept
        # because it is free and strictly fewer launches and pullbacks, which is the axis a
        # LAUNCH-BOUND device is limited by (§5). 🔴 **That GPU benefit is predicted, not
        # measured** -- forward kernels per step go 9 -> 5; whether that shows up needs a
        # `M4_DEVICE=cuda` smoke on Snellius.
        c = @. _sig(gf) * c + _sig(gi) * tanh(gc)
        h = @. _sig(go) * tanh(c)
        Hb[:, t, :] = h
    end
    Hm = copy(Hb)
    Hm2 = reshape(Hm, H, L * B)

    # --- decoder and log-scale head -----------------------------------------------------------
    Y2 = _decoder_skip(decv, ps.V1 * Hm2 .+ ps.cdec, ps, Z2)
    # 🔴 `:constant` MEANS constant: the `Wd` path contributes zero value AND zero gradient, so `Wd`
    # stays at its zero initialisation and the deployed `lstm_step!` (which computes `Wd * h + bd`
    # for every emission mode) sees exactly `bd`. Until 2026-09-23 nothing enforced this -- `Wd`
    # was trained under `:constant` too, so a "constant" head was state-dependent in fact: an
    # `:constant` and a `:state_dependent` fit from one seed came out bit-identical.
    LOGD2 = _logd(Val(spec.emission === :constant), ps, Hm2)
    if spec.uclip !== nothing
        LOGD2 = clamp.(LOGD2, T(spec.uclip[1]), T(spec.uclip[2]))
    end

    return (; Y = reshape(Y2, nout, L, B), LOGD = reshape(LOGD2, nout, L, B),
            MU = reshape(MU2, nz, L, B), SIG = reshape(SIG2, nz, L, B), Hm)
end

"""
    lstm_forward(spec, ps, X::AbstractMatrix, epsz)

Single-segment form, kept so `iwae_nll` and V41 read the way they did. It is the batched method
with `B = 1`, so there is only ever one implementation of the recurrence to get wrong.
"""
function RF.lstm_forward(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractMatrix{T},
                         epsz::AbstractMatrix) where {T}
    nin, L = size(X)
    o = RF.lstm_forward(spec, ps, reshape(X, nin, L, 1), reshape(epsz, spec.n_latent, L, 1))
    return (; Y = reshape(o.Y, :, L), LOGD = reshape(o.LOGD, :, L),
            MU = reshape(o.MU, :, L), SIG = reshape(o.SIG, :, L), Hm = reshape(o.Hm, :, L))
end

# ---------------------------------------------------------------------------------------------
# Rollout forward: the level lags fed from the model's own outputs
# ---------------------------------------------------------------------------------------------

"""
    _rollout_mask(L, burn, K) -> Vector{Bool}

Which steps of an `L`-step segment feed their OWN output back as the next steps' level lag.
`false` for the `burn` warm-up steps (the recorded level is fed, as the deployed closure's
`nwarm` replay does) and at every `K`-th step after it, where the lag is re-anchored to the
recorded level. `K = 1` is teacher forcing everywhere; `K >= L - burn` is one free run of
`L - burn` steps after the warm-up -- deployment's shape.
"""
_rollout_mask(L::Integer, burn::Integer, K::Integer) =
    [s > burn && (s - burn) % K != 0 for s in 1:L]

"Replace rows `r` of `x` by `v`, without mutating (Zygote-safe)."
_swap_rows(x, r::UnitRange, v) = vcat(x[1:(first(r) - 1), :], v, x[(last(r) + 1):end, :])

"""
    _rollout_forward(spec, ps, X, epsz, free)

`lstm_forward` with the level-lag columns of each input taken from the model's own output
wherever `free` says so. Same return value.

🔑 **What is replaced and what is not -- the regime-B surrogate of the online loop.** The deployed
closure pushes `q* + dQ = qhat` into its history, so online the LEVEL lags are the model's own
outputs while the predictor `q*` comes from the solver. Here the lag `k` rows of `x_t`
(`qlag_columns(spec.hist, k)`) are the model's `y_{t-k}` when step `t-k` is free, and the recorded
level otherwise; every predictor column, and the bias, stays recorded. The solver's response to the
model's corrections is what this cannot reproduce -- `q*` is replayed, not re-simulated.

⚠️ **Everything is per step**, unlike `lstm_forward`, which projects the whole segment at once:
`x_t` now depends on `y_{t-1}`, so the encoder, the latent draw (`z_t` reads `x_t`), the input
projection and the decoder all move inside the loop. Gradients flow through the feedback, which is
the point -- and why a long free run can explode (Rik, 2026-09-23): shorten `L` or `K`, or clip.
🔴 `emission = :none` only -- the feedback is then the decoder mean, exactly what the deployed
closure pushes. With an emission head it would push a DRAW, which this does not model.
"""
function _rollout_forward(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractArray{T,3},
                          epsz::AbstractArray, free::AbstractVector{Bool};
                          lagmap = nothing) where {T}
    RF.emission_noise(spec) && error("_rollout_forward: emission = :none only -- the deployed " *
                                     "closure feeds back a DRAW when there is an emission head")
    H = spec.n_hidden
    nin, L, B = size(X)
    length(free) == L || error("_rollout_forward: free has $(length(free)) entries, segment $L")
    nz, nout, hh = spec.n_latent, RF.n_output(spec), spec.hist.h
    encv, hasz = Val(spec.n_encoder > 0), Val(RF.latent_sampled(spec))
    cellv, decv = Val(RF.latent_to_cell(spec)), Val(RF.latent_to_decoder(spec))
    lagrows = [RF.qlag_columns(spec.hist, k) for k in 1:hh]
    # 🔑 `lagmap` turns the model's output into the LEVEL lag it deploys. `nothing` is the level
    # target: the output IS the scaled level. For the `:dQ` target it is `(; a, b)` and the lag is
    # `qstar_lag + a .* y .+ b` -- the replayed predictor lag plus the model's correction, in the
    # input scaling (`a = sigma_out ./ sigma_in`, `b = mu_out ./ sigma_in`), which is exactly
    # `scale_in(q* + dQ)`, what the deployed closure pushes.
    qsrows = lagmap === nothing ? nothing : [RF.qstarlag_columns(spec.hist, k) for k in 1:hh]
    lagmap === nothing || !any(isnothing, qsrows) ||
        error("_rollout_forward: a :dQ-target rollout needs the predictor history (hist_var = :q_star_q)")
    lagval(k, xt, y) = lagmap === nothing ? y : xt[qsrows[k], :] .+ lagmap.a .* y .+ lagmap.b
    any(isnothing, lagrows) && error("_rollout_forward: the history carries no level lags " *
                                     "(hist_var = $(spec.hist.hist_var)) -- nothing to feed back")

    Xs = _timeslices(X, L)
    Es = _timeslices(epsz, L)
    Wh = ps.Wh
    Yb = Zygote.Buffer(similar(X, T, nout, L, B))
    MUb = Zygote.Buffer(similar(X, T, nz, L, B))
    SIGb = Zygote.Buffer(similar(X, T, nz, L, B))
    Hb = Zygote.Buffer(similar(X, T, H, L, B))
    h = zero(similar(X, T, H, B))
    c = zero(similar(X, T, H, B))
    ys = ntuple(_ -> zero(similar(X, T, nout, B)), hh)   # ys[k] = y_{t-k}
    for t in 1:L
        xt = Xs[t]
        for k in 1:hh
            s = t - k
            (s >= 1 && free[s]) && (xt = _swap_rows(xt, lagrows[k], lagval(k, xt, ys[k])))
        end
        MU, SIG, Z = _latent(hasz, encv, ps, xt, Es[t], T, nz, B)
        g = ps.Wx * _cell_input(cellv, xt, Z) .+ ps.b .+ Wh * h
        gi = @view g[1:H, :]
        gf = @view g[(H + 1):(2H), :]
        gc = @view g[(2H + 1):(3H), :]
        go = @view g[(3H + 1):(4H), :]
        c = @. _sig(gf) * c + _sig(gi) * tanh(gc)
        h = @. _sig(go) * tanh(c)
        y = _decoder_skip(decv, ps.V1 * h .+ ps.cdec, ps, Z)
        Yb[:, t, :] = y
        MUb[:, t, :] = MU
        SIGb[:, t, :] = SIG
        Hb[:, t, :] = h
        ys = hh == 1 ? (y,) : (y, ys[1:(end - 1)]...)
    end
    Hm = copy(Hb)
    LOGD2 = _logd(Val(spec.emission === :constant), ps, reshape(Hm, H, L * B))
    if spec.uclip !== nothing
        LOGD2 = clamp.(LOGD2, T(spec.uclip[1]), T(spec.uclip[2]))
    end
    return (; Y = copy(Yb), LOGD = reshape(LOGD2, nout, L, B), MU = copy(MUb),
            SIG = copy(SIGb), Hm)
end

# ---------------------------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------------------------

"""
    elbo(spec, ps, X, Ytrue, score, epsz; beta = 1e-4)

Negative ELBO on one segment, averaged over its scored columns.

`score` is the sub-range `segment_indices` marks as scorable -- the burn-in columns drive the
recurrence and contribute nothing, which is the whole reason they exist.

⚠️ `beta` is the KL weight. The source calls it `lambda` and uses `1e-4`; **`lambda` is the ridge
parameter in this project and the two must never be conflated.**
"""
function RF.elbo(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractArray{T,3}, Ytrue::AbstractArray,
                 score::AbstractUnitRange, epsz::AbstractArray; beta::Real = 1e-4,
                 free::Union{Nothing,AbstractVector{Bool}} = nothing, lagmap = nothing) where {T}
    # `free === nothing` is teacher forcing, the path every existing fit was made on, unchanged.
    # Otherwise the level lags are fed from the model's own outputs -- see `_rollout_forward`.
    out = free === nothing ? RF.lstm_forward(spec, ps, X, epsz) :
          _rollout_forward(spec, ps, X, epsz, free; lagmap)
    nout = RF.n_output(spec)
    B = size(X, 3)
    ns = length(score) * B            # scored steps summed over the batch

    # 🔴 `:none` is a deterministic decoder, so there is no density to evaluate and the
    # reconstruction term is a plain sum of squares -- the source's objective. `Wd`, `bd` and
    # `Araw` never enter the loss, so their gradients are zero and they stay at their zero
    # initialisation; nothing needs to be frozen by hand.
    recon = if RF.emission_noise(spec)
        U = out.LOGD[:, score, :]
        Rres = (Ytrue[:, score, :] .- out.Y[:, score, :]) ./ exp.(U)
        A = _precision_factor(ps.Araw)
        quad = sum(abs2, A * reshape(Rres, nout, :))
        logdet_term = 2 * sum(U) - 2 * ns * sum(log.(diag(A)))
        T(0.5) * (nout * ns * T(log(2 * pi)) + logdet_term + quad)
    else
        T(0.5) * sum(abs2, Ytrue[:, score, :] .- out.Y[:, score, :])
    end
    nll = recon

    kl = if RF.latent_sampled(spec)
        S = out.SIG[:, score, :]
        M = out.MU[:, score, :]
        sum(T(0.5) .* (S .^ 2 .+ M .^ 2 .- one(T)) .- log.(S))
    else
        zero(T)
    end

    # 🔑 Normalised **per scored step**, so the value does not depend on how many segments were
    # batched together. A chunk run as one batch of 8 and the same chunk run as 8 batches of 1 and
    # averaged give the same number, which is what lets the batching be a pure speed change.
    return (nll + T(beta) * kl) / ns
end

"""
    elbo(spec, ps, X::AbstractMatrix, Ytrue, score, epsz; beta)

Single-segment form: the batched method at `B = 1`.
"""
function RF.elbo(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractMatrix{T}, Ytrue::AbstractMatrix,
                 score::AbstractUnitRange, epsz::AbstractMatrix; beta::Real = 1e-4) where {T}
    nin, L = size(X)
    return RF.elbo(spec, ps, reshape(X, nin, L, 1), reshape(Ytrue, :, L, 1), score,
                   reshape(epsz, spec.n_latent, L, 1); beta)
end

# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

"""
    _plateau_step(pl, vl, patience) -> (pl, decay::Bool)

One validation's worth of the decay-on-plateau rule. `pl = (; ref, since)` is the best validation
loss seen **since the last decay** and the number of validations since it improved; `decay` says
`since` has reached `patience`. After acting on a decay the caller resets `pl` to
`_PLATEAU_FRESH`, so the first validation after it becomes the new reference.

🔴 **Why the reference resets at a decay instead of being the all-time best.** Measured on the
2026-09-23 stride scan: the `(400, batch = 2)` control spiked from a best of 9.2e-3 to 3.6e-2 at
update 388. Against the all-time best, each decay then had `patience` validations to get back
below the PRE-spike value at a LOWER rate, never did, and fired again -- at 388, 428, 468, 508,
548 -- taking the rate to `min_lr` in 160 updates and ending the fit at 586, at 15x the loss of
its neighbours. `stride = 20` did the same at 3276-3396 while still descending ~28% per quarter.
Against the best since the decay, a recovering curve keeps setting new references and the rate
drops again only once it has genuinely stopped improving at the current rate.

⚠️ This rule only paces the SCHEDULE. Best-iterate selection and the early stop both keep reading
the all-time best, so a spike can never be returned as the fit and cannot hold off the stop.
Kept pure so the cascade is testable without a fit (V56).
"""
function _plateau_step(pl::NamedTuple, vl::Real, patience::Integer)
    pl = vl < pl.ref ? (; ref = float(vl), since = 0) : (; ref = pl.ref, since = pl.since + 1)
    return pl, pl.since >= patience
end

const _PLATEAU_FRESH = (; ref = Inf, since = 0)

"""
    _window_stalled(upd_axis, bsf, window, rel) -> Bool

Whether the best-so-far validation loss `bsf` (one entry per validation, at the updates in
`upd_axis`) has improved by less than a fraction `rel` over the last `window` updates. `false`
until the history spans `window` updates, and always `false` when `window <= 0` or `rel <= 0`.

🔴 **The gain is relative to `abs` of the earlier value**, because the `:lstm` control's validation
loss is a Gaussian log-density and is NEGATIVE (~ -18): `(old - new) / old` flips sign there, reads
every improvement as a loss, and would stop that fit at its first check.

🔑 Why a window at all (Rik, 2026-09-23, `(400, b2)` rerun, fig14): at `min_lr` the stop keyed on
"no new best for `stop_patience` validations" never fires on a curve that keeps setting tiny new
bests -- that run did 30 000 updates. A windowed gain is a statement about the RATE, which is what
convergence means. Default 0.5% over 500 updates; on the recorded full-batch curves it never fires,
because they are still steep.
"""
function _window_stalled(upd_axis::AbstractVector, bsf::AbstractVector, window::Integer,
                         rel::Real)
    (window <= 0 || rel <= 0 || isempty(bsf)) && return false
    j = searchsortedlast(upd_axis, upd_axis[end] - window)
    j >= 1 || return false
    return (bsf[j] - bsf[end]) / abs(bsf[j]) < rel
end

"Global L2 norm of a gradient NamedTuple (fields that carry no gradient are `nothing`)."
_gnorm(g) = sqrt(sum((x === nothing ? 0.0 : Float64(sum(abs2, x)) for x in values(g)); init = 0.0))

"""
    train_stochlstm(spec, X, Y, steps; kwargs...)

Fit an M4 model to a regressor/target pair.

# Arguments
- `X`: `n_input x N`, the standardised regressor rows **as columns**, increasing in time. This is
  `permutedims` of what `build_history` returns -- the batch builder is row-major because the
  linear cells solve a least-squares system with it, and the recurrence wants time last.
- `Y`: `N_Q x N`, the standardised targets, same ordering.
- `steps`: the physical step index of each column, as `build_history` returns it. Discontinuities
  in it become segment boundaries.

# Keywords
- `L`, `burn`: segment length and burn-in. `L = 500`, `burn = 100` by default -- the configuration
  table's values, kept in step with it so a bare call and a driver call mean the same thing.
- `stride`: distance between the starts of consecutive TRAINING segments. The default `L - burn`
  makes the scored windows exactly tile the record, so every row is scored once per epoch. A
  shorter stride overlaps them, which is **augmentation, not data**: the same rows are re-scored at
  different offsets within a segment. 🔑 It buys two real things -- more optimiser steps per epoch
  (at `L = 500` the default leaves 6 full-length training segments, i.e. one full-batch update per
  epoch, a seventh short one -- the OLDEST rows, since training tiles from the end -- being
  dropped) and
  variation in how long the hidden state has been charged when a given row is scored. ⚠️ The
  gradients from overlapping segments are correlated, so `k x` the segments is not `k x` the
  information; compare runs on optimiser steps, not epochs.
- `epochs`, `batch`, `lr`, `beta`, `seed`. ⚠️ `epochs` is a CAP, not a budget: `stop_patience`
  ends the fit once it has converged, so a point that converges early does not burn the rest.
- `batch`: 🔴 **minibatches are a uniform random sample of the training segments and the final
  partial batch is DROPPED**, so every update covers exactly the same number of segments and an
  update count is comparable across strides. ⚠️ The effective size is `min(batch, nseg)`: at the
  tiling stride there are 6 segments and a batch of 32 can never be filled, and dropping every
  partial batch literally would leave that point with no updates at all.
- `val_every`: 🔴 **optimiser updates between validations, and the unit everything else is
  paced in.** Validation used to run once per epoch, and an epoch is a different amount of
  training at every stride -- 1 update at the tiling stride, 3 at `stride = 20` -- so the plateau
  rule and the early stop were paced differently at every point, which is exactly the confound
  the matched-update budget was supposed to remove. On a fixed update cadence they are not.
  Default 2, at which `patience = 20` is 40 updates.
- `patience`, `lr_decay`, `min_lr`: decay the learning rate by `lr_decay` after `patience`
  VALIDATIONS without an improvement **on the best seen since the last decay**, down to
  `min_lr`. 🔴 Not the all-time best: after a loss spike that reference is out of reach at the
  reduced rate, and the rule then fires every `patience` validations until `min_lr` -- the
  cascade that ended the 2026-09-23 `(400, b2)` control at update 586 (see `_plateau_step`).
  🔴 **`min_lr` is 1e-4, raised from 1e-5 (Rik, 2026-09-23).** The `(400, b2)` rerun reached
  1e-5 at update 2912 -- minibatch noise at batch 2 reads as a plateau -- and then improved 36%
  over the next 27 000 updates at that rate: the floor was too low to be a floor.
- `stop_patience`: 🔴 **early stopping, first rule: once the schedule has bottomed out.** The
  fit ends when the learning rate is at `min_lr` *and* `stop_patience` VALIDATIONS (i.e.
  `stop_patience * val_every` updates) have passed with no new best. 🔑 This is what lets points
  run to CONVERGENCE on their own schedules rather than to a shared budget -- the only comparison
  that is not confounded by whichever budget axis was matched.
- `stop_window`, `stop_rel`: 🔴 **early stopping, second rule: the best has improved by less than
  `stop_rel` (relative) over the last `stop_window` updates**, at any learning rate. Default 0.5%
  over 500 updates (Rik, 2026-09-23); `stop_window = 0` disables it. See `_window_stalled`.
  `history.stopped_early` records whether either rule fired and `history.stop_reason` which --
  `:floor`, `:window`, or `:cap` for a fit that ran to `epochs`.
- `rollout`: 🔴 **rollout training (Rik, 2026-09-23).** `1` (default) is teacher forcing, the path
  every earlier fit used and bit-for-bit unchanged. `K > 1` feeds the model's OWN output back into
  the level-lag columns after the `burn` warm-up, re-anchoring to the recorded level every `K`
  steps; `K >= L - burn` is one free run per segment, deployment's shape. The predictor `q*` stays
  recorded -- the regime-B surrogate of the online loop (`_rollout_forward`). Motivated by the
  online runs: teacher-forced fits settle into a spurious flat state and never reach the upper
  tail once their own output is in the loop (`results_LSTMS.md`, fig15/fig16). `emission = :none`
  only. Validation uses the same rollout.
- `init_ps`: warm-start parameters (e.g. the `ps` a fit file stores), shape-checked against
  `spec`. Rollout training is meant as a fine-tune of a teacher-forced fit.
- `clip`: clip the gradient's global norm to this before Adam; `0` (default) is off.
  `history.gmax` logs the largest norm seen between validations, so an exploding rollout
  gradient is visible rather than silent.

🔴 **Returns the best-validation iterate, not the last one**, and the validation epsilons are drawn
once and held fixed so the curve is a function of the parameters alone. Both matter more than they
look: on R1's record at a constant `lr`, the three latent architectures reached val ≈ −12 near
epoch 91 and were at ≈ −4 by epoch 100, so a last-iterate fit **inverted the architecture ranking**.
`history` carries `update` (the optimiser step each row was recorded at), `best_update`,
`best_index` and `best_val`, so a run that ends far from its best is visible and the curve can be
plotted against updates rather than against an epoch that means something different per point.
- `device_rng`: draw the reparametrisation noise on the device instead of on the host. `false` by
  default. 🔴 **It removes the last host->device transfer per update and costs reproducibility**:
  a different stream, so no fit made with it is comparable to any made without it, and on the CPU
  it bypasses the seeded `rng` entirely. For measuring what the transfer costs, not for fitting.
- `device`: any `Array -> AbstractArray` function placing the parameters and each batch.
  `identity` (the default) is the CPU path; `CuArray` runs the fit on a GPU. 🔑 **The randomness
  stays on the host**: parameters are initialised and `eps` drawn from `Xoshiro(seed)` and only
  then moved, so a device fit and a host fit at the same seed are the same fit to round-off. A
  device RNG would be faster and would void V49's reproducibility property and every number
  already recorded. The returned parameters are always on the host.
  ⚠️ **Works on a GPU and is ~7x slower than the same node's CPU** (2026-09-18) — see `results_LSTMS.md` §5.
- `val_frac`: the trailing fraction of ROWS held out for the reported validation loss -- rows, not
  segments, which is what keeps the scored windows disjoint at any stride (V53).
  ⚠️ This is an *inner* split for early stopping and diagnostics only. Model **selection** is on a
  window disjoint from both training and the online evaluation window -- that is the protocol's
  job, not this function's, and this keyword is not it.

Returns `(ps, history)`.
"""
function RF.train_stochlstm(spec::RF.LSTMSpec, X::AbstractMatrix, Y::AbstractMatrix,
                            steps::AbstractVector{<:Integer};
                            L::Int = 500, burn::Int = 100, epochs::Int = 200, batch::Int = 8,
                            lr::Real = 1e-2, beta::Real = 1e-4, seed::Int = 1,
                            val_frac::Real = 0.2, stride::Int = L - burn,
                            T::Type = Float32, verbose::Bool = true, device = identity,
                            device_rng::Bool = false,
                            patience::Int = 20, lr_decay::Real = 0.3, min_lr::Real = 1e-4,
                            stop_patience::Int = 100, val_every::Int = 2,
                            stop_window::Int = 500, stop_rel::Real = 0.005,
                            rollout::Int = 1, init_ps = nothing, clip::Real = 0,
                            lagmap = nothing)
    rollout >= 1 || error("train_stochlstm: rollout must be >= 1 (1 = teacher forcing); got $rollout")
    (rollout == 1 || !RF.emission_noise(spec)) || error(
        "train_stochlstm: rollout > 1 needs emission = :none -- with an emission head the deployed " *
        "closure feeds back a draw, which the rollout does not model")
    size(X, 2) == size(Y, 2) == length(steps) ||
        error("train_stochlstm: X, Y and steps disagree on the number of columns")
    1 <= stride <= L - burn ||
        error("train_stochlstm: need 1 <= stride <= L - burn = $(L - burn); got $stride. " *
              "A larger stride would leave rows no segment ever scores.")

    # 🔴 **The whole record is staged on the device ONCE, and batches are gathered THERE.** Until
    # 2026-09-18 every batch was assembled on the host and copied per optimiser update — 3 copies
    # each, ~0.2 MB, ~0.6 GB over a 3000-update point. The record itself is far smaller than the
    # batches cut from it (segments overlap, and each is re-copied every epoch), so uploading it
    # once and gathering device-side removes the per-update host traffic entirely for `X` and `Y`.
    # At the production geometry that is 273 kB for `Xd` and 86 kB for `Yd`, uploaded twice, total.
    # ⚠️ **Do not expect this to be fast-er, only clean-er.** The measurement says the loop is
    # launch-bound, not bandwidth-bound: the traffic it removes ran at under 0.1 MB/s on a bus
    # doing tens of GB/s (`results_LSTMS.md` §5). It removes a confound and a per-update host
    # allocation; it does not address the ~17 000-20 000 dependent kernel launches that dominate.
    Xt, Yt = T.(X), T.(Y)
    Xd, Yd = device(Xt), device(Yt)

    # 🔴 **The record is split by ROW, then each side is segmented.** Splitting the segment LIST
    # instead -- `segs[1:end-nval]` / `segs[end-nval+1:end]`, which is what this did until
    # 2026-09-18 -- is safe only at `stride == L - burn`, where the scored windows exactly tile the
    # record and consecutive segments overlap purely in their (unscored) burn-in. At any shorter
    # stride the scored windows overlap, so scored validation rows land inside scored training
    # rows and the validation loss quietly becomes a training loss. Splitting by row cannot do
    # that at any stride.
    # 🔑 The embargo comes free: the validation block's first `burn` rows charge the hidden state
    # and are not scored, so the first scored validation row sits `burn` steps after the last
    # training row -- at `burn = 100` that is about one 1/e time of the level's ACF. With training
    # tiled from the end (below) that last training row is scored at every stride, so the gap is
    # exactly `burn`, not `burn` plus whatever the tiling left over.
    ncol = length(steps)
    ntrain = floor(Int, (1 - val_frac) * ncol)
    0 < ntrain < ncol || error("train_stochlstm: val_frac = $val_frac leaves no split of a " *
                               "$(ncol)-column record")

    # 🔴 **Training keeps only FULL-LENGTH segments, and that is what makes a minibatch a
    # random sample of one population.** Segments of different lengths cannot share a recurrence,
    # so the one short segment used to be batched alone: at the tiling stride that is ONE of the
    # two updates per epoch spent on 13.6% of the scored rows, measured on the 2026-09-18 stride
    # scan. Dropping it buys a single length group, so every batch has the same shape and every
    # update the same weight.
    # 🔴 **Tiled from the END (`anchor = :end`), so the dropped short segment is the OLDEST
    # stretch, not the newest.** Tiled from the start, the full segments stop short of the block's
    # end by a stride-dependent amount -- 379 scored rows at the tiling stride, 19 at `stride =
    # 20`, on the 2879-row HIT block -- and those are exactly the rows adjacent to validation, so
    # the stride scan would have compared points trained on different recent data. Anchored at
    # the end, every stride's last scored training row is `ntrain`. ⚠️ The NUMBER of rows lost
    # still depends on the stride (same 379 vs 19); only where they sit has changed.
    train_all = RF.segment_indices(view(steps, 1:ntrain); L, burn, stride, anchor = :end)
    train_segs = [sg for sg in train_all if length(sg.rows) == L]
    # ⚠️ Validation is NEVER strided. Overlapping validation segments would re-weight some rows
    # more than others for no gain; augmentation is a training-set device.
    val_raw = RF.segment_indices(view(steps, (ntrain + 1):ncol); L, burn)
    val_segs = [(; rows = s.rows .+ ntrain, score = s.score .+ ntrain) for s in val_raw]

    isempty(train_segs) && error("train_stochlstm: no FULL-LENGTH training segment survived " *
                                 "L = $L, burn = $burn, stride = $stride on $(ntrain) columns " *
                                 "($(length(train_all)) shorter ones were dropped)")
    isempty(val_segs) && error("train_stochlstm: no validation segment survived L = $L, " *
                               "burn = $burn on $(ncol - ntrain) columns. Lower `L` or raise " *
                               "`val_frac`.")

    # 🔑 **Everything random is drawn on the HOST with `Xoshiro(seed)` and only then moved.** That
    # is what makes a device fit and a host fit at the same seed the *same* fit, agreeing to
    # floating-point round-off -- which is the only strong test the device path can be given, and
    # is also why `device` is a plain `Array -> Array` function rather than a device RNG. A CURAND
    # stream would be faster and would break V49's reproducibility property and every comparison
    # against a number already in `results_LSTMS.md`. The copies are tiny: one parameter set of
    # ~3 000 values, and `n_latent x L x B` noise, 256 KiB at the largest geometry, against a
    # gradient step measured at ~200 ms.
    rng = Xoshiro(seed)
    # 🔑 `init_ps` warm-starts from an existing fit -- rollout training is a FINE-TUNE of the
    # teacher-forced model, not a fit from scratch. Checked field by field against a fresh init,
    # so a fit of another shape cannot be loaded into this spec silently.
    ps0 = if init_ps === nothing
        RF.init_lstm_params(rng, spec; T)
    else
        tmpl = RF.init_lstm_params(Xoshiro(0), spec; T)
        keys(init_ps) == keys(tmpl) || error("train_stochlstm: init_ps has fields $(keys(init_ps)), " *
                                             "the spec needs $(keys(tmpl))")
        for k in keys(tmpl)
            a, b = getfield(init_ps, k), getfield(tmpl, k)
            (a === nothing) == (b === nothing) && (a === nothing || size(a) == size(b)) ||
                error("train_stochlstm: init_ps.$k does not match the spec")
        end
        map(v -> v === nothing ? nothing : T.(v), init_ps)
    end
    ps = _to_device(device, ps0)
    # ⚠️ `clip > 0` clips the gradient's global norm before Adam -- the guard against a free
    # rollout's exploding gradient (Rik, 2026-09-23). Off by default; `history.gmax` shows
    # whether it is needed.
    rule = clip > 0 ? Optimisers.OptimiserChain(Optimisers.ClipNorm(T(clip)), Optimisers.Adam(T(lr))) :
           Optimisers.Adam(T(lr))
    opt = Optimisers.setup(rule, ps)

    # 🔑 Segments of equal length share one recurrence, so they are grouped once here and batched
    # below. Every segment is exactly `L` long except the last of each contiguous block, so in
    # practice this is one large group plus a couple of singletons -- no padding, no dropped data,
    # and no segment ever shares a batch with one of a different length.
    function group_by_length(segs)
        d = Dict{Int,Vector{Int}}()
        for (i, s) in enumerate(segs)
            push!(get!(d, length(s.rows), Int[]), i)
        end
        return d
    end

    # Stack a set of same-length segments into (n_in, L, B) / (N_Q, L, B) plus the shared scored
    # range. `burn` is the same for every segment, so the local scored range is too.
    # 🔴 **Gathered from the DEVICE-resident record, so this transfers nothing.** `similar(Xd, ...)`
    # allocates wherever `Xd` lives, and each `Xb[:, :, k] = view(Xd, :, rows)` is a device-to-device
    # copy of a strided range — a bulk `copyto!`, not scalar indexing, so a device array accepts it.
    # ⚠️ The batch CANNOT simply be built once and reused: `train_groups` is reshuffled every epoch,
    # so at any stride where the segments outnumber `batch` the *membership* of each chunk changes,
    # not merely its order. Staging the record and gathering per update is what gets the transfers
    # to zero without changing which segments meet in a batch.
    function stack(segs, idx)
        L = length(segs[first(idx)].rows)
        Xb = similar(Xd, T, size(Xd, 1), L, length(idx))
        Yb = similar(Yd, T, size(Yd, 1), L, length(idx))
        for (k, i) in enumerate(idx)
            Xb[:, :, k] = view(Xd, :, segs[i].rows)
            Yb[:, :, k] = view(Yd, :, segs[i].rows)
        end
        s1 = segs[first(idx)]
        sc = (first(s1.score) - first(s1.rows) + 1):(last(s1.score) - first(s1.rows) + 1)
        return Xb, Yb, sc, length(sc) * length(idx)
    end

    # The randomness is drawn OUTSIDE the differentiated function -- the reparametrisation trick --
    # so `elbo` is deterministic given `epsz` and Zygote never sees an RNG.
    #
    # 🔴 **This is the LAST host->device transfer per update, and removing it costs reproducibility,
    # which is why it is opt-in.** Host-drawn (`device_rng = false`, the default): one copy per
    # update, `n_latent x L x B` — 256 kB at the largest geometry — and a device fit equals a host
    # fit at the same seed to round-off, which V54 asserts and every recorded number relies on.
    # Device-drawn (`device_rng = true`): zero transfers, and **a different noise stream**, so the
    # fit is no longer comparable with any existing one, on either device.
    # ⚠️ `randn!` on a device array uses that device's generator; on a host array it uses the
    # GLOBAL rng, **not** the seeded `rng` here — so `device_rng = true` makes even the CPU path
    # non-reproducible. It exists to measure what the transfer costs, not to fit with.
    draw = if device_rng
        (L, B) -> randn!(similar(Xd, T, spec.n_latent, L, B))
    else
        (L, B) -> device(randn(rng, T, spec.n_latent, L, B))
    end

    # 🔑 The training side is now ONE length group by construction, so its geometry is three
    # numbers rather than a dictionary. `group_by_length` survives for validation, which keeps
    # every segment including the short one -- an evaluation must not drop data.
    nseg = length(train_segs)
    batch_eff = min(batch, nseg)
    nbatch = fld(nseg, batch_eff)
    verbose && @info "M4 geometry" nseg batch_eff updates_per_epoch = nbatch dropped_short = (length(train_all) - nseg) dropped_per_epoch = (nseg - nbatch * batch_eff)
    val_groups = group_by_length(val_segs)

    # 🔴 **The validation epsilons are drawn ONCE and reused every epoch.** Re-drawing them makes
    # the reported validation loss a fresh single-sample ELBO estimate, so an epoch-to-epoch
    # comparison mixes "the model changed" with "the noise draw changed" -- and then "best
    # validation" selects partly on a lucky draw. Measured on R1's record: with fresh draws the
    # val loss moved by ~4 nats between consecutive checkpoints while the model was barely
    # changing. A fixed set makes the curve a function of the parameters alone.
    # One fixed batch per validation length-group: the arrays AND the noise draws, built once.
    # 🔑 With `rollout > 1` validation runs the SAME rollout, per segment length -- the objective
    # being optimised is the one early stopping and best-iterate selection read.
    fmask(Lg) = rollout == 1 ? nothing : _rollout_mask(Lg, burn, rollout)
    val_batches = [(stack(val_segs, idx)..., draw(length(val_segs[first(idx)].rows), length(idx)),
                    fmask(length(val_segs[first(idx)].rows)))
                   for (_, idx) in sort(collect(val_groups); by = first)]
    free_tr = fmask(L)

    function validate(p)
        num, den = 0.0, 0
        for (Xb, Yb, sc, ns, eb, fv) in val_batches
            num += RF.elbo(spec, p, Xb, Yb, sc, eb; beta, free = fv, lagmap) * ns
            den += ns
        end
        return num / den
    end

    # 🔴 **The history is indexed by OPTIMISER UPDATE, not by epoch, and `update` is stored
    # beside the curves.** An epoch is a different amount of training at every stride -- 1 update
    # at the tiling stride and 3 at `stride = 20` under the current geometry -- so anything paced
    # by the epoch is paced differently at every point. That is what made the 2026-09-18 stride
    # scan uncomparable, and validating once per epoch would have kept it so even with the
    # batching fixed: the plateau rule and the early stop would still have counted epochs.
    # 🔑 `val_every` is in UPDATES, so `patience` and `stop_patience` are too, identically for
    # every point. At the default `val_every = 2`, `patience = 20` is 40 updates.
    history = (; update = Int[], train = Float64[], val = Float64[], lr = Float64[],
               gmax = Float64[])   # largest gradient norm since the previous validation
    best = (; val = Inf, ps = deepcopy(ps), update = 0, index = 0)
    plateau = _PLATEAU_FRESH
    since_best = 0
    stopped_early = false
    stop_reason = :cap              # :floor (min_lr + stop_patience), :window, or :cap
    bsf = Float64[]                 # best-so-far validation loss, one entry per validation
    cur_lr = T(lr)
    upd = 0
    tot, nb, gmax = 0.0, 0, 0.0

    # 🔴 **A warm start is validated BEFORE its first update, and is itself a candidate for best.**
    # Otherwise a fine-tune can only return an iterate it moved to, and one that makes things worse
    # -- measured 2026-09-23: 10 rollout updates at lr = 1e-3 took the stride-100 fit from 1.01e-3
    # to 1.99e-3 -- comes back worse than it started, with nothing saying so. Row 1 of the history
    # is then update 0 (`train = NaN`: no step has been taken).
    if init_ps !== nothing
        vl0 = validate(ps)
        push!(history.update, 0); push!(history.train, NaN); push!(history.val, vl0)
        push!(history.lr, cur_lr); push!(history.gmax, 0.0)
        best = (; val = vl0, ps = deepcopy(ps), update = 0, index = 1)
        push!(bsf, vl0)
    end

    for epoch in 1:epochs
        # 🔑 Reshuffled every epoch, so which segments meet in a batch changes AND the
        # `nseg - nbatch * batch_eff` that fall outside the last full batch are a different few
        # each time -- dropped from an epoch, never from the fit.
        idxs = shuffle(rng, 1:nseg)
        for k in 1:nbatch
            chunk = view(idxs, ((k - 1) * batch_eff + 1):(k * batch_eff))
            Xb, Yb, sc, _ = stack(train_segs, chunk)
            eb = draw(L, batch_eff)
            loss, gs = Zygote.withgradient(p -> RF.elbo(spec, p, Xb, Yb, sc, eb; beta,
                                                        free = free_tr, lagmap), ps)
            gmax = max(gmax, _gnorm(gs[1]))
            opt, ps = Optimisers.update(opt, ps, gs[1])
            tot += loss
            nb += 1
            upd += 1

            # ⚠️ The final update of the final epoch is always validated, whatever `val_every`
            # divides into. Without it a fit whose update count is not a multiple of `val_every`
            # -- `epochs = 1` in the probe and in several tests -- would return an empty history.
            upd % val_every == 0 || (epoch == epochs && k == nbatch) || continue

            vl = validate(ps)
            push!(history.update, upd)
            push!(history.train, tot / nb)          # the mean since the previous validation
            push!(history.val, vl)
            push!(history.lr, cur_lr)
            push!(history.gmax, gmax)
            tot, nb, gmax = 0.0, 0, 0.0

            # 🔴 **Keep the best iterate, not the last one.** Without this the fit that gets
            # saved is whatever the final update happened to land on. Measured on R1's record: all
            # three latent architectures reached val ~= -12 around epoch 91 and were at ~= -4 by
            # epoch 100, so the last-iterate fit inverted the architecture ranking and the
            # conclusion drawn from it was an artefact of where the optimiser stopped.
            if vl < best.val
                best = (; val = vl, ps = deepcopy(ps), update = upd,
                        index = length(history.val))
                since_best = 0
            else
                # Validations since the ALL-TIME best: what the early stop reads.
                since_best += 1
            end

            # Decay on plateau. The end-of-training oscillation at a constant 1e-3 was several
            # nats wide, which is the other half of why a last-iterate fit was meaningless.
            # 🔴 **Patience counts against the best SINCE THE LAST DECAY, not the all-time best**
            # -- see `_plateau_step`. Best-iterate selection and the early stop above/below still
            # use the all-time best.
            plateau, decay = _plateau_step(plateau, vl, patience)
            if decay && cur_lr > min_lr
                cur_lr = max(T(min_lr), cur_lr * T(lr_decay))
                Optimisers.adjust!(opt, cur_lr)
                plateau = _PLATEAU_FRESH
                verbose && @info "M4 lr decayed" update = upd lr = cur_lr
            end

            # 🔴 **Early stop: at `min_lr` AND `stop_patience` validations past the best.**
            # Three of the five points of the 2026-09-18 stride scan had their best iterate at the
            # final epoch and ended two decades above `min_lr` -- their numbers were where the
            # budget stopped, not where the fit went, and they were not comparable with the two
            # that had converged.
            push!(bsf, best.val)
            if cur_lr <= min_lr && since_best >= stop_patience
                stopped_early = true
                stop_reason = :floor
            # 🔴 **Or: the best improved by < `stop_rel` over the last `stop_window` updates**, at
            # any rate -- see `_window_stalled`. Not gated on `min_lr`, by design (Rik, 2026-09-23).
            elseif _window_stalled(history.update, bsf, stop_window, stop_rel)
                stopped_early = true
                stop_reason = :window
            end
            if stopped_early
                verbose && @info "M4 early stop" reason = stop_reason update = upd best_update = best.update best_val = best.val lr = cur_lr
                break
            end

            # 🔴 FLUSHED. A SLURM log is a redirected stream, and an unflushed heartbeat is
            # indistinguishable from a hang -- which is exactly how a 20-minute job came to be
            # cancelled blind on 2026-09-18.
            if verbose && length(history.val) % 10 == 1
                @info "M4 update $upd (epoch $epoch)" train = history.train[end] val = vl best = best.val
                flush(stderr)
            end
        end
        stopped_early && break
    end

    verbose && @info "M4 done" best_update = best.update best_val = best.val final_val = history.val[end] updates = upd
    # 🔴 Returned on the HOST whatever `device` was. `LSTMWeights`, `save_stochlstm` and JLD2 all
    # expect plain arrays, and a fit that can only be read back on a machine with a GPU is not a
    # fit. `Array` on a host array is a no-op, so the CPU path is untouched.
    # 🔑 The geometry travels with the history. An epoch count is meaningless without it,
    # and deriving it afterwards from `(stride, batch)` is how it came to be wrong twice.
    # ⚠️ `best_update` is the optimiser step the returned iterate came from; `best_index` is
    # its position in the history arrays. They are different numbers whenever `val_every > 1`,
    # and using one where the other belongs is how a curve gets annotated in the wrong place.
    return _to_host(best.ps),
           (; history..., best_update = best.update, best_index = best.index,
            best_val = best.val, stopped_early, stop_reason, updates = upd,
            epochs_run = cld(upd, nbatch), nseg, batch_eff, upd_per_epoch = nbatch, val_every,
            rollout, clip, warm_start = init_ps !== nothing)
end

# ---------------------------------------------------------------------------------------------
# Held-out score
# ---------------------------------------------------------------------------------------------

"""
    iwae_nll(spec, ps, X, Y, score; K = 64, rng = Xoshiro(0))

Held-out negative log-likelihood per scored step, as an IWAE-`K` bound.

The importance weight for sample `k` is `log p(y | z_k) + log p(z_k) - log q(z_k | x)`, and
[`RikFlow.iwae_bound`](@ref) combines them stably.

🔴 **A lower bound on `log p`, so an upper bound on the NLL. It is not comparable to M0's exact
NLL and must never share a column with one** (`plan.md` §3). Report `crps_ensemble` and the rank
histogram beside it -- those two read the same on every cell of the ladder, which is exactly why
§9 makes the rank histogram the ladder-wide calibration metric.
"""
function RF.iwae_nll(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractMatrix{T},
                     Y::AbstractMatrix, score::AbstractUnitRange;
                     K::Int = 64, rng::AbstractRNG = Xoshiro(0)) where {T}
    # 🔴 `emission = :none` has a deterministic decoder, so there is no observation density and no
    # likelihood to bound. Refusing is the honest behaviour: the alternative is a number that looks
    # like an NLL and is not one. Score these cells with `crps_ensemble` and the rank histogram,
    # which are defined for them and read the same on every cell of the ladder.
    RF.emission_noise(spec) || error(
        "iwae_nll: emission = :none has no predictive density, so no likelihood is defined. " *
        "Use crps_ensemble and the rank histogram, or fit with emission = :constant.")
    nout = RF.n_output(spec)
    A = _precision_factor(ps.Araw)
    logdetA = sum(log.(diag(A)))
    L = size(X, 2)

    logw = zeros(Float64, K, length(score))
    for k in 1:K
        epsz = randn(rng, T, spec.n_latent, L)
        out = RF.lstm_forward(spec, ps, X, epsz)
        for (j, t) in enumerate(score)
            u = out.LOGD[:, t]
            r = (Y[:, t] .- out.Y[:, t]) ./ exp.(u)
            logpy = -0.5 * (nout * log(2pi) + 2 * sum(u) - 2 * logdetA + sum(abs2, A * r))
            if RF.latent_sampled(spec)
                z = out.MU[:, t] .+ out.SIG[:, t] .* epsz[:, t]
                logpz = -0.5 * (spec.n_latent * log(2pi) + sum(abs2, z))
                logqz = -0.5 * (spec.n_latent * log(2pi) + sum(abs2, epsz[:, t])) -
                        sum(log.(out.SIG[:, t]))
                logw[k, j] = logpy + logpz - logqz
            else
                logw[k, j] = logpy
            end
        end
    end
    return -mean(RF.iwae_bound(@view logw[:, j]) for j in 1:length(score))
end

# ---------------------------------------------------------------------------------------------
# Conversion to the deployed form
# ---------------------------------------------------------------------------------------------

"""
    LSTMWeights(ps::NamedTuple, spec::LSTMSpec)

Convert trained parameters into the form `lstm_step!` runs on.

Everything is a copy except the covariance. Training carries a free precision factor `A`, so the
implied covariance of the standardised residual is `Sigma_c = (A'A)^{-1}`, whose diagonal is not
unit. The deployed form needs a genuine correlation matrix, so:

    s_i = sqrt(Sigma_c[i, i])        R = diag(s)^{-1} Sigma_c diag(s)^{-1}
    LR  = cholesky(R).L              bd_deployed = bd_trained + log(s)

which is an exact reparametrisation -- `D_dep R D_dep == D_train Sigma_c D_train` -- and not an
approximation. V41 checks it by comparing the two forward passes.
"""
function RF.LSTMWeights(ps::NamedTuple, spec::RF.LSTMSpec)
    A = _precision_factor(ps.Araw)
    T = eltype(ps.Wx)
    Sigc = inv(Symmetric(Matrix(A' * A)))
    s = sqrt.(diag(Sigc))
    R = Sigc ./ (s * s')
    R = (R .+ R') ./ 2                                   # symmetrise away the round-off
    LR = Matrix{T}(cholesky(Symmetric(R)).L)

    return RF.LSTMWeights{T}(
        copy(ps.Wx), copy(ps.Wh), copy(ps.b),
        ps.We === nothing ? nothing : copy(ps.We),
        ps.be === nothing ? nothing : copy(ps.be),
        copy(ps.Bmu), copy(ps.Bsig),
        copy(ps.V1),
        ps.V2 === nothing ? nothing : copy(ps.V2),
        copy(ps.cdec),
        copy(ps.Wd), T.(ps.bd .+ log.(s)),
        LR,
    )
end

end # module
