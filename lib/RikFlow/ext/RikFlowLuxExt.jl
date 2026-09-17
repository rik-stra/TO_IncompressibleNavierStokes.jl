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

# the whole latent path, or a constant stand-in for the deterministic architecture
function _latent(::Val{true}, encv, ps, X, epsz, ::Type{T}, nz, L) where {T}
    E = _encode(encv, ps, X)
    MU = ps.Bmu * E
    SIG = _sp.(ps.Bsig * E)
    return MU, SIG, MU .+ SIG .* epsz
end
_latent(::Val{false}, encv, ps, X, epsz, ::Type{T}, nz, L) where {T} =
    (zeros(T, nz, L), ones(T, nz, L), zeros(T, nz, L))

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
function RF.lstm_forward(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractMatrix{T},
                         epsz::AbstractMatrix) where {T}
    H, L = spec.n_hidden, size(X, 2)

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

    # --- encoder and latent path, whole segment at once --------------------------------------
    MU, SIG, Z = _latent(hasz, encv, ps, X, epsz, T, spec.n_latent, L)

    # --- recurrence ---------------------------------------------------------------------------
    XZ = _cell_input(cellv, X, Z)
    GX = ps.Wx * XZ                                  # every step's input contribution, precomputed

    Hb = Zygote.Buffer(zeros(T, H, L))
    h = zeros(T, H)
    c = zeros(T, H)
    for t in 1:L
        g = GX[:, t] .+ ps.Wh * h .+ ps.b
        i = _sig.(g[1:H])
        f = _sig.(g[(H + 1):(2H)])
        gg = tanh.(g[(2H + 1):(3H)])
        o = _sig.(g[(3H + 1):(4H)])
        c = f .* c .+ i .* gg
        h = o .* tanh.(c)
        Hb[:, t] = h
    end
    Hm = copy(Hb)

    # --- decoder and log-scale head -----------------------------------------------------------
    Y = _decoder_skip(decv, ps.V1 * Hm .+ ps.cdec, ps, Z)
    LOGD = ps.Wd * Hm .+ ps.bd
    if spec.uclip !== nothing
        LOGD = clamp.(LOGD, T(spec.uclip[1]), T(spec.uclip[2]))
    end

    return (; Y, LOGD, MU, SIG, Hm)
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
function RF.elbo(spec::RF.LSTMSpec, ps::NamedTuple, X::AbstractMatrix{T}, Ytrue::AbstractMatrix,
                 score::AbstractUnitRange, epsz::AbstractMatrix; beta::Real = 1e-4) where {T}
    out = RF.lstm_forward(spec, ps, X, epsz)
    nout = RF.n_output(spec)
    ns = length(score)

    U = out.LOGD[:, score]
    Rres = (Ytrue[:, score] .- out.Y[:, score]) ./ exp.(U)
    A = _precision_factor(ps.Araw)
    quad = sum(abs2, A * Rres)
    logdet_term = 2 * sum(U) - 2 * ns * sum(log.(diag(A)))
    nll = T(0.5) * (nout * ns * T(log(2 * pi)) + logdet_term + quad)

    kl = if RF.latent_sampled(spec)
        S = out.SIG[:, score]
        M = out.MU[:, score]
        sum(T(0.5) .* (S .^ 2 .+ M .^ 2 .- one(T)) .- log.(S))
    else
        zero(T)
    end

    return (nll + T(beta) * kl) / ns
end

# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

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
- `L`, `burn`: segment length and burn-in. `L = 200`, `burn = 50` by default; both are sweep axes.
- `epochs`, `batch`, `lr`, `beta`, `seed`.
- `val_frac`: the trailing fraction of segments held out for the reported validation loss.
  ⚠️ This is an *inner* split for early stopping and diagnostics only. Model **selection** is on a
  window disjoint from both training and the online evaluation window -- that is the protocol's
  job, not this function's, and this keyword is not it.

Returns `(ps, history)`.
"""
function RF.train_stochlstm(spec::RF.LSTMSpec, X::AbstractMatrix, Y::AbstractMatrix,
                            steps::AbstractVector{<:Integer};
                            L::Int = 200, burn::Int = 50, epochs::Int = 200, batch::Int = 8,
                            lr::Real = 1e-3, beta::Real = 1e-4, seed::Int = 1,
                            val_frac::Real = 0.2, T::Type = Float32, verbose::Bool = true)
    size(X, 2) == size(Y, 2) == length(steps) ||
        error("train_stochlstm: X, Y and steps disagree on the number of columns")

    Xt, Yt = T.(X), T.(Y)
    segs = RF.segment_indices(steps; L, burn)
    isempty(segs) && error("train_stochlstm: no segment survived L = $L, burn = $burn on a " *
                           "record of $(length(steps)) columns")

    nval = max(1, round(Int, val_frac * length(segs)))
    train_segs, val_segs = segs[1:(end - nval)], segs[(end - nval + 1):end]
    isempty(train_segs) && error("train_stochlstm: val_frac = $val_frac leaves no training segments")

    rng = Xoshiro(seed)
    ps = RF.init_lstm_params(rng, spec; T)
    opt = Optimisers.setup(Optimisers.Adam(T(lr)), ps)

    # The randomness is drawn OUTSIDE the differentiated function -- the reparametrisation trick --
    # so `elbo` is deterministic given `epsz` and Zygote never sees an RNG.
    draw(seg) = randn(rng, T, spec.n_latent, length(seg.rows))

    function segloss(p, seg, e)
        rows = seg.rows
        local_score = (first(seg.score) - first(rows) + 1):(last(seg.score) - first(rows) + 1)
        return RF.elbo(spec, p, Xt[:, rows], Yt[:, rows], local_score, e; beta)
    end

    history = (; train = Float64[], val = Float64[])
    for epoch in 1:epochs
        order = shuffle(rng, eachindex(train_segs))
        tot, nb = 0.0, 0
        for chunk in Iterators.partition(order, batch)
            es = [draw(train_segs[i]) for i in chunk]
            loss, gs = Zygote.withgradient(ps) do p
                # an explicit comprehension rather than a generator: Zygote is reliable on the
                # former and occasionally not on the latter, and the chunk is tiny
                sum([segloss(p, train_segs[i], es[k]) for (k, i) in enumerate(chunk)]) /
                    length(chunk)
            end
            opt, ps = Optimisers.update(opt, ps, gs[1])
            tot += loss
            nb += 1
        end
        push!(history.train, tot / nb)

        vl = mean(segloss(ps, s, draw(s)) for s in val_segs)
        push!(history.val, vl)
        verbose && (epoch % 10 == 1 || epoch == epochs) &&
            @info "M4 epoch $epoch" train = history.train[end] val = vl
    end

    return ps, history
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
