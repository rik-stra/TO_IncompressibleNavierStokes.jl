# M4 -- the stochastic LSTM (STORN / VRNN). Inference side.
#
# 🔑 STDLIB ONLY, and that is the whole design decision. Training happens in
# `ext/RikFlowLuxExt.jl` under Lux; what runs *inside the solver* is the hand-written
# `lstm_step!` below, on plain `Float32` arrays with pre-allocated scratch. Four reasons, in
# order of how much they matter:
#
#   1. The deployed path becomes testable. The verification suite is stdlib-only by design
#      (`test/Project.toml`), so under any other layout the code that actually runs in the
#      solver would be the one piece of the package the matrix cannot reach.
#   2. S4 becomes reachable. The per-step budget is 1.15x TO-LRS with a surrogate share of
#      1.85 ms/step, and `plan.md` §2 S4 says it is *launch-bound, not FLOP-bound*. This cell is
#      ~30 kFLOP at H = 60, N_Q = 6 -- microseconds -- but only if nothing allocates and nothing
#      dispatches generically. 🔴 **Never call `Lux.apply` from the online loop.**
#   3. The base package stays Lux-free, which is what keeps the ~900 M0/M1/M2 jobs from paying
#      Lux load time (`plan.md` §13 P1.0b).
#   4. The port bug becomes a test. The training forward pass is non-mutating and AD-shaped, this
#      one is fused and in-place; they are different code and can drift. V41 is the test that says
#      they have not.
#
# ---------------------------------------------------------------------------------------------
# What the source actually does, and where this differs
# ---------------------------------------------------------------------------------------------
#
# `paper/methods_overview.tex` §"Barthel Sørensen group" writes
#
#     z_t ~ N( mu_z(x_t), sigma_z(x_t) )     mu_z = B x_t,  sigma_z = softplus(C x_t)
#     h_t = LSTM( [x_t ; z_t], h_{t-1} )
#     y_t = f_o( V1 h_t + V2 z_t + c )
#
# 🔑 **That first line is the ENCODER -- the approximate posterior q(z_t | x_t) -- and not the
# prior.** Settled 2026-09-17 against the reference implementation
# (`ben-barthel/learning_dynamics`, `ML_Code/networks_qg.py`), which is unambiguous: the prior is a
# fixed standard normal `p(z) = N(0, I)`, the encoder sees **only `x_t`** -- not the target, not
# `h_{t-1}`, and there is no separate recognition RNN -- and it is a dense `tanh` layer followed by
# the two heads. Because `x_t` is observed at deployment, the same encoder serves training and
# inference, which is why no recognition network appears anywhere in this file.
#
# Three deliberate deviations from the source, each with a reason:
#
#   (a) **A Gaussian emission head, where the source is deterministic.** Their decoder emits a
#       point and the objective is a plain MSE, so their model has no predictive density at all.
#       This project selects on held-out likelihood and has a calibrated-spread criterion (S7), and
#       `plan.md` §3 specifies a *"Gaussian emission head"* for M4. So the decoder here carries the
#       L2 head of `methods_overview.tex` §"State dependent covariance", `Sigma = D R D` with
#       `log d_i` linear in `h_t`. 🔴 This is a deviation, not a reproduction, and the paper must
#       say so where it reports M4.
#   (b) **The mass-conservation penalty is not imported.** It is L4, explicitly out of scope
#       (`plan.md` §24). M4 takes the architecture, not the penalty.
#   (c) **Their KL weight is called `lambda` and is `1e-4`. Here it is `beta`, always** -- `lambda`
#       is the ridge parameter throughout this project and a collision would corrupt §8a and the
#       S2' axis.
#
# ⚠️ The tex also writes the encoder's log-scale matrix as `C`. **`C` is this project's mean
# regression matrix** (`ts_models.jl`), so it is `Bsig` here and in the paper.
#
# ⚠️ One discrepancy between the tex and the reference implementation, resolved by `n_encoder`:
# the tex's equations give a *linear* encoder (`mu_z = B x_t`), the repository puts a dense `tanh`
# layer in front of the heads. `n_encoder = 0` is the tex's form, `n_encoder > 0` the repository's,
# and the default follows the repository because that is the artefact the reported result came
# from.

"""
    LSTMSpec(; hist, n_hidden = 60, n_latent = 60, n_encoder = 60, arch = :vrnn, uclip = nothing)

Shape and architecture of an M4 model.

# Arguments
- `hist`: the [`HistorySpec`](@ref) the regressor is built against. **The same one every other
  cell uses** -- M4 introduces no new input layout, so `build_history`, `HistoryBuffer` and
  `inputvec` are reused unchanged and V1/V2 already cover them.
- `n_hidden`: LSTM width. 60 in the source.
- `n_latent`: dimension of the latent path `z`. 60 in the source -- ⚠️ *not* `N_Q`; an early
  reading of the tex assumed it matched the output width and it does not.
- `n_encoder`: width of the encoder's dense `tanh` layer. `0` drops it, giving the purely linear
  encoder the tex's equations describe. 60 in the source.
- `arch`: one of `:lstm`, `:vaernn`, `:storn`, `:vrnn` -- see the table below.
- `uclip`: `(u_min, u_max)` bounds on the emission log-scale pre-activation, or `nothing`.
  Same device, and the same reason, as [`JointModel`](@ref)'s `uclip`: `d_i = exp(u_i)` grows
  exponentially once the input leaves the training range, and the fraction of steps on which the
  clip activates is a diagnostic that the trajectory has left the regime the head was fitted on.

# The four architectures

Sørensen et al. compare four, and their result is a *ranking* across them, so all four are one
flag apart here rather than four models.

| `arch` | `z` drawn | `z` into the cell | `z` into the decoder | is |
|---|---|---|---|---|
| `:lstm` | no | no | no | deterministic LSTM + Gaussian head |
| `:vaernn` | yes | no | yes | stochasticity at the **output** only |
| `:storn` | yes | yes | no | **STORN** |
| `:vrnn` | yes | yes | yes | **VRNN** |

"Upstream stochasticity", the property they report as less prone to overfitting, is exactly the
`z`-into-the-cell column.
"""
Base.@kwdef struct LSTMSpec
    hist::HistorySpec
    n_hidden::Int = 60
    n_latent::Int = 60
    n_encoder::Int = 60
    arch::Symbol = :vrnn
    uclip::Union{Nothing,Tuple{Float64,Float64}} = nothing

    function LSTMSpec(hist, n_hidden, n_latent, n_encoder, arch, uclip)
        arch in (:lstm, :vaernn, :storn, :vrnn) ||
            error("LSTMSpec: arch must be one of :lstm, :vaernn, :storn, :vrnn; got $(arch)")
        n_hidden > 0 || error("LSTMSpec: n_hidden must be positive")
        n_encoder >= 0 || error("LSTMSpec: n_encoder must be >= 0 (0 = linear encoder)")
        (arch === :lstm || n_latent > 0) ||
            error("LSTMSpec: arch $(arch) has a latent path, so n_latent must be positive")
        new(hist, n_hidden, n_latent, n_encoder, arch, uclip)
    end
end

"""
    latent_sampled(spec)

Whether the architecture has a latent path at all. `false` only for `:lstm`.
"""
latent_sampled(spec::LSTMSpec) = spec.arch !== :lstm

"""
    latent_to_cell(spec)

Whether `z_t` is fed into the recurrence. This is the **upstream stochasticity** that separates
STORN/VRNN from the deterministic and output-only variants, and it is the property Sørensen et al.
report as the one that matters.
"""
latent_to_cell(spec::LSTMSpec) = spec.arch === :storn || spec.arch === :vrnn

"""
    latent_to_decoder(spec)

Whether the decoder carries the `V2 z_t` skip.
"""
latent_to_decoder(spec::LSTMSpec) = spec.arch === :vrnn || spec.arch === :vaernn

"""
    n_input(spec)

Width of the regressor row `x_t`, including its trailing bias entry.
"""
n_input(spec::LSTMSpec) = nfeatures(spec.hist)

"""
    n_output(spec)

Number of QoIs predicted, `N_Q`.
"""
n_output(spec::LSTMSpec) = spec.hist.n_qoi

"""
    n_encoder_out(spec)

Width of whatever the encoder heads read: the dense layer's output, or the raw regressor when
`n_encoder == 0`.
"""
n_encoder_out(spec::LSTMSpec) = spec.n_encoder > 0 ? spec.n_encoder : n_input(spec)

"""
    n_cell_input(spec)

Width of what the LSTM cell actually consumes: the regressor, plus `z` where the architecture
feeds it upstream.
"""
n_cell_input(spec::LSTMSpec) =
    n_input(spec) + (latent_to_cell(spec) ? spec.n_latent : 0)

# ---------------------------------------------------------------------------------------------
# Elementwise helpers
# ---------------------------------------------------------------------------------------------
#
# ⚠️ Underscore-prefixed on purpose. `Lux`/`NNlib` export `sigmoid` and `softplus`, and the
# extension loads both into the same session as this file. Unprefixed names would resolve to
# whichever was imported last.

_sigmoid(x::T) where {T<:Real} = one(T) / (one(T) + exp(-x))

# log(1 + exp(x)), in the branch-stable form. The naive expression overflows for x >~ 89 in
# Float32 and loses the whole value for x <~ -89.
_softplus(x::T) where {T<:Real} = x > zero(T) ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    LSTMWeights

Trained parameters, as plain arrays. No Lux types appear anywhere in this struct -- that is what
lets the online path run, and be tested, without Lux in the environment.

# Gate ordering

🔴 **Fixed here and nowhere else:** rows `1:H` are the input gate `i`, `H+1:2H` the forget gate
`f`, `2H+1:3H` the cell candidate `g`, and `3H+1:4H` the output gate `o`. The training side must
emit this ordering, and V41 is the test that says it did. A transposed convention is silent: the
model trains, the forward pass runs, and the closure is wrong.

# Fields
- `Wx`, `Wh`, `b`: the cell, `4H x n_cell_input`, `4H x H`, `4H`.
- `We`, `be`: the encoder's dense `tanh` layer, `n_encoder x n_input` and `n_encoder`. Both
  `nothing` when `n_encoder == 0`.
- `Bmu`, `Bsig`: the encoder heads, `n_latent x n_encoder_out` each. These parametrise the
  **approximate posterior** `q(z_t | x_t)`; the prior is a fixed `N(0, I)`.
- `V1`, `V2`, `cdec`: the decoder. `V2` is `nothing` unless the architecture carries the skip.
- `Wd`, `bd`: the emission log-scale head, `N_Q x H` and `N_Q`. This is the L2 head of
  `methods_overview.tex` §"State dependent covariance", driven by `h_t` instead of by the raw
  history vector, and it is a deviation from the source -- see the file header, deviation (a).
- `LR`: lower Cholesky factor of the constant correlation matrix `R`, so `R = LR * LR'`.
"""
struct LSTMWeights{T}
    Wx::Matrix{T}
    Wh::Matrix{T}
    b::Vector{T}
    We::Union{Nothing,Matrix{T}}
    be::Union{Nothing,Vector{T}}
    Bmu::Matrix{T}
    Bsig::Matrix{T}
    V1::Matrix{T}
    V2::Union{Nothing,Matrix{T}}
    cdec::Vector{T}
    Wd::Matrix{T}
    bd::Vector{T}
    LR::Matrix{T}
end

"""
    check_shapes(w::LSTMWeights, spec::LSTMSpec)

Assert that a weight set matches a spec. Called on load, because the alternative is a silent
`DimensionMismatch` deep inside the solver on step one of a cluster job.
"""
function check_shapes(w::LSTMWeights, spec::LSTMSpec)
    H, nin, nout, nz = spec.n_hidden, n_input(spec), n_output(spec), spec.n_latent
    ncin, nenc = n_cell_input(spec), n_encoder_out(spec)
    size(w.Wx) == (4H, ncin) || error("Wx is $(size(w.Wx)), expected $((4H, ncin))")
    size(w.Wh) == (4H, H) || error("Wh is $(size(w.Wh)), expected $((4H, H))")
    length(w.b) == 4H || error("b is $(length(w.b)), expected $(4H)")
    size(w.V1) == (nout, H) || error("V1 is $(size(w.V1)), expected $((nout, H))")
    length(w.cdec) == nout || error("cdec is $(length(w.cdec)), expected $(nout)")
    size(w.Wd) == (nout, H) || error("Wd is $(size(w.Wd)), expected $((nout, H))")
    length(w.bd) == nout || error("bd is $(length(w.bd)), expected $(nout)")
    size(w.LR) == (nout, nout) || error("LR is $(size(w.LR)), expected $((nout, nout))")
    if spec.n_encoder > 0 && latent_sampled(spec)
        w.We === nothing && error("n_encoder = $(spec.n_encoder) but We is nothing")
        size(w.We) == (nenc, nin) || error("We is $(size(w.We)), expected $((nenc, nin))")
        length(w.be) == nenc || error("be is $(length(w.be)), expected $(nenc)")
    end
    if latent_sampled(spec)
        size(w.Bmu) == (nz, nenc) || error("Bmu is $(size(w.Bmu)), expected $((nz, nenc))")
        size(w.Bsig) == (nz, nenc) || error("Bsig is $(size(w.Bsig)), expected $((nz, nenc))")
    end
    if latent_to_decoder(spec)
        w.V2 === nothing && error("arch $(spec.arch) carries the V2 skip but V2 is nothing")
        size(w.V2) == (nout, nz) || error("V2 is $(size(w.V2)), expected $((nout, nz))")
    end
    return true
end

"""
    LSTMState(spec, T)

Pre-allocated recurrent state and scratch for one online trajectory. Everything `lstm_step!`
touches lives here, so a step allocates nothing.
"""
mutable struct LSTMState{T}
    h::Vector{T}
    c::Vector{T}
    gates::Vector{T}
    he::Vector{T}
    z::Vector{T}
    muz::Vector{T}
    sigz::Vector{T}
    xz::Vector{T}
    y::Vector{T}
    logd::Vector{T}
    nstep::Int
end

function LSTMState(spec::LSTMSpec, ::Type{T}) where {T}
    H, nz, nout = spec.n_hidden, spec.n_latent, n_output(spec)
    LSTMState{T}(
        zeros(T, H), zeros(T, H), zeros(T, 4H),
        zeros(T, n_encoder_out(spec)),
        zeros(T, nz), zeros(T, nz), zeros(T, nz),
        zeros(T, n_cell_input(spec)),
        zeros(T, nout), zeros(T, nout),
        0,
    )
end

"""
    reset!(st::LSTMState)

Zero the recurrent state. `h_0 = c_0 = 0`, then the warm-up charges it -- see the `StochLSTM`
docstring in `time_series_methods.jl`.
"""
function reset!(st::LSTMState{T}) where {T}
    fill!(st.h, zero(T))
    fill!(st.c, zero(T))
    st.nstep = 0
    return st
end

"""
    lstm_step!(st, w, spec, x; rng = nothing, sample_latent = true)

Advance one step. Mutates `st` and returns `(y, logd, z)` as views into it -- copy them if they
must outlive the next call.

# Arguments
- `x`: the regressor row for this step, length `n_input(spec)`, already standardised.
- `rng`: consumed **only** when `sample_latent` is true and the architecture has a latent path.
- `sample_latent`: `false` substitutes the posterior mean `z = mu_z(x)` and draws nothing.

🔴 **`sample_latent = false` is what the warm-up uses, and it is not a performance shortcut.**
V38 requires that a closure's warm-up replay does not touch the RNG, so that a member seed means
the same thing across closures and across warm-up lengths. The posterior mean keeps that
invariant; sampling during the warm-up would break it.
"""
function lstm_step!(
    st::LSTMState{T}, w::LSTMWeights{T}, spec::LSTMSpec, x::AbstractVector;
    rng = nothing, sample_latent::Bool = true,
) where {T}
    H = spec.n_hidden
    nin = n_input(spec)

    # --- the regressor, converted once ------------------------------------------------------
    @inbounds for k in 1:nin
        st.xz[k] = T(x[k])
    end
    xv = view(st.xz, 1:nin)

    # --- encoder: q(z_t | x_t) --------------------------------------------------------------
    if latent_sampled(spec)
        # ⚠️ The two `mul!` pairs are written out per branch rather than hoisted behind one
        # `enc` variable. Hoisting gives `enc` the type `Union{Vector, SubArray}`, which Julia
        # boxes -- and that box was a per-step heap allocation on exactly the `n_encoder == 0`
        # path, caught by the V40 allocation test. Duplicating two lines is the cheaper trade.
        if spec.n_encoder > 0
            mul!(st.he, w.We, xv)
            @inbounds for k in eachindex(st.he)
                st.he[k] = tanh(st.he[k] + w.be[k])
            end
            mul!(st.muz, w.Bmu, st.he)
            mul!(st.sigz, w.Bsig, st.he)
        else
            mul!(st.muz, w.Bmu, xv)
            mul!(st.sigz, w.Bsig, xv)
        end
        @inbounds for k in eachindex(st.sigz)
            st.sigz[k] = _softplus(st.sigz[k])
        end
        if sample_latent
            rng === nothing && error("lstm_step!: sample_latent = true needs an rng")
            @inbounds for k in eachindex(st.z)
                st.z[k] = st.muz[k] + st.sigz[k] * T(randn(rng))
            end
        else
            copyto!(st.z, st.muz)
        end
    end

    # --- cell input -------------------------------------------------------------------------
    if latent_to_cell(spec)
        @inbounds for k in eachindex(st.z)
            st.xz[nin + k] = st.z[k]
        end
    end

    # --- the cell ---------------------------------------------------------------------------
    mul!(st.gates, w.Wx, st.xz)
    mul!(st.gates, w.Wh, st.h, one(T), one(T))
    @inbounds for k in eachindex(st.gates)
        st.gates[k] += w.b[k]
    end
    @inbounds for k in 1:H
        i = _sigmoid(st.gates[k])
        f = _sigmoid(st.gates[H + k])
        g = tanh(st.gates[2H + k])
        o = _sigmoid(st.gates[3H + k])
        st.c[k] = f * st.c[k] + i * g
        st.h[k] = o * tanh(st.c[k])
    end

    # --- decoder ----------------------------------------------------------------------------
    #
    # f_o is the identity, matching the source's linear output layer. The target is standardised,
    # so a bounded activation would cap the very excursions this model exists to represent.
    mul!(st.y, w.V1, st.h)
    if latent_to_decoder(spec)
        mul!(st.y, w.V2, st.z, one(T), one(T))
    end
    @inbounds for k in eachindex(st.y)
        st.y[k] += w.cdec[k]
    end

    # --- emission log-scale head ------------------------------------------------------------
    mul!(st.logd, w.Wd, st.h)
    @inbounds for k in eachindex(st.logd)
        st.logd[k] += w.bd[k]
    end
    if spec.uclip !== nothing
        lo, hi = T(spec.uclip[1]), T(spec.uclip[2])
        @inbounds for k in eachindex(st.logd)
            st.logd[k] = clamp(st.logd[k], lo, hi)
        end
    end

    st.nstep += 1
    return st.y, st.logd, st.z
end

"""
    sample_emission!(out, st, w, rng)

Draw one realisation of `q^n` given the step just taken: `out = y + D * LR * eps` with
`D = diag(exp.(logd))` and `eps ~ N(0, I)`.

The covariance realised is `Sigma = D R D` with `R = LR * LR'`, which is the constant conditional
correlation form of `methods_overview.tex` §"State dependent covariance" -- positive definite
whenever `R` is and the scales are finite, so the log link needs no constraint.
"""
function sample_emission!(out::AbstractVector{T}, st::LSTMState{T}, w::LSTMWeights{T}, rng) where {T}
    n = length(out)
    @inbounds for k in 1:n
        out[k] = T(randn(rng))
    end
    # out <- LR * out, in place, LR lower triangular
    @inbounds for k in n:-1:1
        acc = zero(T)
        for j in 1:k
            acc += w.LR[k, j] * out[j]
        end
        out[k] = acc
    end
    @inbounds for k in 1:n
        out[k] = st.y[k] + exp(st.logd[k]) * out[k]
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Sequence segmentation for BPTT
# ---------------------------------------------------------------------------------------------

"""
    segment_indices(steps; L, burn, stride = L - burn)

Cut a run of regressor rows into contiguous BPTT segments.

`build_history` returns its rows ordered by increasing physical step, together with those step
indices -- an ordering the AR shift in `ts_fit.jl` already depends on -- so segments are plain
slices. What this function adds is the two things that are easy to get silently wrong:

- **Blocks.** Wherever `steps` jumps by more than one the record is discontinuous, and a segment
  that straddles the gap would train the recurrence across a break in time. Segments never cross
  one.
- **Burn-in.** The first `burn` rows of each segment charge the hidden state and are **excluded
  from the loss**. Without this, every segment contributes a cold-start term from `h = 0`, which
  the model can only fit by learning to predict well from no history -- exactly the behaviour the
  recurrence exists to avoid.

Returns a vector of `(; rows, score)` where `rows` is the full segment (what the forward pass
consumes) and `score` is the sub-range that the loss is summed over. Segments with no scored rows
are dropped.
"""
function segment_indices(steps::AbstractVector{<:Integer}; L::Int, burn::Int,
                         stride::Int = L - burn)
    L > burn >= 0 || error("segment_indices: need L > burn >= 0; got L = $L, burn = $burn")
    stride >= 1 || error("segment_indices: stride must be >= 1; got $stride")
    n = length(steps)
    segs = NamedTuple{(:rows, :score),Tuple{UnitRange{Int},UnitRange{Int}}}[]
    n == 0 && return segs

    # block boundaries: a block is a maximal run of consecutive step indices
    bstart = 1
    for i in 1:n
        isend = i == n || steps[i + 1] != steps[i] + 1
        isend || continue
        blk = bstart:i
        if length(blk) > burn
            s = first(blk)
            while s <= last(blk) - burn
                e = min(s + L - 1, last(blk))
                # a trailing stub with nothing left to score after the burn-in is not a segment
                if e - s + 1 > burn
                    push!(segs, (; rows = s:e, score = (s + burn):e))
                end
                e == last(blk) && break
                s += stride
            end
        end
        bstart = i + 1
    end
    return segs
end

# ---------------------------------------------------------------------------------------------
# Likelihood arithmetic
# ---------------------------------------------------------------------------------------------
#
# Kept here, on already-computed quantities, so the objective the extension optimises and the
# objective the scorer reports are the same code.

"""
    gauss_logpdf(resid, logd, LR)

Log density of `resid` under `N(0, D R D)` with `D = diag(exp.(logd))` and `R = LR * LR'`.

The log-determinant collapses to `2 * sum(logd) + 2 * sum(log(diag(LR)))`, linear in the log-scale
head -- which is the reason `methods_overview.tex` takes a log link here and rejects softplus.
"""
function gauss_logpdf(resid::AbstractVector, logd::AbstractVector, LR::AbstractMatrix)
    n = length(resid)
    T = float(promote_type(eltype(resid), eltype(logd), eltype(LR)))
    logdet2 = zero(T)
    @inbounds for k in 1:n
        logdet2 += 2 * logd[k] + 2 * log(LR[k, k])
    end
    # quad = || LR \ (resid ./ d) ||^2, by forward substitution
    v = Vector{T}(undef, n)
    @inbounds for k in 1:n
        acc = T(resid[k]) / exp(T(logd[k]))
        for j in 1:(k - 1)
            acc -= LR[k, j] * v[j]
        end
        v[k] = acc / LR[k, k]
    end
    quad = zero(T)
    @inbounds for k in 1:n
        quad += v[k]^2
    end
    return -T(0.5) * (n * T(log(2 * pi)) + logdet2 + quad)
end

"""
    kl_diag_gaussian(muq, sigq, mup, sigp)

`KL( N(muq, diag(sigq^2)) || N(mup, diag(sigp^2)) )`, summed over dimensions.

🔑 Report this **per latent dimension** as well as summed. If it goes to zero the latent path is
unused, and `:storn`/`:vrnn` have silently degenerated into `:lstm` with a heteroscedastic head --
which voids the upstream-stochasticity comparison M4 exists for, while every other number in the
table still looks reasonable. Posterior collapse is the standard failure mode of this family and
it is not self-announcing.
"""
function kl_diag_gaussian(muq::AbstractVector, sigq::AbstractVector,
                          mup::AbstractVector, sigp::AbstractVector)
    T = float(promote_type(eltype(muq), eltype(sigq), eltype(mup), eltype(sigp)))
    acc = zero(T)
    @inbounds for k in eachindex(muq)
        sq, sp = T(sigq[k]), T(sigp[k])
        acc += log(sp / sq) + (sq^2 + (T(muq[k]) - T(mup[k]))^2) / (2 * sp^2) - T(0.5)
    end
    return acc
end

"""
    kl_to_standard_normal(mu, sig)

`KL( N(mu, diag(sig^2)) || N(0, I) )`, which is the KL the objective actually uses: the source's
prior is a fixed standard normal, settled against `ML_Code/networks_qg.py`.

Written out rather than deferred to [`kl_diag_gaussian`](@ref) because it is in the training inner
loop and the general form would allocate two unit vectors per step.
"""
function kl_to_standard_normal(mu::AbstractVector, sig::AbstractVector)
    T = float(promote_type(eltype(mu), eltype(sig)))
    acc = zero(T)
    @inbounds for k in eachindex(mu)
        s = T(sig[k])
        acc += T(0.5) * (s^2 + T(mu[k])^2 - one(T)) - log(s)
    end
    return acc
end

"""
    iwae_bound(logw)

`log(1/K * sum(exp.(logw)))` over `K` importance weights, computed stably.

This is what replaces the exact held-out NLL for M4. 🔴 **It is a lower bound on `log p`, so it is
not comparable to M0's exact NLL and must never share a column with it** (`plan.md` §3: a
stochastic latent path has no closed-form one-step predictive density, which is also why the
ensemble rank histogram, not PIT, is the ladder-wide calibration metric).
"""
function iwae_bound(logw::AbstractVector)
    K = length(logw)
    K == 0 && error("iwae_bound: no weights")
    m = maximum(logw)
    isfinite(m) || return m
    acc = zero(float(eltype(logw)))
    @inbounds for k in 1:K
        acc += exp(logw[k] - m)
    end
    return m + log(acc) - log(K)
end
