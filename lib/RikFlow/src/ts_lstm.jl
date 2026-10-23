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
#   (a) **A Gaussian emission head is IMPLEMENTED BUT OFF.** 🔴 `emission` defaults to `:none`
#       (Rik, 2026-09-18): the source's decoder is deterministic, `z` is its only stochasticity,
#       and a second noise channel would let "the noise is in the latent state" stop being true of
#       the fitted model. The head -- `plan.md` §3's *"Gaussian emission head"*, the L2 head of
#       `methods_overview.tex` §"State dependent covariance", `Sigma = D R D` with `log d_i`
#       linear in `h_t` -- stays reachable as `emission = :state_dependent` for when S7 wants a
#       likelihood back, and is still covered by the test suite. Nothing in the M4 experiment
#       uses it; only the `:lstm` control uses `:constant`, and only because a deterministic
#       backbone with no emission noise cannot produce an ensemble at all.
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
    LSTMSpec(; hist, n_hidden = 60, n_latent = 60, n_encoder = 60, arch = :vrnn,
             emission = :none, uclip = nothing)

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
- `emission`: one of `:none`, `:constant`, `:state_dependent` -- see [`emission_noise`](@ref).
  🔴 **`:none` by default**: the source's decoder is deterministic and `z` is its only
  stochasticity. The Gaussian head is implemented and tested but is not used by the M4 experiment.
- `uclip`: `(u_min, u_max)` bounds on the emission log-scale pre-activation, or `nothing`.
  Same device, and the same reason, as [`JointModel`](@ref)'s `uclip`: `d_i = exp(u_i)` grows
  exponentially once the input leaves the training range, and the fraction of steps on which the
  clip activates is a diagnostic that the trajectory has left the regime the head was fitted on.
- `skip`: a linear skip `y += Ws x` from the regressor to the output.
- `window`: `0` (default) carries the recurrent state across solver steps. `W > 0` resets it for
  every prediction and replays the last `W` inputs -- see [`StochLSTM`](@ref).
- `n_layers`: the number of hidden layers of `arch = :dense` (default 2); see [`is_dense`](@ref).
- `n_in`: the network's input width when `scaling.input_map` widens the regressor (`:dense` and
  `:cnn` only); `0` (default) is the regressor's width.
- `n_conv`, `kernel`: `arch = :cnn` only (2026-10-23, step 3a): the channels and the kernel length of
  its two convolutions over the `h` history steps; see [`is_cnn`](@ref).
- `posterior`: `:x` (default, the source's encoder `q(z | x)`) or `:xy` (a conditional VAE:
  `q(z | x, y)` in training, the prior `N(0, I)` at deployment).

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
    # 🔴 OFF by default (Rik, 2026-09-18). The source's decoder is deterministic; the head is kept
    # reachable as `:state_dependent` but nothing in the experiment asks for it. See the header.
    emission::Symbol = :none
    uclip::Union{Nothing,Tuple{Float64,Float64}} = nothing
    # 🔑 A LINEAR SKIP from the regressor to the output, `y += Ws x` (2026-09-24). Without it the
    # output is `V1 h + ...`, a readout of a saturating state, and the fits never learned the
    # large-coefficient cancellation between `q*^n` and `q^{n-1}` that a least-squares map on the
    # same inputs finds at R^2 0.57-0.995: every such fit sat at the h = 0 linear floor
    # (`results_LSTMS.md` §7c). With it the recurrence models what the linear map misses.
    skip::Bool = false
    # 🔑 WINDOW MODE (2026-09-24). `0` is the persistent recurrence every earlier fit uses: the
    # hidden state is carried from one solver step to the next. `W > 0` RESETS the state for every
    # prediction and replays the last `W` inputs, so the deployed prediction is exactly what a
    # training segment of length `W` produces from `h = c = 0`, and training can use every row as
    # a window end. Part of the model, not of the run: stored with the fit, and `train_stochlstm`
    # refuses `L != window`. See `StochLSTM` for the noise tying that makes the replay consistent.
    window::Int = 0
    # 🔑 THE POSTERIOR (2026-09-24). `:x` is the source's: the encoder sees only `x_t`, so the latent
    # can carry no information about the target -- the KL pulls it onto N(0, I), the decoder learns
    # to ignore it, and the one-step spread comes out 10-20x too narrow (§7f). `:xy` is a
    # conditional VAE: in TRAINING the encoder sees `[x_t; y_t]`, so `z_t` must encode the
    # unpredictable part of step t; at DEPLOYMENT `z_t` is drawn from the prior N(0, I) and the
    # encoder is not used at all.
    posterior::Symbol = :x
    # 🔑 THE PRIOR (2026-09-27). `:standard` is N(0, I), i.i.d. per step -- with it the conditional
    # encoder explains each step's residual by that step's own z, the decoder has no use for past z's,
    # and the model's noise is WHITE although the data's residual is coloured (§7j). `:learned` is
    # Chung et al.'s VRNN prior, p(z_t | h_{t-1}) = N(Wm h + bm, softplus(Ws h + bs)^2): the KL then
    # rewards predicting the innovation from the recurrent state -- LEARNED colour. Needs
    # `posterior = :xy` and a recurrent arch; deployment draws each new z from it (`StochLSTM`).
    prior::Symbol = :standard
    # 🔑 DEPTH of the dense MLP (2026-10-22, Rik: step 3a at depth 4). The number of tanh hidden
    # layers, all `n_hidden` wide; `:dense` only. Layers 2..n are stored side by side in `Wh`
    # (`H x (n - 1) H`) and their biases after the first in `b` (`n H`), so the default 2 is the
    # earlier layout exactly -- every existing fit, file and RNG stream is unchanged.
    n_layers::Int = 2
    # 🔑 INPUT WIDTH of the network (2026-10-22, Rik: the predictor increments as extra inputs).
    # `0` (default) is the regressor's own width, `nfeatures(hist)`, as before. A positive value is
    # the width of `P x` when the fit's fixed input map `scaling.input_map = P` WIDENS the regressor
    # (extra inputs that are linear combinations of it); `StochLSTM` checks that `P` is
    # `n_in x nfeatures(hist)`. `:dense` only.
    n_in::Int = 0
    # 🔑 THE CNN of step 3a (2026-10-23, Rik: "build it on the network path"; paper Sec. 6). `arch =
    # :cnn` reads the h history steps of ONE mapped regressor row as a sequence: two tanh convolutions
    # with `n_conv` channels and kernel `kernel`, flattened, the current predictor appended, then the
    # dense layers of `:dense` (two, `n_hidden` wide) and the linear output. Zero for every other arch.
    n_conv::Int = 0
    kernel::Int = 0

    function LSTMSpec(hist, n_hidden, n_latent, n_encoder, arch, emission, uclip, skip, window,
                      posterior, prior, n_layers, n_in, n_conv, kernel)
        arch in (:lstm, :vaernn, :storn, :vrnn, :dense, :cnn) ||
            error("LSTMSpec: arch must be one of :lstm, :vaernn, :storn, :vrnn, :dense, :cnn; got $(arch)")
        (arch !== :dense || window > 0) ||
            error("LSTMSpec: arch = :dense is a feed-forward map of the window -- it needs window > 0")
        if arch === :cnn
            window == 1 || error("LSTMSpec: arch = :cnn reads one regressor row (its lags are the " *
                                 "sequence) -- it needs window = 1; got $window")
            n_latent == 0 && n_encoder == 0 && posterior === :x && prior === :standard && !skip ||
                error("LSTMSpec: arch = :cnn has no latent path, encoder or skip")
            emission !== :none || error("LSTMSpec: arch = :cnn is a deterministic mean -- it needs an " *
                                        "emission head for its noise")
            n_in > 0 || error("LSTMSpec: arch = :cnn reads the mapped input (scaling.input_map): n_in > 0")
            n_layers == 2 || error("LSTMSpec: arch = :cnn has two dense layers after the convolutions")
            n_conv > 0 && kernel >= 1 || error("LSTMSpec: arch = :cnn needs n_conv > 0 and kernel >= 1")
            hist.h - 2(kernel - 1) >= 1 ||
                error("LSTMSpec: kernel $kernel leaves no position after two convolutions of $(hist.h) steps")
        else
            n_conv == 0 && kernel == 0 || error("LSTMSpec: n_conv and kernel are for arch = :cnn only")
        end
        emission in (:none, :constant, :state_dependent) ||
            error("LSTMSpec: emission must be :none, :constant or :state_dependent; got $(emission)")
        (emission !== :none || arch !== :lstm) || error(
            "LSTMSpec: arch = :lstm with emission = :none is a deterministic point predictor -- " *
            "it has no stochasticity at all and cannot produce an ensemble.")
        n_hidden > 0 || error("LSTMSpec: n_hidden must be positive")
        n_encoder >= 0 || error("LSTMSpec: n_encoder must be >= 0 (0 = linear encoder)")
        # 🔑 `arch = :dense` with `n_latent = 0` (2026-09-28, plan step 1, M3ᶠ): a DETERMINISTIC
        # residual MLP on the window -- no latent path at all -- whose only noise is the emission
        # head. It needs that head, exactly as `:lstm` does.
        (arch === :lstm || n_latent > 0 || (arch in (:dense, :cnn) && n_latent == 0)) ||
            error("LSTMSpec: arch $(arch) has a latent path, so n_latent must be positive")
        n_latent >= 0 || error("LSTMSpec: n_latent must be >= 0")
        (arch !== :dense || n_latent > 0 || emission !== :none) || error(
            "LSTMSpec: arch = :dense with n_latent = 0 and emission = :none is a deterministic " *
            "point predictor -- it has no stochasticity at all and cannot produce an ensemble.")
        (arch !== :dense || n_latent > 0 || posterior === :x) ||
            error("LSTMSpec: posterior = :xy needs a latent path; n_latent = 0 has none")
        window >= 0 || error("LSTMSpec: window must be >= 0 (0 = persistent state); got $window")
        posterior in (:x, :xy) || error("LSTMSpec: posterior must be :x or :xy; got $posterior")
        prior in (:standard, :learned) || error("LSTMSpec: prior must be :standard or :learned; got $prior")
        (prior === :standard || (posterior === :xy && arch in (:storn, :vrnn, :vaernn))) ||
            error("LSTMSpec: prior = :learned needs posterior = :xy and a recurrent latent arch")
        (posterior === :x || arch !== :lstm) ||
            error("LSTMSpec: posterior = :xy needs a latent path; arch = :lstm has none")
        n_layers >= 2 || error("LSTMSpec: n_layers must be >= 2; got $n_layers")
        (n_layers == 2 || arch === :dense) ||
            error("LSTMSpec: n_layers is the depth of the dense MLP; arch $(arch) has none")
        n_in >= 0 || error("LSTMSpec: n_in must be >= 0 (0 = the regressor's width); got $n_in")
        (n_in == 0 || arch in (:dense, :cnn)) ||
            error("LSTMSpec: n_in (a widened input) is implemented for arch = :dense and :cnn only")
        new(hist, n_hidden, n_latent, n_encoder, arch, emission, uclip, skip, window, posterior, prior,
            n_layers, n_in, n_conv, kernel)
    end
end

"""
    emission_noise(spec)

Whether the decoder adds observation noise on top of the latent path.

🔑 **`:none` is the source's design and removes the competing noise channel.** Barthel Sørensen et
al.'s decoder is deterministic: `z` is their *only* stochasticity. A state-dependent emission head
gives the model a second way to produce spread, with nothing in the objective allocating between
them -- so "the noise is in the latent state" can stop being true of the *fitted* model even where
the architecture flag says it is.

| `emission` | `Sigma` | reconstruction term | predictive spread comes from |
|---|---|---|---|
| `:none` | — | mean squared error | the latent path **only** |
| `:constant` | `D R D`, `D` constant | Gaussian log-density | latent + a fixed output scale |
| `:state_dependent` | `D(h_t) R D(h_t)` | Gaussian log-density | latent + a state-dependent scale |

🔴 **`:none` has no predictive density, so it has no likelihood** — `iwae_nll` is undefined and
refuses rather than returning a number. Ensemble scoring is unaffected: drawing `z` still gives an
ensemble, so `crps_ensemble`, the rank histogram and spread--skill all read normally. That is the
trade: the source's design, scored the way the rest of the ladder is scored, minus the likelihood.
"""
emission_noise(spec::LSTMSpec) = spec.emission !== :none

"""
    latent_sampled(spec)

Whether the architecture has a latent path at all. `false` for `:lstm` and `:cnn`.
"""
latent_sampled(spec::LSTMSpec) = spec.arch !== :lstm && spec.arch !== :cnn

"""
    latent_to_cell(spec)

Whether `z_t` is fed into the recurrence. This is the **upstream stochasticity** that separates
STORN/VRNN from the deterministic and output-only variants, and it is the property Sørensen et al.
report as the one that matters.
"""
latent_to_cell(spec::LSTMSpec) = spec.arch === :storn || spec.arch === :vrnn || spec.arch === :dense

"""
    is_dense(spec)

`arch = :dense` (Phase D, 2026-09-27): no recurrence -- a two-layer tanh MLP on the whole window,
`y_n = V1 tanh(Wh tanh(Wx [x_{n-W+1..n}; z_{n-W+1..n}] + b1) + b2) + cdec (+ Ws x_n)`. `Wx` is
`H x W (n_input + n_latent)` (all inputs first, then all latents), `Wh` is `H x H`, `b = [b1; b2]`.
With `n_layers = L > 2` (2026-10-22) there are `L` hidden layers: `Wh = [W_2 ... W_L]` is
`H x (L - 1) H` and `b = [b1; ...; bL]`; see [`dense_hidden`](@ref).
`n_latent = 0` is allowed here (and only here among the latent archs): a deterministic residual MLP
whose noise is the emission head alone (M3ᶠ, plan step 1).
"""
is_dense(spec::LSTMSpec) = spec.arch === :dense

"""
    is_cnn(spec)

`arch = :cnn` (2026-10-23, step 3a; paper Sec. 6): the history of ONE mapped regressor row `x` as a
sequence over the `h` lags, oldest first, with the channels `Cv.seq_index` picks from `x`; two tanh
convolutions (kernel `spec.kernel`, `spec.n_conv` channels, no padding), flattened; the current
predictor `x[Cv.q_index]` appended; then `:dense`'s two hidden layers (`Wx`, `Wh`, `b`) and the linear
output `V1 h + cdec` -- `cnn_mean_fit.jl`'s `mkmodel` (SEQ = lags, ACT = tanh) exactly, see
[`cnn_features`](@ref). No latent path; its noise is the emission head (and the emission options of
`StochLSTM`).
"""
is_cnn(spec::LSTMSpec) = spec.arch === :cnn

"Number of positions left after `cnn`'s two convolutions of the `h` history steps."
cnn_positions(spec::LSTMSpec) = spec.hist.h - 2(spec.kernel - 1)

"""
    dense_hidden(n_layers, Wh, b, h1, H)

Hidden layers 2..`n_layers` of the dense MLP, from the first layer's output `h1` (`H` rows, any
number of columns): `h_k = tanh(W_k h_{k-1} + b_k)` with `W_k` the `k - 1`-th `H x H` block of `Wh`
and `b_k` the `k`-th `H` entries of `b`. The training forward pass (Lux extension) and the deployed
`dense_window!` both call it, so the two cannot drift apart. Non-mutating, so Zygote differentiates it.
"""
function dense_hidden(n_layers::Int, Wh::AbstractMatrix, b::AbstractVector, h1::AbstractVecOrMat, H::Int)
    h = h1
    for k in 2:n_layers
        h = tanh.(Wh[:, ((k - 2) * H + 1):((k - 1) * H)] * h .+ b[((k - 1) * H + 1):(k * H)])
    end
    return h
end

"""
    latent_to_decoder(spec)

Whether the decoder carries the `V2 z_t` skip.
"""
latent_to_decoder(spec::LSTMSpec) = spec.arch === :vrnn || spec.arch === :vaernn

"""
    n_input(spec)

Width of the network's input row `x_t`: the regressor's, including its trailing bias entry, or
`spec.n_in` when a fixed input map widens it (see `LSTMSpec`'s `n_in`).
"""
n_input(spec::LSTMSpec) = spec.n_in > 0 ? spec.n_in : nfeatures(spec.hist)

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
n_encoder_out(spec::LSTMSpec) = spec.n_encoder > 0 ? spec.n_encoder : n_encoder_in(spec)

"""
    n_encoder_in(spec)

Width of what the encoder reads: the regressor, plus the target when `posterior = :xy`.
"""
n_encoder_in(spec::LSTMSpec) = n_input(spec) + (spec.posterior === :xy ? n_output(spec) : 0)

"""
    n_cell_input(spec)

Width of what the LSTM cell actually consumes: the regressor, plus `z` where the architecture
feeds it upstream.
"""
n_cell_input(spec::LSTMSpec) =
    is_cnn(spec) ? cnn_positions(spec) * spec.n_conv + n_output(spec) :     # the dense layers' input
    is_dense(spec) ? spec.window * (n_input(spec) + spec.n_latent) :
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
- `Ws`: the linear skip, `N_Q x n_input`, added to the decoder output; `nothing` unless
  `spec.skip`. Last, so the 13-argument form every older call site uses still constructs.
- `Cv`: `arch = :cnn` only, else `nothing` (2026-10-23): the convolutions and the sequence layout,
  `(; W1, b1, W2, b2, seq_index, q_index)` -- `W1` `kernel x n_ch x n_conv`, `W2` `kernel x n_conv x
  n_conv` (Lux's `Conv` layout), `seq_index` (`h x n_ch`) the entries of the mapped input `x` that
  form the sequence (row t = lag h - t + 1, oldest first), `q_index` (`N_Q`) those of the current
  predictor. `Wx`/`Wh`/`b`/`V1`/`cdec` are then the dense layers after the convolutions.
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
    Ws::Union{Nothing,Matrix{T}}
    # the learned prior (`spec.prior = :learned`): (; Wm, bm, Ws, bs), n_latent x H / n_latent
    P::Union{Nothing,NamedTuple}
    # `arch = :cnn`: the convolutions and the sequence layout, see above
    Cv::Union{Nothing,NamedTuple}
end
LSTMWeights{T}(Wx, Wh, b, We, be, Bmu, Bsig, V1, V2, cdec, Wd, bd, LR) where {T} =
    LSTMWeights{T}(Wx, Wh, b, We, be, Bmu, Bsig, V1, V2, cdec, Wd, bd, LR, nothing, nothing, nothing)
LSTMWeights{T}(Wx, Wh, b, We, be, Bmu, Bsig, V1, V2, cdec, Wd, bd, LR, Ws) where {T} =
    LSTMWeights{T}(Wx, Wh, b, We, be, Bmu, Bsig, V1, V2, cdec, Wd, bd, LR, Ws, nothing, nothing)
LSTMWeights{T}(Wx, Wh, b, We, be, Bmu, Bsig, V1, V2, cdec, Wd, bd, LR, Ws, P) where {T} =
    LSTMWeights{T}(Wx, Wh, b, We, be, Bmu, Bsig, V1, V2, cdec, Wd, bd, LR, Ws, P, nothing)

"""
    check_shapes(w::LSTMWeights, spec::LSTMSpec)

Assert that a weight set matches a spec. Called on load, because the alternative is a silent
`DimensionMismatch` deep inside the solver on step one of a cluster job.
"""
function check_shapes(w::LSTMWeights, spec::LSTMSpec)
    H, nin, nout, nz = spec.n_hidden, n_input(spec), n_output(spec), spec.n_latent
    ncin, nenc = n_cell_input(spec), n_encoder_out(spec)
    ff = is_dense(spec) || is_cnn(spec)  # a feed-forward net: dense layers instead of LSTM gates
    G = ff ? H : 4H                      # rows of the first layer: LSTM gates, or the dense layer
    size(w.Wx) == (G, ncin) || error("Wx is $(size(w.Wx)), expected $((G, ncin))")
    nh = ff ? (spec.n_layers - 1) * H : H                  # columns of Wh: the stacked hidden layers
    size(w.Wh) == (G, nh) || error("Wh is $(size(w.Wh)), expected $((G, nh))")
    nb = ff ? spec.n_layers * H : 4H
    if is_cnn(spec)
        cv = w.Cv
        cv === nothing && error("arch = :cnn but the convolution weights Cv are nothing")
        K, C = spec.kernel, spec.n_conv
        size(cv.seq_index, 1) == spec.hist.h ||
            error("seq_index has $(size(cv.seq_index, 1)) rows, expected h = $(spec.hist.h)")
        nch = size(cv.seq_index, 2)
        size(cv.W1) == (K, nch, C) || error("W1 is $(size(cv.W1)), expected $((K, nch, C))")
        size(cv.W2) == (K, C, C) || error("W2 is $(size(cv.W2)), expected $((K, C, C))")
        length(cv.b1) == C && length(cv.b2) == C || error("b1 / b2 must have n_conv = $C entries")
        length(cv.q_index) == nout || error("q_index has $(length(cv.q_index)) entries, expected $nout")
        all(i -> 1 <= i <= nin, cv.seq_index) && all(i -> 1 <= i <= nin, cv.q_index) ||
            error("seq_index / q_index point outside the $nin inputs")
    else
        w.Cv === nothing || error("Cv (convolution weights) is set but arch = $(spec.arch)")
    end
    length(w.b) == nb || error("b is $(length(w.b)), expected $(nb)")
    size(w.V1) == (nout, H) || error("V1 is $(size(w.V1)), expected $((nout, H))")
    length(w.cdec) == nout || error("cdec is $(length(w.cdec)), expected $(nout)")
    size(w.Wd) == (nout, H) || error("Wd is $(size(w.Wd)), expected $((nout, H))")
    length(w.bd) == nout || error("bd is $(length(w.bd)), expected $(nout)")
    size(w.LR) == (nout, nout) || error("LR is $(size(w.LR)), expected $((nout, nout))")
    if spec.n_encoder > 0 && latent_sampled(spec)
        w.We === nothing && error("n_encoder = $(spec.n_encoder) but We is nothing")
        size(w.We) == (nenc, n_encoder_in(spec)) ||
            error("We is $(size(w.We)), expected $((nenc, n_encoder_in(spec)))")
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
    if spec.prior === :learned
        w.P === nothing && error("spec.prior = :learned but the prior weights P are nothing")
        size(w.P.Wm) == (nz, H) && size(w.P.Ws) == (nz, H) && length(w.P.bm) == nz &&
            length(w.P.bs) == nz || error("learned prior weights have the wrong shape")
    end
    if spec.skip
        w.Ws === nothing && error("spec.skip but Ws is nothing")
        size(w.Ws) == (nout, nin) || error("Ws is $(size(w.Ws)), expected $((nout, nin))")
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
    lstm_step!(st, w, spec, x; rng = nothing, sample_latent = true, eps = nothing)

Advance one step. Mutates `st` and returns `(y, logd, z)` as views into it -- copy them if they
must outlive the next call.

# Arguments
- `x`: the regressor row for this step, length `n_input(spec)`, already standardised.
- `rng`: consumed **only** when `sample_latent` is true, `eps` is `nothing` and the architecture
  has a latent path.
- `sample_latent`: `false` substitutes the posterior mean `z = mu_z(x)` and draws nothing.
- `eps`: a given standard-normal draw, length `n_latent`, used as `z = mu_z + sigma_z .* eps`
  instead of drawing from `rng`. What window mode replays with (`StochLSTM`): the draw belongs to
  the physical step, not to the window it is replayed in. Takes precedence over `sample_latent`.

🔴 **`sample_latent = false` is what the warm-up uses, and it is not a performance shortcut.**
V38 requires that a closure's warm-up replay does not touch the RNG, so that a member seed means
the same thing across closures and across warm-up lengths. The posterior mean keeps that
invariant; sampling during the warm-up would break it.
"""
function lstm_step!(
    st::LSTMState{T}, w::LSTMWeights{T}, spec::LSTMSpec, x::AbstractVector;
    rng = nothing, sample_latent::Bool = true, eps = nothing,
) where {T}
    H = spec.n_hidden
    nin = n_input(spec)

    # --- the regressor, converted once ------------------------------------------------------
    @inbounds for k in 1:nin
        st.xz[k] = T(x[k])
    end
    xv = view(st.xz, 1:nin)

    # --- encoder: q(z_t | x_t) --------------------------------------------------------------
    # `posterior = :xy`: the encoder needs the target, which deployment does not have -- `z_t` is
    # drawn from the prior N(0, I) instead: `mu_z = 0`, `sigma_z = 1`, so `z = eps`
    if latent_sampled(spec) && spec.posterior === :xy
        if eps !== nothing
            @inbounds for k in eachindex(st.z)
                st.z[k] = T(eps[k])
            end
        elseif sample_latent
            rng === nothing && error("lstm_step!: sample_latent = true needs an rng")
            @inbounds for k in eachindex(st.z)
                st.z[k] = T(randn(rng))
            end
        else
            fill!(st.z, zero(T))
        end
    elseif latent_sampled(spec)
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
        if eps !== nothing
            @inbounds for k in eachindex(st.z)
                st.z[k] = st.muz[k] + st.sigz[k] * T(eps[k])
            end
        elseif sample_latent
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
    spec.skip && mul!(st.y, w.Ws, xv, one(T), one(T))
    @inbounds for k in eachindex(st.y)
        st.y[k] += w.cdec[k]
    end

    # --- emission log-scale head ------------------------------------------------------------
    # `:none` has no emission noise at all, so there is no scale to compute. `:constant` needs no
    # branch: its `Wd` is identically zero, so the same matrix product returns `bd`.
    if !emission_noise(spec)
        fill!(st.logd, zero(T))
        st.nstep += 1
        return st.y, st.logd, st.z
    end
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
    sample_emission!(out, st, w, spec, rng)

Draw one realisation of `q^n` given the step just taken: `out = y + D * LR * eps` with
`D = diag(exp.(logd))` and `eps ~ N(0, I)`.

The covariance realised is `Sigma = D R D` with `R = LR * LR'`, which is the constant conditional
correlation form of `methods_overview.tex` §"State dependent covariance" -- positive definite
whenever `R` is and the scales are finite, so the log link needs no constraint.
"""
function sample_emission!(out::AbstractVector{T}, st::LSTMState{T}, w::LSTMWeights{T},
                          spec::LSTMSpec, rng) where {T}
    # 🔴 `:none` is a deterministic decoder: the prediction IS the mean, and all the spread in an
    # ensemble comes from the latent draw upstream. Nothing is added here and the rng is untouched.
    if !emission_noise(spec)
        copyto!(out, st.y)
        return out
    end
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

"""
    learned_prior_z!(z, st, w, eps)

`prior = :learned`: `z = Wm h + bm + softplus(Ws h + bs) .* eps` from the CURRENT recurrent state
`st.h` (= h_{t-1} before the step is taken). Writes `z` in place.
"""
function learned_prior_z!(z::AbstractVector, st::LSTMState{T}, w::LSTMWeights{T}, eps) where {T}
    P = w.P
    mu = P.Wm * st.h .+ P.bm
    sg = _softplus.(P.Ws * st.h .+ P.bs)
    @inbounds for k in eachindex(z)
        z[k] = mu[k] + sg[k] * T(eps[k])
    end
    return z
end

"""
    dense_window!(st, w, spec, xwin, ewin)

`arch = :dense`, deployed: the output for the LAST column of the window, from the stored inputs
`xwin` (`n_input x W`) and latent draws `ewin` (`n_latent x W`). Writes `st.y` and `st.logd` exactly
as `lstm_step!` does for the recurrent architectures. The latent of every window step is formed
from its own stored draw, as the training forward pass forms it (`posterior = :xy`: z = eps).
"""
function dense_window!(st::LSTMState{T}, w::LSTMWeights{T}, spec::LSTMSpec, xwin::AbstractMatrix,
                       ewin::AbstractMatrix) where {T}
    H, nin, nz, W = spec.n_hidden, n_input(spec), spec.n_latent, spec.window
    size(xwin, 2) == W || error("dense_window!: the window holds $(size(xwin, 2)) steps, needs $W")
    F = zeros(T, W * (nin + nz))
    for t in 1:W
        F[((t - 1) * nin + 1):(t * nin)] .= T.(view(xwin, :, t))
        xt = view(xwin, :, t)
        if spec.posterior === :xy
            z = T.(view(ewin, :, t))
        else
            enc = spec.n_encoder > 0 ? tanh.(w.We * xt .+ w.be) : xt
            z = w.Bmu * enc .+ _softplus.(w.Bsig * enc) .* T.(view(ewin, :, t))
        end
        F[(W * nin + (t - 1) * nz + 1):(W * nin + t * nz)] .= z
    end
    h1 = tanh.(w.Wx * F .+ w.b[1:H])
    h2 = dense_hidden(spec.n_layers, w.Wh, w.b, h1, H)       # the last hidden layer
    st.h .= h2
    st.y .= w.V1 * h2 .+ w.cdec
    spec.skip && (st.y .+= w.Ws * T.(view(xwin, :, W)))
    if emission_noise(spec)
        st.logd .= w.Wd * h2 .+ w.bd
        spec.uclip !== nothing && (st.logd .= clamp.(st.logd, T(spec.uclip[1]), T(spec.uclip[2])))
    else
        fill!(st.logd, zero(T))
    end
    return st.y, st.logd
end

"""
    conv1d_tanh(A, W, b)

One tanh convolution layer as Lux's `Conv((K,), cin => cout, tanh)` computes it (NNlib's `conv`, a true
convolution: the kernel is flipped), no padding, stride 1: `A` is `L x cin`, `W` `K x cin x cout`,
`b` `cout`; returns `(L - K + 1) x cout`, `Y[t, o] = tanh(b[o] + Σ_c Σ_j W[j, c, o] A[t + K - j, c])`.
"""
function conv1d_tanh(A::AbstractMatrix{T}, W::AbstractArray{T,3}, b::AbstractVector{T}) where {T}
    K, cin, cout = size(W)
    size(A, 2) == cin || error("conv1d_tanh: the input has $(size(A, 2)) channels, the kernel $cin")
    L = size(A, 1) - K + 1
    Y = zeros(T, L, cout)
    @inbounds for o in 1:cout, t in 1:L
        s = b[o]
        for c in 1:cin, j in 1:K
            s += W[j, c, o] * A[t + K - j, c]
        end
        Y[t, o] = tanh(s)
    end
    return Y
end

"""
    cnn_features(w, spec, x)

`arch = :cnn`: the input of the dense layers for the mapped regressor row `x` -- the sequence
`A[t, c] = x[seq_index[t, c]]`, two `conv1d_tanh` layers, flattened column-major as Lux's
`FlattenLayer` does (`vec`), and the current predictor `x[q_index]` appended (Lux's
`Parallel(vcat, conv, NoOpLayer())`).
"""
function cnn_features(w::LSTMWeights{T}, spec::LSTMSpec, x::AbstractVector) where {T}
    cv = w.Cv
    S, nch = size(cv.seq_index)
    A = T[T(x[cv.seq_index[t, c]]) for t in 1:S, c in 1:nch]
    h2 = conv1d_tanh(conv1d_tanh(A, cv.W1, cv.b1), cv.W2, cv.b2)
    return vcat(vec(h2), T[T(x[i]) for i in cv.q_index])
end

"""
    cnn_window!(st, w, spec, xwin)

`arch = :cnn`, deployed: the output for the last (only) column of the window `xwin` (`n_input x 1`),
written to `st.y` and `st.logd` as `dense_window!` writes them: `cnn_features`, then the two hidden
layers (`dense_hidden`, shared with `:dense` and the training side) and `V1 h + cdec`.
"""
function cnn_window!(st::LSTMState{T}, w::LSTMWeights{T}, spec::LSTMSpec, xwin::AbstractMatrix) where {T}
    H = spec.n_hidden
    F = cnn_features(w, spec, view(xwin, :, size(xwin, 2)))
    h1 = tanh.(w.Wx * F .+ w.b[1:H])
    h2 = dense_hidden(spec.n_layers, w.Wh, w.b, h1, H)
    st.h .= h2
    st.y .= w.V1 * h2 .+ w.cdec
    st.logd .= w.Wd * h2 .+ w.bd
    spec.uclip !== nothing && (st.logd .= clamp.(st.logd, T(spec.uclip[1]), T(spec.uclip[2])))
    return st.y, st.logd
end

"""
    cnn_weights(net, spec; seq_index, q_index, bd, LR, T = Float32)

`LSTMWeights` of a CNN trained by `cnn_mean_fit.jl` (SEQ = lags): `net` is the Lux parameter tree of its
`Chain(Parallel(vcat, Chain(Conv, Conv, FlattenLayer()), NoOpLayer()), Dense, Dense, Dense)` as plain
arrays (`layer_1.layer_1.layer_{1,2}` the convolutions, `layer_{2,3}` the hidden dense layers,
`layer_4` the output); `bd` (log sd) and `LR` (lower Cholesky factor of the correlation) the constant
emission head. Plain arrays in, plain arrays out: no Lux here.
"""
function cnn_weights(net, spec::LSTMSpec; seq_index::AbstractMatrix{<:Integer},
                     q_index::AbstractVector{<:Integer}, bd::AbstractVector, LR::AbstractMatrix,
                     T::Type = Float32)
    is_cnn(spec) || error("cnn_weights: spec.arch is $(spec.arch), not :cnn")
    c1, c2 = net.layer_1.layer_1.layer_1, net.layer_1.layer_1.layer_2
    d1, d2, o = net.layer_2, net.layer_3, net.layer_4
    H, nq = spec.n_hidden, n_output(spec)
    Cv = (; W1 = Array{T,3}(c1.weight), b1 = Vector{T}(vec(c1.bias)), W2 = Array{T,3}(c2.weight),
          b2 = Vector{T}(vec(c2.bias)), seq_index = Matrix{Int}(seq_index), q_index = Vector{Int}(q_index))
    w = LSTMWeights{T}(Matrix{T}(d1.weight), Matrix{T}(d2.weight), Vector{T}(vcat(vec(d1.bias), vec(d2.bias))),
                       nothing, nothing, zeros(T, 0, 0), zeros(T, 0, 0), Matrix{T}(o.weight), nothing,
                       Vector{T}(vec(o.bias)), zeros(T, nq, H), Vector{T}(bd), Matrix{T}(LR), nothing, nothing, Cv)
    check_shapes(w, spec)
    return w
end

# ---------------------------------------------------------------------------------------------
# Sequence segmentation for BPTT
# ---------------------------------------------------------------------------------------------

"""
    segment_indices(steps; L, burn, stride = L - burn, anchor = :start)

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

- **Anchor.** `anchor = :start` (the default) lays the first segment at the start of each block
  and walks forward, so any short remainder is the block's LAST segment. `anchor = :end` lays the
  last segment flush with the block's end and walks backward, so the short remainder is the FIRST
  segment. 🔑 **Training uses `:end`**: a fit keeps only full-length segments, and under `:start`
  the dropped remainder is the most recent stretch of the training block -- the rows right before
  validation -- and its size depends on the stride (13.6% of the scored rows at the tiling stride,
  0.7% at `stride = 20`, on the 2879-row HIT block). Under `:end` the full segments always reach
  the last training row, and what is dropped is the oldest stretch. ⚠️ The COUNT dropped is the
  same under both anchors; only which rows it is changes. Validation keeps `:start`, and keeps its
  short segment, because an evaluation must not drop data.

Returns a vector of `(; rows, score)` where `rows` is the full segment (what the forward pass
consumes) and `score` is the sub-range that the loss is summed over, in increasing time order
under either anchor. Segments with no scored rows are dropped.
"""
function segment_indices(steps::AbstractVector{<:Integer}; L::Int, burn::Int,
                         stride::Int = L - burn, anchor::Symbol = :start)
    L > burn >= 0 || error("segment_indices: need L > burn >= 0; got L = $L, burn = $burn")
    stride >= 1 || error("segment_indices: stride must be >= 1; got $stride")
    anchor in (:start, :end) ||
        error("segment_indices: anchor must be :start or :end; got $(repr(anchor))")
    n = length(steps)
    segs = NamedTuple{(:rows, :score),Tuple{UnitRange{Int},UnitRange{Int}}}[]
    n == 0 && return segs

    # block boundaries: a block is a maximal run of consecutive step indices
    bstart = 1
    for i in 1:n
        isend = i == n || steps[i + 1] != steps[i] + 1
        isend || continue
        blk = bstart:i
        if length(blk) > burn && anchor === :end
            # Mirror of the forward walk below: the last segment ends at the block's last row and
            # each earlier one ends `stride` before it; a leading stub keeps what is left.
            nseg0 = length(segs)
            e = last(blk)
            while e - first(blk) + 1 > burn
                s = max(e - L + 1, first(blk))
                push!(segs, (; rows = s:e, score = (s + burn):e))
                s == first(blk) && break
                e -= stride
            end
            reverse!(view(segs, (nseg0 + 1):length(segs)))
        elseif length(blk) > burn
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

# ---------------------------------------------------------------------------------------------
# Training entry points -- implemented in ext/RikFlowLuxExt.jl
# ---------------------------------------------------------------------------------------------
#
# Declared here so the extension has something to add methods to, and so that calling one without
# the extension loaded says what is missing instead of raising a bare MethodError about a name the
# reader has never seen.

const _LUX_EXT_HINT = """
RikFlow: this is implemented in the Lux extension, which is not loaded. Run under
`lib/RikFlow/training` (or any environment with Lux, Optimisers and Zygote) and

    using RikFlow, Lux, Optimisers, Zygote

Training deliberately lives outside the package: the online path is stdlib-only so that the ~900
M0/M1/M2 jobs, and the verification suite, never pay Lux load time.
"""

"""
    init_lstm_params(rng, spec; T = Float32)

Initial training parameters for an M4 model, as a NamedTuple of plain arrays.
Implemented in the Lux extension.
"""
init_lstm_params(args...; kwargs...) = error(_LUX_EXT_HINT)

"""
    lstm_forward(spec, ps, X, epsz)

Non-mutating, AD-shaped forward pass over a whole segment. The training counterpart of
[`lstm_step!`](@ref); V41 is the test that the two agree. Implemented in the Lux extension.
"""
lstm_forward(args...; kwargs...) = error(_LUX_EXT_HINT)

"""
    elbo(spec, ps, X, Y, score, epsz; beta)

Negative ELBO on one segment, averaged over its scored columns. Implemented in the Lux extension.
"""
elbo(args...; kwargs...) = error(_LUX_EXT_HINT)

"""
    train_stochlstm(spec, X, Y, steps; kwargs...)

Fit an M4 model. Implemented in the Lux extension.
"""
train_stochlstm(args...; kwargs...) = error(_LUX_EXT_HINT)

"""
    m4_device(name) -> f

Resolve a device name into the `Array -> AbstractArray` function `train_stochlstm`'s `device`
keyword wants. `"cpu"` gives `identity`; `"cuda"`/`"gpu"` gives `CuArray`, and **throws if no
device is actually present** rather than falling back to the host silently.

One resolver for all three drivers, so `M4_DEVICE` cannot come to mean three different things.
Implemented in the Lux extension.
"""
m4_device(args...; kwargs...) = error(_LUX_EXT_HINT)

"""
    iwae_nll(spec, ps, X, Y, score; K, rng)

Held-out negative log-likelihood, as an IWAE-K bound. Implemented in the Lux extension.

🔴 A **lower** bound on `log p`, so an **upper** bound on the NLL. Not comparable to M0's exact
NLL and never in the same column as one.
"""
iwae_nll(args...; kwargs...) = error(_LUX_EXT_HINT)

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
