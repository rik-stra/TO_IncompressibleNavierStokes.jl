# Saving and loading a fitted M4 model.
#
# ⚠️ Not stdlib -- this is the one M4 file that needs JLD2, which is why it is separate from
# `ts_lstm.jl` and `ts_lstm_online.jl` rather than appended to either. Both of those stay loadable
# by the stdlib-only verification suite.
#
# 🔑 **Everything is stored as plain arrays and a NamedTuple of the spec's fields, never as the
# structs themselves.** JLD2 can serialise a struct, but then the file is only readable by a
# checkout whose `LSTMSpec` still has exactly those fields in that order -- and this project's
# stated convention is that the next release is breaking. A fit that outlives one field rename is
# worth the twelve lines.

"""
    save_stochlstm(path, spec, weights, scaling; extras...)

Write a fitted M4 model. `extras` are stored alongside and returned by [`load_stochlstm`](@ref) --
use it for the training window, the seed, `beta`, the history and anything else the run needs to be
reproducible.
"""
function save_stochlstm(path::AbstractString, spec::LSTMSpec, weights::LSTMWeights, scaling;
                        extras...)
    specnt = (; h = spec.hist.h, n_qoi = spec.hist.n_qoi, hist_var = spec.hist.hist_var,
              include_predictor = spec.hist.include_predictor,
              n_hidden = spec.n_hidden, n_latent = spec.n_latent, n_encoder = spec.n_encoder,
              # 🔴 `emission` was missing from this tuple until 2026-09-18, so a fit made with one
              # emission mode came back deployed under whichever mode `LSTMSpec` defaulted to --
              # silently, and with `check_shapes` passing, because the head's weights exist in
              # every mode and only the *use* of them differs. It decides whether the deployed
              # closure adds observation noise at all, so it is part of the model.
              emission = spec.emission,
              arch = spec.arch, uclip = spec.uclip)
    wnt = (; weights.Wx, weights.Wh, weights.b, weights.We, weights.be,
           weights.Bmu, weights.Bsig, weights.V1, weights.V2, weights.cdec,
           weights.Wd, weights.bd, weights.LR)
    mkpath(dirname(path))
    jldsave(path; spec = specnt, weights = wnt, scaling, version = 1, extras = (; extras...))
    return path
end

# A file written before `emission` was stored carries whatever mode was the `LSTMSpec` default
# when it was made, which was `:state_dependent`. Assume that and say so, rather than silently
# adopting today's `:none` and deploying a different model from the one that was fitted.
function _stored_emission(s)
    hasproperty(s, :emission) && return s.emission
    @warn "this fit predates the stored `emission` field; assuming :state_dependent, the default " *
          "in force when it was written. Refit if that is not what it was."
    return :state_dependent
end

"""
    load_stochlstm(path)

Read back what [`save_stochlstm`](@ref) wrote, as `(; spec, weights, scaling, extras)`.

`check_shapes` runs here, so a spec/weight mismatch is reported at load rather than as a
`DimensionMismatch` on step one of a cluster job.
"""
function load_stochlstm(path::AbstractString)
    d = load(path)
    haskey(d, "version") && d["version"] == 1 ||
        error("load_stochlstm: $path has version $(get(d, "version", missing)), expected 1")
    s = d["spec"]
    spec = LSTMSpec(;
        hist = HistorySpec(; h = s.h, n_qoi = s.n_qoi, hist_var = s.hist_var,
                           include_predictor = s.include_predictor),
        n_hidden = s.n_hidden, n_latent = s.n_latent, n_encoder = s.n_encoder,
        arch = s.arch, uclip = s.uclip, emission = _stored_emission(s))
    w = d["weights"]
    T = eltype(w.Wx)
    weights = LSTMWeights{T}(w.Wx, w.Wh, w.b, w.We, w.be, w.Bmu, w.Bsig,
                             w.V1, w.V2, w.cdec, w.Wd, w.bd, w.LR)
    check_shapes(weights, spec)
    return (; spec, weights, scaling = d["scaling"], extras = d["extras"])
end
