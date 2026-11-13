# A copy of an exported network closure (StochLSTM_seed<s>.jld2) with its deployed noise sd scaled by a constant
# factor on every QoI (Σ -> f² Σ): the k1a noise-level test (2026-11-13; Rik: "run the k1a noise test"; card
# analysis/cards/step4.md). Only the log-sd bias `bd` changes, by log(f); the mean (every other weight, the skip Ws,
# the scaling) and the correlation `LR` are copied byte for byte. Deployment draws out = y + exp(logd) .* (LR ε) with
# logd = Wd h + bd (src/ts_lstm.jl `cnn_window!`, `sample_emission!`), so this needs Wd = 0 (a constant emission)
# and no uclip, both checked. No src change: Snellius deploys the copy with its own code.
#
# Usage, from lib/RikFlow:  julia --project=. exp_square_HIT/tools/scale_closure_noise.jl <src dir> <dst dir> <factor> [seed=1]
#   dirs under exp_square_HIT/output/TO_LSTM, e.g. step3/closures/LinReg1_nt_CNN_crps_nt_headcorr
#   step3/closures/LinReg1_nt_CNN_crps_nt_headcorr_sx1.08 1.08
# Checks after writing: load_stochlstm on both; every weight but bd identical (===, element for element); exp(bd)
# ratio = f to 1e-14; one step of the deployed closure on random windows: the mean y bit-identical, logd shifted by
# log(f), and Σ = D R D scaled by f² to 1e-13.
using JLD2, LinearAlgebra, Printf, Random, SHA
using RikFlow
const RF = RikFlow

length(ARGS) >= 3 || error("usage: scale_closure_noise.jl <src dir> <dst dir> <factor> [seed=1]")
const TO = joinpath(@__DIR__, "..", "output", "TO_LSTM")
src, dst, f = joinpath(TO, ARGS[1]), joinpath(TO, ARGS[2]), parse(Float64, ARGS[3])
seed = length(ARGS) >= 4 ? parse(Int, replace(ARGS[4], "seed=" => "")) : 1
f > 0 || error("the factor must be positive")
fs, fd = joinpath(src, "StochLSTM_seed$seed.jld2"), joinpath(dst, "StochLSTM_seed$seed.jld2")
isfile(fs) || error("no closure at $fs")
ispath(fd) && error("$fd exists; not overwriting")

d = load(fs)
w = d["weights"]
s = d["spec"]
s.emission === :constant || error("emission is $(s.emission), not :constant: logd would not be bd alone")
s.uclip === nothing || error("uclip = $(s.uclip): a clamp on logd could break the exact scaling")
all(iszero, w.Wd) || error("Wd is not zero: logd depends on the state, scaling bd is not a constant factor")
w2 = merge(w, (; bd = w.bd .+ log(f)))
ex = merge(d["extras"], (; noise_scale = f, noise_scale_source = ARGS[1],
                          noise_scale_source_sha256 = bytes2hex(open(sha256, fs)),
                          noise_scale_note = "deployed sd x $f on every QoI (Sigma x $(f^2)); mean and correlation unchanged"))
mkpath(dst)
jldsave(fd; spec = s, weights = w2, scaling = d["scaling"], version = d["version"], extras = ex)

# --- checks -------------------------------------------------------------------------------------------------------
a, b = RF.load_stochlstm(fs), RF.load_stochlstm(fd)
for k in fieldnames(typeof(a.weights))
    k === :bd && continue
    x, y = getfield(a.weights, k), getfield(b.weights, k)
    (x === nothing && y === nothing) || (x isa NamedTuple ? x == y : (size(x) == size(y) && all(x .=== y))) ||
        error("weights.$k differs")
end
string(a.scaling) == string(b.scaling) || error("scaling differs")
r = exp.(b.weights.bd) ./ exp.(a.weights.bd)
maximum(abs, r .- f) < 1e-14 || error("exp(bd) ratio off: $r")
# one deployed step on random windows: y identical, logd + log(f), Σ = D R D x f²
rng = Xoshiro(20261113)
nin = size(a.weights.Ws, 2)
T = eltype(a.weights.Wx)
maxdy, maxdl = 0.0, 0.0
for t in 1:20
    x = reshape(randn(rng, nin), :, 1)
    sta, stb = RF.LSTMState(a.spec, T), RF.LSTMState(b.spec, T)
    RF.cnn_window!(sta, a.weights, a.spec, x)
    RF.cnn_window!(stb, b.weights, b.spec, x)
    global maxdy = max(maxdy, maximum(abs, sta.y .- stb.y))
    global maxdl = max(maxdl, maximum(abs, (stb.logd .- sta.logd) .- log(f)))
end
maxdy == 0 || error("the mean differs: max |dy| = $maxdy")
maxdl < 1e-13 || error("logd shift off by $maxdl")
R = a.weights.LR * a.weights.LR'
Sa, Sb = Diagonal(exp.(a.weights.bd)) * R * Diagonal(exp.(a.weights.bd)), Diagonal(exp.(b.weights.bd)) * R * Diagonal(exp.(b.weights.bd))
maximum(abs, Sb ./ Sa .- f^2) < 1e-13 || error("Σ ratio off")
@printf("wrote %s\n  factor %.4f: sd %s -> %s (target units)\n  mean bit-identical on 20 random windows (max |dy| = %g); logd shifted by log(f) to %.1e; Σ x %.6f elementwise to %.1e\n  sha256 %s\n",
        fd, f, join((@sprintf("%.4f", v) for v in exp.(a.weights.bd)), " "), join((@sprintf("%.4f", v) for v in exp.(b.weights.bd)), " "),
        maxdy, maxdl, f^2, maximum(abs, Sb ./ Sa .- f^2), bytes2hex(open(sha256, fd)))
