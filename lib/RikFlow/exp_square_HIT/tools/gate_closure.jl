# A copy of a step-4 closure (a CNN on a frozen LinReg, `skip`) whose NETWORK output is gated by each QoI's own level
# q*^n: the step-4 mechanism test 2 (2026-11-13; Rik: "do all of them"; card analysis/cards/step4.md, "Mechanism test
# 2"). `mode = extremes`: the network acts only outside the central 90 % of the level; `bulk`: only inside it. The
# thresholds are the quantiles of the tracked record's q* on 1-10 TU (record columns 400-4000, the fits' window): the
# network acts fully beyond the 2.5 / 97.5 % quantiles, not at all inside 5-95 %, with half-cosine tapers between
# (`RF.level_gate_weights`); the two modes sum to the ungated closure. Everything else (weights, skip, noise, scaling)
# is copied unchanged; the gate is stored as `weights.Cv.level_gate` and applied by `StochLSTM` (src, V103).
#
# Usage, from lib/RikFlow:  julia --project=. exp_square_HIT/tools/gate_closure.jl <src dir> <dst dir> extremes|bulk [seed=1]
#   dirs under exp_square_HIT/output/TO_LSTM, e.g. step3/closures/LinReg1_nt_CNN_crps_nt_headcorr
#   step3/closures/LinReg1_nt_CNN_crps_nt_headcorr_gext extremes
using JLD2, LinearAlgebra, Printf, Random, SHA, Statistics
using RikFlow
const RF = RikFlow

length(ARGS) >= 3 || error("usage: gate_closure.jl <src dir> <dst dir> extremes|bulk [seed=1]")
const TO = joinpath(@__DIR__, "..", "output", "TO_LSTM")
src, dst, mode = joinpath(TO, ARGS[1]), joinpath(TO, ARGS[2]), Symbol(ARGS[3])
mode in (:extremes, :bulk) || error("mode must be extremes or bulk")
seed = length(ARGS) >= 4 ? parse(Int, replace(ARGS[4], "seed=" => "")) : 1
fs, fd = joinpath(src, "StochLSTM_seed$seed.jld2"), joinpath(dst, "StochLSTM_seed$seed.jld2")
isfile(fs) || error("no closure at $fs")
ispath(fd) && error("$fd exists; not overwriting")

# the thresholds: q* of the tracked record on 1-10 TU (columns 400-4000, `cfg.train_range` of the fits)
const REC = joinpath(@__DIR__, "..", "..", "analysis", "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")
qs = load(REC, "q_star")[:, 400:4000]
Qn(p) = [quantile(qs[i, :], p) for i in axes(qs, 1)]
lg = (; mode, lo_full = Qn(0.025), lo_edge = Qn(0.05), hi_edge = Qn(0.95), hi_full = Qn(0.975))

a = RF.load_stochlstm(fs)
RF.is_cnn(a.spec) && a.spec.skip && !RF.is_generator(a.spec) || error("not a step-4 closure (cnn, skip, not a generator)")
RF.cnn_level_gate(a.weights) === nothing || error("the source is gated already")
w2 = RF.with_level_gate(a.weights, lg)
RF.check_shapes(w2, a.spec)
d = load(fs)
wnt = merge(d["weights"], (; Cv = w2.Cv))
ex = merge(d["extras"], (; level_gate_mode = mode, level_gate_source = ARGS[1],
                          level_gate_source_sha256 = bytes2hex(open(sha256, fs)),
                          level_gate_note = "network output x per-QoI weight of its own level q*: $(mode)-only; thresholds = " *
                                            "quantiles 2.5/5/95/97.5 % of the record's q* on 1-10 TU (cols 400-4000)"))
mkpath(dst)
jldsave(fd; spec = d["spec"], weights = wnt, scaling = d["scaling"], version = d["version"], extras = ex)

# --- checks: the copy loads; every weight but Cv is the source's; Cv is the source's plus the gate; the gated
# window output is skip + w .* network for random windows at levels in each zone
b = RF.load_stochlstm(fd)
for k in fieldnames(typeof(a.weights))
    k === :Cv && continue
    x, y = getfield(a.weights, k), getfield(b.weights, k)
    (x === nothing && y === nothing) || x == y || error("weights.$k differs")
end
all(getfield(b.weights.Cv, k) == getfield(a.weights.Cv, k) for k in keys(a.weights.Cv)) || error("Cv differs")
RF.cnn_level_gate(b.weights) == lg || error("the stored gate differs")
string(a.scaling) == string(b.scaling) || error("scaling differs")
T = eltype(a.weights.Wx)
rng = Xoshiro(20261113)
for lev in (lg.lo_full .- 1, (lg.lo_full .+ lg.lo_edge) ./ 2, (lg.lo_edge .+ lg.hi_edge) ./ 2, lg.hi_full .+ 1)
    g = RF.level_gate_weights(RF.cnn_level_gate(b.weights), lev)
    x = reshape(randn(rng, size(a.weights.Ws, 2)), :, 1)
    s1, s0, sg = RF.LSTMState(a.spec, T), RF.LSTMState(a.spec, T), RF.LSTMState(a.spec, T)
    RF.cnn_window!(s1, a.weights, a.spec, x)
    RF.cnn_window!(s0, a.weights, a.spec, x; gate = zeros(length(g)))
    RF.cnn_window!(sg, b.weights, b.spec, x; gate = g)
    maximum(abs, sg.y .- (s0.y .+ g .* (s1.y .- s0.y))) <= 1e-12 * max(1, maximum(abs, s1.y)) || error("gated output off")
    @printf("  levels %-26s weights %s\n", lev === lg.lo_full .- 1 ? "below lo_full" : "", join((@sprintf("%.2f", v) for v in g), " "))
end
fmt(v) = join((@sprintf("%.4g", x) for x in v), " ")
@printf("wrote %s (%s-only)\n  thresholds (raw q*, Z06 E06 Z715 E715 Z1632 E1632):\n    2.5 %% %s\n    5 %%   %s\n    95 %%  %s\n    97.5 %% %s\n  sha256 %s\n",
        fd, mode, fmt(lg.lo_full), fmt(lg.lo_edge), fmt(lg.hi_edge), fmt(lg.hi_full), bytes2hex(open(sha256, fd)))
