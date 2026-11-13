# Mechanism test 4 (card analysis/cards/step5_reach_k1a.md; Rik, 2026-11-13: "do all of them"): a copy of an exported network
# closure with a CONSTANT offset added to its deployed correction of one QoI every step, `scaling.dq_offset` (physical units,
# `ts_lstm_online.jl` `get_next_item_timeseries`: dQ .+= dq_offset after the output scaling; refused for a generator). Default:
# k1a with −0.015 target sd in E[0,6], B's measured bulk offset (critical reviews 2026-11-13). Weights, spec and every other
# scaling key are copied unchanged (Σ unchanged). As the step-4 session's `scale_closure_noise.jl`. No src change.
#
# Usage, from lib/RikFlow:  julia --project=. exp_square_HIT/tools/offset_closure.jl <src dir> <dst dir> <qoi> <offset in target sd>
#   e.g. step3/closures/LinReg1_nt_CNN_crps_nt_headcorr step3/closures/LinReg1_nt_CNN_crps_nt_headcorr_off2m0.015 2 -0.015
# Checks after writing: load_stochlstm on both; weights identical (element for element); scaling identical except dq_offset;
# the deployed closure (stochastic = false, the same warm-up and predictor) at the first predicted step on 20 random record
# windows: dQ differs by exactly the offset in the QoI and by 0 elsewhere; Σ unchanged (bd, LR identical).
using JLD2, LinearAlgebra, Printf, Random, SHA, Statistics
using RikFlow
const RF = RikFlow

length(ARGS) == 4 || error("usage: offset_closure.jl <src dir> <dst dir> <qoi> <offset in target sd>")
const TO = joinpath(@__DIR__, "..", "output", "TO_LSTM")
src, dst, iq, off_sd = joinpath(TO, ARGS[1]), joinpath(TO, ARGS[2]), parse(Int, ARGS[3]), parse(Float64, ARGS[4])
fs = joinpath(src, only(filter(f -> startswith(f, "StochLSTM_seed") && endswith(f, ".jld2"), readdir(src))))
fd = joinpath(dst, basename(fs))
ispath(fd) && error("$fd exists; not overwriting")
d = load(fs)
sc = d["scaling"]
hasproperty(sc, :dq_offset) && error("the source already carries a dq_offset")
d["spec"].emission === :generator && error("dq_offset is refused for a generator")
so = vec(sc.out_scaling.sigma)
nq = length(so)
offv = zeros(nq); offv[iq] = off_sd * so[iq]                           # physical units
sc2 = merge(sc, (; dq_offset = offv))
ex = merge(d["extras"], (; dq_offset_target_sd = off_sd, dq_offset_qoi = iq, dq_offset_source = ARGS[1],
                          dq_offset_source_sha256 = bytes2hex(open(sha256, fs)),
                          dq_offset_note = "constant offset $(off_sd) target sd added to QoI $iq's deployed correction every step; weights, Sigma unchanged"))
mkpath(dst)
jldsave(fd; spec = d["spec"], weights = d["weights"], scaling = sc2, version = d["version"], extras = ex)

# --- checks -------------------------------------------------------------------------------------------------------
a, b = RF.load_stochlstm(fs), RF.load_stochlstm(fd)
for k in fieldnames(typeof(a.weights))
    x, y = getfield(a.weights, k), getfield(b.weights, k)
    (x === nothing && y === nothing) || (x isa NamedTuple ? x == y : (size(x) == size(y) && all(x .=== y))) || error("weights.$k differs")
end
for k in propertynames(a.scaling)
    string(getproperty(a.scaling, k)) == string(getproperty(b.scaling, k)) || error("scaling.$k differs")
end
propertynames(b.scaling) == (propertynames(a.scaling)..., :dq_offset) || error("unexpected scaling keys")
# the deployed closures on the same windows of the record: the first predicted step
REF = load(joinpath(@__DIR__, "..", "..", "analysis", "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
Q, QS = REF["q"], REF["q_star"]
const maxd = let rng = Xoshiro(20261113), nwarm = 100, md = zeros(nq)
for t in 1:20
    n = rand(rng, 4001:39000)
    warm = Q[:, (n - nwarm + 1):n] .- QS[:, (n - nwarm):(n - 1)]
    ma = RF.StochLSTM(a.spec, a.weights, a.scaling; spinnup_data = warm, rng = Xoshiro(1), gate = 0.0, stochastic = false)
    mb = RF.StochLSTM(b.spec, b.weights, b.scaling; spinnup_data = copy(warm), rng = Xoshiro(1), gate = 0.0, stochastic = false)
    for k in 1:nwarm
        RF.get_next_item_timeseries(ma, QS[:, n - nwarm + k - 1]); RF.get_next_item_timeseries(mb, QS[:, n - nwarm + k - 1])
    end
    da, db = RF.get_next_item_timeseries(ma, QS[:, n]), RF.get_next_item_timeseries(mb, QS[:, n])
    md .= max.(md, abs.((db .- da) .- offv))
end
md
end
maximum(maxd) <= 1e-12 * maximum(abs, so) || error("the deployed offset is off: $maxd")
@printf("wrote %s\n  QoI %d offset %+.4f target sd = %+.6e physical (so = %.6e); relative to the reference mean level of QoI %d: see the card\n",
        fd, iq, off_sd, offv[iq], so[iq], iq)
@printf("  weights identical; scaling identical but dq_offset; first predicted step on 20 record windows: dQ(b) - dQ(a) - offset, max %s\n",
        join((@sprintf("%.1e", x) for x in maxd), " "))
@printf("  sha256 %s\n", bytes2hex(open(sha256, fd)))
