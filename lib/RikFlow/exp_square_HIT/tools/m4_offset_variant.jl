# Write a copy of a fit with a per-QoI correction offset (closed-loop calibration, 2026-09-27).
#
#     julia --project=lib/RikFlow/training m4_offset_variant.jl <src fit dir> <dst fit dir> c1,...,c6
#
# `c` is in units of the RECORD's sd(dQ_j); it is stored in physical units as `scaling.dq_offset`,
# which the deployed closure adds to every correction. Everything else is copied unchanged.
using RikFlow, JLD2, Statistics
const RF = RikFlow
src, dst, cs = ARGS[1], ARGS[2], ARGS[3]
# optional 4th argument: a noise scale (window mode), stored as `scaling.noise_scale`
ns = length(ARGS) >= 4 && ARGS[4] != "-" ? parse(Float64, ARGS[4]) : nothing
# optional 5th argument `rel`: a STATE-PROPORTIONAL offset, c .* q* ./ mean(q*) over 1-10 TU
rel = length(ARGS) >= 5 && ARGS[5] == "rel"
# optional 6th argument: per-QoI emission-noise multipliers e1,...,e6 (`scaling.emission_scale`)
es = length(ARGS) >= 6 && ARGS[6] != "-" ? parse.(Float64, split(ARGS[6], ",")) : nothing
# optional 7th argument: per-QoI AR(1) coefficients of the emission noise (`scaling.eta_ar`)
ar = length(ARGS) >= 7 ? parse.(Float64, split(ARGS[7], ",")) : nothing
TO = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
ref = load(normpath(joinpath(@__DIR__, "..", "..", "analysis", "data",
                             "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")))
sdQ = vec(std(ref["dQ"]; dims = 2))
c = parse.(Float64, split(cs, ","))
fit = RF.load_stochlstm(joinpath(TO, src, "StochLSTM_seed1.jld2"))
sc = all(iszero, c) ? fit.scaling : merge(fit.scaling, (; dq_offset = c .* sdQ))
ns === nothing || (sc = merge(sc, (; noise_scale = ns)))
es === nothing || (sc = merge(sc, (; emission_scale = es, eta_ar = ar)))
ar === nothing || (sc = merge(sc, (; eta_ar = ar)))
rel && (sc = merge(sc, (; offset_ref = vec(mean(ref["q_star"][:, 400:3999]; dims = 2)))))
mkpath(joinpath(TO, dst))
RF.save_stochlstm(joinpath(TO, dst, "StochLSTM_seed1.jld2"), fit.spec, fit.weights, sc;
                  merge(fit.extras, (; offset_from = src, offset_sd = c, noise_scale = ns, offset_rel = rel, emission_scale = es, eta_ar = ar))...)
println("wrote $(dst) with offset (dQ sd) $(c), noise scale $(ns), state-proportional $(rel), emission scale $(es), AR $(ar)")
