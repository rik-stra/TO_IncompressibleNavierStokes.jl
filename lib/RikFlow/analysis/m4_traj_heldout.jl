# One teacher-forced trajectory per trained M4 model on the HELD-OUT selection window.
#
#     RIKFLOW_QOI_CACHE=... julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/analysis/m4_traj_heldout.jl
#
# ⚠️ **Runs under the training project**: converting a trained `ps` to deployed `LSTMWeights`
# needs the Lux extension, which `analysis/` deliberately does not carry. It writes
# `analysis/output/m4_traj_heldout.jld2`; `plot_m4_traj.jl` draws it under `--project=analysis`.
# The split exists because plotting and Lux cannot share an environment here.
#
# 🔴 The window, burn-in and forcing are `postrun_lstm.jl`'s, so these trajectories and the
# scores there are the same experiment: teacher-forced on the recorded history, the model never
# consuming its own output. ⚠️ ONE latent draw per model -- the spread between a model and the
# truth mixes "the model is wrong" with "this draw was unlucky". Read shape and envelope here;
# calibration is CRPS and the rank histogram (§4).
#
using RikFlow, Lux, Optimisers, Zygote, JLD2, Random, Statistics, Printf
const RF = RikFlow

const TO = get(ENV, "TO_LSTM_DIR",
                normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM")))
const OUT = get(ENV, "OUT_FILE", joinpath(@__DIR__, "output", "m4_traj_heldout.jld2"))
include(joinpath(@__DIR__, "..", "exp_square_HIT", "tools", "m4_data.jl"))
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(TO, "..", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
rec = load_m4_qois(; track_file)
println("record: q $(size(rec.q))  q_star $(size(rec.q_star))")

inputs = load(joinpath(TO,"inputs_lstm.jld2"), "inputs")
cfg = inputs[2]
a_tr, b_tr = cfg.train_range
_, in_scaling = RF._normalise(rec.q[:, a_tr:(b_tr-1)]; normalization = cfg.normalization)
println("train_range = ", cfg.train_range, "  in_scaling mu ", size(in_scaling.mu), " sigma ", size(in_scaling.sigma))

a = b_tr; b = min(size(rec.q,2), b_tr + 3600)
println("eval window = ($a, $b)")

models = Any[]   # (label, group, spec, weights)
sc = load(joinpath(TO,"stride_scan_StochLSTM2_seed1.jld2"))
spec_s = sc["spec"]
for r in sc["results"]
    push!(models, (label = "stride $(r.stride), batch $(r.batch)", group = "stride scan",
                   spec = spec_s, w = RF.LSTMWeights(r.ps, spec_s)))
end
for (cell, tag) in [(2,":storn"), (5,":vrnn"), (7,":lstm control")]
    p = joinpath(TO, "StochLSTM$cell", "StochLSTM_seed1.jld2")
    isfile(p) || (println("missing $p"); continue)
    f = RF.load_stochlstm(p)
    push!(models, (label = "StochLSTM$cell $tag", group = "cells", spec = f.spec, w = f.weights))
end
println("models: ", length(models))

# --- the window, exactly as postrun_lstm.jl builds it -------------------------------------------
spec0 = models[1].spec
qs  = RF.scale_input(rec.q[:, a:b], in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b-1)], in_scaling)
X, Y, steps = RF.build_history(spec0.hist, qss, qs)
Xc, Yc = permutedims(Float32.(X)), permutedims(Float32.(Y))
N = size(Xc,2); nq = RF.n_output(spec0)
burn = min(max(spec0.hist.h + 50, get(cfg, :burn, 0)), N - 10)
score = (burn+1):N
println("N = $N  burn = $burn  scored = $(length(score))  nq = $nq")

function one_traj(spec, w, Xc, N, burn, nq; member = 1)
    st = RF.LSTMState(spec, Float32); rng = Xoshiro(20_000 + member)
    draw = zeros(Float32, nq); out = zeros(Float32, N - burn, nq)
    for t in 1:N
        RF.lstm_step!(st, w, spec, view(Xc,:,t); rng, sample_latent = true)
        RF.sample_emission!(draw, st, w, spec, rng)
        t > burn && (out[t-burn,:] .= draw)
    end
    return out
end

preds = Array{Float32,3}(undef, length(score), nq, length(models))
for (i,m) in enumerate(models)
    preds[:,:,i] = one_traj(m.spec, m.w, Xc, N, burn, nq)
    e = preds[:,:,i] .- permutedims(Yc[:,score])
    @printf("%-28s rmse(scaled) per QoI: %s\n", m.label,
            join((@sprintf("%.3f", sqrt(mean(e[:,j].^2))) for j in 1:nq), " "))
end

truth = permutedims(Yc[:, score])
mu = vec(in_scaling.mu); sg = vec(in_scaling.sigma)
mkpath(dirname(OUT))
jldsave(OUT;
    labels = [m.label for m in models], groups = [m.group for m in models],
    steps = steps[score] .+ (a - 1), preds, truth, mu, sg,
    eval_window = (a,b), train_range = cfg.train_range, burn, member = 1)
println("wrote ", OUT)
