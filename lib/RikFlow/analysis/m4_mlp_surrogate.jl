# An MLP surrogate of the LF solver's QoI step, and its closed-loop go/no-go (Rik, 2026-09-25).
#
#     julia --project=lib/RikFlow/training lib/RikFlow/analysis/m4_mlp_surrogate.jl fitdir...
#
# 🔑 S depends ONLY on the current step (Rik): q*_{n+1} = q*_n + dQ_n + Delta, with
#     Delta ~ N( mu(u_n), diag(sigma(u_n)^2) ),   u_n = [q*_n; dQ_n]  (standardised)
# i.e. the network predicts the pure solver step on top of the corrected level, not the near-identity
# level itself (the least-squares surrogate of §7h(a) got that near-identity slope just above 1 and
# amplified every deviation). The Gaussian residual stands in for what the six QoIs cannot see -- the
# forcing and the rest of the field. Trained by Gaussian NLL on 1-10 TU (rows 400-3999), last 20% of
# rows validating, best iterate kept. A LINEAR head on the same input is fitted the same way as a
# control.
#
# Go/no-go: each surrogate drives the deployed closures in closed loop from the online IC (column 1,
# 100-step recorded warm-up, the online replicas' seeds, 8000 steps), and the 20 TU mean offset / sd
# ratio are printed beside the GPU runs of the same fits (same statistic as m4_online_moments.jl).

using RikFlow, Lux, Optimisers, Zygote, JLD2, Statistics, Printf, Random, LinearAlgebra
const RF = RikFlow

TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
ref = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
qr, qsr, dQr = ref["q"], ref["q_star"], ref["dQ"]
nq = size(qr, 1)

# --- data: u_n = [q*_n; dQ_n] -> Delta_n = q*_{n+1} - q*_n - dQ_n, rows 400-3999 ------------------
rows = 400:3998                                   # n + 1 <= 3999 stays inside 1-10 TU
U = vcat(qsr[:, rows], dQr[:, rows])
Dl = qsr[:, rows .+ 1] .- qsr[:, rows] .- dQr[:, rows]
mu_u, sd_u = vec(mean(U; dims = 2)), vec(std(U; dims = 2))
mu_d, sd_d = vec(mean(Dl; dims = 2)), vec(std(Dl; dims = 2))
Us = Float32.((U .- mu_u) ./ sd_u); Ds = Float32.((Dl .- mu_d) ./ sd_d)
ntr = floor(Int, 0.8 * size(Us, 2))
tr, va = 1:ntr, (ntr + 1):size(Us, 2)
@printf("solver step Delta: sd per QoI %s (vs sd(dQ) %s)\n", join((@sprintf("%.3g", x) for x in sd_d), " "),
        join((@sprintf("%.3g", x) for x in vec(std(dQr[:, rows]; dims = 2))), " "))

# --- the two heads -------------------------------------------------------------------------------
nh = parse(Int, get(ENV, "SURR_NH", "64"))
mlp = Chain(Dense(2nq => nh, tanh), Dense(nh => nh, tanh), Dense(nh => 2nq))
lin = Dense(2nq => 2nq)
nll(y, out) = begin
    m, ls = out[1:nq, :], out[(nq + 1):end, :]
    mean(sum(0.5 .* ((y .- m) ./ exp.(ls)) .^ 2 .+ ls; dims = 1))
end
function fit_head(model; epochs = 4000, lr = 1e-3, seed = 1)
    ps, st = Lux.setup(Xoshiro(seed), model)
    opt = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)
    loss(p, idx) = nll(Ds[:, idx], first(model(Us[:, idx], p, st)))
    best = (; v = Inf, ps = deepcopy(ps), ep = 0)
    for ep in 1:epochs
        g = Zygote.gradient(p -> loss(p, tr), ps)[1]
        opt, ps = Optimisers.update(opt, ps, g)
        if ep % 20 == 0
            v = loss(ps, va)
            v < best.v && (best = (; v, ps = deepcopy(ps), ep))
        end
    end
    return best.ps, st, best.v, best.ep, loss(best.ps, tr)
end
heads = Dict{String,Any}()
for (name, model) in (("mlp", mlp), ("lin", lin))
    ps, st, v, ep, t = fit_head(model)
    o = first(model(Us[:, va], ps, st))
    r2 = [1 - mean(abs2, Ds[k, va] .- o[k, :]) / var(Ds[k, va]) for k in 1:nq]
    @printf("%-4s head: best val NLL %.4f at epoch %d (train %.4f); val R^2 of the step mean %s\n", name, v, ep, t,
            join((@sprintf("%.3f", x) for x in r2), " "))
    heads[name] = (; model, ps, st)
end
mkpath(joinpath(TO, "surrogate"))
jldsave(joinpath(TO, "surrogate", "mlp_S.jld2"); ps_mlp = heads["mlp"].ps, ps_lin = heads["lin"].ps, nh,
        mu_u, sd_u, mu_d, sd_d, rows)

"One surrogate step: q*_{n+1} from (q*_n, dQ_n), with the residual drawn from `rng` (or its mean)."
function surr_step(h, qs, dq, rng; noise = true)
    u = Float32.((vcat(qs, dq) .- mu_u) ./ sd_u)
    o = Float64.(vec(first(h.model(reshape(u, :, 1), h.ps, h.st))))
    z = o[1:nq] .+ (noise ? exp.(o[(nq + 1):end]) .* randn(rng, nq) : zeros(nq))
    return qs .+ dq .+ mu_d .+ sd_d .* z
end

const NSTEP = 8000
const NWARM = 100
mu, sd = vec(mean(qr; dims = 2)), vec(std(qr; dims = 2))
function closed_loop(h, fit, seed; noise = true)
    m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = dQr[:, 1:NWARM], rng = Xoshiro(seed))
    srng = Xoshiro(10_000 + seed)
    q = zeros(nq, NSTEP + 1); q[:, 1] = qr[:, 1]
    qs = copy(qsr[:, 1])
    for n in 1:NSTEP
        d = RF.get_next_item_timeseries(m, qs)
        q[:, n + 1] = qs .+ d
        all(isfinite, q[:, n + 1]) && maximum(abs, (q[:, n + 1] .- mu) ./ sd) < 50 || return q[:, 1:(n + 1)], true
        qs = surr_step(h, qs, d, srng; noise)
    end
    return q, false
end
stat(q) = ((vec(mean(q; dims = 2)) .- mu) ./ sd, vec(std(q; dims = 2)) ./ sd)
fmt(v) = join((@sprintf("%6.2f", x) for x in v), "")

# sanity: the surrogate with the RECORD's own corrections replayed step by step (open-loop in dQ)
for name in ("mlp", "lin")
    h = heads[name]; rng = Xoshiro(7)
    q = zeros(nq, NSTEP + 1); q[:, 1] = qr[:, 1]; qs = copy(qsr[:, 1])
    for n in 1:NSTEP
        q[:, n + 1] = qs .+ dQr[:, n]; qs = surr_step(h, qs, dQr[:, n], rng)
    end
    dm, sr = stat(q)
    @printf("%-4s surrogate, recorded dQ replayed:        dmean %s | sdr %s\n", name, fmt(dm), fmt(sr))
end

seeds = (234 + 1 + 2, 234 + 2 + 2, 234 + 3 + 2)
for d in ARGS
    fit = RF.load_stochlstm(joinpath(TO, d, "StochLSTM_seed1.jld2"))
    for name in ("mlp", "lin"), (i, s) in enumerate(seeds)
        q, blew = closed_loop(heads[name], fit, s)
        if blew
            @printf("%-20s %-4s r%d  diverged at step %d\n", basename(d), name, i, size(q, 2) - 1); continue
        end
        dm, sr = stat(q)
        @printf("%-20s %-4s r%d  dmean %s | sdr %s\n", basename(d), name, i, fmt(dm), fmt(sr))
    end
    for f in sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d\.jld2$", x), readdir(joinpath(TO, d))))
        o = load(joinpath(TO, d, f), "data_online")
        size(o.q, 2) < NSTEP + 1 && continue
        dm, sr = stat(o.q[:, 1:(NSTEP + 1)])
        @printf("%-20s GPU  r%s  dmean %s | sdr %s\n", basename(d), f[end-5], fmt(dm), fmt(sr))
    end
end
