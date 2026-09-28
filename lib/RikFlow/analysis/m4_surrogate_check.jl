# Go/no-go for closed-loop training (plan, "next after §7g"): does a cheap surrogate of the LF
# solver's QoI step reproduce the ONLINE bias of the deployed closures?
#
#     julia --project=lib/RikFlow/training lib/RikFlow/analysis/m4_surrogate_check.jl [p] fitdir...
#
# 🔑 Why a surrogate at all: a rollout that REPLAYS the recorded q* cannot see the bias -- the replayed
# q* pins the level (§7: rollout/teacher-forced 1.00x for the dQ target). The bias lives in the
# solver's response, q*_n = S(q_{n-1}, ...): a slightly-too-large correction raises the next q*.
#
# The surrogate linearises S AROUND THE RECORDED TRAJECTORY:
#     q*_n = q*_n^rec + sum_{k=1..p} A_k (q_{n-k} - q_{n-k}^rec)
# The recorded q* keeps the OU forcing and the true dynamics along the record; the A_k (least squares
# of q* on the lagged level, 1-10 TU only, in physical units) carry a DEVIATION forward. The deployed
# closure (`StochLSTM`, exactly as online: same warm-up, seeds, 8000 steps) is run against it, and the
# 20 TU mean offset / sd ratio are computed as `m4_online_moments.jl` does -- to be compared with the
# GPU runs of the same fit. If they agree in sign and ordering across lambda, S is a usable training
# environment; if not, closed-loop training needs the real solver.

using RikFlow, JLD2, Statistics, Printf, Random, LinearAlgebra
const RF = RikFlow

p = length(ARGS) >= 1 && all(isdigit, ARGS[1]) ? parse(Int, ARGS[1]) : 1
dirs = filter(a -> !all(isdigit, a), ARGS)
TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
ref = load(get(ENV, "QOI_CACHE", joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2")))
qr, qsr, dQr = ref["q"], ref["q_star"], ref["dQ"]          # q: 6 x (N+1) incl. t = 0; q_star, dQ: 6 x N
mu, sd = vec(mean(qr; dims = 2)), vec(std(qr; dims = 2))
nq = size(qr, 1)
const NSTEP = 8000
const NWARM = 100

# --- the surrogate's A_k: least squares of q*_n - mean on [q_{n-1}; ...; q_{n-p}] over 1-10 TU ------
# alignment: q_star[:, n] is the LF step from q[:, n] (the state at the start of step n), i.e. it
# depends on q^{n-1} = q[:, n], q^{n-2} = q[:, n-1], ...
rows = (400 + p):3999
X = reduce(vcat, [qr[:, rows .- (k - 1)] for k in 1:p]); X = vcat(X, ones(1, length(rows)))
Ys = qsr[:, rows]
C = X' \ Ys'
A = [permutedims(C[((k - 1) * nq + 1):(k * nq), :]) for k in 1:p]     # A_k: q* <- q_{n-k}
res = Ys .- C' * X
@printf("surrogate p = %d: one-step R^2 of q* per QoI %s ; spectral radius of A_1 %.4f\n", p,
        join((@sprintf("%.5f", 1 - var(res[k, :]) / var(Ys[k, :])) for k in 1:nq), " "),
        maximum(abs, eigvals(A[1])))

function run_surrogate(fit, seed)
    m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = dQr[:, 1:NWARM],
                     rng = Xoshiro(seed))
    q = zeros(nq, NSTEP + 1)
    q[:, 1] = qr[:, 1]
    for n in 1:NSTEP
        qs = copy(qsr[:, n])
        for k in 1:p
            n - k + 1 >= 1 && (qs .+= A[k] * (q[:, n - k + 1] .- qr[:, n - k + 1]))
        end
        d = RF.get_next_item_timeseries(m, qs)
        q[:, n + 1] = qs .+ d
        all(isfinite, q[:, n + 1]) || return q[:, 1:n], true
    end
    return q, false
end

seeds = (234 + 1 + 2, 234 + 2 + 2, 234 + 3 + 2)       # the online replicas' closure seeds
@printf("%-26s %-9s %s | %s\n", "fit", "source", "dmean per QoI (ref sd)", "sd ratio per QoI")
for d in dirs
    fit = RF.load_stochlstm(joinpath(TO, d, "StochLSTM_seed1.jld2"))
    for (i, s) in enumerate(seeds)
        q, blew = run_surrogate(fit, s)
        if blew || size(q, 2) < NSTEP + 1
            @printf("%-26s surr r%d   diverged at step %d\n", basename(d), i, size(q, 2)); continue
        end
        dm = (vec(mean(q; dims = 2)) .- mu) ./ sd; sr = vec(std(q; dims = 2)) ./ sd
        @printf("%-26s surr r%d   %s | %s\n", basename(d), i, join((@sprintf("%6.2f", x) for x in dm), ""),
                join((@sprintf("%5.2f", x) for x in sr), ""))
    end
    # the GPU runs of the same fit, same statistic
    for f in sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d\.jld2$", x), readdir(joinpath(TO, d))))
        o = load(joinpath(TO, d, f), "data_online")
        size(o.q, 2) < NSTEP + 1 && continue
        q = o.q[:, 1:(NSTEP + 1)]
        dm = (vec(mean(q; dims = 2)) .- mu) ./ sd; sr = vec(std(q; dims = 2)) ./ sd
        @printf("%-26s online r%s %s | %s\n", basename(d), f[end-5], join((@sprintf("%6.2f", x) for x in dm), ""),
                join((@sprintf("%5.2f", x) for x in sr), ""))
    end
end
