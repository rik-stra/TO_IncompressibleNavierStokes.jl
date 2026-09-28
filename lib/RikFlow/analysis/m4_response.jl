# The LF solver's MEASURED response of q* to a correction, and a linear-response surrogate built on
# it (plan option A, 2026-09-25).
#
#     julia --project=lib/RikFlow/training lib/RikFlow/analysis/m4_response.jl [K] fitdir...
#
# Data: `output/TO_LSTM/response/`, written by the GPU replay runs of `12_online_StochLSTM.jl` with
# `RIKFLOW_PERT`: the solver driven by the recorded dQ (`base_a`, `base_b`: identical), or by the
# recorded dQ plus a sustained step `+-delta` in QoI j from step m on (`s<m>_q<j>_<p|m>`).
# q* is reconstructed exactly as q[:, n+1] - dQ[:, n] (the online path does not store it).
#
# 1. Noise floor: base_a vs base_b.
# 2. Step response per unit correction, S_k[:, j] = (q*+ - q*-)_{m+k} / (2 delta_j), k = 1..K, at the
#    three start times m; its spread across m, and the symmetric part (q*+ + q*- - 2 q*base) as a
#    linearity check.
# 3. The kernel G_k = S_k - S_{k-1}, averaged over m; compared with the least-squares Jacobian of
#    §7h(a) (spectral radius 1.013, which amplified deviations).
# 4. Go/no-go: q*_n = q*_n^rec + sum_{k=1..K} G_k (dQhat_{n-k} - dQ^rec_{n-k}) driving the deployed
#    closures from the online IC, against the GPU runs of the same fits.

using RikFlow, JLD2, Statistics, Printf, Random, LinearAlgebra
const RF = RikFlow

K = length(ARGS) >= 1 && all(isdigit, ARGS[1]) ? parse(Int, ARGS[1]) : 200
dirs = filter(a -> !all(isdigit, a), ARGS)
TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
RD = joinpath(TO, "response")
ref = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
qr, qsr, dQr = ref["q"], ref["q_star"], ref["dQ"]
nq = size(qr, 1); sdq = vec(std(qr; dims = 2))
fmt(v; d = 3) = join((@sprintf("%9.*f", d, x) for x in v), "")

function loadrun(tag)
    f = only(filter(x -> endswith(x, "_$(tag).jld2"), readdir(RD)))
    o = load(joinpath(RD, f), "data_online")
    n = size(o.dQ, 2)
    return (; q = o.q, dQ = o.dQ, qs = o.q[:, 2:(n + 1)] .- o.dQ)   # qs[:, n] = q*_n
end

# --- 1. noise floor ---------------------------------------------------------------------------------
A, B = loadrun("base_a"), loadrun("base_b")
d0 = maximum(abs.(A.qs .- B.qs) ./ sdq; dims = 1)
@printf("noise floor, base_a vs base_b, max over QoIs in q sd: step 100 %.2e, 500 %.2e, 1000 %.2e, 2000 %.2e\n",
        d0[100], d0[500], d0[1000], d0[2000])
@printf("replay vs record, max over QoIs in q sd:              step 100 %.2e, 500 %.2e, 1000 %.2e, 2000 %.2e\n",
        (maximum(abs.(A.qs[:, n] .- qsr[:, n]) ./ sdq) for n in (100, 500, 1000, 2000))...)

# --- 2. step responses ------------------------------------------------------------------------------
ms = (200, 1000, 1800)
S = zeros(nq, nq, K, length(ms))           # S[:, j, k, im]: response of q*_{m+k} to a unit step in j
sym = zeros(length(ms), nq)                # symmetric/antisymmetric size ratio (linearity)
for (im, m) in enumerate(ms), j in 1:nq
    P, M = loadrun("s$(m)_q$(j)_p"), loadrun("s$(m)_q$(j)_m")
    rr = m:size(P.dQ, 2)
    delta = mean(P.dQ[j, rr] .- A.dQ[j, rr])                  # the step actually applied
    ks = (m + 1):(m + K)
    anti = (P.qs[:, ks] .- M.qs[:, ks]) ./ (2delta)
    S[:, j, :, im] = anti
    symm = (P.qs[:, ks] .+ M.qs[:, ks] .- 2 .* A.qs[:, ks]) ./ (2delta)
    sym[im, j] = norm(symm) / norm(anti)
end
@printf("linearity: |symmetric| / |antisymmetric| per start time (max over j): %s\n",
        join((@sprintf("%.3f", maximum(sym[i, :])) for i in 1:length(ms)), " "))
Sm = dropdims(mean(S; dims = 4); dims = 4)
G = cat(Sm[:, :, 1:1], diff(Sm; dims = 3); dims = 3)      # impulse kernel G_k = S_k - S_{k-1}
println("step response, diagonal S_k[j, j] (q* per unit correction), mean over start times:")
for k in (1, 2, 5, 10, 25, 50, 100, 200)
    k <= K || continue
    @printf("  k = %3d  %s   spread over m: %s\n", k, fmt([Sm[j, j, k] for j in 1:nq]),
            fmt([std(S[j, j, k, :]) for j in 1:nq]))
end
@printf("one-step kernel G_1: spectral radius %.4f; diagonal %s\n", maximum(abs, eigvals(G[:, :, 1])),
        fmt(diag(G[:, :, 1])))
jldsave(joinpath(RD, "response_kernel.jld2"); G, S, ms, K)

# --- 4. linear-response surrogate, closed loop -------------------------------------------------------
const NSTEP = 8000
const NWARM = 100
mu = vec(mean(qr; dims = 2))
function closed_loop(fit, seed)
    m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = dQr[:, 1:NWARM], rng = Xoshiro(seed))
    q = zeros(nq, NSTEP + 1); q[:, 1] = qr[:, 1]
    dd = zeros(nq, NSTEP)                          # dQhat - dQ^rec
    for n in 1:NSTEP
        qs = copy(qsr[:, n])
        for k in 1:min(K, n - 1)
            qs .+= view(G, :, :, k) * view(dd, :, n - k)
        end
        d = RF.get_next_item_timeseries(m, qs)
        dd[:, n] = d .- dQr[:, n]
        q[:, n + 1] = qs .+ d
        all(isfinite, q[:, n + 1]) && maximum(abs, (q[:, n + 1] .- mu) ./ sdq) < 50 || return q[:, 1:(n + 1)], true
    end
    return q, false
end
stat(q) = ((vec(mean(q; dims = 2)) .- mu) ./ sdq, vec(std(q; dims = 2)) ./ sdq)
seeds = (234 + 1 + 2, 234 + 2 + 2, 234 + 3 + 2)
for d in dirs
    fit = RF.load_stochlstm(joinpath(TO, d, "StochLSTM_seed1.jld2"))
    for (i, s) in enumerate(seeds)
        q, blew = closed_loop(fit, s)
        blew && (@printf("%-20s LRS  r%d  diverged at step %d\n", basename(d), i, size(q, 2) - 1); continue)
        dm, sr = stat(q)
        @printf("%-20s LRS  r%d  dmean %s | sdr %s\n", basename(d), i, fmt(dm; d = 2), fmt(sr; d = 2))
    end
    for f in sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d\.jld2$", x), readdir(joinpath(TO, d))))
        o = load(joinpath(TO, d, f), "data_online")
        size(o.q, 2) < NSTEP + 1 && continue
        dm, sr = stat(o.q[:, 1:(NSTEP + 1)])
        @printf("%-20s GPU  r%s  dmean %s | sdr %s\n", basename(d), f[end-5], fmt(dm; d = 2), fmt(sr; d = 2))
    end
end
